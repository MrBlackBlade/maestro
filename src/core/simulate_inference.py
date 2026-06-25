"""Offline and shared helpers for MAESTRO physiological → affect inference."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn

from src.core.affect_bridge import MoodMatch, affect_to_mood_match
from src.core.config import Config
from src.core.pipeline import MAESTROInferencePipeline

# ---------------------------------------------------------------------------
# Defaults (match CASE preprocessing / training notebook)
# ---------------------------------------------------------------------------
DEFAULT_CFG: dict[str, Any] = {
    "fs_physio": 1000,
    "fs_annot": 20,
    "baseline_video_id": 10,
    "label_neutral": 5.0,
}

DEFAULT_VALENCE_CKPT = Config.MODEL_CKPT_DIR / "lstm_2" / "fold_21_sub21_valence.pt"
DEFAULT_AROUSAL_CKPT = Config.MODEL_CKPT_DIR / "lstm_1" / "fold_15_sub15_arousal.pt"
DEFAULT_DATASET_PATH = Config.DATASETS_DIR / "case_processed.h5"

# Feature extractors (heartpy / neurokit) need a sufficiently long window.
# simulate_inference tiles each short CASE window 100× (~16 s at 1 kHz).
DEFAULT_TILE_REPEAT = 100
MIN_SIGNAL_SAMPLES = 1_000
RECOMMENDED_SIGNAL_SAMPLES = 16_000


# =====================================================================
# LSTM model (as built in the training notebook)
# =====================================================================
class SingleTargetLSTM(nn.Module):
    """LSTM regressor for a single continuous physiological target."""

    def __init__(self, input_size: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.ln(last_hidden)
        last_hidden = self.drop(last_hidden)
        return self.head(last_hidden)


@dataclass(frozen=True)
class LoadedModels:
    valence_model: nn.Module
    arousal_model: nn.Module
    scaler_v: tuple[np.ndarray, np.ndarray]
    scaler_a: tuple[np.ndarray, np.ndarray]
    n_features: int
    device: torch.device


@dataclass(frozen=True)
class InferenceResult:
    valence: float
    arousal: float
    valence_norm: float
    arousal_norm: float
    mood: MoodMatch
    music_params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "valence_norm": self.valence_norm,
            "arousal_norm": self.arousal_norm,
            "mood": {
                "id": self.mood.mood_id,
                "name": self.mood.mood_name,
                "distance": self.mood.distance,
                "circumplex_valence": self.mood.circumplex_valence,
                "circumplex_arousal": self.mood.circumplex_arousal,
            },
            "music_params": self.music_params,
        }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_signal_dict(signals: dict[str, Any], min_samples: int = MIN_SIGNAL_SAMPLES) -> None:
    """Ensure BVP/GSR/SKT arrays exist, match in length, and are long enough."""
    required = ("bvp", "gsr", "skt")
    missing = [k for k in required if k not in signals]
    if missing:
        raise ValueError(f"Missing signal keys: {missing}. Expected keys: {list(required)}")

    lengths = {k: len(np.asarray(signals[k], dtype=np.float64).ravel()) for k in required}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"bvp/gsr/skt must have equal length, got {lengths}")

    n = next(iter(unique_lengths))
    if n < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples per channel at fs_physio=1000; got {n}."
        )


def normalize_signal_dict(signals: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert incoming lists/arrays to 1-D float64 numpy arrays."""
    validate_signal_dict(signals)
    return {k: np.asarray(signals[k], dtype=np.float64).ravel() for k in ("bvp", "gsr", "skt")}


def tile_window_sample(sample: np.ndarray, repeat: int = DEFAULT_TILE_REPEAT) -> dict[str, np.ndarray]:
    """Expand a short CASE window (T×3) into long 1-D streams for each modality."""
    return {
        "bvp": np.tile(sample[:, 0], repeat),
        "gsr": np.tile(sample[:, 1], repeat),
        "skt": np.tile(sample[:, 2], repeat),
    }


def generate_dummy_signals(
    n_samples: int = RECOMMENDED_SIGNAL_SAMPLES,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Synthetic BVP/GSR/SKT for WebSocket testing when no HDF5 file is available."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_samples / DEFAULT_CFG["fs_physio"], n_samples, endpoint=False)
    return {
        "bvp": 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.05 * rng.standard_normal(n_samples),
        "gsr": 0.3 + 0.1 * np.sin(2 * np.pi * 0.05 * t) + 0.02 * rng.standard_normal(n_samples),
        "skt": 32.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t) + 0.01 * rng.standard_normal(n_samples),
    }


def load_lstm_models(
    valence_path: os.PathLike | str = DEFAULT_VALENCE_CKPT,
    arousal_path: os.PathLike | str = DEFAULT_AROUSAL_CKPT,
    device: torch.device | None = None,
) -> LoadedModels:
    device = device or get_device()
    val_path = Path(valence_path)
    aro_path = Path(arousal_path)

    val_checkpoint = torch.load(val_path, map_location=device)
    aro_checkpoint = torch.load(aro_path, map_location=device)

    n_features = int(val_checkpoint.get("n_features", 57))

    valence_model = SingleTargetLSTM(input_size=n_features).to(device)
    arousal_model = SingleTargetLSTM(input_size=n_features).to(device)

    valence_model.load_state_dict(val_checkpoint["model_state_dict"])
    arousal_model.load_state_dict(aro_checkpoint["model_state_dict"])

    valence_model.eval()
    arousal_model.eval()
    
    # FIX 1: Derive Standard Scaler paths from checkpoint filename patterns
    def _get_scaler_paths(ckpt_path: Path) -> tuple[Path, Path]:
        prefix = ckpt_path.stem.rsplit('_', 1)[0]  # Extracts fold_15_sub15
        mean_p = ckpt_path.parent / f"{prefix}_scaler_mean.npy"
        scale_p = ckpt_path.parent / f"{prefix}_scaler_scale.npy"
        return mean_p, scale_p

    v_mean_path, v_scale_path = _get_scaler_paths(val_path)
    a_mean_path, a_scale_path = _get_scaler_paths(aro_path)

    scaler_v = (np.load(v_mean_path), np.load(v_scale_path))
    scaler_a = (np.load(a_mean_path), np.load(a_scale_path))

    return LoadedModels(
        valence_model=valence_model,
        arousal_model=arousal_model,
        scaler_v=scaler_v,
        scaler_a=scaler_a,
        n_features=n_features,
        device=device,
    )


def create_pipeline(
    models: LoadedModels,
    cfg: dict[str, Any] | None = None,
) -> MAESTROInferencePipeline:
    return MAESTROInferencePipeline(
        model_v=models.valence_model,
        model_a=models.arousal_model,
        n_features=models.n_features,
        cfg=cfg or DEFAULT_CFG.copy(),
        scaler_v=models.scaler_v,
        scaler_a=models.scaler_a,
    )


def load_dataset_windows(
    data_path: os.PathLike | str = DEFAULT_DATASET_PATH,
    base_idx: int = 0,
    pred_idx: int = 10,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Load baseline + prediction windows from case_processed.h5."""
    with h5py.File(data_path, "r") as f:
        x_data = f["x"]
        total_windows = int(x_data.shape[0])
        pred_idx = min(max(1, pred_idx), total_windows - 1)
        base_sample = x_data[base_idx]
        pred_sample = x_data[pred_idx]

    meta = {
        "total_windows": total_windows,
        "base_idx": base_idx,
        "pred_idx": pred_idx,
        "seq_len": int(base_sample.shape[0]),
    }
    baseline = tile_window_sample(base_sample)
    prediction = tile_window_sample(pred_sample)
    return baseline, prediction, meta


def run_inference(
    pipeline: MAESTROInferencePipeline,
    window_signals: dict[str, Any],
) -> InferenceResult:
    """Run predict() and map outputs to mood + serializable dict."""
    signals = normalize_signal_dict(window_signals)
    result = pipeline.predict(signals)
    mood = affect_to_mood_match(
        valence_joystick=result["valence"],
        arousal_joystick=result["arousal"],
    )
    return InferenceResult(
        valence=result["valence"],
        arousal=result["arousal"],
        valence_norm=result["valence_norm"],
        arousal_norm=result["arousal_norm"],
        mood=mood,
        music_params=result["music_params"],
    )


def calibrate_and_predict(
    pipeline: MAESTROInferencePipeline,
    baseline_signals: dict[str, Any],
    window_signals: dict[str, Any],
) -> InferenceResult:
    pipeline.calibrate(normalize_signal_dict(baseline_signals))
    return run_inference(pipeline, window_signals)


def run_generator_subprocess(
    valence_norm: float,
    arousal_norm: float,
    model_name: str = "neg_cfg_generator",
) -> None:
    cmd = [
        "python",
        "-m",
        f"src.models.{model_name}",
        "generate",
        "--valence",
        str(round(valence_norm, 4)),
        "--arousal",
        str(round(arousal_norm, 4)),
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    print("\n--- 1. Loading Preprocessed Data ---")
    data_path = DEFAULT_DATASET_PATH
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    with h5py.File(data_path, "r") as f:
        total_windows = f["x"].shape[0]

    pred_idx_str = input(
        f"Select a window index [1-{total_windows - 1}] to use as the PREDICTION input (default=10): "
    ).strip()
    pred_idx = (
        int(pred_idx_str)
        if pred_idx_str.isdigit() and 1 <= int(pred_idx_str) < total_windows
        else min(10, total_windows - 1)
    )

    baseline_signals, window_signals, meta = load_dataset_windows(data_path, pred_idx=pred_idx)
    print(
        f"\nUsing Window {meta['base_idx']} (Start of Session) as Baseline, "
        f"and Window {meta['pred_idx']} for Prediction."
    )

    print("\n--- 2. Setting up LSTM Models ---")
    models = load_lstm_models(device=device)
    pipeline = create_pipeline(models)
    print("LSTM weights loaded successfully.")

    print("\n--- 3. Running MAESTROInferencePipeline ---")
    result = calibrate_and_predict(pipeline, baseline_signals, window_signals)

    print(f"Computed Valence Output (Normalized): {result.valence_norm:.4f}")
    print(f"Computed Arousal Output (Normalized): {result.arousal_norm:.4f}")
    print(f"Inverse-Transformed Valence (Joystick): {result.valence:.4f}")
    print(f"Inverse-Transformed Arousal (Joystick): {result.arousal:.4f}")

    print("\n--- 4. Mapping to Mood ---")
    print(f"Deducted Mood Class: {result.mood.mood_name.upper()} (ID: {result.mood.mood_id})")
    print(f"Distance to nearest prototype in circumplex space: {result.mood.distance:.4f}")

    print("\n--- 5. Passing to Generator ---")
    available_models = [
        "generator",
        "minimal_generator",
        "mood_generator",
        "neg_cfg_generator",
        "chrollo"
    ]
    print("Available generator models:")
    for i, model in enumerate(available_models, 1):
        print(f"  [{i}] {model}")

    choice = input("Select a model to use for generation [1-4] (default=4): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(available_models):
        selected_model = available_models[int(choice) - 1]
    elif choice in available_models:
        selected_model = choice
    else:
        selected_model = "chrollo"

    run_generator_subprocess(result.valence_norm, result.arousal_norm, selected_model)


if __name__ == "__main__":
    main()