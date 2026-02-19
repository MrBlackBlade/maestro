"""
Centralized configuration for the Emotion → MIDI Transformer.
All hyperparameters, paths, and settings in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class TokenizerConfig:
    """MIDI tokenizer vocabulary settings."""
    num_pitches: int = 128         # MIDI pitches 0-127
    num_velocity_bins: int = 32    # Velocity quantized to 32 bins
    num_time_shift_bins: int = 100 # Time shifts: 10ms to 1000ms in 10ms steps
    num_instruments: int = 128     # General MIDI program numbers 0-127
    max_seq_len: int = 2048        # Maximum token sequence length

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including all token types and special tokens."""
        return (
            self.num_pitches          # NOTE_ON
            + self.num_pitches        # NOTE_OFF
            + self.num_velocity_bins  # VELOCITY
            + self.num_time_shift_bins  # TIME_SHIFT
            + self.num_instruments    # INSTRUMENT
            + 1                       # BAR
            + 3                       # BOS, EOS, PAD
        )


@dataclass
class ModelConfig:
    """Transformer model hyperparameters."""
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 2048
    emotion_dim: int = 2  # valence + arousal


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 100
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 10
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42


@dataclass
class PathConfig:
    """File and directory paths."""
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    dataset_dir: Path = field(default=None)
    memo_audio_dir: Path = field(default=None)
    memo_annotations_dir: Path = field(default=None)
    memo_features_dir: Path = field(default=None)
    transcribed_midi_dir: Path = field(default=None)
    processed_data_path: Path = field(default=None)
    checkpoint_dir: Path = field(default=None)
    output_dir: Path = field(default=None)

    def __post_init__(self):
        if self.dataset_dir is None:
            self.dataset_dir = self.project_root / "datasets" / "Memo2496"
        if self.memo_audio_dir is None:
            self.memo_audio_dir = self.dataset_dir / "MusicRawData"
        if self.memo_annotations_dir is None:
            self.memo_annotations_dir = self.dataset_dir / "Annotations"
        if self.memo_features_dir is None:
            self.memo_features_dir = self.dataset_dir / "Features"
        if self.transcribed_midi_dir is None:
            self.transcribed_midi_dir = self.dataset_dir / "transcribed_midi"
        if self.processed_data_path is None:
            self.processed_data_path = self.dataset_dir / "emotion_midi_processed.h5"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "checkpoints"
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"


@dataclass
class GenerateConfig:
    """Inference/generation settings."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    max_gen_len: int = 1024  # Max tokens to generate


@dataclass
class MaestroConfig:
    """Top-level config combining all sub-configs."""
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# Global default config instance
DEFAULT_CONFIG = MaestroConfig()
