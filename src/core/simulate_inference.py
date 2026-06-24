import torch
import torch.nn as nn
import numpy as np
import h5py
import os

from src.core.affect_bridge import affect_to_mood_match
from src.core.pipeline import MAESTROInferencePipeline

# =====================================================================
# Define LSTM Model (as built in the training notebook)
# =====================================================================
class SingleTargetLSTM(nn.Module):
    """LSTM regressor for a single continuous physiological target."""
    def __init__(self, input_size: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden,
            num_layers  = 1,
            batch_first = True,
        )
        self.ln   = nn.LayerNorm(hidden)
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
        lstm_out, _  = self.lstm(x)
        last_hidden  = lstm_out[:, -1, :]
        last_hidden  = self.ln(last_hidden)
        last_hidden  = self.drop(last_hidden)
        return self.head(last_hidden)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # =====================================================================
    # 1. Load the dataset sample
    # =====================================================================
    print("\n--- 1. Loading Preprocessed Data ---")
    data_path = 'datasets/case_processed.h5'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    with h5py.File(data_path, 'r') as f:
        x_data = f['x']
        total_windows = x_data.shape[0]
        seq_len = x_data.shape[1]
        
        print(f"Dataset contains {total_windows} windows of length {seq_len} (BVP, GSR, SKT).")
        
        # The baseline is strictly the first window of the session (Video 10 - startVid)
        base_idx = 0
        
        pred_idx_str = input(f"Select a window index [1-{total_windows-1}] to use as the PREDICTION input (default=10): ").strip()
        pred_idx = int(pred_idx_str) if pred_idx_str.isdigit() and 1 <= int(pred_idx_str) < total_windows else min(10, total_windows - 1)
        
        base_sample = x_data[base_idx]
        pred_sample = x_data[pred_idx]
        
    print(f"\nUsing Window {base_idx} (Start of Session) as Baseline, and Window {pred_idx} for Prediction.")
    
    # Repeat the tiny sample to create a stable ~16-second window for the signal filters
    base_bvp = np.tile(base_sample[:, 0], 100)
    base_gsr = np.tile(base_sample[:, 1], 100)
    base_skt = np.tile(base_sample[:, 2], 100)
    
    pred_bvp = np.tile(pred_sample[:, 0], 100)
    pred_gsr = np.tile(pred_sample[:, 1], 100)
    pred_skt = np.tile(pred_sample[:, 2], 100)

    # =====================================================================
    # 2. Setup and Load LSTM Models
    # =====================================================================
    print("\n--- 2. Setting up LSTM Models ---")
    val_path = 'models/lstm_2/fold_21_sub21_valence.pt'
    aro_path = 'models/lstm_1/fold_15_sub15_arousal.pt'

    val_checkpoint = torch.load(val_path, map_location=device)
    aro_checkpoint = torch.load(aro_path, map_location=device)
    
    v_n_features = val_checkpoint.get('n_features', 57)
    a_n_features = aro_checkpoint.get('n_features', 57)
    
    valence_model = SingleTargetLSTM(input_size=v_n_features).to(device)
    arousal_model = SingleTargetLSTM(input_size=a_n_features).to(device)

    valence_model.load_state_dict(val_checkpoint['model_state_dict'])
    arousal_model.load_state_dict(aro_checkpoint['model_state_dict'])
    print("LSTM weights loaded successfully.")

    # =====================================================================
    # 3. Instantiate Pipeline & Run Calibration
    # =====================================================================
    print("\n--- 3. Running MAESTROInferencePipeline ---")
    
    cfg = {
        'fs_physio': 1000,
        'fs_annot': 20,
        'baseline_video_id': 10,
        'label_neutral': 5.0
    }
    
    pipeline = MAESTROInferencePipeline(
        model_v=valence_model,
        model_a=arousal_model,
        n_features=v_n_features,
        cfg=cfg
    )
    
    # We use the selected window as the baseline to simulate a valid session
    baseline_signals = {
        'bvp': base_bvp,
        'gsr': base_gsr,
        'skt': base_skt
    }
    pipeline.calibrate(baseline_signals)

    # =====================================================================
    # 4. Predict Valence/Arousal from Window
    # =====================================================================
    window_signals = {
        'bvp': pred_bvp,
        'gsr': pred_gsr,
        'skt': pred_skt
    }
    
    result = pipeline.predict(window_signals)
    
    val_pred_norm = result['valence_norm']
    aro_pred_norm = result['arousal_norm']
    
    print(f"Computed Valence Output (Normalized): {val_pred_norm:.4f}")
    print(f"Computed Arousal Output (Normalized): {aro_pred_norm:.4f}")
    print(f"Inverse-Transformed Valence (Joystick): {result['valence']:.4f}")
    print(f"Inverse-Transformed Arousal (Joystick): {result['arousal']:.4f}")

    # =====================================================================
    # 5. Map the Affect space to Mood Class
    # =====================================================================
    print("\n--- 4. Mapping to Mood ---")
    mood_match = affect_to_mood_match(valence_joystick=result['valence'], arousal_joystick=result['arousal'])
    print(f"Deducted Mood Class: {mood_match.mood_name.upper()} (ID: {mood_match.mood_id})")
    print(f"Distance to nearest prototype in circumplex space: {mood_match.distance:.4f}")

    # =====================================================================
    # 6. Feed to Generator using its exact inference script
    # =====================================================================
    print("\n--- 5. Passing to Generator ---")
    
    available_models = [
        "generator",
        "minimal_generator",
        "mood_generator",
        "neg_cfg_generator"
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
        selected_model = "neg_cfg_generator"
        
    import subprocess
    cmd = [
        "python", "-m", f"src.models.{selected_model}",
        "generate",
        "--valence", str(round(val_pred_norm, 4)),
        "--arousal", str(round(aro_pred_norm, 4)),
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
