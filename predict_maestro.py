import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import neurokit2 as nk
import heartpy as hp
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Model Architecture
# ==========================================
class LSTMEmotionRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.bn = nn.BatchNorm1d(self.hidden_size)
        
        def head():
            return nn.Sequential(
                nn.Linear(self.hidden_size, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )
            
        self.valence_head = head()
        self.arousal_head = head()
        
    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :] 
        last_step = self.bn(last_step)
        return self.valence_head(last_step), self.arousal_head(last_step)

# ==========================================
# 2. Preprocessing & Normalization
# ==========================================
class PhysiologicalPreprocessor:
    def __init__(self, fs: int = 1000):
        self.fs = fs

    def _bandpass(self, sig, low, high):
        nyq = self.fs / 2
        b, a = butter(4, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, sig)

    def _lowpass(self, sig, cutoff):
        nyq = self.fs / 2
        b, a = butter(4, cutoff / nyq, btype='low')
        return filtfilt(b, a, sig)

    def preprocess(self, df):
        processed = df.copy()
        if 'bvp' in df.columns:
            processed['bvp'] = self._bandpass(df['bvp'].values, 0.5, 3.5)
        if 'eda' in df.columns:
            processed['eda'] = self._lowpass(df['eda'].values, 3.0)
        if 'skt' in df.columns:
            processed['skt'] = signal.medfilt(df['skt'].values, kernel_size=51)
        return processed

class BaselineReductionNormalizer:
    def __init__(self, fs: int = 1000, baseline_sec: int = 60):
        self.baseline_samples = fs * baseline_sec

    def normalize(self, df: pd.DataFrame, cols: list):
        df_norm = df.copy()
        for col in cols:
            if col in df_norm.columns:
                baseline_mean = df_norm[col].iloc[:self.baseline_samples].mean()
                df_norm[col] = df_norm[col] - baseline_mean
        return df_norm

# ==========================================
# 3. Feature Extraction
# ==========================================
class FeatureExtractor:
    def __init__(self, fs=1000):
        self.fs = fs
        self.stat_funs = {
            'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max,
            'range': lambda x: np.max(x) - np.min(x),
            'p25': lambda x: np.percentile(x, 25), 'p75': lambda x: np.percentile(x, 75)
        }

    def extract_window(self, sig_dict):
        feats = {}
        for name, sig in sig_dict.items():
            for s_name, func in self.stat_funs.items():
                feats[f"{name}_{s_name}"] = func(sig)
        
        # BVP/HRV Features
        if 'bvp' in sig_dict:
            try:
                working_data, measures = hp.process(sig_dict['bvp'], sample_rate=self.fs)
                feats.update({f"hrv_{k}": v for k, v in measures.items() if isinstance(v, (int, float))})
            except: pass

        # EDA Features
        if 'eda' in sig_dict:
            try:
                signals, _ = nk.eda_process(sig_dict['eda'], sampling_rate=self.fs)
                feats['eda_phasic_mean'] = signals['EDA_Phasic'].mean()
                feats['eda_tonic_mean'] = signals['EDA_Tonic'].mean()
            except: pass
            
        return feats

# ==========================================
# 4. Main Inference Logic
# ==========================================
def predict_subject(csv_path, model_path, device='cpu'):
    # Load Data
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    
    # Preprocess
    preprocessor = PhysiologicalPreprocessor(fs=1000)
    df_clean = preprocessor.preprocess(df)
    
    # Normalize (Baseline Reduction)
    normalizer = BaselineReductionNormalizer(fs=1000, baseline_sec=60)
    df_norm = normalizer.normalize(df_clean, ['bvp', 'eda', 'skt'])
    
    # Windowing & Feature Extraction (60s window, 30s stride)
    extractor = FeatureExtractor(fs=1000)
    win_size, stride = 60 * 1000, 30 * 1000
    all_features = []
    
    print("Extracting features...")
    for start in range(0, len(df_norm) - win_size, stride):
        window = df_norm.iloc[start : start + win_size]
        sigs = {k: window[k].values for k in ['bvp', 'eda', 'skt'] if k in window.columns}
        all_features.append(extractor.extract_window(sigs))
    
    features_df = pd.DataFrame(all_features).fillna(0)
    X = torch.FloatTensor(features_df.values).unsqueeze(1).to(device) # [Batch, Seq=1, Features]

    # Load Model
    input_dim = X.shape[2]
    model = LSTMEmotionRegressor(input_size=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        valences, arousals = model(X)
    
    results = pd.DataFrame({
        'Window_Start_Sec': np.arange(len(all_features)) * 30,
        'Predicted_Valence': valences.squeeze().cpu().numpy(),
        'Predicted_Arousal': arousals.squeeze().cpu().numpy()
    })
    
    return results

if __name__ == "__main__":
    # Example Usage
    MODEL_FILE = "E:\Graduation Project\maestro\outputs\maestro_affective_model.pt"
    SUBJECT_CSV = "E:\Graduation Project\maestro\data\interpolated\physiological\sub_7.csv" # Update path accordingly
    
    if Path(MODEL_FILE).exists() and Path(SUBJECT_CSV).exists():
        predictions = predict_subject(SUBJECT_CSV, MODEL_FILE)
        print("\n--- Predictions ---")
        print(predictions.head(10))
        predictions.to_csv("subject_predictions.csv", index=False)
        print("\nSaved results to subject_predictions.csv")
    else:
        print("Error: Ensure model file and subject CSV path are correct.")