import pandas as pd
import numpy as np
import h5py
import neurokit2 as nk
import os
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_DIR = "E:/Gam3a/Grad-proj"
CASE_DIR = f"{DATA_DIR}/CASE_full"
CASE_DATA_PATH = f"{CASE_DIR}/data/interpolated/physiological"
CASE_ANNOTATION_PATH = f"{CASE_DIR}/data/interpolated/annotations"

JOYSTICK_DELAY_MS = 8000
WINDOW_SIZE_MS = 1000   # 1 second windows for feature stats
WINDOW_STRIDE_MS = 500  # 0.5 second stride
SAMPLING_RATE = 1000    # 1000 Hz
SEQUENCE_LENGTH = 10    # Look back 10 windows (history) for the GRU

# Write outputs into this repo's datasets/ folder (so the notebook can load it consistently)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_align_data(subject_id: int) -> pd.DataFrame:
    """Loads physiological and annotation data, aligns them, and returns a merged DataFrame."""
    if subject_id < 1 or subject_id > 30:
        raise ValueError(f"Subject ID must be between 1 and 30, got {subject_id}")

    data_path = f"{CASE_DATA_PATH}/sub_{subject_id}.csv"
    annotation_path = f"{CASE_ANNOTATION_PATH}/sub_{subject_id}.csv"
    
    if not os.path.exists(data_path) or not os.path.exists(annotation_path):
        print(f"Warning: Data for subject {subject_id} not found.")
        return None

    # Load data
    data = pd.read_csv(data_path)
    annotation = pd.read_csv(annotation_path)

    # Clean data (keep only relevant columns for now)
    # We need 'bvp', 'gsr', 'skt' (skin temp), and 'time'
    # 'daqtime' in data is 'time'
    data.rename(columns={"daqtime": "time"}, inplace=True)
    
    # Annotation 'jstime' is 'time'
    # Apply joystick delay
    annotation["jstime"] = annotation["jstime"] - JOYSTICK_DELAY_MS
    annotation = annotation[annotation["jstime"] >= 0]
    annotation.rename(columns={"jstime": "time"}, inplace=True)

    # IMPORTANT:
    # Physiological is 1000Hz (1ms), annotations are ~20Hz (50ms). An inner join will drop ~98% of rows.
    # We instead assign each physiological sample the most recent annotation (step-wise hold),
    # which is the standard alignment for continuous annotations.
    data = data.sort_values("time").reset_index(drop=True)
    annotation = annotation.sort_values("time").reset_index(drop=True)
    join = pd.merge_asof(data, annotation, on="time", direction="backward")
    # Drop leading rows before first available annotation
    join = join.dropna(subset=["valence", "arousal"]).reset_index(drop=True)
    
    return join

def extract_features(df: pd.DataFrame, sampling_rate: int = 1000) -> pd.DataFrame:
    """Extracts physiological features using NeuroKit2."""
    
    # 1. Process GSR (EDA)
    # We extract the phasic component (SCR) so emotion-related "spikes" are visible.
    gsr_clean = nk.eda_clean(df["gsr"].values, sampling_rate=sampling_rate)
    eda_phasic = nk.eda_phasic(gsr_clean, sampling_rate=sampling_rate, method="highpass")
    df["gsr_phasic"] = eda_phasic["EDA_Phasic"].values
    
    # 2. Process BVP to get Heart Rate
    try:
        # Raw BVP is high-frequency and noisy; convert it to a smooth BPM line via peaks.
        bvp_clean = nk.ppg_clean(df["bvp"].values, sampling_rate=sampling_rate, method="elgendi")
        peaks = nk.ppg_findpeaks(bvp_clean, sampling_rate=sampling_rate, method="elgendi")
        peak_idx = peaks.get("PPG_Peaks", None)
        if peak_idx is None or len(peak_idx) < 3:
            raise ValueError("Too few PPG peaks detected")
        hr = nk.signal_rate(peak_idx, sampling_rate=sampling_rate, desired_length=len(bvp_clean))
        df["heart_rate"] = hr
    except Exception as e:
        print(f"Error processing PPG: {e}. Using raw BVP as fallback (suboptimal).")
        df["heart_rate"] = df["bvp"]  # Fallback
        
    # 3. Skin Temperature (skt)
    # Use delta/gradient so the model sees warming/cooling rather than a near-constant baseline.
    skt = df["skt"].values.astype(float)
    df["skt_delta"] = np.gradient(skt) * sampling_rate  # approx. change per second
    
    return df

def normalize_subject_data(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Z-score normalization per subject."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def create_feature_sequences(df: pd.DataFrame, feature_cols: List[str], label_cols: List[str], 
                             window_size_ms: int, stride_ms: int, sampling_rate: int, 
                             seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Computes window statistics (mean/std/slope) for each window.
    2. Stacks 'seq_len' of these statistic-windows into a sequence for the GRU.
    """
    window_size_samples = int(window_size_ms * sampling_rate / 1000)
    stride_samples = int(stride_ms * sampling_rate / 1000)
    
    # 1. Extract window stats
    stats_list = []
    labels_list = []
    
    data_values = df[feature_cols].values
    label_values = df[label_cols].values
    n_samples = len(df)
    
    # We iterate through the raw data to build "statistic windows"
    for i in range(0, n_samples - window_size_samples, stride_samples):
        window = data_values[i : i + window_size_samples]
        
        # Compute stats for this window
        # features: [gsr_phasic, heart_rate, skt_delta]
        # We want mean, std, and maybe slope/diff
        w_mean = np.mean(window, axis=0)
        w_std = np.std(window, axis=0)
        
        # Simple slope (end - start) or gradient mean
        # For skt_delta, mean is already a slope-like feature.
        # Let's add explicit slope for GSR and HR: (last - first)
        w_slope = window[-1] - window[0]
        
        # Concatenate: [mean(3), std(3), slope(3)] -> 9 features
        window_stats = np.concatenate([w_mean, w_std, w_slope])
        
        stats_list.append(window_stats)
        # Label at the end of this window
        labels_list.append(label_values[i + window_size_samples - 1])
        
    stats_arr = np.array(stats_list)
    labels_arr = np.array(labels_list)
    
    if len(stats_arr) <= seq_len:
        return np.array([]), np.array([])
        
    # 2. Create sequences of history (Feature-Sequence)
    # X: (N, seq_len, num_stats)
    # y: (N, 2)
    X_seq = []
    y_seq = []
    
    for i in range(seq_len, len(stats_arr)):
        # Sequence of last 'seq_len' statistic-windows
        X_seq.append(stats_arr[i - seq_len : i])
        y_seq.append(labels_arr[i])
        
    return np.array(X_seq), np.array(y_seq)

def main():
    all_X = []
    all_y = []
    all_subjects = [] # To track which subject each window belongs to
    
    feature_cols = ["gsr_phasic", "heart_rate", "skt_delta"]
    label_cols = ['valence', 'arousal']
    
    print("Starting preprocessing...")
    
    for subject_id in range(1, 31):
        print(f"Processing Subject {subject_id}...")
        
        # 1. Load
        df = load_and_align_data(subject_id)
        if df is None or df.empty:
            continue
            
        # 2. Extract Features
        try:
            df = extract_features(df, SAMPLING_RATE)
        except Exception as e:
            print(f"Failed to extract features for subject {subject_id}: {e}")
            continue

        # Drop rows where feature extraction produced NaNs (common at edges / poor peak detection)
        df = df.dropna(subset=feature_cols + label_cols).reset_index(drop=True)
        if df.empty:
            print(f"Subject {subject_id}: empty after dropping NaNs, skipping.")
            continue

        # 3. Normalize
        df = normalize_subject_data(df, feature_cols)
        
        # 4. Feature-Sequence Creation
        # Note: We pass raw SAMPLING_RATE (1000Hz) because we are windowing the raw 1000Hz data now,
        # not downsampling first. We calculate stats on the 1000Hz windows.
        X_sub, y_sub = create_feature_sequences(
            df, feature_cols, label_cols, WINDOW_SIZE_MS, WINDOW_STRIDE_MS, SAMPLING_RATE, SEQUENCE_LENGTH
        )
        
        if len(X_sub) > 0:
            all_X.append(X_sub)
            all_y.append(y_sub)
            # Create subject ID array for these windows
            subjects_sub = np.full((len(X_sub), 1), subject_id)
            all_subjects.append(subjects_sub)
            
    # Concatenate all
    if all_X:
        final_X = np.concatenate(all_X, axis=0)
        final_y = np.concatenate(all_y, axis=0)
        final_subjects = np.concatenate(all_subjects, axis=0)
        
        print(f"Final X shape: {final_X.shape}")
        print(f"Final y shape: {final_y.shape}")
        
        # Save to H5
        output_file = os.path.join(OUTPUT_DIR, "case_processed_v2.h5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('x', data=final_X)
            f.create_dataset('y', data=final_y)
            f.create_dataset('subject_ids', data=final_subjects)
            
        print(f"Saved processed data to {output_file}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()

