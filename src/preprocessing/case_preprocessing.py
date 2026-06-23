from typing import Dict, List

import h5py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = "datasets"

CASE_DIR = f"{DATA_DIR}/CASE_full"

CASE_DATA_PATH = f"{CASE_DIR}/data/interpolated/physiological"
CASE_ANNOTATION_PATH = f"{CASE_DIR}/data/interpolated/annotations"

JOYSTICK_DELAY_MS = 8000

WINDOW_SIZE_MS = 8000
WINDOW_STRIDE_MS = 100

def get_batch_windows(data: pd.DataFrame, batch_size_ms: int) -> List[pd.DataFrame]:
    refresh_rate = data["time"].diff().mean().round(0)
    batch_size_samples = int(batch_size_ms / refresh_rate)
    batch_size_stride_ms = WINDOW_STRIDE_MS
    batch_windows: List[pd.DataFrame] = []
    for i in range(0, len(data) - batch_size_samples + 1, batch_size_stride_ms):
        batch: pd.DataFrame = data.iloc[i:i+batch_size_samples]
        batch.loc[:, "time"] = batch["time"] - batch["time"].iloc[0]
        
        batch_windows.append(batch)
    return batch_windows

def get_subject_data(subject_id: int) -> List[pd.DataFrame]:
    if subject_id < 1 or subject_id > 30:
        raise ValueError(f"Subject ID must be between 1 and 30, got {subject_id}")

    # Load data and annotation
    data = pd.read_csv(f"{CASE_DATA_PATH}/sub_{subject_id}.csv")
    annotation = pd.read_csv(f"{CASE_ANNOTATION_PATH}/sub_{subject_id}.csv")

    # Clean data
    data.drop(columns=["video", "ecg", "rsp", "emg_zygo", "emg_coru", "emg_trap"], inplace=True)
    annotation.drop(columns=["video"], inplace=True)

    # Add arbitrary delay to jstime (Subject moves the joystick after feeling the emotion by a few seconds)
    annotation["jstime"] = annotation["jstime"] - JOYSTICK_DELAY_MS

    # Remove data before the first joystick movement
    annotation = annotation[annotation["jstime"] >=0]

    # Join data and annotation
    data.rename(columns={"daqtime": "time"}, inplace=True)
    annotation.rename(columns={"jstime": "time"}, inplace=True)
    join = pd.merge(data, annotation, on="time", how="inner")

    # Remove time column
    # join.drop(columns=["time"], inplace=True)


    # Scale data
    columns_to_exclude = ["time"]
    columns_to_scale = [col for col in join.columns if col not in columns_to_exclude]
    scaler = StandardScaler()
    scaler = scaler.fit(join[columns_to_scale])
    join_scaled_part = scaler.transform(join[columns_to_scale])
    join_scaled_part = pd.DataFrame(join_scaled_part, columns=columns_to_scale)

    # Combine scaled and unscaled columns
    join_scaled = join.copy()
    join_scaled[columns_to_scale] = join_scaled_part
    

    # Normalize data
    # for column in join.columns:
    #     join[column] = (join[column] - join[column].mean()) / join[column].std()

    
    batch_windows = get_batch_windows(join, WINDOW_SIZE_MS)


    return batch_windows, scaler

def convert_to_numpy(batch_windows: List[pd.DataFrame]) -> np.ndarray:
    x = []
    y = []
    for batch in batch_windows:
        x_df = batch[["bvp", "gsr", "skt"]]
        x.append(x_df.values)
        y_df = batch[["valence", "arousal"]]
        v_a = y_df.iloc[-1]
        y.append(v_a)

    return np.array(x), np.array(y)


if __name__ == "__main__":
    # subject_data: Dict[int, pd.DataFrame] = {}
    all_windows: List[pd.DataFrame] = []
    for i in range(1, 31):
        batch_windows, scaler = get_subject_data(i)
        # subject_data[i] = batch_windows
        all_windows.extend(batch_windows)

    x, y = convert_to_numpy(all_windows)
    print(x.shape)
    print(y.shape)
    with h5py.File(f'{DATA_DIR}/case_processed.h5', 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
    
