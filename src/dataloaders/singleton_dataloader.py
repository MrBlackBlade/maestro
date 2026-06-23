import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

from src.core.config import Config

class SingletonDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len
        
        # CORRECTED: Just subtract seq_len. 
        # If len is 10 and seq_len is 3, we have 7 valid starting positions (0 to 6)
        self.num_samples = len(token_ids) - self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # .indices() safely calculates the start, stop, and step 
            # within the bounds of self.num_samples
            return [self[i] for i in range(*idx.indices(self.num_samples))]
        
        # 1. Support negative indexing (e.g., dataset[-1] gets the very last valid sequence)
        if idx < 0:
            idx = self.num_samples + idx
            
        # 2. Enforce strict boundaries so PyTorch DataLoaders behave properly
        if idx >= self.num_samples or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.num_samples}")
            
        # 3. Grab the chunk
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        
        # 4. Split into input and target
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

def get_token_ids(token_path):
    token_ids = np.load(token_path)
    return token_ids

def get_singleton_dataloader(token_path, seq_len, batch_size=4, shuffle=False, num_workers=0):
    token_ids = get_token_ids(token_path)
    dataset = SingletonDataset(token_ids, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    token_path = Config.DATASETS_DIR / "XMIDI_angry_classical_0HP7PK58_tokens.npy"
    dataloader = get_singleton_dataloader(token_path, seq_len=256)
    for x, y in dataloader:
        print(x.shape)
        print(y.shape)
        break