"""
Cached Dataset - Loads pre-tokenized sequences from .npy files.

This is MUCH faster than tokenizing MIDI files on-the-fly during training.
Use this after running 0_preprocess_tokens.py.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from src.core.config import Config


class CachedDataset(Dataset):
    """
    Fast dataset that loads pre-tokenized sequences from .npy files.
    
    Each sample returns:
        tokens  - LongTensor  [SEQ_LEN]     (REMI token ids)
        mood_id - LongTensor  scalar         (mood category)
        genre_id- LongTensor  scalar         (genre category)
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenized_dir: str | Path,
        seq_len: int = Config.SEQ_LEN,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenized_dir = Path(tokenized_dir)
        self.seq_len = seq_len

        # Filter to only files that have been pre-processed
        def has_tokenized_file(filename):
            npy_name = filename.replace(".midi", ".npy")
            return (self.tokenized_dir / npy_name).exists()

        exists_mask = self.df["filename"].apply(has_tokenized_file)
        self.df = self.df[exists_mask].reset_index(drop=True)
        print(f"[CachedDataset] Loaded {len(self.df)} pre-tokenized samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = self.tokenized_dir / row["filename"].replace(".midi", ".npy")

        # Load pre-tokenized sequence (FAST - just numpy load)
        token_ids = np.load(npy_path).astype(np.int64).tolist()

        # Pad / random-crop to seq_len
        if len(token_ids) == 0:
            token_ids = [0] * self.seq_len
        elif len(token_ids) < self.seq_len:
            token_ids = token_ids + [0] * (self.seq_len - len(token_ids))
        elif len(token_ids) > self.seq_len:
            # Random crop for data augmentation
            start = np.random.randint(0, len(token_ids) - self.seq_len + 1)
            token_ids = token_ids[start : start + self.seq_len]
        # else: len(token_ids) == self.seq_len, use as-is

        tokens = torch.tensor(token_ids, dtype=torch.long)
        mood_id = torch.tensor(row["mood_id"], dtype=torch.long)
        genre_id = torch.tensor(row["genre_id"], dtype=torch.long)

        return tokens, mood_id, genre_id


def get_cached_dataloader(
    csv_path: str | Path = Config.METADATA_CSV,
    tokenized_dir: str | Path = Config.DATA_DIR / "tokenized",
    batch_size: int = Config.BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 4,  # Use multiple workers for parallel loading
    persistent_workers: bool = True,  # Keep workers alive between epochs
    prefetch_factor: int = 2,  # Prefetch batches
):
    """
    Get a DataLoader using pre-tokenized cached data.
    
    This is MUCH faster than the on-the-fly tokenization approach.
    """
    dataset = CachedDataset(csv_path, tokenized_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return loader

if __name__ == "__main__":
    dataloader = get_cached_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=Config.PERSISTENT_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
    )
    for tokens, mood_id, genre_id in dataloader:
        print(tokens.shape)
        print(mood_id.shape)
        print(genre_id.shape)
        break
