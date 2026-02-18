"""
PyTorch Dataset that loads pre-processed metadata (CSV) and tokenizes
MIDI files on-the-fly from the XMIDI dataset.
"""
import os
import sys
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from symusic import Score

from src.config import Config
from src.utils import get_tokenizer


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout/stderr (for suppressing C++ library debug output)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class XmidiDataset(Dataset):
    """
    Each sample returns:
        tokens  – LongTensor  [SEQ_LEN]     (REMI token ids)
        mood_id – LongTensor  scalar         (mood category)
        genre_id– LongTensor  scalar         (genre category)
    """

    def __init__(
        self,
        csv_path: str | Path,
        midi_folder: str | Path,
        tokenizer,
        seq_len: int = Config.SEQ_LEN,
    ):
        self.df = pd.read_csv(csv_path)
        self.midi_folder = Path(midi_folder)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Keep only rows whose MIDI files actually exist on disk
        exists_mask = self.df["filename"].apply(
            lambda fn: (self.midi_folder / fn).exists()
        )
        self.df = self.df[exists_mask].reset_index(drop=True)
        print(f"[Dataset] Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        midi_path = self.midi_folder / row["filename"]

        # ---------- tokenize ----------
        # Suppress C++ library debug output (read_file messages) to speed up training
        try:
            with suppress_stdout_stderr():
                score = Score(str(midi_path))
                tok_result = self.tokenizer.encode(score)
            # miditok returns list[TokSequence] or single TokSequence
            if isinstance(tok_result, list):
                token_ids = tok_result[0].ids
            else:
                token_ids = tok_result.ids
        except Exception:
            # If a file fails to tokenize, return zeros (will be filtered by collate)
            token_ids = []

        # ---------- pad / random-crop to seq_len ----------
        if len(token_ids) == 0:
            token_ids = [0] * self.seq_len
        elif len(token_ids) < self.seq_len:
            # Pad with 0 (PAD token)
            token_ids = token_ids + [0] * (self.seq_len - len(token_ids))
        elif len(token_ids) > self.seq_len:
            # Random crop for data augmentation
            start = np.random.randint(0, len(token_ids) - self.seq_len + 1)
            token_ids = token_ids[start : start + self.seq_len]
        # else: len(token_ids) == self.seq_len, use as-is (no crop needed)

        tokens = torch.tensor(token_ids, dtype=torch.long)
        mood_id = torch.tensor(row["mood_id"], dtype=torch.long)
        genre_id = torch.tensor(row["genre_id"], dtype=torch.long)

        return tokens, mood_id, genre_id


def get_dataloader(
    csv_path: str | Path = Config.METADATA_CSV,
    midi_folder: str | Path = Config.XMIDI_DATASET_DIR,
    tokenizer=None,
    batch_size: int = Config.BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """Convenience function used by training scripts."""
    if tokenizer is None:
        tokenizer = get_tokenizer(Config.TOKENIZER_PARAMS_PATH)
    dataset = XmidiDataset(csv_path, midi_folder, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, tokenizer

