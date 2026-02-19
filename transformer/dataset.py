"""
Dataset class for Emotion → MIDI Transformer.

Loads preprocessed (emotion, midi_tokens) pairs from an HDF5 file
and prepares them for transformer training.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from .config import DEFAULT_CONFIG, TokenizerConfig
except ImportError:
    from config import DEFAULT_CONFIG, MaestroConfig, TokenizerConfig
    DEFAULT_CONFIG = MaestroConfig()


class EmotionMIDIDataset(Dataset):
    """
    PyTorch Dataset for (emotion, MIDI token sequence) pairs.

    Expects an HDF5 file with the following datasets:
        - 'emotions':  (N, 2) float32 — [valence, arousal] per sample
        - 'tokens':    (N,) variable-length arrays of int32 — token sequences
        - 'lengths':   (N,) int32 — original length of each token sequence
    """

    def __init__(
        self,
        h5_path: str,
        max_seq_len: int = None,
        pad_token: int = None,
        tokenizer_cfg: TokenizerConfig = None,
    ):
        self.h5_path = Path(h5_path)
        cfg = tokenizer_cfg or DEFAULT_CONFIG.tokenizer
        self.max_seq_len = max_seq_len or cfg.max_seq_len
        self.pad_token = pad_token if pad_token is not None else (cfg.vocab_size - 1)

        # Load everything into memory (dataset is small enough)
        with h5py.File(self.h5_path, "r") as f:
            self.emotions = f["emotions"][:]  # (N, 2) float32
            self.lengths = f["lengths"][:]    # (N,) int32

            # Load variable-length token sequences
            raw_tokens = f["tokens"]
            self.token_sequences = []
            for i in range(len(self.emotions)):
                self.token_sequences.append(raw_tokens[i][:])

        self.num_samples = len(self.emotions)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            emotion:      (2,) float32 — [valence, arousal]
            input_tokens: (max_seq_len,) int64 — padded input sequence (shifted right)
            target_tokens:(max_seq_len,) int64 — padded target sequence (next-token)
            padding_mask: (max_seq_len + 1,) bool — True for padded positions
                          (+1 because the model prepends an emotion token)
        """
        emotion = torch.tensor(self.emotions[idx], dtype=torch.float32)
        tokens = self.token_sequences[idx].astype(np.int64)

        # Truncate if necessary
        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[: self.max_seq_len + 1]

        # Input = all tokens except last, Target = all tokens except first
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        # Calculate actual length (before padding)
        actual_len = len(input_tokens)

        # Pad to max_seq_len
        input_padded = np.full(self.max_seq_len, self.pad_token, dtype=np.int64)
        target_padded = np.full(self.max_seq_len, self.pad_token, dtype=np.int64)

        input_padded[:actual_len] = input_tokens
        target_padded[:actual_len] = target_tokens

        # Padding mask: True where padded
        # +1 for the emotion token prepended by the model (never padded)
        padding_mask = np.ones(self.max_seq_len + 1, dtype=bool)
        padding_mask[0] = False                   # emotion token
        padding_mask[1: actual_len + 1] = False   # real tokens

        return (
            emotion,
            torch.tensor(input_padded, dtype=torch.long),
            torch.tensor(target_padded, dtype=torch.long),
            torch.tensor(padding_mask, dtype=torch.bool),
        )


def create_data_splits(
    h5_path: str,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    max_seq_len: int = None,
    tokenizer_cfg: TokenizerConfig = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train/val/test splits from the HDF5 file.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    full_dataset = EmotionMIDIDataset(
        h5_path,
        max_seq_len=max_seq_len,
        tokenizer_cfg=tokenizer_cfg,
    )

    total = len(full_dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    return train_ds, val_ds, test_ds


def create_dataloaders(
    h5_path: str,
    batch_size: int = 16,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    max_seq_len: int = None,
    tokenizer_cfg: TokenizerConfig = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = create_data_splits(
        h5_path, val_split, test_split, seed, max_seq_len, tokenizer_cfg
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_test():
    """Create a small mock HDF5 file and test the dataset pipeline."""
    import tempfile
    import os

    print("=" * 60)
    print("EmotionMIDIDataset — Self-Test")
    print("=" * 60)

    cfg = DEFAULT_CONFIG.tokenizer
    max_seq = 128  # Use shorter sequences for testing

    # Create mock HDF5
    print("\n[1] Creating mock HDF5 dataset...")
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        mock_path = f.name

    N = 50  # 50 samples
    with h5py.File(mock_path, "w") as f:
        emotions = np.random.uniform(-1, 1, (N, 2)).astype(np.float32)
        f.create_dataset("emotions", data=emotions)

        # Variable-length token sequences
        dt = h5py.vlen_dtype(np.int32)
        tokens_ds = f.create_dataset("tokens", shape=(N,), dtype=dt)
        lengths = np.zeros(N, dtype=np.int32)

        for i in range(N):
            seq_len = np.random.randint(10, max_seq)
            seq = np.random.randint(0, cfg.vocab_size - 1, size=seq_len).astype(np.int32)
            tokens_ds[i] = seq
            lengths[i] = seq_len

        f.create_dataset("lengths", data=lengths)

    print(f"  Created {N} samples at: {mock_path}")

    # Test dataset loading
    print("\n[2] Testing EmotionMIDIDataset...")
    dataset = EmotionMIDIDataset(mock_path, max_seq_len=max_seq)
    print(f"  Dataset size: {len(dataset)}")

    emotion, input_tok, target_tok, pad_mask = dataset[0]
    print(f"  emotion:      {emotion.shape} {emotion.dtype}")
    print(f"  input_tokens: {input_tok.shape} {input_tok.dtype}")
    print(f"  target_tokens:{target_tok.shape} {target_tok.dtype}")
    print(f"  padding_mask: {pad_mask.shape} {pad_mask.dtype}")

    assert emotion.shape == (2,)
    assert input_tok.shape == (max_seq,)
    assert target_tok.shape == (max_seq,)
    assert pad_mask.shape == (max_seq + 1,)
    print("  ✓ Shapes correct")

    # Test dataloaders
    print("\n[3] Testing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        mock_path,
        batch_size=8,
        val_split=0.2,
        test_split=0.2,
        max_seq_len=max_seq,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    for batch in train_loader:
        emo, inp, tgt, mask = batch
        print(f"  Batch: emo={emo.shape}, inp={inp.shape}, tgt={tgt.shape}, mask={mask.shape}")
        break

    print("  ✓ DataLoaders work")

    # Clean up
    os.unlink(mock_path)

    print("\n" + "=" * 60)
    print("✓ All self-tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion MIDI Dataset")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    args = parser.parse_args()

    if args.test:
        _run_self_test()
    else:
        parser.print_help()
