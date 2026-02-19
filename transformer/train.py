"""
Training loop for the Emotion → MIDI Transformer.

Features:
    - Cross-entropy loss with label smoothing
    - AdamW optimizer + cosine annealing LR
    - Gradient clipping
    - Early stopping with patience
    - Model checkpointing
    - CUDA support
    - Detailed training logs

Usage:
    python -m transformer.train [--data path/to/processed.h5]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    from .config import DEFAULT_CONFIG, MaestroConfig
    from .model import EmotionMusicTransformer
    from .dataset import create_dataloaders
    from .tokenizer import MIDITokenizer
except ImportError:
    from config import DEFAULT_CONFIG, MaestroConfig
    from model import EmotionMusicTransformer
    from dataset import create_dataloaders
    from tokenizer import MIDITokenizer
    DEFAULT_CONFIG = MaestroConfig()


class Trainer:
    """Handles the full training lifecycle."""

    def __init__(
        self,
        config: MaestroConfig = None,
        data_path: Optional[str] = None,
    ):
        self.cfg = config or DEFAULT_CONFIG
        self.data_path = Path(data_path) if data_path else self.cfg.paths.processed_data_path
        self.device = torch.device(self.cfg.device)

        # Ensure checkpoint dir exists
        self.cfg.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_training()

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_time": [],
        }

    def _setup_model(self):
        """Initialize the transformer model."""
        tokenizer = MIDITokenizer(self.cfg.tokenizer)
        self.vocab_size = tokenizer.vocab_size
        self.pad_token = tokenizer.pad_token_id

        self.model = EmotionMusicTransformer(
            vocab_size=self.vocab_size,
            model_cfg=self.cfg.model,
        ).to(self.device)

        param_count = self.model.count_parameters()
        print(f"Model parameters: {param_count:,}")
        print(f"Device: {self.device}")

    def _setup_data(self):
        """Create data loaders."""
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            h5_path=str(self.data_path),
            batch_size=self.cfg.train.batch_size,
            val_split=self.cfg.train.val_split,
            test_split=self.cfg.train.test_split,
            seed=self.cfg.train.seed,
            max_seq_len=self.cfg.tokenizer.max_seq_len,
            tokenizer_cfg=self.cfg.tokenizer,
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches:   {len(self.val_loader)}")
        print(f"Test batches:  {len(self.test_loader)}")

    def _setup_training(self):
        """Initialize optimizer, scheduler, and loss function."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        # Cross-entropy loss ignoring padding tokens
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_token,
            label_smoothing=0.1,
        )

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (emotion, input_tokens, target_tokens, padding_mask) in enumerate(self.train_loader):
            emotion = emotion.to(self.device)
            input_tokens = input_tokens.to(self.device)
            target_tokens = target_tokens.to(self.device)
            padding_mask = padding_mask.to(self.device)

            # Forward pass
            logits = self.model(input_tokens, emotion, src_key_padding_mask=padding_mask)

            # The model outputs T+1 logits (emotion position + T token positions)
            # We only need the token positions for loss (skip emotion position)
            # logits[:, 1:, :] corresponds to predictions for each input token position
            token_logits = logits[:, 1:, :]  # (B, T, vocab_size)

            # Reshape for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = self.criterion(
                token_logits.reshape(-1, self.vocab_size),
                target_tokens.reshape(-1),
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.train.grad_clip_norm,
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for emotion, input_tokens, target_tokens, padding_mask in self.val_loader:
            emotion = emotion.to(self.device)
            input_tokens = input_tokens.to(self.device)
            target_tokens = target_tokens.to(self.device)
            padding_mask = padding_mask.to(self.device)

            logits = self.model(input_tokens, emotion, src_key_padding_mask=padding_mask)
            token_logits = logits[:, 1:, :]

            loss = self.criterion(
                token_logits.reshape(-1, self.vocab_size),
                target_tokens.reshape(-1),
            )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "vocab_size": self.vocab_size,
            "model_config": {
                "d_model": self.cfg.model.d_model,
                "nhead": self.cfg.model.nhead,
                "num_layers": self.cfg.model.num_layers,
                "dim_feedforward": self.cfg.model.dim_feedforward,
                "dropout": self.cfg.model.dropout,
                "max_seq_len": self.cfg.model.max_seq_len,
                "emotion_dim": self.cfg.model.emotion_dim,
            },
            "train_history": self.train_history,
        }

        # Save per-epoch checkpoint
        epoch_path = self.cfg.paths.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}_loss{val_loss:.4f}.pt"
        torch.save(checkpoint, epoch_path)

        # Save best
        if is_best:
            best_path = self.cfg.paths.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"    ★ New best model saved! (val_loss={self.best_val_loss:.5f})")

    def train(self):
        """Full training loop with early stopping."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"  Epochs:     {self.cfg.train.num_epochs}")
        print(f"  Batch size: {self.cfg.train.batch_size}")
        print(f"  LR:         {self.cfg.train.learning_rate}")
        print(f"  Patience:   {self.cfg.train.early_stop_patience}")
        print()

        for epoch in range(1, self.cfg.train.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start

            # Track history
            self.train_history["train_loss"].append(train_loss)
            self.train_history["val_loss"].append(val_loss)
            self.train_history["learning_rate"].append(current_lr)
            self.train_history["epoch_time"].append(epoch_time)

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)

            # Log
            print(
                f"Epoch {epoch:3d}/{self.cfg.train.num_epochs} | "
                f"lr {current_lr:.1e} | "
                f"train {train_loss:.5f} | "
                f"val {val_loss:.5f} | "
                f"{'★ best' if is_best else f'patience {self.patience_counter}/{self.cfg.train.early_stop_patience}'} | "
                f"{epoch_time:.1f}s"
            )

            # Early stopping
            if self.patience_counter >= self.cfg.train.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.cfg.train.early_stop_patience} epochs)")
                break

        # Final evaluation on test set
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)
        self._final_eval()

        # Save training history
        history_path = self.cfg.paths.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.train_history, f, indent=2)
        print(f"\nTraining history saved to: {history_path}")

    @torch.no_grad()
    def _final_eval(self):
        """Evaluate on test set using best model."""
        # Load best model
        best_path = self.cfg.paths.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded best model from epoch {checkpoint['epoch']}")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for emotion, input_tokens, target_tokens, padding_mask in self.test_loader:
            emotion = emotion.to(self.device)
            input_tokens = input_tokens.to(self.device)
            target_tokens = target_tokens.to(self.device)
            padding_mask = padding_mask.to(self.device)

            logits = self.model(input_tokens, emotion, src_key_padding_mask=padding_mask)
            token_logits = logits[:, 1:, :]

            loss = self.criterion(
                token_logits.reshape(-1, self.vocab_size),
                target_tokens.reshape(-1),
            )

            total_loss += loss.item()
            num_batches += 1

        test_loss = total_loss / max(num_batches, 1)
        print(f"  Test loss: {test_loss:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Train Emotion → MIDI Transformer")
    parser.add_argument("--data", type=str, help="Path to preprocessed HDF5 file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    config = MaestroConfig()

    if args.epochs:
        config.train.num_epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr
    if args.device:
        config.device = args.device

    trainer = Trainer(config=config, data_path=args.data)
    trainer.train()


if __name__ == "__main__":
    main()
