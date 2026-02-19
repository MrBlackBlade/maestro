"""
Step 2 – Train the Music Generator (Autoregressive Transformer).

The model learns: "Given this sequence of MIDI tokens + a Mood + a Genre,
what is the most likely *next* token?"

Training uses standard Next-Token Prediction with Cross-Entropy Loss.

Usage
-----
    cd music-generator-test
    python 2_train_generator.py
"""
import sys
import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from src.dataset_cached import get_cached_dataloader
from src.model_generator import MusicGenerator
from src.utils import get_tokenizer


def train():
    device = Config.DEVICE
    print(f"Device: {device}")

    # ---- Data ----
    tokenizer = get_tokenizer(Config.TOKENIZER_PARAMS_PATH)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # Check if pre-processed tokens exist
    if not Config.TOKENIZED_DIR.exists() or len(list(Config.TOKENIZED_DIR.glob("*.npy"))) == 0:
        print("\n" + "=" * 60)
        print("ERROR: Pre-processed tokens not found!")
        print("=" * 60)
        print("Run 'python 0_preprocess_tokens.py' first to cache all tokenized sequences.")
        print("This will eliminate the disk I/O bottleneck and speed up training significantly.")
        print("=" * 60)
        return

    loader = get_cached_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=Config.PERSISTENT_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
    )
    print(f"Batches per epoch: {len(loader)}")
    print(f"Using {Config.NUM_WORKERS} parallel workers for data loading")

    # ---- Model ----
    model = MusicGenerator(
        vocab_size=vocab_size,
        num_moods=Config.NUM_MOODS,
        num_genres=Config.NUM_GENRES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Generator parameters: {total_params:,}")

    # ---- Optimizer & Loss ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD tokens

    # ---- Checkpoints directory ----
    Config.GENERATOR_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    best_loss = float("inf")
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Update progress bar less frequently to reduce I/O overhead
        # Only update every 10 batches or use mininterval for time-based updates
        loop = tqdm(loader, desc=f"Epoch {epoch}/{Config.EPOCHS}", mininterval=1.0)
        for batch_idx, (tokens, moods, genres) in enumerate(loop):
            tokens = tokens.to(device)   # [B, SEQ_LEN]
            moods = moods.to(device)     # [B]
            genres = genres.to(device)   # [B]

            # Next-token prediction: input = tokens[:-1], target = tokens[1:]
            # Example: if tokens = [1, 5, 10, 20], then:
            #   inp = [1, 5, 10]  (predict next token given previous)
            #   tgt = [5, 10, 20] (the actual next tokens)
            inp = tokens[:, :-1]  # [B, SEQ_LEN-1]
            tgt = tokens[:, 1:]    # [B, SEQ_LEN-1]

            # Forward: model predicts logits for each position
            logits = model(inp, moods, genres)  # [B, SEQ_LEN-1, vocab_size]

            # Loss calculation (Cross-Entropy):
            # 1. Reshape logits: [B, SEQ_LEN-1, vocab_size] -> [B*(SEQ_LEN-1), vocab_size]
            # 2. Reshape targets: [B, SEQ_LEN-1] -> [B*(SEQ_LEN-1)]
            # 3. For each position, compute CE between predicted distribution and true token
            # 4. ignore_index=0 means padding tokens (id=0) don't contribute to loss
            loss = criterion(
                logits.reshape(-1, vocab_size),  # [B*(SEQ_LEN-1), vocab_size]
                tgt.reshape(-1),                  # [B*(SEQ_LEN-1)]
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if Config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar less frequently (every 10 batches) to reduce I/O overhead
            # This prevents the progress bar from blocking GPU computation
            if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
                loop.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{epoch_loss/num_batches:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        lr = scheduler.get_last_lr()[0]
        print(f"  => Epoch {epoch}  avg_loss={avg_loss:.4f}  lr={lr:.2e}")

        # ---- Save checkpoint every epoch ----
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": vocab_size,
            "num_moods": Config.NUM_MOODS,
            "num_genres": Config.NUM_GENRES,
            "loss": avg_loss,
        }
        torch.save(ckpt, Config.GENERATOR_CKPT_DIR / f"gen_epoch_{epoch}.pt")

        # Save best model separately for easy loading
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, Config.GENERATOR_CKPT_DIR / "generator_best.pt")
            print(f"  => New best model saved (loss={best_loss:.4f})")

    # ---- Final "latest" checkpoint ----
    torch.save(ckpt, Config.GENERATOR_CKPT_DIR / "generator_latest.pt")
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints at: {Config.GENERATOR_CKPT_DIR}")
    print("Next step: python 3_train_refiner.py")


if __name__ == "__main__":
    train()

