"""
Step 3 - Train the Levenshtein Refiner (Non-Autoregressive Denoising Model).

Training strategy
-----------------
1. Take a **clean** XMIDI token sequence (the "ground truth").
2. **Corrupt** it by randomly replacing a fraction of tokens with random ids.
3. Ask the Refiner to:
   a) Predict which tokens were corrupted  (Deletion head - binary CE).
   b) Reconstruct the original token       (Token head   - CE on all positions).

This teaches the model to "fix" low-quality or incoherent music.

Usage
-----
    cd music-generator-test
    python 3_train_refiner.py
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from src.dataset_cached import get_cached_dataloader
from src.model_refiner import LevenshteinRefiner
from src.utils import get_tokenizer


# ======================================================================
# Corruption function
# ======================================================================
def corrupt_sequence(
    tokens: torch.Tensor,
    vocab_size: int,
    noise_level: float = Config.NOISE_LEVEL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly replace *noise_level* fraction of non-padding tokens with
    random token ids.

    Returns
    -------
    corrupted : LongTensor [B, S]  - the noisy version
    is_noisy  : LongTensor [B, S]  - binary label (1 where corrupted)
    """
    corrupted = tokens.clone()
    # Only corrupt non-padding positions
    non_pad = tokens != 0                                       # [B, S]
    noise_mask = (torch.rand_like(tokens.float()) < noise_level) & non_pad  # [B, S]

    # Replace with random valid token ids (1 .. vocab_size-1)
    random_ids = torch.randint(1, vocab_size, tokens.shape, device=tokens.device)
    corrupted[noise_mask] = random_ids[noise_mask]

    is_noisy = noise_mask.long()  # 1 = corrupted, 0 = clean
    return corrupted, is_noisy


# ======================================================================
# Training loop
# ======================================================================
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
    model = LevenshteinRefiner(
        vocab_size=vocab_size,
        num_moods=Config.NUM_MOODS,
        num_genres=Config.NUM_GENRES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Refiner parameters: {total_params:,}")

    # ---- Optimizer & Loss ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Two losses: deletion (binary CE) and token reconstruction (CE)
    deletion_criterion = nn.CrossEntropyLoss()
    token_criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

    # ---- Checkpoints directory ----
    Config.REFINER_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Training ----
    best_loss = float("inf")
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_del_loss = 0.0
        epoch_tok_loss = 0.0
        num_batches = 0

        # Update progress bar less frequently to reduce I/O overhead
        loop = tqdm(loader, desc=f"Epoch {epoch}/{Config.EPOCHS}", mininterval=1.0)
        for batch_idx, (clean_tokens, moods, genres) in enumerate(loop):
            clean_tokens = clean_tokens.to(device)   # [B, S]
            moods = moods.to(device)
            genres = genres.to(device)

            # 1. Corrupt
            noisy_tokens, noise_labels = corrupt_sequence(
                clean_tokens, vocab_size
            )

            # 2. Forward
            del_logits, tok_logits = model(noisy_tokens, moods, genres)

            # 3. Losses
            # Deletion loss: did the model find the corrupted positions?
            del_loss = deletion_criterion(
                del_logits.reshape(-1, 2),
                noise_labels.reshape(-1),
            )
            # Token reconstruction loss: can it predict the original tokens?
            tok_loss = token_criterion(
                tok_logits.reshape(-1, vocab_size),
                clean_tokens.reshape(-1),
            )

            # Combined loss (weighted sum)
            loss = tok_loss + 0.5 * del_loss

            # 4. Backward
            optimizer.zero_grad()
            loss.backward()
            if Config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_del_loss += del_loss.item()
            epoch_tok_loss += tok_loss.item()
            num_batches += 1
            
            # Update progress bar less frequently (every 10 batches) to reduce I/O overhead
            if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
                loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    del_l=f"{del_loss.item():.4f}",
                    tok_l=f"{tok_loss.item():.4f}",
                    avg_loss=f"{epoch_loss/num_batches:.4f}",
                )

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_del = epoch_del_loss / max(num_batches, 1)
        avg_tok = epoch_tok_loss / max(num_batches, 1)
        lr = scheduler.get_last_lr()[0]
        print(
            f"  => Epoch {epoch}  loss={avg_loss:.4f}  "
            f"del={avg_del:.4f}  tok={avg_tok:.4f}  lr={lr:.2e}"
        )

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
        torch.save(ckpt, Config.REFINER_CKPT_DIR / f"ref_epoch_{epoch}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, Config.REFINER_CKPT_DIR / "refiner_best.pt")
            print(f"  => New best refiner saved (loss={best_loss:.4f})")

    # ---- Final checkpoint ----
    torch.save(ckpt, Config.REFINER_CKPT_DIR / "refiner_latest.pt")
    print(f"\nRefiner training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints at: {Config.REFINER_CKPT_DIR}")
    print("Next step: streamlit run app.py")


if __name__ == "__main__":
    train()

