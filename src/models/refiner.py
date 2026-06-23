"""
Music Generator - Autoregressive Decoder-only Transformer (GPT-style).

Conditioned on discrete **Mood** and **Genre** embeddings which are added to
every token embedding so the model always "knows" the requested style.

Key design choices
------------------
* Learned positional embeddings (up to MAX_SEQ_LEN).
* Causal (upper-triangular) mask prevents the model from seeing future tokens.
* KV-cache friendly: during inference you can call the model with just the
  last token and manually manage the cache (not shown here but trivial to add).
"""
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import get_tokenizer, save_midi
# from src.dataloaders.full_dataloader import get_full_dataloader
from src.dataloaders.singleton_dataloader import get_singleton_dataloader
from src.dataloaders.dataset_cached import get_cached_dataloader
from src.models.general_model_handler import GeneralModelHandler


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

class ModelRefiner(nn.Module):
  def __init__(
      self,
      vocab_size: int,
      d_model: int = Config.D_MODEL,
      nhead: int = Config.NUM_HEADS,
      num_layers: int = Config.REFINER_NUM_LAYERS,
      dim_feedforward: int = Config.DIM_FEEDFORWARD,
      dropout: float = Config.DROPOUT,
      max_seq_len: int = Config.MAX_SEQ_LEN,
      num_moods: int = Config.NUM_MOODS,
      num_genres: int = Config.NUM_GENRES,
  ):
      super().__init__()
      self.d_model = d_model
      self.vocab_size = vocab_size

      # ---- Embeddings ----
      self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
      self.pos_emb = nn.Embedding(max_seq_len, d_model)
      self.mood_emb = nn.Embedding(num_moods, d_model)
      self.genre_emb = nn.Embedding(num_genres, d_model)

      # ---- Bidirectional Transformer Encoder ----
      encoder_layer = nn.TransformerEncoderLayer(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          dropout=dropout,
          batch_first=True,
          norm_first=True,
      )
      self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

      # ---- Output Heads ----
      # Deletion classifier: 0 = Keep, 1 = Delete
      self.deletion_head = nn.Linear(d_model, 2)
      # Token reconstruction / replacement
      self.token_head = nn.Linear(d_model, vocab_size)

      self.emb_norm = nn.LayerNorm(d_model)
      self.drop = nn.Dropout(dropout)

      self._init_weights()

  def _init_weights(self):
      nn.init.normal_(self.token_emb.weight, std=0.02)
      nn.init.normal_(self.pos_emb.weight, std=0.02)
      nn.init.normal_(self.mood_emb.weight, std=0.02)
      nn.init.normal_(self.genre_emb.weight, std=0.02)

  # ------------------------------------------------------------------
  def forward(
      self,
      x: torch.Tensor,
      mood_id: torch.Tensor,
      genre_id: torch.Tensor,
  ):
      """
      Parameters
      ----------
      x        : LongTensor  [B, S]   - (possibly corrupted) token ids
      mood_id  : LongTensor  [B]
      genre_id : LongTensor  [B]

      Returns
      -------
      del_logits : FloatTensor [B, S, 2]           - keep/delete per token
      tok_logits : FloatTensor [B, S, vocab_size]   - replacement prediction
      """
      B, S = x.shape
      device = x.device

      # 1. Embeddings + conditioning
      positions = torch.arange(S, device=device).unsqueeze(0)
      h = self.token_emb(x) + self.pos_emb(positions)
      cond = self.mood_emb(mood_id).unsqueeze(1) + self.genre_emb(genre_id).unsqueeze(1)
      h = h + cond

      h = self.emb_norm(h)
      h = self.drop(h)

      # 2. Bidirectional encoding (no causal mask - sees everything)
      # Create a padding mask: positions where x == 0 are padding
      padding_mask = (x == 0)  # [B, S], True where padded
      latent = self.encoder(h, src_key_padding_mask=padding_mask)

      # 3. Heads
      del_logits = self.deletion_head(latent)   # [B, S, 2]
      tok_logits = self.token_head(latent)       # [B, S, V]

      return del_logits, tok_logits

class ModelRefinerHandler(GeneralModelHandler):
    MODEL_NAME = "refiner_0"

    def __init__(self, model: nn.Module, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, scheduler, self.MODEL_NAME)
        self.tok_criterion, self.del_criterion = criterion

    def train_step(self, batch):
        clean_tokens, moods, genres = batch
        
        clean_tokens = clean_tokens.to(self.device)   # [B, S]
        moods = moods.to(self.device)
        genres = genres.to(self.device)

        # 1. Corrupt
        noisy_tokens, noise_labels = corrupt_sequence(
            clean_tokens, vocab_size
        )

        # 2. Forward
        del_logits, tok_logits = self.model(noisy_tokens, moods, genres)

        # 3. Losses
        # Deletion loss: did the model find the corrupted positions?
        del_loss = self.del_criterion(
            del_logits.reshape(-1, 2),
            noise_labels.reshape(-1),
        )
        # Token reconstruction loss: can it predict the original tokens?
        tok_loss = self.tok_criterion(
            tok_logits.reshape(-1, vocab_size),
            clean_tokens.reshape(-1),
        )

        # Combined loss (weighted sum)
        loss = tok_loss + 0.5 * del_loss
        
        return loss

if __name__ == "__main__":
    device = Config.DEVICE
    print(f"Device: {device}")

    # ---- Data ----
    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    ## TO BE IMPLEMENTED
    # dataloader = get_full_dataloader()
    dataloader = get_cached_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=Config.PERSISTENT_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
    )
    
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Using {Config.NUM_WORKERS} parallel workers for data loading")

    # ---- Model ----
    model = ModelRefiner(
        vocab_size=vocab_size,
        num_moods=Config.NUM_MOODS,
        num_genres=Config.NUM_GENRES,
    ).to(device)
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

    handler = ModelRefinerHandler(
      model=model,
      optimizer=optimizer,
      scheduler=scheduler,
      criterion=(token_criterion, deletion_criterion)
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Refiner parameters: {total_params:,}")

    handler.train(dataloader=dataloader, epochs=Config.EPOCHS)

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(Config.DEVICE)
        x_batch = x_batch[0, 0:1].unsqueeze(0)
        print(x_batch.shape)
        generated_tokens = handler.generate(x_batch)
        print(generated_tokens[:20])
        save_midi(generated_tokens, tokenizer, "generated_midi.mid")
        break