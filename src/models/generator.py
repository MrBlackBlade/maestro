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

class ModelGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = Config.D_MODEL,
        nhead: int = Config.NUM_HEADS,
        num_layers: int = Config.NUM_LAYERS,
        dim_feedforward: int = Config.DIM_FEEDFORWARD,
        dropout: float = Config.DROPOUT,
        max_seq_len: int = Config.MAX_SEQ_LEN,
        num_moods: int = Config.NUM_MOODS,
        num_genres: int = Config.NUM_GENRES,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # ---- Token + Position embeddings ----
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # ---- Conditioning embeddings (discrete) ----
        self.mood_emb = nn.Embedding(num_moods, d_model)
        self.genre_emb = nn.Embedding(num_genres, d_model)

        # ---- Transformer Decoder stack ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # ---- Output projection ----
        self.fc_out = nn.Linear(d_model, vocab_size)

        # ---- Layer Norm on embeddings ----
        self.emb_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Xavier-uniform for embeddings, normal for linear layers."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.mood_emb.weight, std=0.02)
        nn.init.normal_(self.genre_emb.weight, std=0.02)
        nn.init.normal_(self.fc_out.weight, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular boolean mask (True = masked / ignored)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mood_id: torch.Tensor,
        genre_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : LongTensor  [B, S]   - input token ids
        mood_id  : LongTensor  [B]      - mood category index
        genre_id : LongTensor  [B]      - genre category index

        Returns
        -------
        logits   : FloatTensor [B, S, vocab_size]
        """
        B, S = x.shape
        device = x.device

        # 1. Embed tokens + positions
        positions = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
        h = self.token_emb(x) * math.sqrt(self.d_model)
        h = h + self.pos_emb(positions)

        # 2. Add mood + genre conditioning (broadcast over sequence dim)
        cond = self.mood_emb(mood_id).unsqueeze(1) + self.genre_emb(genre_id).unsqueeze(1)
        h = h + cond  # [B, S, D]

        h = self.emb_norm(h)
        h = self.drop(h)

        # 3. Causal self-attention
        mask = self._make_causal_mask(S, device)

        # TransformerDecoder expects (tgt, memory). We use self-attention only
        # by passing h as both tgt and memory with a causal mask on tgt.
        out = self.transformer(tgt=h, memory=h, tgt_mask=mask)

        # 4. Project to vocabulary
        logits = self.fc_out(out)  # [B, S, V]
        return logits

class ModelGeneratorHandler(GeneralModelHandler):
    MODEL_NAME = "generator_0"

    def __init__(self, model: nn.Module, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, scheduler, self.MODEL_NAME)
        self.criterion = criterion

    def train_step(self, batch):
        tokens, moods, genres = batch
        
        tokens = tokens.to(self.device)
        moods = moods.to(self.device)
        genres = genres.to(self.device)
        
        inp = tokens[:, :-1]  # [B, SEQ_LEN-1]
        tgt = tokens[:, 1:]    # [B, SEQ_LEN-1]

        # Forward: model predicts logits for each position
        logits = self.model(inp, moods, genres)  # [B, SEQ_LEN-1, vocab_size]

        # Loss calculation (Cross-Entropy):
        # 1. Reshape logits: [B, SEQ_LEN-1, vocab_size] -> [B*(SEQ_LEN-1), vocab_size]
        # 2. Reshape targets: [B, SEQ_LEN-1] -> [B*(SEQ_LEN-1)]
        # 3. For each position, compute CE between predicted distribution and true token
        # 4. ignore_index=0 means padding tokens (id=0) don't contribute to loss
        loss = self.criterion(
            logits.reshape(-1, vocab_size),  # [B*(SEQ_LEN-1), vocab_size]
            tgt.reshape(-1),                  # [B*(SEQ_LEN-1)]
        )

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
    model = ModelGenerator(
        vocab_size=vocab_size,
        num_moods=Config.NUM_MOODS,
        num_genres=Config.NUM_GENRES,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    
    handler = ModelGeneratorHandler(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Generator parameters: {total_params:,}")
    
    handler.train(dataloader=dataloader, epochs=Config.EPOCHS)

    ## Example inference after training (using the first batch from the dataloader)
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(Config.DEVICE)
        x_batch = x_batch[0, 0:1].unsqueeze(0)
        print(x_batch.shape)
        generated_tokens = handler.generate(x_batch)
        print(generated_tokens[:20])
        save_midi(generated_tokens, tokenizer, "generated_midi.mid")
        break