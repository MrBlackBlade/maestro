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
import argparse
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import get_tokenizer, save_midi, top_k_top_p_sample
# from src.dataloaders.full_dataloader import get_full_dataloader
from src.dataloaders import singleton_dataloader
from src.dataloaders.singleton_dataloader import get_singleton_dataloader
# from src.dataloaders.dataset_cached import get_cached_dataloader
# from src.dataloaders.modified_dataset_cached import get_modified_cached_dataloader
from src.dataloaders.mood_dataset_cached import get_mood_cached_dataloader
from src.models.general_model_handler import GeneralModelHandler
from src.models.cached_transformer import (
    CachedTransformerEncoderLayer,
    CachedTransformerEncoder,
    KVCache,
)

class MoodClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = Config.D_MODEL,
        nhead: int = Config.NUM_HEADS,
        num_layers: int = Config.CLASSIFIER_NUM_LAYERS,
        dim_feedforward: int = Config.DIM_FEEDFORWARD,
        dropout: float = Config.DROPOUT,
        max_seq_len: int = Config.MAX_SEQ_LEN,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # ONLY Tokens and Positions! No mood_emb.
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = CachedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = CachedTransformerEncoder(encoder_layer, num_layers=num_layers)

        # Maps the final transformer hidden state to mood scores
        self.mood_classifier = nn.Linear(self.d_model, Config.NUM_MOODS)

        self.emb_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.mood_classifier.weight, std=0.02)
        nn.init.zeros_(self.mood_classifier.bias)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, S = x.shape
        device = x.device

        # Standard transformer embedding step
        positions = torch.arange(start_pos, start_pos + S, device=device).unsqueeze(0)
        h = self.token_emb(x) * math.sqrt(self.d_model)
        h = h + self.pos_emb(positions)

        h = self.emb_norm(h)
        h = self.drop(h)

        if kv_cache is not None:
            out = self.transformer(h, kv_cache=kv_cache)
        else:
            out = self.transformer(h, is_causal=True)

        logits = self.mood_classifier(out) # [B, S, NUM_MOODS]
        return logits

class MoodClassifierHandler(GeneralModelHandler):
    MODEL_NAME = "classifier_4"

    def __init__(self, model: nn.Module, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, scheduler, self.MODEL_NAME)
        self.criterion = criterion

    def train_step(self, batch):
        # We don't pass 'moods' sequence to the model anymore!
        tokens, _, true_mood = batch 
        
        tokens = tokens.to(self.device)
        true_mood = true_mood.to(self.device)
        
        # Chop off the last token just like the generator does
        inp = tokens[:, :-1]    # [B, SEQ_LEN - 1]

        # Forward pass: purely analyzing the notes
        logits = self.model(inp)  # [B, SEQ_LEN - 1, NUM_MOODS]

        B, S, _ = logits.shape

        # ── Position-warmup loss (Fix 1) ──────────────────────────────────
        # Skip the first CLASSIFIER_WARMUP positions from the loss entirely.
        # Early positions have seen too few tokens to carry mood signal; including
        # them causes the model to learn a near-uniform "hedge" distribution as
        # its global minimum. Only computing loss over positions with adequate
        # context forces the classifier to learn genuine mood features.
        W = min(Config.CLASSIFIER_WARMUP, S - 1)  # guard: must leave at least 1 position
        mood_target = true_mood.unsqueeze(1).expand(B, S)   # [B, S]

        mood_loss = F.cross_entropy(
            logits[:, W:, :].reshape(-1, Config.NUM_MOODS),
            mood_target[:, W:].reshape(-1),
        )

        return mood_loss

    @torch.no_grad()
    def inference(
        self, 
        tokens: torch.Tensor, 
        kv_cache: KVCache | None = None, 
        start_pos: int = 0
    ):
        """
        Evaluates a sequence of tokens and predicts the mood.
        
        Parameters:
        tokens: Tensor of shape [S] or [B, S]
        
        Returns:
        predicted_mood_id: The integer ID of the most likely mood (Shape: [B])
        mood_probs: The full softmax probability distribution (Shape: [B, NUM_MOODS])
        """
        self.model.eval()
        self.model.to(self.device)

        # 1. Format the input safely (ensure it has a Batch dimension)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            
        tokens = tokens.to(self.device)

        # 2. Forward Pass
        # Returns shape: [B, S, NUM_MOODS]
        logits = self.model(tokens, kv_cache=kv_cache, start_pos=start_pos)

        # 3. We only care about the model's judgment at the very end of the sequence
        last_logits = logits[:, -1, :] # Shape: [B, NUM_MOODS]

        # 4. Convert raw logits to percentages (probabilities)
        mood_probs = F.softmax(last_logits, dim=-1)

        # 5. Get the definitive answer (the mood ID with the highest probability)
        predicted_mood_id = torch.argmax(mood_probs, dim=-1)

        # Return both the raw ID and the probabilities in case you want to print them
        return predicted_mood_id, mood_probs
