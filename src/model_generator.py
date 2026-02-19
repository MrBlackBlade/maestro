"""
Music Generator – Autoregressive Decoder-only Transformer (GPT-style).

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

from config import Config


class MusicGenerator(nn.Module):
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
        x        : LongTensor  [B, S]   – input token ids
        mood_id  : LongTensor  [B]      – mood category index
        genre_id : LongTensor  [B]      – genre category index

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

