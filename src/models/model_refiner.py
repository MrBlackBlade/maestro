"""
Music Refiner - Levenshtein Transformer (Non-Autoregressive).

Takes a "draft" token sequence (from the Generator) and **edits** it in
parallel: it can *delete* bad tokens and *replace* tokens to make the
sequence more coherent and better aligned with the requested Mood/Genre.

Training strategy
-----------------
The refiner is trained as a **denoising autoencoder**: we take a clean XMIDI
sequence, corrupt it (random token replacements / deletions), and ask the
model to reconstruct the original.

Architecture
------------
* Bidirectional Transformer **Encoder** (sees the full corrupted sequence).
* Two output heads:
    - **Deletion head** - binary classifier per token (keep / delete).
    - **Token head**    - predicts the correct token at each position
                          (replacement / reconstruction).
"""
import torch
import torch.nn as nn

from src.core.config import Config


class LevenshteinRefiner(nn.Module):
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

