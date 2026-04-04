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

class MoodModelGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = Config.D_MODEL,
        nhead: int = Config.NUM_HEADS,
        num_layers: int = Config.NUM_LAYERS,
        dim_feedforward: int = Config.DIM_FEEDFORWARD,
        dropout: float = Config.DROPOUT,
        max_seq_len: int = Config.MAX_SEQ_LEN,
        num_moods: int = Config.NUM_MOODS + 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # ---- Token + Position embeddings ----
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # ---- Conditioning embeddings (discrete) ----
        self.mood_emb = nn.Embedding(num_moods, d_model)

        # ---- Transformer stack (decoder-only, GPT-style) ----
        # CachedTransformerEncoder is a drop-in for nn.TransformerEncoder
        # that threads an optional KVCache through each layer during inference.
        # Parameter names match PyTorch's built-in layers for checkpoint compat.
        encoder_layer = CachedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = CachedTransformerEncoder(encoder_layer, num_layers=num_layers)

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
        nn.init.normal_(self.fc_out.weight, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mood_id: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : LongTensor  [B, S]   - input token ids
        mood_id   : LongTensor  [B, S]   - mood category index
        kv_cache  : optional KVCache for cached inference (None = training)
        start_pos : positional-embedding offset for the first token in ``x``

        Returns
        -------
        logits    : FloatTensor [B, S, vocab_size]
        """
        B, S = x.shape
        device = x.device

        # Keep conditioning length aligned with token sequence length.
        if mood_id.dim() == 1:
            mood_id = mood_id.unsqueeze(1)
        if mood_id.size(0) != B:
            raise ValueError(f"Batch mismatch: x has batch {B}, mood_id has batch {mood_id.size(0)}")
        if mood_id.size(1) != S:
            if mood_id.size(1) == 1:
                mood_id = mood_id.expand(B, S)
            elif mood_id.size(1) > S:
                mood_id = mood_id[:, -S:]
            else:
                pad = mood_id[:, -1:].expand(B, S - mood_id.size(1))
                mood_id = torch.cat([mood_id, pad], dim=1)

        positions = torch.arange(start_pos, start_pos + S, device=device).unsqueeze(0)
        h = self.token_emb(x) * math.sqrt(self.d_model)
        h = h + self.pos_emb(positions)
        cond = self.mood_emb(mood_id)
        h = h + cond

        h = self.emb_norm(h)
        h = self.drop(h)

        if kv_cache is not None:
            out = self.transformer(h, kv_cache=kv_cache)
        else:
            out = self.transformer(h, is_causal=True)

        logits = self.fc_out(out)
        return logits

class MoodModelGeneratorHandler(GeneralModelHandler):
    MODEL_NAME = "generator_2"

    def __init__(self, model: nn.Module, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, scheduler, self.MODEL_NAME)
        self.criterion = criterion

    def train_step(self, batch):
        tokens, moods = batch
        
        tokens = tokens.to(self.device)
        moods = moods.to(self.device)
        
        inp = tokens[:, :-1]    # [B, SEQ_LEN]
        tgt = tokens[:, 1:]     # [B, SEQ_LEN]

        # Forward: model predicts logits for each position
        logits = self.model(inp, moods)  # [B, SEQ_LEN, vocab_size]

        # Loss calculation (Cross-Entropy):
        # 1. Reshape logits: [B, SEQ_LEN-1, vocab_size] -> [B*(SEQ_LEN-1), vocab_size]
        # 2. Reshape targets: [B, SEQ_LEN-1] -> [B*(SEQ_LEN-1)]
        # 3. For each position, compute CE between predicted distribution and true token
        # 4. ignore_index=0 means padding tokens (id=0) don't contribute to loss
        loss = self.criterion(
            logits.reshape(-1, self.model.vocab_size),
            tgt.reshape(-1),
        )

        return loss
    
    # def generate(
    #     self,
    #     mood: str,
    #     genre: str,
    #     start: list[int] | None = None,
    #     target_length: int = 4096,
    #     temperature: float = Config.TEMPERATURE,
    #     top_k: int = Config.TOP_K,
    #     top_p: float = Config.TOP_P,
    # ):
    #     self.model.eval()
    #     self.model.to(self.device)

    #     tokenizer = get_tokenizer()

    #     if start is None or len(start) == 0:
    #         bos_id = 1
    #         if hasattr(tokenizer, "special_tokens_ids") and len(tokenizer.special_tokens_ids) > 1:
    #             bos_id = tokenizer.special_tokens_ids[1]
    #         bos_id = min(bos_id, self.model.vocab_size - 1)
    #         start = [bos_id]

    #     m_id = torch.tensor([Config.MOOD_TO_ID[mood]], device=self.device)
    #     g_id = torch.tensor([Config.GENRE_TO_ID[genre]], device=self.device)
    #     sequence = torch.tensor([start], dtype=torch.long, device=self.device)

    #     progress = tqdm(range(target_length), desc="Generating MIDI")
    #     with torch.no_grad():
    #         for _ in progress:
    #             ctx = sequence[:, -(Config.SEQ_LEN-1):]
    #             logits = self.model(ctx, m_id, g_id)
    #             next_logits = logits[:, -1, :]

    #             next_token = top_k_top_p_sample(
    #                 next_logits, top_k, top_p, temperature, self.model.vocab_size
    #             )
    #             next_token = torch.clamp(next_token, 0, self.model.vocab_size - 1)
    #             sequence = torch.cat([sequence, next_token], dim=1)

    #     return sequence.squeeze(0).cpu().tolist()
    @torch.inference_mode()
    def generate_single_step(
        self,
        current_tokens: torch.Tensor,                   # Tensor Shape: (1, seq_len)
        current_moods: torch.Tensor,                    # Tensor Shape: (1, seq_len)
        target_mood_id: int,                            # Int: The mood you want for THIS specific step
        uncond_mood_id: int = Config.NUM_MOODS,         # Int: Your unconditional/null mood ID
        cfg_scale=3.0,                                  # Float: Guidance strength
        temperature=1.20,                               # Float: Randomness/creativity
        top_p=0.95,                                     # Float: Nucleus filtering threshold
        cond_cache: KVCache | None = None,              # Pre-allocated conditional KV cache
        uncond_cache: KVCache | None = None,            # Pre-allocated unconditional KV cache
    ):
        use_cache = Config.USE_KV_CACHE and cond_cache is not None

        if use_cache:
            if cond_cache.is_full():
                # Cache overflow: reset and partial refill.  Keep half the
                # window so the next ~MAX_SEQ_LEN/2 steps run from cache.
                cond_cache.reset()
                uncond_cache.reset()
                refill_len = min(current_tokens.size(1), Config.MAX_SEQ_LEN // 2)
                ctx = current_tokens[:, -refill_len:].to(self.device)
                cond_mood_seq = current_moods[:, -refill_len:].to(self.device)
                uncond_mood_seq = torch.full_like(cond_mood_seq, uncond_mood_id)
                start_pos = 0
            else:
                # Normal cached decode: process only the latest token.
                ctx = current_tokens[:, -1:].to(self.device)
                cond_mood_seq = current_moods[:, -1:].to(self.device)
                uncond_mood_seq = torch.full_like(cond_mood_seq, uncond_mood_id)
                start_pos = cond_cache.seq_len

            cond_logits = self.model(
                ctx, cond_mood_seq, kv_cache=cond_cache, start_pos=start_pos,
            )[:, -1, :]
            uncond_logits = self.model(
                ctx, uncond_mood_seq, kv_cache=uncond_cache, start_pos=start_pos,
            )[:, -1, :]
        else:
            # No cache: full-context forward (original behaviour).
            ctx_len = min(current_tokens.size(1), Config.SEQ_LEN)
            ctx = current_tokens[:, -ctx_len:].to(self.device)
            cond_mood_seq = current_moods[:, -ctx_len:].to(self.device)
            uncond_mood_seq = torch.full((1, ctx_len), uncond_mood_id, dtype=torch.long, device=self.device)

            cond_logits = self.model(ctx, cond_mood_seq)[:, -1, :]
            uncond_logits = self.model(ctx, uncond_mood_seq)[:, -1, :]

        # ---- Classifier-Free Guidance ----
        final_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)

        # ---- Temperature ----
        final_logits = final_logits / temperature

        # ---- Top-p (Nucleus) Filtering ----
        probs = F.softmax(final_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # ---- Sample ----
        next_token = torch.multinomial(probs, num_samples=1)
        next_mood = torch.full((1, 1), target_mood_id, dtype=torch.long, device=self.device)

        updated_tokens = torch.cat((current_tokens, next_token), dim=1)
        updated_moods = torch.cat((current_moods, next_mood), dim=1)
        return updated_tokens, updated_moods, next_token
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoodModelGenerator – train or generate")
    sub = parser.add_subparsers(dest="command")

    tr = sub.add_parser("train")
    tr.add_argument("--epochs", type=int, default=Config.EPOCHS)
    tr.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    tr.add_argument("--resume-epoch", type=int, default=None,
                     help="Resume from this checkpoint epoch before training")

    gen = sub.add_parser("generate")
    gen.add_argument("--epoch", type=int, default=None,
                      help="Checkpoint epoch to load (default: best)")
    gen.add_argument("--mood", type=str, default="magnificent", choices=Config.MOODS)
    gen.add_argument("--length", type=int, default=4096)
    gen.add_argument("--output", type=str, default="generated_midi.mid")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    device = Config.DEVICE
    print(f"Device: {device}")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    model = MoodModelGenerator(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs if args.command == "train" else Config.EPOCHS,
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    handler = MoodModelGeneratorHandler(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Generator parameters: {total_params:,}")

    # ── Train ────────────────────────────────────────────────────────────
    if args.command == "train":
        if args.resume_epoch is not None:
            handler.load_checkpoint(epoch=args.resume_epoch)
            print(f"Resumed from epoch {args.resume_epoch}")

        dataloader = get_mood_cached_dataloader(
            batch_size=args.batch_size,
            num_workers=Config.NUM_WORKERS,
            persistent_workers=Config.PERSISTENT_WORKERS,
            prefetch_factor=Config.PREFETCH_FACTOR,
        )
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Using {Config.NUM_WORKERS} parallel workers for data loading")
        handler.train(dataloader=dataloader, epochs=args.epochs)

    # ── Generate ─────────────────────────────────────────────────────────
    elif args.command == "generate":
        handler.load_checkpoint(epoch=args.epoch)
        model.eval()

        if Config.USE_KV_CACHE:
            cond_cache = KVCache.from_model(model)
            uncond_cache = KVCache.from_model(model)
        else:
            cond_cache = uncond_cache = None

        target_mood_id = Config.MOOD_TO_ID[args.mood]
        current_tokens = torch.tensor([[1]], device=device)
        current_moods = torch.tensor([[target_mood_id]], device=device)

        for step in tqdm(range(args.length), desc="Generating MIDI"):
            current_tokens, current_moods, next_token = handler.generate_single_step(
                current_tokens, current_moods, target_mood_id,
                cond_cache=cond_cache, uncond_cache=uncond_cache,
            )

        generated_tokens = current_tokens.squeeze(0).cpu().tolist()
        save_midi(generated_tokens, tokenizer, args.output)
        print(f"Saved {len(generated_tokens)} tokens to {args.output}")