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
from src.dataloaders.modified_dataset_cached import get_modified_cached_dataloader
from src.models.general_model_handler import GeneralModelHandler
from src.models.cached_transformer import (
    CachedTransformerEncoderLayer,
    CachedTransformerEncoder,
    KVCache,
)

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

        # ---- Transformer stack (decoder-only, GPT-style) ----
        # CachedTransformerEncoder is a drop-in for nn.TransformerEncoder
        # that threads an optional KVCache through each layer during inference.
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
        nn.init.normal_(self.genre_emb.weight, std=0.02)
        nn.init.normal_(self.fc_out.weight, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mood_id: torch.Tensor,
        genre_id: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : LongTensor  [B, S]   - input token ids
        mood_id   : LongTensor  [B]      - mood category index
        genre_id  : LongTensor  [B]      - genre category index
        kv_cache  : optional KVCache for cached inference (None = training)
        start_pos : positional-embedding offset for the first token in ``x``

        Returns
        -------
        logits    : FloatTensor [B, S, vocab_size]
        """
        B, S = x.shape
        device = x.device

        positions = torch.arange(start_pos, start_pos + S, device=device).unsqueeze(0)
        h = self.token_emb(x) * math.sqrt(self.d_model)
        h = h + self.pos_emb(positions)

        cond = self.mood_emb(mood_id).unsqueeze(1) + self.genre_emb(genre_id).unsqueeze(1)
        h = h + cond

        h = self.emb_norm(h)
        h = self.drop(h)

        if kv_cache is not None:
            out = self.transformer(h, kv_cache=kv_cache)
        else:
            out = self.transformer(h, is_causal=True)

        logits = self.fc_out(out)
        return logits

class ModelGeneratorHandler(GeneralModelHandler):
    MODEL_NAME = "generator_1"

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
            logits.reshape(-1, self.model.vocab_size),
            tgt.reshape(-1),
        )

        return loss
    
    def generate(
        self,
        mood: str,
        genre: str,
        start: list[int] | None = None,
        target_length: int = 4096,
        temperature: float = Config.TEMPERATURE,
        top_k: int = Config.TOP_K,
        top_p: float = Config.TOP_P,
    ):
        self.model.eval()
        self.model.to(self.device)

        tokenizer = get_tokenizer()

        if start is None or len(start) == 0:
            bos_id = 1
            if hasattr(tokenizer, "special_tokens_ids") and len(tokenizer.special_tokens_ids) > 1:
                bos_id = tokenizer.special_tokens_ids[1]
            bos_id = min(bos_id, self.model.vocab_size - 1)
            start = [bos_id]

        m_id = torch.tensor([Config.MOOD_TO_ID[mood]], device=self.device)
        g_id = torch.tensor([Config.GENRE_TO_ID[genre]], device=self.device)
        sequence = torch.tensor([start], dtype=torch.long, device=self.device)

        cache = KVCache.from_model(self.model) if Config.USE_KV_CACHE else None

        progress = tqdm(range(target_length), desc="Generating MIDI")
        with torch.inference_mode():
            for _ in progress:
                if cache is not None:
                    if cache.is_full():
                        cache.reset()
                        refill_len = min(sequence.size(1), Config.MAX_SEQ_LEN // 2)
                        ctx = sequence[:, -refill_len:]
                        start_pos = 0
                    else:
                        ctx = sequence[:, -1:]
                        start_pos = cache.seq_len
                    logits = self.model(ctx, m_id, g_id, kv_cache=cache, start_pos=start_pos)
                else:
                    ctx = sequence[:, -(Config.SEQ_LEN - 1):]
                    logits = self.model(ctx, m_id, g_id)

                next_logits = logits[:, -1, :]
                next_token = top_k_top_p_sample(
                    next_logits, top_k, top_p, temperature, self.model.vocab_size
                )
                next_token = torch.clamp(next_token, 0, self.model.vocab_size - 1)
                sequence = torch.cat([sequence, next_token], dim=1)

        return sequence.squeeze(0).cpu().tolist()
    
if __name__ == "__main__":
    device = Config.DEVICE
    print(f"Device: {device}")

    # ---- Data ----
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # ---- Model ----
    model = ModelGenerator(
        vocab_size=vocab_size,
        num_moods=Config.NUM_MOODS,
        num_genres=Config.NUM_GENRES,
    ).to(device)
    
    parser = argparse.ArgumentParser(description="Generator – train or generate")
    sub = parser.add_subparsers(dest="command")

    tr = sub.add_parser("train")
    tr.add_argument("--epochs", type=int, default=Config.EPOCHS)
    tr.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    tr.add_argument("--resume-epoch", type=int, default=None,
                    help="Resume from this checkpoint epoch before training")

    gen = sub.add_parser("generate")
    gen.add_argument("--epoch", type=int, default=None,
                     help="Checkpoint epoch to load (default: best)")
    gen.add_argument("--length", type=int, default=4096)
    gen.add_argument("--seed-length", type=int, default=1,
                     help="Number of initial tokens to seed generation")
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=10)
    gen.add_argument("--output", type=str, default="generated_midi.mid")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

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
    
    if args.command == "train":
        dataloader = get_modified_cached_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=Config.PERSISTENT_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
        sample_factor=1.0
        )

        if args.resume_epoch is not None:
            handler.load_checkpoint(epoch=args.resume_epoch)
            start_epoch = args.resume_epoch + 1
            print(f"Resumed from epoch {args.resume_epoch}, continuing at epoch {start_epoch}")
        else:
            start_epoch = 1
        
        handler.train(dataloader=dataloader, epochs=args.epochs, start_epoch=start_epoch)
        
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Using {Config.NUM_WORKERS} parallel workers for data loading")


        total_params = sum(p.numel() for p in model.parameters())
        print(f"Generator parameters: {total_params:,}")

    elif args.command == "generate":
        if args.epoch is not None:
            handler.load_checkpoint(epoch=args.epoch)
            print(f"Loaded checkpoint from epoch {args.epoch}")
        else:
            handler.load_checkpoint()
            print("Loaded best checkpoint")

        start_tokens = [1] * max(1, args.seed_length)
        x_batch = torch.tensor([start_tokens], dtype=torch.long, device=device)
        generated_tokens = handler.generate(
            x_batch,
            target_length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(generated_tokens[:20])
        save_midi(generated_tokens, tokenizer, args.output)