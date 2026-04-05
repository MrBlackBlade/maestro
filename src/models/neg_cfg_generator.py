
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import get_tokenizer, save_midi
from src.dataloaders.mood_dataset_cached import get_mood_cached_dataloader
from src.models.general_model_handler import GeneralModelHandler
from src.models.mood_generator import MoodModelGenerator
from src.models.cached_transformer import KVCache


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NegCFGGenerator(MoodModelGenerator):
    """MoodModelGenerator + linear mood classifier head."""

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)
        self.mood_classifier = nn.Linear(self.d_model, Config.NUM_MOODS)
        nn.init.normal_(self.mood_classifier.weight, std=0.02)
        nn.init.zeros_(self.mood_classifier.bias)

    def forward(
        self,
        x: torch.Tensor,
        mood_id: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, S = x.shape
        device = x.device

        if mood_id.dim() == 1:
            mood_id = mood_id.unsqueeze(1)
        if mood_id.size(0) != B:
            raise ValueError(
                f"Batch mismatch: x has batch {B}, mood_id has batch {mood_id.size(0)}"
            )
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
        h = h + self.mood_emb(mood_id)

        h = self.emb_norm(h)
        h = self.drop(h)

        if kv_cache is not None:
            out = self.transformer(h, kv_cache=kv_cache)
        else:
            out = self.transformer(h, is_causal=True)

        logits = self.fc_out(out)
        if return_hidden:
            return logits, out
        return logits


# ---------------------------------------------------------------------------
# Handler (training + generation)
# ---------------------------------------------------------------------------

class NegCFGGeneratorHandler(GeneralModelHandler):
    MODEL_NAME = "generator_3"

    def __init__(self, model: nn.Module, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, scheduler, self.MODEL_NAME)
        self.criterion = criterion

    # ── Training ─────────────────────────────────────────────────────────

    def train_step(self, batch):
        tokens, moods, true_mood = batch

        tokens = tokens.to(self.device)
        moods = moods.to(self.device)
        true_mood = true_mood.to(self.device)

        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]

        logits, hidden = self.model(inp, moods, return_hidden=True)

        token_loss = self.criterion(
            logits.reshape(-1, self.model.vocab_size),
            tgt.reshape(-1),
        )

        cls_input = hidden.detach() if Config.MOOD_CLASSIFIER_DETACH else hidden
        mood_logits = self.model.mood_classifier(cls_input)
        B, S, _ = mood_logits.shape
        mood_target = true_mood.unsqueeze(1).expand(B, S)
        mood_loss = F.cross_entropy(
            mood_logits.reshape(-1, Config.NUM_MOODS),
            mood_target.reshape(-1),
        )

        return token_loss + Config.MOOD_LOSS_WEIGHT * mood_loss

    # ── Inference helpers ────────────────────────────────────────────────

    def _build_mood_batch(self, seq_len: int, uncond_mood_id: int) -> torch.Tensor:
        """Build ``[NUM_MOODS + 1, seq_len]`` mood tensor.

        Row 0 is unconditional; row ``i + 1`` uses mood ``i``.
        """
        num_branches = Config.NUM_MOODS + 1
        mood_batch = torch.empty(
            num_branches, seq_len, dtype=torch.long, device=self.device,
        )
        mood_batch[0].fill_(uncond_mood_id)
        for i in range(Config.NUM_MOODS):
            mood_batch[i + 1].fill_(i)
        return mood_batch

    # ── Generation (negative CFG) ────────────────────────────────────────

    @torch.inference_mode()
    def generate_single_step(
        self,
        current_tokens: torch.Tensor,
        current_moods: torch.Tensor,
        target_mood_id: int,
        uncond_mood_id: int = Config.NUM_MOODS,
        pos_cfg_scale: float = Config.POS_CFG_SCALE,
        neg_cfg_scale: float = Config.NEG_CFG_SCALE,
        temperature: float = 1.20,
        top_p: float = 0.95,
        cache: KVCache | None = None,
    ):
        """One autoregressive step with negative classifier-free guidance.

        Runs a single batched forward pass over ``NUM_MOODS + 1`` branches
        (unconditional + one per mood), uses the classifier to identify
        penalty moods, and combines logits via the negative CFG formula.

        Returns ``(updated_tokens, updated_moods, next_token)``.
        """
        num_branches = Config.NUM_MOODS + 1
        use_cache = Config.USE_KV_CACHE and cache is not None

        if use_cache:
            if cache.is_full():
                cache.reset()
                refill_len = min(current_tokens.size(1), Config.MAX_SEQ_LEN // 2)
                ctx = current_tokens[:, -refill_len:].to(self.device)
                ctx = ctx.expand(num_branches, -1)
                start_pos = 0
            else:
                ctx = current_tokens[:, -1:].to(self.device)
                ctx = ctx.expand(num_branches, -1)
                start_pos = cache.seq_len

            mood_batch = self._build_mood_batch(ctx.size(1), uncond_mood_id)
            logits, hidden = self.model(
                ctx, mood_batch,
                kv_cache=cache, start_pos=start_pos, return_hidden=True,
            )
        else:
            ctx_len = min(current_tokens.size(1), Config.SEQ_LEN)
            ctx = current_tokens[:, -ctx_len:].to(self.device)
            ctx = ctx.expand(num_branches, -1)

            mood_batch = self._build_mood_batch(ctx.size(1), uncond_mood_id)
            logits, hidden = self.model(ctx, mood_batch, return_hidden=True)

        # Last-position logits and hidden states for every branch
        last_logits = logits[:, -1, :]     # [num_branches, V]
        last_hidden = hidden[:, -1, :]     # [num_branches, D]

        # ── Classifier → penalty selection ───────────────────────────────
        target_branch = target_mood_id + 1
        mood_probs = F.softmax(
            self.model.mood_classifier(last_hidden[target_branch]), dim=-1,
        )  # [NUM_MOODS]

        target_prob = mood_probs[target_mood_id]
        penalty_mask = mood_probs > target_prob
        penalty_mask[target_mood_id] = False

        # ── Negative CFG logit combination ───────────────────────────────
        uncond_logits = last_logits[0]
        cond_logits = last_logits[target_branch]

        final_logits = uncond_logits + pos_cfg_scale * (cond_logits - uncond_logits)

        penalty_indices = penalty_mask.nonzero(as_tuple=True)[0]
        if penalty_indices.numel() > 0:
            branch_indices = penalty_indices + 1
            penalty_logits = last_logits[branch_indices]               # [K, V]
            scales = neg_cfg_scale * mood_probs[penalty_indices]       # [K]
            diffs = penalty_logits - uncond_logits.unsqueeze(0)        # [K, V]
            final_logits = final_logits - (scales.unsqueeze(1) * diffs).sum(0)

        # ── Temperature + top-p + sample ─────────────────────────────────
        final_logits = final_logits / temperature

        probs = F.softmax(final_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False

        scatter_mask = remove_mask.scatter(0, sorted_indices, remove_mask)
        probs[scatter_mask] = 0.0
        probs = probs / probs.sum()

        next_token = torch.multinomial(probs.unsqueeze(0), num_samples=1)
        next_mood = torch.full(
            (1, 1), target_mood_id, dtype=torch.long, device=self.device,
        )

        updated_tokens = torch.cat((current_tokens, next_token), dim=1)
        updated_moods = torch.cat((current_moods, next_mood), dim=1)
        return updated_tokens, updated_moods, next_token


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NegCFGGenerator – train or generate with negative CFG",
    )
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

    model = NegCFGGenerator(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs if args.command == "train" else Config.EPOCHS,
        eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    handler = NegCFGGeneratorHandler(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"NegCFGGenerator parameters: {total_params:,}")

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

        num_branches = Config.NUM_MOODS + 1
        if Config.USE_KV_CACHE:
            cache = KVCache.from_model(model, batch_size=num_branches)
        else:
            cache = None

        target_mood_id = Config.MOOD_TO_ID[args.mood]
        current_tokens = torch.tensor([[1]], device=device)
        current_moods = torch.tensor([[target_mood_id]], device=device)

        for step in tqdm(range(args.length), desc="Generating MIDI"):
            current_tokens, current_moods, _ = handler.generate_single_step(
                current_tokens, current_moods, target_mood_id,
                cache=cache,
            )

        generated_tokens = current_tokens.squeeze(0).cpu().tolist()
        save_midi(generated_tokens, tokenizer, args.output)
        print(f"Saved {len(generated_tokens)} tokens to {args.output}")
