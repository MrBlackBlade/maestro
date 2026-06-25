
import argparse
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.core.config import Config
from src.core.audio_engine import AudioEngine
from src.core.utils import get_tokenizer, save_midi
from src.dataloaders.mood_dataset_cached import get_mood_cached_dataloader
from src.models.general_model_handler import GeneralModelHandler
from src.models.mood_generator import MoodModelGenerator
from src.models.mood_classifier import MoodClassifier
from src.models.mood_classifier import MoodClassifierHandler
from src.models.cached_transformer import KVCache


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Chrollo(MoodModelGenerator):
    """MoodModelGenerator + secondary mood classifier."""

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        mood_id: torch.Tensor,
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
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

        return logits


# ---------------------------------------------------------------------------
# Handler (training + generation)
# ---------------------------------------------------------------------------

class ChrolloHandler(GeneralModelHandler):
    MODEL_NAME = "chrollo_0"

    def __init__(self, model: nn.Module, optimizer, criterion, scheduler, classifier_handler: MoodClassifierHandler):
        super().__init__(model, optimizer, scheduler, self.MODEL_NAME)
        self.criterion = criterion
        self.classifier_handler = classifier_handler
        self.dynamic_temperature = 0
        self.current_entropy = 0
        self.delta_entropy_list = []

    # ── Training ─────────────────────────────────────────────────────────

    def train_step(self, batch):
        tokens, moods, *_ = batch
        
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
        temperature: float = Config.D_TEMP_MIN,
        top_p: float = 0.95,
        generator_cache: KVCache | None = None,
        classifier_cache: KVCache | None = None,
    ):
        """One autoregressive step with negative classifier-free guidance.

        Runs a single batched forward pass over ``NUM_MOODS + 1`` branches
        (unconditional + one per mood), uses the classifier to identify
        penalty moods, and combines logits via the negative CFG formula.

        Returns ``(updated_tokens, updated_moods, next_token)``.
        """
        num_branches = Config.NUM_MOODS + 1
        use_cache = Config.USE_KV_CACHE and (
            generator_cache is not None
            and classifier_cache is not None
        )

        if use_cache:
            if generator_cache.is_full():
                # ── CRITICAL: Synchronize resets! ──
                generator_cache.reset()
                classifier_cache.reset() 

                refill_len = min(current_tokens.size(1), Config.MAX_SEQ_LEN // 2)
                
                # 1. Base sequence for the classifier [1, refill_len]
                classifier_ctx = current_tokens[:, -refill_len:].to(self.device)
                
                # 2. Expanded sequence for the generator [num_branches, refill_len]
                ctx = classifier_ctx.expand(num_branches, -1)
                start_pos = 0
            else:
                # 1. Newest token only [1, 1]
                classifier_ctx = current_tokens[:, -1:].to(self.device)
                
                # 2. Expand for generator [num_branches, 1]
                ctx = classifier_ctx.expand(num_branches, -1)
                start_pos = generator_cache.seq_len

            mood_batch = self._build_mood_batch(ctx.size(1), uncond_mood_id)
            logits = self.model(
                ctx, mood_batch,
                kv_cache=generator_cache, start_pos=start_pos
            )
            
        else:
            ctx_len = min(current_tokens.size(1), Config.SEQ_LEN)
            
            # Base sequence
            classifier_ctx = current_tokens[:, -ctx_len:].to(self.device)
            
            # Expanded sequence
            ctx = classifier_ctx.expand(num_branches, -1)

            mood_batch = self._build_mood_batch(ctx.size(1), uncond_mood_id)
            logits = self.model(ctx, mood_batch)

        # Last-position logits and hidden states for every branch
        last_logits = logits[:, -1, :]     # [num_branches, V]

        # ── Classifier → penalty selection ───────────────────────────────
        target_branch = target_mood_id + 1
        _, mood_probs = self.classifier_handler.inference(
            tokens=classifier_ctx,
            kv_cache=classifier_cache,
            start_pos=start_pos
        )
        
        # Since the handler returns shape [B, NUM_MOODS] and our batch size is 1,
        # we squeeze it to get a clean 1D tensor: [NUM_MOODS]
        mood_probs = mood_probs.squeeze(0)

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

        # ── Shannon-Entropy + Dynamic Temperature ────────────────────────
        raw_probs = F.softmax(final_logits, dim=-1)
        entropy = -(raw_probs * torch.log(raw_probs + 1e-9)).sum().item()
        delta_entropy = np.abs(entropy - self.current_entropy)
        if len(self.delta_entropy_list) < 100:
            self.delta_entropy_list.append(delta_entropy)
        else:
            self.delta_entropy_list.pop(0)
            self.delta_entropy_list.append(delta_entropy)
        avg_delta_entropy = np.average(self.delta_entropy_list)
        self.current_entropy = entropy

        if len(self.delta_entropy_list) == 100:
            if avg_delta_entropy < Config.ENTROPY_LOW:
                # Model is overly confident / looping. Build pressure over time.
                self.dynamic_temperature += Config.D_TEMP_UP
            if avg_delta_entropy > Config.ENTROPY_HIGH:
                # Model is being creative. Release the pressure.
                self.dynamic_temperature = max(0.0, self.dynamic_temperature - Config.D_TEMP_DOWN)

            # Cap the maximum pressure so it doesn't devolve into pure noise
            self.dynamic_temperature = min(self.dynamic_temperature, Config.D_TEMP_MAX)

        current_temp = temperature + self.dynamic_temperature

        # ── Temperature + top-p + sample ─────────────────────────────────
        final_logits = final_logits / current_temp

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
        return updated_tokens, updated_moods, next_token#, avg_delta_entropy, current_temp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chrollo - train or generate with negative CFG",
    )
    sub = parser.add_subparsers(dest="command")

    tr = sub.add_parser("train")
    tr.add_argument("--target", type=str, choices=Config.TRAIN_CHOICES, default=Config.DEFAULT_TRAIN_CHOICE)
    tr.add_argument("--epochs", type=int, default=Config.EPOCHS)
    tr.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    tr.add_argument("--resume-epoch", type=int, default=None,
                     help="Resume from this checkpoint epoch before training")

    gen = sub.add_parser("generate")
    gen.add_argument("--epoch", type=int, default=None,
                      help="Checkpoint epoch to load (default: best)")
    gen.add_argument("--mood", type=str, default="romantic", choices=Config.MOODS)
    gen.add_argument("--transition-mood", type=str, default="magnificent", choices=Config.MOODS,
                      help="Mood to transition to during generation")
    gen.add_argument("--transition-step", type=int, default=1024,
                      help="Step at which to transition the mood")
    gen.add_argument("--length", type=int, default=6000)
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

    mood_classifier = MoodClassifier(vocab_size=vocab_size).to(device)
    mood_classifier_optimizer = torch.optim.AdamW(
        mood_classifier.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
    )
    mood_classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        mood_classifier_optimizer,
        T_max=args.epochs if args.command == "train" else Config.EPOCHS,
        eta_min=1e-6,
    )
    mood_classifier_criterion = nn.CrossEntropyLoss()
    mood_classifier_handler = MoodClassifierHandler(
        model=mood_classifier, optimizer=mood_classifier_optimizer, scheduler=mood_classifier_scheduler, criterion=mood_classifier_criterion,
    )

    chrollo = Chrollo(vocab_size=vocab_size).to(device)
    chrollo_optimizer = torch.optim.AdamW(
        chrollo.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
    )
    chrollo_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        chrollo_optimizer,
        T_max=args.epochs if args.command == "train" else Config.EPOCHS,
        eta_min=1e-6,
    )
    chrollo_criterion = nn.CrossEntropyLoss(ignore_index=0)
    chrollo_handler = ChrolloHandler(
        model=chrollo, optimizer=chrollo_optimizer, scheduler=chrollo_scheduler, criterion=chrollo_criterion,
        classifier_handler=mood_classifier_handler,
    )

    total_params = sum(p.numel() for p in chrollo.parameters())
    print(f"Chrollo parameters: {total_params:,}")

    # ── Train ────────────────────────────────────────────────────────────
    if args.command == "train":
        start_epoch = 1
        if args.resume_epoch is not None:
            chrollo_handler.load_checkpoint(epoch=args.resume_epoch)
            start_epoch = args.resume_epoch + 1
            print(f"Resumed from epoch {args.resume_epoch}, continuing at epoch {start_epoch}")

        dataloader = get_mood_cached_dataloader(
            batch_size=args.batch_size,
            num_workers=Config.NUM_WORKERS,
            persistent_workers=Config.PERSISTENT_WORKERS,
            prefetch_factor=Config.PREFETCH_FACTOR,
        )
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Using {Config.NUM_WORKERS} parallel workers for data loading")
        if args.target == "generator":
            chrollo_handler.train(dataloader=dataloader, epochs=args.epochs, start_epoch=start_epoch)
        elif args.target == "classifier":
            mood_classifier_handler.train(dataloader=dataloader, epochs=args.epochs, start_epoch=start_epoch)

    # ── Generate ─────────────────────────────────────────────────────────
    elif args.command == "generate":
        # entropy_list = []
        # temp_list = []
        chrollo_handler.load_checkpoint(epoch=args.epoch)
        chrollo.eval()

        num_branches = Config.NUM_MOODS + 1
        if Config.USE_KV_CACHE:
            generator_cache = KVCache.from_model(chrollo, batch_size=num_branches)
            classifier_cache = KVCache.from_model(mood_classifier)
        else:
            generator_cache = None
            classifier_cache = None
        
        try: 
            audio_engine = AudioEngine()
            target_mood_id = Config.MOOD_TO_ID[args.mood]
            
            transition_mood_id = None
            if hasattr(args, 'transition_mood') and args.transition_mood is not None:
                transition_mood_id = Config.MOOD_TO_ID[args.transition_mood]
                
            current_tokens = torch.tensor([[1]], device=device)
            current_moods = torch.tensor([[target_mood_id]], device=device)
            audio_engine.push_token(1)

            with tqdm(total=args.length, desc="Generating MIDI") as pbar:
                step = 0
                while step < args.length:
                    if transition_mood_id is not None and step == args.transition_step:
                        target_mood_id = transition_mood_id
                        print(f"\n[Step {step}] Transitioning mood to: {args.transition_mood}")
                        
                    while (audio_engine.audio_queue.qsize() > 1):
                        time.sleep(0.1)
                    current_tokens, current_moods, next_token = chrollo_handler.generate_single_step(
                        current_tokens, current_moods, target_mood_id,
                        generator_cache=generator_cache,
                        classifier_cache=classifier_cache,
                    )
                    # entropy_list.append(entropy)
                    # temp_list.append(current_temp)
                    audio_engine.push_token(next_token.item())
                    step += 1
                    pbar.update(1)
        finally:
            audio_engine.push_token(4, stop=True)
            audio_engine.playback_done.wait()


        generated_tokens = current_tokens.squeeze(0).cpu().tolist()
        save_midi(generated_tokens, tokenizer, args.output)
        # import json
        # json.dump(entropy_list, open("entropy.json", "w"), indent=4)
        # json.dump(temp_list, open("temperature.json", "w"), indent=4)
        print(f"Saved {len(generated_tokens)} tokens to {args.output}")
