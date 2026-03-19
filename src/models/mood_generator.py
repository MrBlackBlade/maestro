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
        # We use TransformerEncoder (self-attention only) with a causal mask.
        # NOT TransformerDecoder, which has cross-attention that would leak
        # future information through the unmasked memory path.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : LongTensor  [B, S]   - input token ids
        mood_id  : LongTensor  [B, S]   - mood category index

        Returns
        -------
        logits   : FloatTensor [B, S, vocab_size]
        """
        B, S = x.shape
        device = x.device

        positions = torch.arange(S, device=device).unsqueeze(0)
        h = self.token_emb(x) * math.sqrt(self.d_model)
        h = h + self.pos_emb(positions)
        print(h.shape)
        print(mood_id.shape)
        cond = self.mood_emb(mood_id)
        h = h + cond

        h = self.emb_norm(h)
        h = self.drop(h)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        out = self.transformer(h, mask=causal_mask, is_causal=True)

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
    @torch.no_grad()
    def generate_single_step(
        self, 
        current_tokens: torch.Tensor,                   # Tensor Shape: (1, seq_len)
        target_mood_id: int,                            # Int: The mood you want for THIS specific step
        uncond_mood_id: int = Config.NUM_MOODS,         # Int: Your unconditional/null mood ID
        cfg_scale=3.0,                                  # Float: Guidance strength
        temperature=1.05,                               # Float: Randomness/creativity
        top_p=0.95,                                     # Float: Nucleus filtering threshold
        # rep_penalty=1.15,                             # Float: Repetition penalty factor (> 1.0)
        # rep_window=50                                 # Int: How many past tokens to look at for penalty
    ):
        self.model.eval()
        self.model.to(self.device)
        
        # 1. Build the mood sequences for the current context
        # This floods the entire context window with your target mood to force the pivot
        cond_mood_seq = torch.full((1, Config.SEQ_LEN), target_mood_id, dtype=torch.long, device=device)
        uncond_mood_seq = torch.full((1, Config.SEQ_LEN), uncond_mood_id, dtype=torch.long, device=device)
        
        # 2. Dual Forward Pass for CFG
        cond_logits = self.model(current_tokens, cond_mood_seq)[:, -1, :]
        uncond_logits = self.model(current_tokens, uncond_mood_seq)[:, -1, :]
        
        # 3. Apply Classifier-Free Guidance
        final_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
        
        # # 4. Apply Repetition Penalty
        # if rep_penalty > 1.0:
        #     # Look only at the most recent tokens up to rep_window
        #     recent_tokens = current_tokens[0, -rep_window:].tolist()
        #     for token in set(recent_tokens):
        #         # If logit is positive, divide to reduce it. If negative, multiply to make it more negative.
        #         if final_logits[0, token] > 0:
        #             final_logits[0, token] /= rep_penalty
        #         else:
        #             final_logits[0, token] *= rep_penalty
                    
        # 5. Apply Temperature
        final_logits = final_logits / temperature
        
        # 6. Apply Top-p (Nucleus) Filtering
        probs = F.softmax(final_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        
        # Renormalize the probabilities so they sum to 1.0 again
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # 7. Sample the next token
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 8. Append to the sequence
        updated_tokens = torch.cat((current_tokens, next_token), dim=1)
        
        return updated_tokens, next_token
    
if __name__ == "__main__":
    device = Config.DEVICE
    print(f"Device: {device}")

    # ---- Data ----
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    ## TO BE IMPLEMENTED
    # dataloader = get_full_dataloader()
    dataloader = get_mood_cached_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=Config.PERSISTENT_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
        sample_factor=0.2
    )
    
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Using {Config.NUM_WORKERS} parallel workers for data loading")

    # ---- Model ----
    model = MoodModelGenerator(
        vocab_size=vocab_size,
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
    
    handler = MoodModelGeneratorHandler(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Generator parameters: {total_params:,}")
    
    handler.train(dataloader=dataloader, epochs=10)
    # handler.load_checkpoint(epoch=9)

    current_tokens = torch.tensor([[1]], device=Config.DEVICE)
    current_mood = Config.MOOD_TO_ID["romantic"]
    target_length = 4096
    for step in range(target_length):
        if step == 2048:
            current_mood = Config.MOOD_TO_ID["warm"]
        current_tokens, next_token = handler.generate_single_step(current_tokens, current_mood)

    generated_tokens = current_tokens.squeeze(0).cpu().tolist()
    print(generated_tokens[0:20])
    print(generated_tokens[2048:2068])
    save_midi(generated_tokens, tokenizer, "generated_midi.mid")