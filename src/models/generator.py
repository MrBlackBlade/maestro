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
from src.dataloaders.modified_dataset_cached import get_modified_cached_dataloader
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
    
    def generate(self, mood, genre, start=[1]):
        self.model.eval()
        self.model.to(self.device)

        target_length = 4096
        # window_size = Config.MAX_SEQ_LEN

        # current_bar = set()

        m_id = torch.tensor([Config.MOOD_TO_ID[mood]], device=Config.DEVICE)
        g_id = torch.tensor([Config.GENRE_TO_ID[genre]], device=Config.DEVICE)

        # Start with BOS token (id=1 in most tokenizers, fallback to 1)
        #bos_id = tokenizer.special_tokens_ids[1] if hasattr(tokenizer, "special_tokens_ids") else 1
        # Ensure BOS token is valid
        # bos_id = min(bos_id, self.model.vocab_size - 1)
        # bos_token = 1
        sequence = torch.tensor([start], dtype=torch.long, device=Config.DEVICE)

        position_token_ids = set()
        pitch_token_ids = set()
        program_token_ids = set() 
        bar_token_ids = set()
        tokenizer = get_tokenizer()
        for tok_str, tok_id in tokenizer.vocab.items():
            if "Position".lower() in tok_str.lower():
                position_token_ids.add(tok_id)
            elif "Pitch".lower() in tok_str.lower():
                pitch_token_ids.add(tok_id)
            elif "Program".lower() in tok_str.lower():
                program_token_ids.add(tok_id)
            elif "Bar".lower() in tok_str.lower():
                bar_token_ids.add(tok_id)

        dict_decoder = {tok_id: tok_str for tok_str, tok_id in tokenizer.vocab.items()}


        current_bar = set()
        current_program = None
        current_position = None
        current_pitch = None

        progress = tqdm(range(target_length), desc="Generating MIDI")
        with torch.no_grad():
            for i in progress:
                # Sliding window: keep last MAX_SEQ_LEN tokens
                # Get Context Window
                ctx = sequence[:, -Config.MAX_SEQ_LEN:]

                # Inference
                logits = self.model(ctx, m_id, g_id)

                # Get Next Token Logits
                next_logits = logits[:, -1, :]  # last position
                
                # Mask out the last token
                last_token_id = sequence[0, -1].item()
                next_logits[0, last_token_id] = float('-inf')

                # If we're about to generate a pitch (last token was position), mask any pitch that would duplicate a note already in this bar
                for program, position, pitch in current_bar:
                    next_logits[0, pitch] = float('-inf')

                # Sample
                next_token = top_k_top_p_sample(next_logits, 0, 0, 1, self.model.vocab_size)
                # Clamp Token to Valid Range
                next_token = torch.clamp(next_token, 0, self.model.vocab_size - 1)
                # Append Next Token to Sequence
                sequence = torch.cat([sequence, next_token], dim=1)

                # Update state from the token we just generated
                next_token_id = next_token.item()
                if next_token_id in bar_token_ids:
                    current_bar = set()
                elif next_token_id in program_token_ids:
                    current_program = next_token_id
                elif next_token_id in position_token_ids:
                    current_position = next_token_id
                elif next_token_id in pitch_token_ids:
                    # if current_program is not None and current_position is not None:
                    note = (current_program, current_position, next_token_id)
                    current_bar.add(note)

                # if (i + 1) % 10 == 0 or i == target_length - 1:
                #     progress.update(10)
                #     progress.set_postfix(f"Generated {i + 1}/{target_length} tokens")
                #     # status_text.text(f"Generated {i + 1}/{target_length} tokens")
        
        final_token_list = sequence.squeeze(0).cpu().tolist()

        return final_token_list
    
if __name__ == "__main__":
    device = Config.DEVICE
    print(f"Device: {device}")

    # ---- Data ----
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    ## TO BE IMPLEMENTED
    # dataloader = get_full_dataloader()
    dataloader = get_modified_cached_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        persistent_workers=Config.PERSISTENT_WORKERS,
        prefetch_factor=Config.PREFETCH_FACTOR,
        sample_factor=1.0
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
    
    # handler.train(dataloader=dataloader, epochs=Config.EPOCHS)
    handler.load_checkpoint(epoch=3)

    ## Example inference after training (using the first batch from the dataloader)
    simple = get_singleton_dataloader(Config.TOKENIZED_DIR / "XMIDI_warm_jazz_5AKYJWEA.npy", seq_len=1024)
    token_sequence = []
    for x, y in simple:
        token_sequence = x[0,:64].tolist()
        break
    generated_tokens = handler.generate(mood="angry", genre="classical", start=token_sequence)
    print(generated_tokens[:20])
    save_midi(generated_tokens, tokenizer, "generated_midi.mid")
    # token_sequence = [1]
    # generated_tokens = handler.generate(mood="exciting", genre="classical", start=token_sequence)
    # print(generated_tokens[:20])
    # save_midi(generated_tokens, tokenizer, "generated_midi.mid")