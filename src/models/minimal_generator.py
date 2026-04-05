import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math

from tqdm import tqdm

from src.core.config import Config
from src.core.utils import get_tokenizer, save_midi
from src.dataloaders.singleton_dataloader import get_singleton_dataloader
from src.models.general_model_handler import GeneralModelHandler
from src.models.cached_transformer import (
    CachedTransformerEncoderLayer,
    CachedTransformerEncoder,
    KVCache,
)

from miditok import REMI, TokSequence

class MinimalGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_seq_len=Config.MAX_SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # ==========================================
        # 1. Embeddings (Translating integers to vectors)
        # ==========================================
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # The model doesn't inherently understand the order of music. 
        # We use a learned positional embedding to tell it where it is in time.
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # ==========================================
        # 2. Transformer Blocks
        # ==========================================
        # CachedTransformerEncoder is a drop-in for nn.TransformerEncoder
        # that threads an optional KVCache through each layer during inference.
        layer = CachedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = CachedTransformerEncoder(layer, num_layers=num_layers)
        
        # ==========================================
        # 3. Output Head
        # ==========================================
        # Projects the final transformer representations back into vocabulary probabilities
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, kv_cache=None, start_pos=0):
        B, T = x.size()
        
        positions = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(0).expand(B, T)
        
        embeds = self.token_emb(x) * math.sqrt(self.d_model)
        x = embeds + self.pos_emb(positions)
        
        if kv_cache is not None:
            out = self.transformer(x, kv_cache=kv_cache)
        else:
            out = self.transformer(x, is_causal=True)
        
        logits = self.fc_out(out)
        return logits

class MinimalGeneratorHandler(GeneralModelHandler):
    MODEL_NAME = "minimal_generator_0"

    def __init__(self, model: nn.Module, optimizer, criterion, scheduler):
        super().__init__(model, optimizer, scheduler, self.MODEL_NAME)
        self.criterion = criterion

    def train_step(self, batch):
        x_batch, y_batch = batch
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        logits = self.model(x_batch)                            #[B, SEQ_LEN, vocab_size]
        logits_flat = logits.view(-1, self.model.vocab_size)    #[B*SEQ_LEN, vocab_size]

        y_flat = y_batch.view(-1)                               #[B*SEQ_LEN]

        loss = self.criterion(logits_flat, y_flat)

        return loss
    
    def generate(
        self,
        x_batch,
        target_length: int = 4096,
        temperature: float = 1.0,
        top_k: int = 10,
    ):
        self.model.eval()
        self.model.to(self.device)

        seed_tokens = x_batch.to(self.device)
        window_size = Config.MAX_SEQ_LEN
        generated_tokens = seed_tokens

        cache = KVCache.from_model(self.model) if Config.USE_KV_CACHE else None

        with torch.inference_mode():
            for i in range(target_length - 1):
                if cache is not None:
                    if cache.is_full():
                        cache.reset()
                        refill_len = min(generated_tokens.size(1), Config.MAX_SEQ_LEN // 2)
                        window = generated_tokens[:, -refill_len:]
                        start_pos = 0
                    else:
                        window = generated_tokens[:, -1:]
                        start_pos = cache.seq_len
                    logits = self.model(window, kv_cache=cache, start_pos=start_pos)
                else:
                    window = generated_tokens[:, -window_size:]
                    logits = self.model(window)

                next_token_logits = logits[:, -1, :]

                last_token_id = generated_tokens[0, -1].item()
                next_token_logits[0, last_token_id] = float('-inf')

                scaled_logits = next_token_logits / temperature
                top_k_values, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)

                filtered_logits = torch.full_like(scaled_logits, float('-inf'))
                filtered_logits.scatter_(1, top_k_indices, top_k_values)

                probs = F.softmax(filtered_logits, dim=-1)

                next_token_id = torch.multinomial(probs, num_samples=1)
                generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)

        final_token_list = generated_tokens.squeeze(0).cpu().tolist()

        return final_token_list

if __name__ == "__main__":
    #tokenizer = get_tokenizer(Config.TOKENIZER_PARAMS_PATH)
    device = Config.DEVICE
    
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size # Fetch dynamically from miditok
    model = MinimalGenerator(vocab_size=vocab_size)

    # # Test a forward pass with our single batch from earlier
    # # x_batch shape: [4, 256]
    token_path = Config.DATASETS_DIR / "XMIDI_angry_classical_0HP7PK58_tokens.npy"

    parser = argparse.ArgumentParser(description="MinimalGenerator – train or generate")
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

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    handler = MinimalGeneratorHandler(model, optimizer, criterion, scheduler)

    if args.command == "train":
        dataloader = get_singleton_dataloader(token_path, seq_len=1024)

        if args.resume_epoch is not None:
            handler.load_checkpoint(epoch=args.resume_epoch)
            start_epoch = args.resume_epoch + 1
            print(f"Resumed from epoch {args.resume_epoch}, continuing at epoch {start_epoch}")
        else:
            start_epoch = 1

        handler.train(dataloader=dataloader, epochs=args.epochs, start_epoch=start_epoch)

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
