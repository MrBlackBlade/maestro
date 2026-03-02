import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math

from tqdm import tqdm

from src.core.config import Config
from src.core.utils import get_tokenizer
from src.dataloaders.singleton_dataloader import get_singleton_dataloader


from miditok import REMI, TokSequence

class MinimalGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_seq_len=Config.MAX_SEQ_LEN):
        super().__init__()
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
        # We use PyTorch's native layers. For a decoder-only model, 
        # we can actually use the "EncoderLayer" as long as we force a causal mask on it.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.2,
            batch_first=True  # Keeps tensors as [Batch, Sequence, Feature]
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # ==========================================
        # 3. Output Head
        # ==========================================
        # Projects the final transformer representations back into vocabulary probabilities
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size() # Batch size, Sequence Length
        
        # 1. Create position indices (0, 1, 2 ... T-1)
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        
        # 2. Combine Token + Positional Embeddings
        # (Standard practice is to scale token embeddings by sqrt(d_model))
        embeds = self.token_emb(x) * math.sqrt(self.d_model)
        x = embeds + self.pos_emb(positions)
        
        # 3. The Causal Mask (CRITICAL)
        # We generate a square matrix that hides the future tokens from the current token.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        # 4. Pass through the transformer
        # is_causal=True triggers PyTorch's highly optimized FlashAttention under the hood
        out = self.transformer(x, mask=causal_mask, is_causal=True)
        
        # 5. Get the final logits (predictions)
        logits = self.fc_out(out)
        
        return logits

class ModelHandler:
    def __init__(self, model: nn.Module, dataloader, optimizer, criterion, tokenizer):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion

        self.vocab_size = tokenizer.vocab_size

        self.device = Config.DEVICE

    def train(self, epochs):
        self.model.train()
        self.model.to(self.device)

        for epoch in range(epochs):
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs}", mininterval=1.0)
            for batch_idx, (x_batch, y_batch) in enumerate(loop):

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x_batch)

                logits_flat = logits.view(-1, self.vocab_size)
                y_flat = y_batch.view(-1)

                loss = self.criterion(logits_flat, y_flat)
                loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    def generate(self, x_batch):
        self.model.eval()
        self.model.to(self.device)

        seed_tokens = x_batch.to(self.device)
        target_length = 4096
        window_size = Config.MAX_SEQ_LEN
        generated_tokens = seed_tokens

        temperature = 1
        top_k = 10

        current_bar = set()

        with torch.no_grad():
            for i in range(target_length - 1):
                # Get Context Window
                window = generated_tokens[:, -window_size:]
                
                # Inference
                logits = self.model(window)
                next_token_logits = logits[:, -1, :]

                # Mask out the last token
                last_token_id = generated_tokens[0, -1].item()
                next_token_logits[0, last_token_id] = float('-inf')

                # Sample
                scaled_logits = next_token_logits / temperature
                top_k_values, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)

                # Create a tensor of -infinity, then scatter top K values back into it
                filtered_logits = torch.full_like(scaled_logits, float('-inf'))
                filtered_logits.scatter_(1, top_k_indices, top_k_values)
                
                # Convert to probabilities
                probs = F.softmax(filtered_logits, dim=-1)
                
                next_token_id = torch.multinomial(probs, num_samples=1)
                generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)
        
        final_token_list = generated_tokens.squeeze(0).cpu().tolist()

        return final_token_list

def save_midi(token_list: list, tokenizer, output_path: str):
    #tok_sequence = TokSequence(token_list)
    midi = tokenizer.decode(token_list)
    midi.dump_midi(output_path)

if __name__ == "__main__":
    #tokenizer = get_tokenizer(Config.TOKENIZER_PARAMS_PATH)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size # Fetch dynamically from miditok
    model = MinimalGenerator(vocab_size=vocab_size)

    # # Test a forward pass with our single batch from earlier
    # # x_batch shape: [4, 256]
    token_path = Config.DATASETS_DIR / "XMIDI_angry_classical_0HP7PK58_tokens.npy"

    dataloader = get_singleton_dataloader(token_path, seq_len=1024)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    handler = ModelHandler(model, dataloader, optimizer, criterion, tokenizer)
    handler.train(epochs=5)

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(Config.DEVICE)
        x_batch = x_batch[0, 0:1].unsqueeze(0)
        print(x_batch.shape)
        generated_tokens = handler.generate(x_batch)
        print(generated_tokens[:20])
        save_midi(generated_tokens, tokenizer, "generated_midi.mid")
        break
