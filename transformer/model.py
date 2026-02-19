"""
Emotion-Conditioned Music Transformer.

Decoder-only (GPT-style) transformer that generates MIDI token sequences
conditioned on an emotion vector (valence, arousal).
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import ModelConfig, TokenizerConfig, DEFAULT_CONFIG
except ImportError:
    from config import ModelConfig, TokenizerConfig, MaestroConfig
    DEFAULT_CONFIG = MaestroConfig()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EmotionMusicTransformer(nn.Module):
    """
    Decoder-only transformer for emotion-conditioned MIDI generation.

    Architecture:
        1. Emotion (valence, arousal) → projected to d_model, prepended as a
           conditioning token at position 0.
        2. MIDI tokens → embedding + positional encoding.
        3. N transformer decoder layers with causal masking.
        4. Linear head over vocab for next-token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        model_cfg: ModelConfig = None,
    ):
        super().__init__()
        cfg = model_cfg or DEFAULT_CONFIG.model
        self.cfg = cfg
        self.vocab_size = vocab_size

        # ── Emotion conditioning ──────────────────────────────────────────
        self.emotion_proj = nn.Sequential(
            nn.Linear(cfg.emotion_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # ── Token embedding ──────────────────────────────────────────────
        self.token_embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len + 1, cfg.dropout)

        # ── Transformer decoder stack ────────────────────────────────────
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=cfg.num_layers,
        )

        # ── Output head ──────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output head
        self.head.weight = self.token_embed.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        emotion: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing).

        Args:
            tokens: (B, T) integer token IDs (input sequence).
            emotion: (B, 2) emotion vector [valence, arousal].
            src_key_padding_mask: (B, T+1) True for padded positions.

        Returns:
            logits: (B, T+1, vocab_size) next-token prediction logits.
                    The first position corresponds to the emotion token's prediction.
        """
        B, T = tokens.shape

        # Emotion conditioning → (B, 1, d_model)
        emo_embed = self.emotion_proj(emotion).unsqueeze(1)

        # Token embedding → (B, T, d_model)
        tok_embed = self.token_embed(tokens)

        # Concatenate: [emotion_token, token_0, token_1, ..., token_T-1]
        # Shape: (B, T+1, d_model)
        x = torch.cat([emo_embed, tok_embed], dim=1)

        # Positional encoding
        x = self.pos_encoding(x)

        # Causal mask: prevent attending to future tokens
        seq_len = T + 1
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tokens.device
        )

        # Apply transformer
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        emotion: torch.Tensor,
        max_len: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        bos_token: int = None,
        eos_token: int = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation conditioned on emotion.

        Args:
            emotion: (1, 2) or (B, 2) emotion vector.
            max_len: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Keep only top-k logits.
            top_p: Nucleus sampling threshold.
            bos_token: BOS token ID.
            eos_token: EOS token ID.

        Returns:
            generated: (B, generated_len) token IDs including BOS.
        """
        self.eval()
        device = next(self.parameters()).device

        if emotion.dim() == 1:
            emotion = emotion.unsqueeze(0)
        emotion = emotion.to(device)
        B = emotion.size(0)

        # Start with BOS token
        if bos_token is None:
            from .config import DEFAULT_CONFIG as _cfg
            from .tokenizer import MIDITokenizer
            _tok = MIDITokenizer(_cfg.tokenizer)
            bos_token = _tok.bos_token_id
            eos_token = _tok.eos_token_id

        generated = torch.full((B, 1), bos_token, dtype=torch.long, device=device)

        for _ in range(max_len):
            # Truncate to max seq length if needed
            input_tokens = generated[:, -(self.cfg.max_seq_len):]

            # Forward pass
            logits = self.forward(input_tokens, emotion)

            # Get logits for the last position
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < min_values,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have generated EOS
            if eos_token is not None and (next_token == eos_token).all():
                break

        return generated

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_test():
    """Smoke test: verify forward and generate with random data."""
    print("=" * 60)
    print("EmotionMusicTransformer -- Self-Test")
    print("=" * 60)

    tok_cfg = DEFAULT_CONFIG.tokenizer
    model_cfg = DEFAULT_CONFIG.model

    vocab_size = tok_cfg.vocab_size
    model = EmotionMusicTransformer(vocab_size=vocab_size, model_cfg=model_cfg)
    device = "cpu"
    model = model.to(device)

    print(f"Vocab size:  {vocab_size}")
    print(f"Parameters:  {model.count_parameters():,}")
    print(f"Device:      {device}")
    print(f"Model config: d_model={model_cfg.d_model}, layers={model_cfg.num_layers}, heads={model_cfg.nhead}")

    # Test forward pass
    print("\n[1] Testing forward pass...")
    B, T = 4, 64
    tokens = torch.randint(0, vocab_size, (B, T), device=device)
    emotion = torch.randn(B, 2, device=device)

    logits = model(tokens, emotion)
    expected_shape = (B, T + 1, vocab_size)
    print(f"  Input:  tokens={tokens.shape}, emotion={emotion.shape}")
    print(f"  Output: logits={logits.shape}")
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    print(f"  [OK] Shape correct: {expected_shape}")

    # Test with padding mask
    print("\n[2] Testing with padding mask...")
    pad_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
    pad_mask[:, -10:] = True  # last 10 positions are padding
    logits_masked = model(tokens, emotion, src_key_padding_mask=pad_mask)
    assert logits_masked.shape == expected_shape
    print(f"  [OK] Padding mask works")

    # Test generation
    print("\n[3] Testing generation...")
    from transformer.tokenizer import MIDITokenizer
    tokenizer = MIDITokenizer(tok_cfg)
    emo = torch.tensor([[0.5, 0.7]], device=device)
    gen_tokens = model.generate(
        emo,
        max_len=32,
        temperature=1.0,
        top_k=10,
        bos_token=tokenizer.bos_token_id,
        eos_token=tokenizer.eos_token_id,
    )
    print(f"  Generated shape: {gen_tokens.shape}")
    print(f"  Generated tokens: {gen_tokens[0, :10].tolist()}...")
    print(f"  [OK] Generation works")

    print("\n" + "=" * 60)
    print("[OK] All self-tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Music Transformer")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    args = parser.parse_args()

    if args.test:
        _run_self_test()
    else:
        parser.print_help()
