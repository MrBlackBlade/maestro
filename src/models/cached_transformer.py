"""
Drop-in cached replacements for nn.TransformerEncoderLayer / nn.TransformerEncoder.

Checkpoint-compatible: parameter names (self_attn, linear1, linear2, norm1, norm2,
layers.*) match PyTorch's built-in implementations so existing state_dicts load
without remapping.

Two modes controlled by the ``kv_cache`` argument to ``forward()``:

  Training  (kv_cache=None)
      Delegates to nn.MultiheadAttention which uses F.scaled_dot_product_attention
      internally (FlashAttention / memory-efficient backend selected automatically).

  Cached inference (kv_cache provided)
      Manually extracts Q/K/V projection weights from nn.MultiheadAttention,
      writes K/V into a pre-allocated static buffer (no torch.cat), and calls
      F.scaled_dot_product_attention with only the new token(s) as query.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    """Pre-allocated key/value cache for autoregressive transformer inference.

    Buffers are fixed-size tensors allocated once at construction. Each
    generation step writes into the next slot via index assignment — no
    ``torch.cat``, no memory fragmentation, predictable GPU footprint.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.seq_len = 0

        if torch.device(device).type == "cuda":
            needed = self.estimate_memory(
                num_layers, batch_size, num_heads, max_seq_len, head_dim, dtype,
            )
            free, _ = torch.cuda.mem_get_info(device)
            if needed > free * 0.9:
                raise RuntimeError(
                    f"KVCache needs {needed / 1e6:.1f} MB but only "
                    f"{free / 1e6:.1f} MB free on {device}"
                )

        shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = [
            torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    # ------------------------------------------------------------------
    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V into the buffer and return full K/V up to this point.

        Does **not** advance ``seq_len`` — the encoder calls :meth:`advance`
        once after all layers have processed.

        Parameters
        ----------
        new_k, new_v : ``[B, num_heads, new_len, head_dim]``

        Returns
        -------
        (K_full, V_full) each ``[B, num_heads, seq_len + new_len, head_dim]``
        """
        new_len = new_k.size(2)
        end = self.seq_len + new_len
        self.k_cache[layer_idx][:, :, self.seq_len:end, :] = new_k
        self.v_cache[layer_idx][:, :, self.seq_len:end, :] = new_v
        return (
            self.k_cache[layer_idx][:, :, :end, :],
            self.v_cache[layer_idx][:, :, :end, :],
        )

    def advance(self, num_tokens: int = 1) -> None:
        """Bump the position pointer after all layers have processed."""
        self.seq_len += num_tokens

    def is_full(self) -> bool:
        return self.seq_len >= self.max_seq_len

    def reset(self) -> None:
        """Zero all buffers and rewind to position 0."""
        for i in range(self.num_layers):
            self.k_cache[i].zero_()
            self.v_cache[i].zero_()
        self.seq_len = 0

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_memory(
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        """Return estimated memory consumption in bytes."""
        elem = torch.tensor([], dtype=dtype).element_size()
        return 2 * num_layers * batch_size * num_heads * max_seq_len * head_dim * elem

    @classmethod
    def from_model(cls, model: nn.Module, batch_size: int = 1) -> "KVCache":
        """Create a cache pre-sized for *model*'s transformer stack."""
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        layer0 = model.transformer.layers[0]
        return cls(
            num_layers=model.transformer.num_layers,
            batch_size=batch_size,
            num_heads=layer0.nhead,
            max_seq_len=model.pos_emb.num_embeddings,
            head_dim=layer0.head_dim,
            dtype=dtype,
            device=device,
        )


# ---------------------------------------------------------------------------
# Cached Transformer Encoder Layer
# ---------------------------------------------------------------------------

class CachedTransformerEncoderLayer(nn.Module):
    """``nn.TransformerEncoderLayer`` with optional KV-cache support.

    Sub-module names (``self_attn``, ``linear1``, ``linear2``, ``norm1``,
    ``norm2``) are identical to PyTorch's implementation so that
    ``state_dict`` keys match existing checkpoints.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True,
        norm_first: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_first = norm_first

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    # ------------------------------------------------------------------
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )

    # ------------------------------------------------------------------
    def _qkv_proj(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project *x* into Q, K, V heads using ``self.self_attn`` weights.

        Returns (Q, K, V) each shaped ``[B, nhead, S, head_dim]``.
        """
        B, S, _ = x.shape
        W = self.self_attn.in_proj_weight          # [3*d, d]
        b = self.self_attn.in_proj_bias             # [3*d]
        d = self.d_model

        Q = F.linear(x, W[:d], b[:d])
        K = F.linear(x, W[d : 2 * d], b[d : 2 * d])
        V = F.linear(x, W[2 * d :], b[2 * d :])

        Q = Q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        return Q, K, V

    def _out_proj(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Merge heads and apply the output projection."""
        B = attn_out.size(0)
        S = attn_out.size(2)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.self_attn.out_proj(attn_out)

    # ------------------------------------------------------------------
    def _cached_attn(
        self,
        x: torch.Tensor,
        kv_cache: KVCache,
        layer_idx: int,
    ) -> torch.Tensor:
        """Self-attention with KV cache (inference only)."""
        S = x.size(1)
        assert kv_cache.seq_len == 0 or S == 1, (
            f"Multi-token input (S={S}) with non-empty cache "
            f"(seq_len={kv_cache.seq_len}) would produce an incorrect causal mask"
        )

        Q, K_new, V_new = self._qkv_proj(x)
        K_full, V_full = kv_cache.update(layer_idx, K_new, V_new)

        attn_out = F.scaled_dot_product_attention(
            Q, K_full, V_full, is_causal=(S > 1),
        )
        return self._out_proj(attn_out)

    # ------------------------------------------------------------------
    def _sdpa_attn(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        """Self-attention via F.scaled_dot_product_attention (no cache).

        Calls SDPA directly instead of going through
        ``nn.MultiheadAttention.forward()``, which guarantees the
        FlashAttention / memory-efficient backend is used regardless of
        PyTorch-version quirks around ``is_causal`` + ``attn_mask``.
        """
        Q, K, V = self._qkv_proj(x)
        dp = self.self_attn.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            Q, K, V, is_causal=is_causal, dropout_p=dp,
        )
        return self._out_proj(attn_out)

    # ------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:

        if self.norm_first:
            x = self.norm1(src)
            if kv_cache is not None:
                attn_out = self._cached_attn(x, kv_cache, layer_idx)
            elif is_causal and src_mask is None and src_key_padding_mask is None:
                attn_out = self._sdpa_attn(x, is_causal=True)
            else:
                attn_out = self.self_attn(
                    x, x, x,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                    need_weights=False,
                )[0]
            src = src + self.dropout1(attn_out)
            src = src + self._ff_block(self.norm2(src))
        else:
            if kv_cache is not None:
                attn_out = self._cached_attn(src, kv_cache, layer_idx)
            elif is_causal and src_mask is None and src_key_padding_mask is None:
                attn_out = self._sdpa_attn(src, is_causal=True)
            else:
                attn_out = self.self_attn(
                    src, src, src,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                    need_weights=False,
                )[0]
            src = self.norm1(src + self.dropout1(attn_out))
            src = self.norm2(src + self._ff_block(src))

        return src


# ---------------------------------------------------------------------------
# Cached Transformer Encoder
# ---------------------------------------------------------------------------

class CachedTransformerEncoder(nn.Module):
    """``nn.TransformerEncoder`` with KV-cache threading.

    Uses ``self.layers`` (``nn.ModuleList``) for checkpoint-compatible naming.
    """

    def __init__(
        self,
        encoder_layer: CachedTransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        output = src
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                kv_cache=kv_cache,
                layer_idx=i,
            )

        if self.norm is not None:
            output = self.norm(output)

        if kv_cache is not None:
            kv_cache.advance(src.size(1))

        return output
