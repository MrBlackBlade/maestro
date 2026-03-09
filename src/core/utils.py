"""
Shared utilities: tokenizer creation, token encode/decode helpers.
Both the Generator and Refiner use the same tokenizer so they share a vocabulary.
"""
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from miditok import REMI, TokenizerConfig
from symusic import Score
import torch
import torch.nn.functional as F

from src.core.config import Config

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout/stderr (for suppressing C++ library debug output)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def get_tokenizer(trained_path: str | Path | None = None) -> REMI:
    """
    Return a REMI tokenizer.

    If *trained_path* points to an existing tokenizer JSON produced by
    ``tokenizer.save()``, it is loaded from disk (this preserves the
    vocabulary built during pre-processing).  Otherwise a fresh tokenizer
    with the project-wide config is returned.
    """
    if trained_path and Path(trained_path).exists():
        tokenizer = REMI(params=Path(trained_path))
        return tokenizer

    config = TokenizerConfig(
        num_velocities=Config.NUM_VELOCITIES,
        use_chords=Config.USE_CHORDS,
        use_programs=Config.USE_PROGRAMS,
        one_token_stream_for_programs=True,
    )
    tokenizer = REMI(config)
    return tokenizer


def encode_midi_file(tokenizer: REMI, midi_path: str | Path) -> list[int]:
    """
    Tokenize a single MIDI file and return a flat list of token ids.
    """
    score = Score(str(midi_path))
    tok_result = tokenizer.encode(score)

    # miditok may return a list of TokSequence (one per track) or a single
    # TokSequence.  We always take the first track.
    if isinstance(tok_result, list):
        return tok_result[0].ids
    return tok_result.ids


def decode_token_ids(tokenizer: REMI, token_ids: list[int], output_path: str | Path):
    """
    Convert a list of token ids back to a MIDI file and save it.
    """
    midi = tokenizer.decode(token_ids)
    midi.dump_midi(str(output_path))


def save_mappings(mood_map: dict, genre_map: dict, path: str | Path):
    """Persist label->id mappings so the app can load them later."""
    data = {"mood_to_id": mood_map, "genre_to_id": genre_map}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        
def save_midi(token_list: list, tokenizer, output_path: str):
    #tok_sequence = TokSequence(token_list)
    midi = tokenizer.decode(token_list)
    midi.dump_midi(output_path)
    
def load_mappings(path: str | Path) -> tuple[dict, dict]:
    """Load label->id mappings saved during pre-processing."""
    with open(path) as f:
        data = json.load(f)
    return data["mood_to_id"], data["genre_to_id"]

def top_k_top_p_sample(logits: torch.Tensor, top_k: int, top_p: float, temperature: float, vocab_size: int):
    """
    Apply temperature scaling, top-k, and nucleus (top-p) filtering,
    then sample a single token.
    
    Parameters
    ----------
    logits : torch.Tensor [B, vocab_size]
        Raw logits from the model
    top_k : int
        Top-k sampling parameter
    top_p : float
        Nucleus sampling parameter
    temperature : float
        Temperature scaling
    vocab_size : int
        Valid vocabulary size (to clamp logits)
    """
    # Clamp logits to valid vocabulary range
    if logits.size(-1) > vocab_size:
        logits = logits[:, :vocab_size]
    
    logits = logits / max(temperature, 1e-8)

    # Top-k
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_val, torch.full_like(logits, -float("inf")), logits)

    # Top-p (nucleus)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = -float("inf")
        # Scatter back to original positions
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
