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

