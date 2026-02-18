"""
Shared utilities: tokenizer creation, token encode/decode helpers.
Both the Generator and Refiner use the same tokenizer so they share a vocabulary.
"""
import json
from pathlib import Path

from miditok import REMI, TokenizerConfig
from symusic import Score

from src.config import Config


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
        pitch_range=Config.PITCH_RANGE,
        num_velocities=Config.NUM_VELOCITIES,
        use_chords=Config.USE_CHORDS,
        use_programs=Config.USE_PROGRAMS,
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


def load_mappings(path: str | Path) -> tuple[dict, dict]:
    """Load label->id mappings saved during pre-processing."""
    with open(path) as f:
        data = json.load(f)
    return data["mood_to_id"], data["genre_to_id"]

