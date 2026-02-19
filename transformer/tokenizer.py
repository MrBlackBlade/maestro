"""
Custom MIDI Tokenizer for the Emotion → MIDI Transformer.

Implements a REMI-like tokenization scheme that converts MIDI files
into integer token sequences and vice versa.

Token vocabulary:
    NOTE_ON_{0-127}       : Note-on events for each MIDI pitch
    NOTE_OFF_{0-127}      : Note-off events for each MIDI pitch  
    VELOCITY_{0-31}       : Velocity quantized into 32 bins
    TIME_SHIFT_{0-99}     : Time shifts from 10ms to 1000ms (10ms steps)
    INSTRUMENT_{0-127}    : General MIDI program change
    BAR                   : Bar/measure boundary marker
    BOS / EOS / PAD       : Special tokens
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi

# Allow running as module or standalone
try:
    from .config import TokenizerConfig, DEFAULT_CONFIG
except ImportError:
    from config import TokenizerConfig, MaestroConfig
    DEFAULT_CONFIG = MaestroConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Token type constants — offsets into the vocabulary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TokenLayout:
    """Computes and stores the offset ranges for each token type."""
    cfg: TokenizerConfig

    # ── Offsets (computed once) ────────────────────────────────────────────
    @property
    def note_on_offset(self) -> int:
        return 0

    @property
    def note_off_offset(self) -> int:
        return self.cfg.num_pitches

    @property
    def velocity_offset(self) -> int:
        return self.note_off_offset + self.cfg.num_pitches

    @property
    def time_shift_offset(self) -> int:
        return self.velocity_offset + self.cfg.num_velocity_bins

    @property
    def instrument_offset(self) -> int:
        return self.time_shift_offset + self.cfg.num_time_shift_bins

    @property
    def bar_token(self) -> int:
        return self.instrument_offset + self.cfg.num_instruments

    @property
    def bos_token(self) -> int:
        return self.bar_token + 1

    @property
    def eos_token(self) -> int:
        return self.bos_token + 1

    @property
    def pad_token(self) -> int:
        return self.eos_token + 1

    @property
    def vocab_size(self) -> int:
        return self.pad_token + 1


# ─────────────────────────────────────────────────────────────────────────────
# MIDI Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class MIDITokenizer:
    """
    Converts between MIDI files and integer token sequences.

    Usage:
        tokenizer = MIDITokenizer()
        tokens = tokenizer.midi_to_tokens("song.mid")
        midi = tokenizer.tokens_to_midi(tokens)
        midi.write("reconstructed.mid")
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.cfg = config or DEFAULT_CONFIG.tokenizer
        self.layout = TokenLayout(self.cfg)

        # Velocity bin edges (0-127 → 32 bins)
        self._vel_bin_edges = np.linspace(0, 127, self.cfg.num_velocity_bins + 1)

        # Time shift bin size in seconds (10ms per bin)
        self._time_step_sec = 0.01  # 10ms

        # Build human-readable token name lookup (for debugging)
        self._token_names: Dict[int, str] = self._build_token_names()

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self.layout.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.layout.pad_token

    @property
    def bos_token_id(self) -> int:
        return self.layout.bos_token

    @property
    def eos_token_id(self) -> int:
        return self.layout.eos_token

    def midi_to_tokens(self, midi_path: str) -> List[int]:
        """
        Convert a MIDI file to a list of integer tokens.

        Args:
            midi_path: Path to a .mid/.midi file.

        Returns:
            List of integer token IDs, wrapped with BOS/EOS.
        """
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        return self._encode_midi(midi)

    def tokens_to_midi(
        self,
        tokens: List[int],
        default_tempo: float = 120.0,
    ) -> pretty_midi.PrettyMIDI:
        """
        Convert a list of integer tokens back to a PrettyMIDI object.

        Args:
            tokens: List of integer token IDs.
            default_tempo: Tempo for the output MIDI (BPM).

        Returns:
            A PrettyMIDI object.
        """
        return self._decode_tokens(tokens, default_tempo)

    def encode(self, midi_path: str) -> List[int]:
        """Alias for midi_to_tokens."""
        return self.midi_to_tokens(midi_path)

    def decode(self, tokens: List[int], default_tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
        """Alias for tokens_to_midi."""
        return self.tokens_to_midi(tokens, default_tempo)

    def save_midi(self, tokens: List[int], output_path: str, tempo: float = 120.0) -> None:
        """Decode tokens and write to a MIDI file."""
        midi = self.tokens_to_midi(tokens, tempo)
        midi.write(str(output_path))

    def token_to_name(self, token_id: int) -> str:
        """Return a human-readable name for a token ID."""
        return self._token_names.get(token_id, f"UNKNOWN_{token_id}")

    def tokens_to_names(self, tokens: List[int]) -> List[str]:
        """Return human-readable names for a list of token IDs."""
        return [self.token_to_name(t) for t in tokens]

    # ── Encoding (MIDI → Tokens) ─────────────────────────────────────────

    def _encode_midi(self, midi: pretty_midi.PrettyMIDI) -> List[int]:
        """Core encoding logic: PrettyMIDI → token list."""
        # Collect all events: (time, priority, group, token_id)
        # Priority ensures correct order within the same timestamp:
        #   0 = INSTRUMENT (must come before note to set context)
        #   1 = VELOCITY   (must come before note to set context)
        #   2 = NOTE_ON    (after instrument + velocity are set)
        #   3 = NOTE_OFF   (processed independently)
        events: List[Tuple[float, int, int, int]] = []
        group_counter = 0

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue  # Skip drum tracks for simplicity

            program = instrument.program
            inst_token = self.layout.instrument_offset + min(program, self.cfg.num_instruments - 1)

            for note in instrument.notes:
                # Each note gets a unique group so its INST+VEL+NOTE_ON stay together
                group_counter += 1

                # Velocity bin
                vel_bin = self._velocity_to_bin(note.velocity)
                vel_token = self.layout.velocity_offset + vel_bin

                # Note on group: instrument → velocity → note_on
                note_on_token = self.layout.note_on_offset + note.pitch
                events.append((note.start, 0, group_counter, inst_token))
                events.append((note.start, 1, group_counter, vel_token))
                events.append((note.start, 2, group_counter, note_on_token))

                # Note off (independent, no group dependency)
                note_off_token = self.layout.note_off_offset + note.pitch
                events.append((note.end, 3, group_counter, note_off_token))

        # Sort by: time → group → priority (keeps each note's INST+VEL+NOTE_ON together)
        events.sort(key=lambda e: (e[0], e[2], e[1]))

        # Convert to tokens with time shifts
        tokens: List[int] = [self.layout.bos_token]
        current_time = 0.0

        for event_time, _priority, _group, token_id in events:
            # Insert time shift tokens
            dt = event_time - current_time
            if dt > 0:
                tokens.extend(self._encode_time_shift(dt))
                current_time = event_time

            tokens.append(token_id)

        tokens.append(self.layout.eos_token)
        return tokens

    def _encode_time_shift(self, dt_seconds: float) -> List[int]:
        """
        Encode a time delta as one or more TIME_SHIFT tokens.
        Each TIME_SHIFT token represents 10ms to 1000ms.
        Large gaps require multiple tokens.
        """
        tokens = []
        remaining = dt_seconds

        while remaining > self._time_step_sec * 0.5:  # > 5ms threshold
            # Number of 10ms steps, capped at max bin
            steps = min(
                round(remaining / self._time_step_sec),
                self.cfg.num_time_shift_bins,
            )
            steps = max(steps, 1)  # minimum 1 step

            token = self.layout.time_shift_offset + steps - 1  # 0-indexed
            tokens.append(token)

            remaining -= steps * self._time_step_sec

        return tokens

    def _velocity_to_bin(self, velocity: int) -> int:
        """Quantize MIDI velocity (0-127) to a bin index."""
        bin_idx = int(np.digitize(velocity, self._vel_bin_edges[1:])) 
        return min(bin_idx, self.cfg.num_velocity_bins - 1)

    def _bin_to_velocity(self, bin_idx: int) -> int:
        """Convert a velocity bin back to a representative MIDI velocity."""
        # Use the midpoint of the bin
        low = self._vel_bin_edges[bin_idx]
        high = self._vel_bin_edges[min(bin_idx + 1, len(self._vel_bin_edges) - 1)]
        return int((low + high) / 2)

    # ── Decoding (Tokens → MIDI) ─────────────────────────────────────────

    def _decode_tokens(
        self,
        tokens: List[int],
        default_tempo: float,
    ) -> pretty_midi.PrettyMIDI:
        """Core decoding logic: token list → PrettyMIDI."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=default_tempo)

        # Track state
        current_time = 0.0
        current_velocity = 64  # Default velocity
        current_program = 0    # Default instrument (Acoustic Grand Piano)

        # Instrument cache: program → PrettyMIDI.Instrument
        instruments: Dict[int, pretty_midi.Instrument] = {}

        # Open notes: (program, pitch) → start_time
        open_notes: Dict[Tuple[int, int], Tuple[float, int]] = {}

        for token in tokens:
            if token == self.layout.bos_token or token == self.layout.eos_token:
                continue
            if token == self.layout.pad_token:
                continue

            if token == self.layout.bar_token:
                # Bar marker — currently decorative, no action needed
                continue

            # ── TIME_SHIFT ────────────────────────────────────────────
            if self.layout.time_shift_offset <= token < self.layout.time_shift_offset + self.cfg.num_time_shift_bins:
                steps = token - self.layout.time_shift_offset + 1
                current_time += steps * self._time_step_sec
                continue

            # ── INSTRUMENT ────────────────────────────────────────────
            if self.layout.instrument_offset <= token < self.layout.instrument_offset + self.cfg.num_instruments:
                current_program = token - self.layout.instrument_offset
                continue

            # ── VELOCITY ──────────────────────────────────────────────
            if self.layout.velocity_offset <= token < self.layout.velocity_offset + self.cfg.num_velocity_bins:
                vel_bin = token - self.layout.velocity_offset
                current_velocity = self._bin_to_velocity(vel_bin)
                continue

            # ── NOTE_ON ───────────────────────────────────────────────
            if self.layout.note_on_offset <= token < self.layout.note_on_offset + self.cfg.num_pitches:
                pitch = token - self.layout.note_on_offset
                key = (current_program, pitch)

                # Close any previously open note for this pitch+program
                if key in open_notes:
                    start_time, vel = open_notes.pop(key)
                    self._add_note(instruments, current_program, pitch, vel, start_time, current_time)

                open_notes[key] = (current_time, current_velocity)
                continue

            # ── NOTE_OFF ──────────────────────────────────────────────
            if self.layout.note_off_offset <= token < self.layout.note_off_offset + self.cfg.num_pitches:
                pitch = token - self.layout.note_off_offset

                # Search all programs for an open note with this pitch
                # (NOTE_OFF tokens don't carry instrument context)
                matched_key = None
                for key in open_notes:
                    if key[1] == pitch:
                        matched_key = key
                        break

                if matched_key is not None:
                    start_time, vel = open_notes.pop(matched_key)
                    self._add_note(instruments, matched_key[0], pitch, vel, start_time, current_time)
                continue

        # Close any remaining open notes
        for (program, pitch), (start_time, vel) in open_notes.items():
            end_time = current_time + 0.25  # default quarter-note duration
            self._add_note(instruments, program, pitch, vel, start_time, end_time)

        # Add all instruments to MIDI
        for inst in instruments.values():
            midi.instruments.append(inst)

        return midi

    def _add_note(
        self,
        instruments: Dict[int, pretty_midi.Instrument],
        program: int,
        pitch: int,
        velocity: int,
        start: float,
        end: float,
    ) -> None:
        """Add a note to the appropriate instrument, creating it if needed."""
        if end <= start:
            end = start + 0.01  # Minimum duration

        if program not in instruments:
            instruments[program] = pretty_midi.Instrument(
                program=program,
                name=pretty_midi.program_to_instrument_name(program),
            )

        note = pretty_midi.Note(
            velocity=min(max(velocity, 0), 127),
            pitch=min(max(pitch, 0), 127),
            start=start,
            end=end,
        )
        instruments[program].notes.append(note)

    # ── Utility ───────────────────────────────────────────────────────────

    def _build_token_names(self) -> Dict[int, str]:
        """Build a lookup from token ID to human-readable name."""
        names = {}
        L = self.layout

        for i in range(self.cfg.num_pitches):
            note_name = pretty_midi.note_number_to_name(i)
            names[L.note_on_offset + i] = f"NOTE_ON_{note_name}"
            names[L.note_off_offset + i] = f"NOTE_OFF_{note_name}"

        for i in range(self.cfg.num_velocity_bins):
            names[L.velocity_offset + i] = f"VELOCITY_{i}"

        for i in range(self.cfg.num_time_shift_bins):
            ms = (i + 1) * 10
            names[L.time_shift_offset + i] = f"TIME_SHIFT_{ms}ms"

        for i in range(self.cfg.num_instruments):
            try:
                inst_name = pretty_midi.program_to_instrument_name(i)
            except Exception:
                inst_name = f"Program_{i}"
            names[L.instrument_offset + i] = f"INSTRUMENT_{inst_name}"

        names[L.bar_token] = "BAR"
        names[L.bos_token] = "BOS"
        names[L.eos_token] = "EOS"
        names[L.pad_token] = "PAD"

        return names


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_test():
    """Round-trip test: create synthetic MIDI → tokenize → decode → verify."""
    import tempfile
    import os

    print("=" * 60)
    print("MIDI Tokenizer — Self-Test")
    print("=" * 60)

    tokenizer = MIDITokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"PAD={tokenizer.pad_token_id}, BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}")

    # 1. Create a synthetic MIDI
    print("\n[1] Creating synthetic MIDI...")
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    # Piano (program 0)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    test_notes = [
        (60, 80, 0.0, 0.5),    # C4
        (64, 90, 0.5, 1.0),    # E4
        (67, 100, 1.0, 1.5),   # G4
        (72, 70, 1.5, 2.0),    # C5
        (60, 85, 2.0, 2.5),    # C4 again
    ]
    for pitch, vel, start, end in test_notes:
        piano.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
    midi.instruments.append(piano)

    # Strings (program 48)
    strings = pretty_midi.Instrument(program=48, name="Strings")
    for pitch, vel, start, end in [(55, 60, 0.0, 2.0), (59, 60, 0.0, 2.0)]:
        strings.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
    midi.instruments.append(strings)

    # Save original
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        original_path = f.name
        midi.write(original_path)
    print(f"  Original MIDI saved to: {original_path}")
    print(f"  Instruments: Piano ({len(piano.notes)} notes), Strings ({len(strings.notes)} notes)")

    # 2. Tokenize
    print("\n[2] Tokenizing...")
    tokens = tokenizer.midi_to_tokens(original_path)
    print(f"  Token count: {len(tokens)}")
    print(f"  First 20 tokens: {tokenizer.tokens_to_names(tokens[:20])}")
    print(f"  Last 5 tokens:   {tokenizer.tokens_to_names(tokens[-5:])}")

    # 3. Decode
    print("\n[3] Decoding back to MIDI...")
    reconstructed = tokenizer.tokens_to_midi(tokens)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        recon_path = f.name
        reconstructed.write(recon_path)
    print(f"  Reconstructed MIDI saved to: {recon_path}")

    # 4. Verify
    print("\n[4] Verification...")
    original_notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            original_notes.append((inst.program, note.pitch, note.velocity, round(note.start, 2), round(note.end, 2)))
    original_notes.sort()

    recon_notes = []
    for inst in reconstructed.instruments:
        for note in inst.notes:
            recon_notes.append((inst.program, note.pitch, note.velocity, round(note.start, 2), round(note.end, 2)))
    recon_notes.sort()

    print(f"  Original notes:      {len(original_notes)}")
    print(f"  Reconstructed notes: {len(recon_notes)}")

    # Check note count matches
    assert len(recon_notes) == len(original_notes), (
        f"Note count mismatch: {len(recon_notes)} vs {len(original_notes)}"
    )

    # Check pitches and timing (velocity may differ slightly due to binning)
    pitch_matches = 0
    timing_matches = 0
    # Timing tolerance: 15ms accounts for 10ms quantization + rounding
    TIMING_TOL = 0.015
    for orig, recon in zip(original_notes, recon_notes):
        if orig[0] == recon[0] and orig[1] == recon[1]:  # program + pitch
            pitch_matches += 1
        start_ok = abs(orig[3] - recon[3]) < TIMING_TOL
        end_ok = abs(orig[4] - recon[4]) < TIMING_TOL
        if start_ok and end_ok:
            timing_matches += 1
        else:
            print(f"    Timing diff: orig=({orig[3]:.3f}, {orig[4]:.3f}) "
                  f"recon=({recon[3]:.3f}, {recon[4]:.3f}) "
                  f"delta_start={abs(orig[3]-recon[3]):.3f} delta_end={abs(orig[4]-recon[4]):.3f}")

    print(f"  Pitch matches:  {pitch_matches}/{len(original_notes)}")
    print(f"  Timing matches: {timing_matches}/{len(original_notes)} (within {TIMING_TOL*1000:.0f}ms)")

    assert pitch_matches == len(original_notes), "Pitch mismatch detected!"
    assert timing_matches == len(original_notes), "Timing mismatch detected!"

    # Clean up
    os.unlink(original_path)
    os.unlink(recon_path)

    print("\n" + "=" * 60)
    print("[OK] All self-tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI Tokenizer")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    parser.add_argument("--encode", type=str, help="Encode a MIDI file and print tokens")
    parser.add_argument("--decode", type=str, help="Decode a token file back to MIDI (not implemented in CLI)")
    args = parser.parse_args()

    if args.test:
        _run_self_test()
    elif args.encode:
        tokenizer = MIDITokenizer()
        tokens = tokenizer.midi_to_tokens(args.encode)
        print(f"Tokens ({len(tokens)}):")
        print(tokens)
        print(f"\nHuman-readable:")
        for i, name in enumerate(tokenizer.tokens_to_names(tokens)):
            print(f"  [{i:4d}] {name}")
    else:
        parser.print_help()
