"""
Real-time MIDI playback for token-by-token generation.

Parses REMI tokens on the fly and sends MIDI events to the OS synthesizer
via pygame.midi, timing notes according to Bar/Position tokens and a
configurable BPM.
"""

import time
import threading
from typing import Optional

import pygame.midi


class RealtimeMidiPlayer:
    """Streams REMI tokens to the OS MIDI synthesizer in real time.

    Maintains internal state (bar, position, program, pitch, velocity) and
    fires a MIDI note-on the moment a Duration token completes a note event.
    Timing is derived from Bar/Position tokens so playback matches musical time.
    """

    _BEATS_PER_BAR = 4  # 4/4 assumed

    def __init__(self, tokenizer, bpm: float = 120.0):
        pygame.midi.init()
        port = pygame.midi.get_default_output_id()
        if port < 0:
            pygame.midi.quit()
            raise RuntimeError("No MIDI output device found")
        self._out = pygame.midi.Output(port)

        self._bpm = bpm
        self._seconds_per_beat = 60.0 / bpm

        # Build token lookup: id -> (type_str, value_str)
        self._tok_lookup: dict[int, tuple[str, str]] = {}
        max_position = 0
        for tok_str, tok_id in tokenizer.vocab.items():
            typ, val = self._parse_tok(tok_str)
            self._tok_lookup[tok_id] = (typ, val)
            if typ == "Position":
                try:
                    max_position = max(max_position, int(val))
                except ValueError:
                    pass

        self._positions_per_bar = max_position + 1 if max_position > 0 else 48
        self._recalc_position_timing()

        # Per-program MIDI channel allocation (channel 9 reserved for drums)
        self._program_channels: dict[int, int] = {}
        self._next_channel = 0

        # Note accumulation state
        self._cur_program: int = 0
        self._cur_pitch: Optional[int] = None
        self._cur_velocity: int = 80
        self._cur_position: int = 0

        # Timing
        self._bar_wall: float = 0.0
        self._started: bool = False

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tok(tok_str: str) -> tuple[str, str]:
        parts = tok_str.split("_", 1)
        return (parts[0], parts[1] if len(parts) > 1 else "")

    def _recalc_position_timing(self) -> None:
        self._seconds_per_position = (
            self._BEATS_PER_BAR * self._seconds_per_beat
        ) / self._positions_per_bar

    # ------------------------------------------------------------------
    def _duration_seconds(self, val: str) -> float:
        """Convert a miditok duration value (e.g. '1.0.8') to seconds."""
        parts = val.split(".")
        try:
            beats = int(parts[0]) if len(parts) > 0 else 0
            sub = int(parts[1]) if len(parts) > 1 else 0
            ticks = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            try:
                return (int(val) / 480.0) * self._seconds_per_beat
            except ValueError:
                return 0.25 * self._seconds_per_beat

        sub_per_beat = max(self._positions_per_bar // self._BEATS_PER_BAR, 1)
        total_beats = beats + sub / sub_per_beat + ticks / (sub_per_beat * 8)
        return total_beats * self._seconds_per_beat

    # ------------------------------------------------------------------
    def _channel_for_program(self, program: int) -> int:
        """Return (and lazily allocate) a MIDI channel for *program*."""
        if program in self._program_channels:
            return self._program_channels[program]
        ch = self._next_channel
        if ch == 9:
            ch = 10
        if ch > 15:
            ch = 0
        self._program_channels[program] = ch
        self._out.set_instrument(program, ch)
        self._next_channel = min(ch + 1, 15)
        return ch

    # ------------------------------------------------------------------
    def _play_note(self, pitch: int, velocity: int, duration_sec: float) -> None:
        ch = self._channel_for_program(self._cur_program)
        vel = min(max(velocity, 1), 127)
        self._out.note_on(pitch, vel, ch)

        def _off():
            try:
                self._out.note_off(pitch, 0, ch)
            except Exception:
                pass

        threading.Timer(max(duration_sec, 0.05), _off).start()

    # ------------------------------------------------------------------
    def _wait_until(self, target: float) -> None:
        """Sleep until *target* (perf_counter timestamp) if it is in the future."""
        delta = target - time.perf_counter()
        if delta > 0:
            time.sleep(delta)

    # ------------------------------------------------------------------
    def feed_token(self, token_id: int) -> None:
        """Process one REMI token, potentially sleeping and/or playing a note."""
        entry = self._tok_lookup.get(token_id)
        if entry is None:
            return

        typ, val = entry

        if typ == "Bar":
            if self._started:
                self._wait_until(
                    self._bar_wall + self._BEATS_PER_BAR * self._seconds_per_beat
                )
                self._bar_wall = time.perf_counter()
            else:
                self._bar_wall = time.perf_counter()
                self._started = True
            self._cur_position = 0

        elif typ == "Position":
            try:
                pos = int(val)
            except ValueError:
                return
            if self._started:
                self._wait_until(self._bar_wall + pos * self._seconds_per_position)
            self._cur_position = pos

        elif typ == "Program":
            try:
                self._cur_program = int(val)
            except ValueError:
                pass

        elif typ == "Pitch":
            try:
                self._cur_pitch = int(val)
            except ValueError:
                pass

        elif typ == "Velocity":
            try:
                self._cur_velocity = min(max(int(val), 1), 127)
            except ValueError:
                pass

        elif typ == "Duration":
            if self._cur_pitch is not None:
                dur = self._duration_seconds(val)
                self._play_note(self._cur_pitch, self._cur_velocity, dur)
                self._cur_pitch = None

        elif typ == "Tempo":
            try:
                new_bpm = float(val)
                if new_bpm > 0:
                    self._bpm = new_bpm
                    self._seconds_per_beat = 60.0 / new_bpm
                    self._recalc_position_timing()
            except ValueError:
                pass

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Send All-Notes-Off on every channel and release the device."""
        time.sleep(0.5)
        for ch in range(16):
            self._out.write_short(0xB0 | ch, 123, 0)
        self._out.close()
        pygame.midi.quit()
