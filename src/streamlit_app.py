"""
MAESTRO – Continuous Mood-Driven MIDI Generation.

Architecture overview
---------------------
Three background threads run while the Streamlit UI stays responsive:

  1. **Generation thread** – calls ``generate_single_step`` in a loop,
     producing one token at a time.  Tokens are accumulated into "bars";
     a bar boundary is detected whenever the model emits token-id 4
     (``Bar_None``).  Completed bars are placed on a **bounded queue**
     (``maxsize = MAX_BARS_AHEAD = 2``).  When the queue is full the
     generator *blocks*, which is the throttling mechanism that prevents
     it from racing arbitrarily far ahead of the audio player.

  2. **Mood-watcher thread** – polls the Streamlit slider values at 10 Hz
     and pushes a new mood to the generation state only when the sliders
     have moved far enough (Euclidean distance ≥ ``SLIDER_CHANGE_THRESHOLD``).

  3. **Playback thread** – pulls bars from the queue one at a time,
     decodes each bar's token IDs into a ``symusic.Score`` via the REMI
     tokenizer, renders the Score to a temporary WAV file through the
     FluidSynth CLI (v2.5.2), and plays it synchronously through
     ``sounddevice``.  While a bar is playing the queue slot is occupied;
     once playback finishes the slot is freed and the generator can
     produce more.

Throttling & mood-change latency
---------------------------------
Because the queue holds at most 2 bars, when the user changes mood at
most 2 "stale" bars (generated under the old mood) are already queued
and must play out.  After those, the generator immediately picks up the
new mood for all subsequent bars.
"""

import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sounddevice as sd
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
from miditok import TokSequence

from src.core.config import Config
from src.core.utils import get_tokenizer, save_midi
from src.models.mood_generator import MoodModelGenerator, MoodModelGeneratorHandler

# ── FluidSynth / playback settings ──────────────────────────────────────────
# SoundFont is loaded strictly from the project root.
SOUNDFONT = str(Config.PROJECT_ROOT / "SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2")
SAMPLE_RATE = 44100

# Token id emitted by the REMI tokenizer at every bar boundary ("Bar_None").
BAR_TOKEN_ID = 4

# How many completed bars may sit in the queue before the generator blocks.
# A value of 2 means the generator stays at most 2 bars ahead of the player,
# keeping mood-change latency low while avoiding audio gaps.
MAX_BARS_AHEAD = 2

# ── Valence-Arousal → Mood mapping (Russell's Circumplex Model) ─────────────
# Each mood is placed on a 2-D plane: Valence (0-10, negative→positive) and
# Arousal (0-10, calm→energetic).  The UI exposes two sliders and we map the
# (valence, arousal) point to the nearest mood centre below.
MOOD_CENTERS = {
    "angry":       (2, 9),
    "exciting":    (7, 9),
    "fear":        (1, 7),
    "funny":       (7, 6),
    "happy":       (9, 7),
    "lazy":        (4, 1),
    "magnificent": (9, 9),
    "quiet":       (6, 2),
    "romantic":    (8, 4),
    "sad":         (2, 2),
    "warm":        (7, 3),
}

# Minimum Euclidean distance the sliders must move (from the last accepted
# point) before a mood change is pushed to the generator.  Prevents jitter
# from tiny slider movements triggering rapid mood switches.
SLIDER_CHANGE_THRESHOLD = 2.0


def valence_arousal_to_mood(valence: float, arousal: float) -> str:
    """Return the mood label whose centre is closest to (valence, arousal)."""
    best, best_d = "", float("inf")
    for mood, (v, a) in MOOD_CENTERS.items():
        d = math.hypot(valence - v, arousal - a)
        if d < best_d:
            best, best_d = mood, d
    return best


# ── Model loading (cached across Streamlit reruns) ──────────────────────────
@st.cache_resource
def load_model():
    tokenizer = get_tokenizer()
    model = MoodModelGenerator(vocab_size=tokenizer.vocab_size).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    handler = MoodModelGeneratorHandler(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
    )
    handler.load_checkpoint()
    return handler, tokenizer


# ── Thread-safe shared state ────────────────────────────────────────────────
class GenerationState:
    """
    Shared mutable state accessed by all three threads + the Streamlit UI.

    Every read/write of mutable fields goes through ``self.lock`` to
    avoid data races.  The ``bar_queue`` is inherently thread-safe
    (``queue.Queue``).
    """

    def __init__(self, initial_mood: str):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()   # set by Stop button or restart

        mid = Config.MOOD_TO_ID[initial_mood]
        self.target_mood_id: int = mid        # mood id the generator should use NOW
        self.active_mood: str = initial_mood  # human-readable label for the UI

        # Full token / mood history (kept for saving MIDI at the end).
        # Shape: (1, seq_len) — single-batch tensors on the model device.
        self.tokens = torch.tensor([[1]], device=Config.DEVICE)
        self.moods = torch.tensor([[mid]], device=Config.DEVICE)
        self.token_count: int = 1

        # Counters shown in the UI so the user can see the gen/play gap.
        self.bars_generated: int = 0
        self.bars_played: int = 0
        self.running: bool = False

        # Slider state — written by the UI thread, read by the mood watcher.
        center = MOOD_CENTERS[initial_mood]
        self.accepted_valence: float = float(center[0])
        self.accepted_arousal: float = float(center[1])
        self.slider_valence: float = self.accepted_valence
        self.slider_arousal: float = self.accepted_arousal

        # ── The key data structure ──
        # A bounded FIFO of completed bars (each bar = list[int] of token ids).
        # maxsize=MAX_BARS_AHEAD means the generator blocks on put() when the
        # player is more than MAX_BARS_AHEAD bars behind → automatic throttle.
        self.bar_queue: queue.Queue = queue.Queue(maxsize=MAX_BARS_AHEAD)

    def push_mood(self, mood: str, v: float, a: float):
        """Called by the mood-watcher thread when a significant change is detected."""
        with self.lock:
            prev = self.active_mood
            self.target_mood_id = Config.MOOD_TO_ID[mood]
            self.active_mood = mood
            self.accepted_valence = v
            self.accepted_arousal = a
        print(
            f"[mood] {prev!r} -> {mood!r} | v={v:.1f} a={a:.1f}",
            flush=True,
        )

    def set_sliders(self, v: float, a: float):
        """Called by the Streamlit UI on every rerun to relay slider positions."""
        with self.lock:
            self.slider_valence = v
            self.slider_arousal = a

    def get_sliders(self):
        """Read current slider positions (called by mood-watcher thread)."""
        with self.lock:
            return self.slider_valence, self.slider_arousal

    def get_target_mood_id(self) -> int:
        """Read the mood id that should condition the next generated token."""
        with self.lock:
            return self.target_mood_id

    def snapshot(self):
        """Return a consistent tuple of stats for the Streamlit metrics row."""
        with self.lock:
            return (
                self.token_count,
                self.active_mood,
                self.running,
                self.bars_generated,
                self.bars_played,
            )


# ── Thread 1: continuous generation ─────────────────────────────────────────
def generation_worker(handler: MoodModelGeneratorHandler, state: GenerationState):
    """
    Generate tokens one at a time.  Accumulate them into bars (split at
    token id == BAR_TOKEN_ID).  When a bar is complete, enqueue it for
    playback.  The bounded queue blocks the generator when the player is
    more than MAX_BARS_AHEAD bars behind.
    """
    state.running = True

    # Accumulator for the current bar's token ids.  Flushed into the
    # bar_queue every time a BAR_TOKEN_ID is generated.
    current_bar_ids: list[int] = []

    try:
        while not state.stop_event.is_set():
            # 1. Read the mood the user currently wants.
            mood_id = state.get_target_mood_id()

            # 2. Snapshot the running sequence so far (no copy — tensors are
            #    replaced atomically via = so this is safe with the lock).
            with state.lock:
                tokens, moods = state.tokens, state.moods

            # 3. Run one autoregressive step: predict the next token.
            tokens, moods, next_token = handler.generate_single_step(
                tokens, moods, mood_id
            )
            next_id = next_token.item()

            # 4. Write the extended sequence back to shared state.
            with state.lock:
                state.tokens = tokens
                state.moods = moods
                state.token_count = tokens.size(1)

            # 5. Bar splitting — id == BAR_TOKEN_ID means "start of a new bar".
            #    When we see it (and we already have tokens for the previous
            #    bar), we package the previous bar and put it on the queue.
            if next_id == BAR_TOKEN_ID and len(current_bar_ids) > 0:
                bar_copy = list(current_bar_ids)

                # put() blocks when the queue is full (2 bars waiting).
                # This is the throttle: the generator sleeps here until the
                # player finishes a bar and frees a slot.  We use a timeout
                # so we can still honour stop_event while waiting.
                while not state.stop_event.is_set():
                    try:
                        state.bar_queue.put(bar_copy, timeout=0.5)
                        with state.lock:
                            state.bars_generated += 1
                        print(
                            f"[gen] bar #{state.bars_generated} enqueued "
                            f"({len(bar_copy)} tokens)",
                            flush=True,
                        )
                        break
                    except queue.Full:
                        continue

                # Begin the next bar — its first token is this BAR_TOKEN_ID.
                current_bar_ids = [next_id]
            else:
                current_bar_ids.append(next_id)
    finally:
        # Flush whatever partial bar remains (e.g. after the user hits Stop).
        if current_bar_ids:
            try:
                state.bar_queue.put(current_bar_ids, timeout=1.0)
            except queue.Full:
                pass
        # Sentinel (None) tells the playback thread that no more bars are coming.
        try:
            state.bar_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        state.running = False


# ── Thread 2: mood watcher ──────────────────────────────────────────────────
def mood_watcher(state: GenerationState):
    """
    Poll the slider values at ~10 Hz.  When the (valence, arousal) point
    has drifted far enough from the last accepted position AND the nearest
    mood label has actually changed, push the new mood to the generation
    state.  The generator picks it up on its next iteration.
    """
    while not state.stop_event.is_set():
        v, a = state.get_sliders()
        new_mood = valence_arousal_to_mood(v, a)
        dist = math.hypot(v - state.accepted_valence, a - state.accepted_arousal)

        if dist >= SLIDER_CHANGE_THRESHOLD and new_mood != state.active_mood:
            state.push_mood(new_mood, v, a)

        time.sleep(0.1)


# ── Thread 3: live playback ─────────────────────────────────────────────────
def _play_score(score, soundfont: str):
    """
    Render a symusic Score to audio and play it synchronously.

    Pipeline:  Score → temp .mid → FluidSynth CLI → temp .wav → sounddevice

    FluidSynth 2.5.x requires option flags *before* positional args:
        fluidsynth -ni -F output.wav -r 44100 soundfont.sf2 input.mid
    (midi2audio puts -F *after* the files which causes the "-F is an
    illegal option at this place" error, so we call the binary directly.)

    Playback is blocking (sd.wait) – the caller stays here until the
    entire bar has finished playing.
    """
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mid_f:
        mid_path = mid_f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
        wav_path = wav_f.name
    try:
        score.dump_midi(mid_path)
        subprocess.run(
            [
                "fluidsynth", "-ni",
                "-F", wav_path,
                "-r", str(SAMPLE_RATE),
                soundfont,
                mid_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        audio, sr = sf.read(wav_path)
        sd.play(audio, sr)
        sd.wait()
    finally:
        os.unlink(mid_path)
        os.unlink(wav_path)


def playback_worker(state: GenerationState, tokenizer):
    """
    Consumer side of the bar queue.

    For each bar:
      1. Wrap the token-id list in a ``TokSequence`` and decode it back
         into a ``symusic.Score`` (contains MIDI note events for one bar).
      2. Call ``_play_score`` which renders the Score to WAV via FluidSynth
         and plays it synchronously (blocks until the bar finishes playing).
      3. After playback, bump ``bars_played`` and free the queue slot so
         the generator can continue.

    A ``None`` sentinel on the queue means the generator has stopped —
    exit the loop gracefully.
    """
    while not state.stop_event.is_set():
        try:
            bar_ids = state.bar_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # None is the sentinel pushed by the generator when it exits.
        if bar_ids is None:
            break

        try:
            tok_seq = TokSequence(ids=bar_ids)
            score = tokenizer.decode(tok_seq)
            _play_score(score, SOUNDFONT)
            with state.lock:
                state.bars_played += 1
            print(f"[play] bar #{state.bars_played} done", flush=True)
        except Exception as e:
            print(f"[play] error: {e}", flush=True)

    sd.stop()


# ── Streamlit UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="MAESTRO Live", layout="wide")
st.title("MAESTRO – Continuous Mood Generation")

handler, tokenizer = load_model()

if "state" not in st.session_state:
    st.session_state.state = None

with st.sidebar:
    st.header("Mood Control")
    valence = st.slider("Valence  (negative → positive)", 0, 10, 5)
    arousal = st.slider("Arousal  (calm → energetic)", 0, 10, 5)
    mapped_mood = valence_arousal_to_mood(valence, arousal)
    st.markdown(f"**Mapped mood:** `{mapped_mood}`")

    col_start, col_stop = st.columns(2)
    btn_start = col_start.button("Start")
    btn_stop = col_stop.button("Stop")
    btn_save = st.button("Save MIDI")

state: GenerationState | None = st.session_state.state

if btn_start:
    if state is not None:
        state.stop_event.set()
    state = GenerationState(initial_mood=mapped_mood)
    st.session_state.state = state
    threading.Thread(target=generation_worker, args=(handler, state), daemon=True).start()
    threading.Thread(target=mood_watcher, args=(state,), daemon=True).start()
    threading.Thread(target=playback_worker, args=(state, tokenizer), daemon=True).start()

if btn_stop and state is not None:
    state.stop_event.set()

if state is not None and state.running:
    state.set_sliders(float(valence), float(arousal))

if state is not None:
    count, mood, running, bars_gen, bars_played = state.snapshot()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "Generating" if running else "Stopped")
    c2.metric("Tokens", f"{count:,}")
    c3.metric("Active Mood", mood)
    c4.metric("Bars", f"{bars_played} / {bars_gen}")

    if running:
        time.sleep(0.5)
        st.rerun()

if btn_save and state is not None:
    with state.lock:
        token_list = state.tokens.squeeze(0).cpu().tolist()
    save_midi(token_list, tokenizer, "generated_live.mid")
    st.success("Saved to `generated_live.mid`")
