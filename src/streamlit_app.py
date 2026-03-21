"""
MAESTRO – Continuous Mood-Driven MIDI Generation.

Three background threads run while the Streamlit UI stays responsive:
  1. Generation thread  – calls generate_single_step in a tight loop.
  2. Mood-watcher thread – polls slider values and pushes a new mood
     only when the sliders moved far enough from the last accepted point.
  3. Playback thread    – decodes tokens → Score → audio and plays live.
"""

import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd
import streamlit as st
import torch
import torch.nn as nn
from symusic import Synthesizer

from src.core.config import Config
from src.core.utils import get_tokenizer, save_midi
from src.models.mood_generator import MoodModelGenerator, MoodModelGeneratorHandler

# ── Playback settings ──────────────────────────────────────────────────────
SOUNDFONT_PATH = "SOUNDFONT PATH"            # <-- change this to your .sf2 path
SAMPLE_RATE = 44100
PLAYBACK_CHUNK = 256                         # tokens to accumulate before playing

# ── Valence-Arousal → Mood mapping ─────────────────────────────────────────
# Positions follow Russell's Circumplex Model:
#   Valence 0-10 (negative → positive), Arousal 0-10 (calm → energetic)
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

SLIDER_CHANGE_THRESHOLD = 2.0


def valence_arousal_to_mood(valence: float, arousal: float) -> str:
    best, best_d = "", float("inf")
    for mood, (v, a) in MOOD_CENTERS.items():
        d = math.hypot(valence - v, arousal - a)
        if d < best_d:
            best, best_d = mood, d
    return best


# ── Model loading (cached across Streamlit reruns) ─────────────────────────
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


# ── Thread-safe shared state ───────────────────────────────────────────────
class GenerationState:
    def __init__(self, initial_mood: str):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        mid = Config.MOOD_TO_ID[initial_mood]
        self.target_mood_id: int = mid
        self.active_mood: str = initial_mood

        self.tokens = torch.tensor([[1]], device=Config.DEVICE)
        self.moods = torch.tensor([[mid]], device=Config.DEVICE)
        self.token_count: int = 1
        self.running: bool = False

        center = MOOD_CENTERS[initial_mood]
        self.accepted_valence: float = float(center[0])
        self.accepted_arousal: float = float(center[1])
        self.slider_valence: float = self.accepted_valence
        self.slider_arousal: float = self.accepted_arousal

        self.last_played_count: int = 0
        self.last_audio_samples: int = 0

    def push_mood(self, mood: str, v: float, a: float):
        with self.lock:
            prev = self.active_mood
            prev_id = self.target_mood_id
            self.target_mood_id = Config.MOOD_TO_ID[mood]
            self.active_mood = mood
            self.accepted_valence = v
            self.accepted_arousal = a
            new_id = self.target_mood_id
        print(
            f"[mood] PUSH: {prev!r} (id={prev_id}) -> {mood!r} (id={new_id}) | "
            f"sliders v={v:.1f} a={a:.1f}",
            flush=True,
        )

    def set_sliders(self, v: float, a: float):
        with self.lock:
            self.slider_valence = v
            self.slider_arousal = a

    def get_sliders(self):
        with self.lock:
            return self.slider_valence, self.slider_arousal

    def get_target_mood_id(self) -> int:
        with self.lock:
            return self.target_mood_id

    def snapshot(self):
        with self.lock:
            return self.token_count, self.active_mood, self.running


# ── Thread 1: continuous generation ────────────────────────────────────────
def generation_worker(handler: MoodModelGeneratorHandler, state: GenerationState):
    id_to_mood = {i: m for m, i in Config.MOOD_TO_ID.items()}
    state.running = True
    last_mood_id: int | None = None
    try:
        while not state.stop_event.is_set():
            mood_id = state.get_target_mood_id()
            if mood_id != last_mood_id:
                name = id_to_mood.get(mood_id, "?")
                if last_mood_id is None:
                    print(
                        f"[gen] started: target_mood_id={mood_id} ({name!r})",
                        flush=True,
                    )
                else:
                    print(
                        f"[gen] picked up new mood: target_mood_id={mood_id} ({name!r}) "
                        f"(was {last_mood_id} {id_to_mood.get(last_mood_id, '?')!r})",
                        flush=True,
                    )
                last_mood_id = mood_id

            with state.lock:
                tokens, moods = state.tokens, state.moods

            tokens, moods, _ = handler.generate_single_step(tokens, moods, mood_id)

            with state.lock:
                state.tokens = tokens
                state.moods = moods
                state.token_count = tokens.size(1)
    finally:
        state.running = False


# ── Thread 2: mood watcher ─────────────────────────────────────────────────
def mood_watcher(state: GenerationState):
    while not state.stop_event.is_set():
        v, a = state.get_sliders()
        new_mood = valence_arousal_to_mood(v, a)
        dist = math.hypot(v - state.accepted_valence, a - state.accepted_arousal)

        if dist >= SLIDER_CHANGE_THRESHOLD and new_mood != state.active_mood:
            state.push_mood(new_mood, v, a)

        time.sleep(0.1)


# ── Thread 3: live playback ────────────────────────────────────────────────
def playback_worker(state: GenerationState, tokenizer):
    synth = Synthesizer(SOUNDFONT_PATH, sample_rate=SAMPLE_RATE)

    while not state.stop_event.is_set():
        with state.lock:
            new_tokens = state.token_count - state.last_played_count

        if new_tokens < PLAYBACK_CHUNK:
            time.sleep(0.1)
            continue

        with state.lock:
            token_list = state.tokens.squeeze(0).cpu().tolist()
            start_sample = state.last_audio_samples

        try:
            score = tokenizer.decode(token_list)
            audio_array = synth.render(score)
            playable_audio = np.ascontiguousarray(audio_array.T)

            new_audio = playable_audio[start_sample:]
            if len(new_audio) > 0:
                sd.play(new_audio, samplerate=SAMPLE_RATE)
                sd.wait()

            with state.lock:
                state.last_audio_samples = len(playable_audio)
                state.last_played_count = state.token_count
        except Exception:
            time.sleep(0.1)

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
    count, mood, running = state.snapshot()
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", "Generating" if running else "Stopped")
    c2.metric("Tokens", f"{count:,}")
    c3.metric("Active Mood", mood)

    if running:
        time.sleep(0.5)
        st.rerun()

if btn_save and state is not None:
    with state.lock:
        token_list = state.tokens.squeeze(0).cpu().tolist()
    save_midi(token_list, tokenizer, "generated_live.mid")
    st.success("Saved to `generated_live.mid`")
