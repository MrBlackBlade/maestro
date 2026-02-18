"""
Streamlit App – XMIDI Live Music Studio

Combines the trained Generator and Refiner models to produce music from
user-selected Mood and Genre.  Outputs a downloadable MIDI file and,
if FluidSynth is installed, plays the audio in the browser.

Usage
-----
    cd music-generator-test
    streamlit run app.py
"""
import sys
import os
import tempfile
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.model_generator import MusicGenerator
from src.model_refiner import LevenshteinRefiner
from src.utils import get_tokenizer

# ======================================================================
# Page config
# ======================================================================
st.set_page_config(page_title="XMIDI Live Studio", page_icon="🎹", layout="wide")

# ======================================================================
# Cached model loaders (run once, survive reruns)
# ======================================================================
DEVICE = Config.DEVICE


@st.cache_resource
def load_generator():
    """Load the Generator checkpoint independently."""
    ckpt_path = Config.GENERATOR_CKPT_DIR / "generator_best.pt"
    if not ckpt_path.exists():
        ckpt_path = Config.GENERATOR_CKPT_DIR / "generator_latest.pt"
    if not ckpt_path.exists():
        return None

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = MusicGenerator(
        vocab_size=checkpoint["vocab_size"],
        num_moods=checkpoint["num_moods"],
        num_genres=checkpoint["num_genres"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_refiner():
    """Load the Refiner checkpoint independently."""
    ckpt_path = Config.REFINER_CKPT_DIR / "refiner_best.pt"
    if not ckpt_path.exists():
        ckpt_path = Config.REFINER_CKPT_DIR / "refiner_latest.pt"
    if not ckpt_path.exists():
        return None

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = LevenshteinRefiner(
        vocab_size=checkpoint["vocab_size"],
        num_moods=checkpoint["num_moods"],
        num_genres=checkpoint["num_genres"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_tokenizer():
    return get_tokenizer(Config.TOKENIZER_PARAMS_PATH)


# ======================================================================
# Sampling helpers
# ======================================================================
def top_k_top_p_sample(logits: torch.Tensor, top_k: int, top_p: float, temperature: float):
    """
    Apply temperature scaling, top-k, and nucleus (top-p) filtering,
    then sample a single token.
    """
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
        # Scatter back
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


# ======================================================================
# UI
# ======================================================================
st.title("🎹 XMIDI Live Music Studio")
st.markdown("Generate emotion-driven music using AI.  Select a **Mood** and **Genre**, then hit *Generate*.")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("🎛️ Controls")

    mood_names = [m.capitalize() for m in Config.MOODS]
    selected_mood = st.selectbox("Mood", mood_names, index=4)  # default: Happy

    genre_names = [g.capitalize() for g in Config.GENRES]
    selected_genre = st.selectbox("Genre", genre_names, index=3)  # default: Pop

    st.divider()
    generate_length = st.slider("Tokens to generate", 64, 1024, Config.GENERATE_LENGTH, step=64)
    temperature = st.slider("Temperature", 0.1, 2.0, Config.TEMPERATURE, step=0.05)
    top_k = st.slider("Top-K", 0, 200, Config.TOP_K)
    top_p = st.slider("Top-P (nucleus)", 0.0, 1.0, Config.TOP_P, step=0.05)

    st.divider()
    use_refiner = st.checkbox("Apply Refiner (polish output)", value=True)
    refiner_passes = st.slider("Refiner passes", 1, 3, 1) if use_refiner else 1

# Map UI labels back to IDs
mood_id = Config.MOOD_TO_ID[selected_mood.lower()]
genre_id = Config.GENRE_TO_ID[selected_genre.lower()]

# ---- Main area ----
col1, col2 = st.columns([2, 1])

with col1:
    generate_btn = st.button("🎵 Generate Music", type="primary", use_container_width=True)

with col2:
    st.info(f"**Mood:** {selected_mood}  |  **Genre:** {selected_genre}")

if generate_btn:
    gen_model = load_generator()
    ref_model = load_refiner() if use_refiner else None
    tokenizer = load_tokenizer()

    if gen_model is None:
        st.error(
            "Generator checkpoint not found!  "
            "Run `python 2_train_generator.py` first to train the model."
        )
        st.stop()

    # ---- 1. GENERATE DRAFT ----
    st.subheader("1️⃣ Generating draft...")
    progress = st.progress(0)
    status_text = st.empty()

    m_id = torch.tensor([mood_id], device=DEVICE)
    g_id = torch.tensor([genre_id], device=DEVICE)

    # Start with BOS token (id=1 in most tokenizers, fallback to 1)
    bos_id = tokenizer.special_tokens_ids.get("BOS", 1) if hasattr(tokenizer, "special_tokens_ids") else 1
    sequence = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for i in range(generate_length):
            # Sliding window: keep last MAX_SEQ_LEN tokens
            ctx = sequence[:, -Config.MAX_SEQ_LEN :]
            logits = gen_model(ctx, m_id, g_id)
            next_logits = logits[:, -1, :]  # last position
            next_token = top_k_top_p_sample(next_logits, top_k, top_p, temperature)
            sequence = torch.cat([sequence, next_token], dim=1)

            if (i + 1) % 10 == 0 or i == generate_length - 1:
                progress.progress((i + 1) / generate_length)
                status_text.text(f"Generated {i + 1}/{generate_length} tokens")

    draft_ids = sequence[0].cpu().tolist()
    status_text.text(f"Draft complete: {len(draft_ids)} tokens")

    # ---- 2. REFINE (optional) ----
    if use_refiner and ref_model is not None:
        st.subheader("2️⃣ Refining & polishing...")
        refined = sequence.clone()
        with torch.no_grad():
            for p in range(refiner_passes):
                del_logits, tok_logits = ref_model(refined, m_id, g_id)
                # Replace every position with the refiner's top prediction
                refined = torch.argmax(tok_logits, dim=-1)  # [B, S]
                st.text(f"  Refiner pass {p + 1}/{refiner_passes} done")
        final_ids = refined[0].cpu().tolist()
    else:
        final_ids = draft_ids
        if use_refiner and ref_model is None:
            st.warning("Refiner checkpoint not found – skipping refinement.")

    # ---- 3. CONVERT TO MIDI & PLAY ----
    st.subheader("3️⃣ Your AI Composition")

    # Decode tokens back to MIDI
    try:
        midi_obj = tokenizer.decode(final_ids)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            midi_path = tmp.name
            midi_obj.dump_midi(midi_path)

        # Download button
        with open(midi_path, "rb") as f:
            midi_bytes = f.read()
        st.download_button(
            label="⬇️ Download MIDI",
            data=midi_bytes,
            file_name=f"generated_{selected_mood}_{selected_genre}.mid",
            mime="audio/midi",
        )

        # Try to play audio (requires midi2audio + FluidSynth)
        try:
            from midi2audio import FluidSynth

            # Common soundfont locations
            sf_candidates = [
                Path(__file__).parent / "FluidR3_GM.sf2",
                Path(__file__).parent / "soundfont.sf2",
                Path(r"C:\soundfonts\FluidR3_GM.sf2"),
            ]
            sf_path = None
            for candidate in sf_candidates:
                if candidate.exists():
                    sf_path = str(candidate)
                    break

            if sf_path:
                wav_path = midi_path.replace(".mid", ".wav")
                fs = FluidSynth(sf_path)
                fs.midi_to_audio(midi_path, wav_path)
                st.audio(wav_path, format="audio/wav")
            else:
                st.info(
                    "💡 To hear audio in-browser, place a SoundFont file "
                    "(e.g. `FluidR3_GM.sf2`) in the project root and install FluidSynth."
                )
        except ImportError:
            st.info(
                "💡 Install `midi2audio` and FluidSynth for in-browser playback: "
                "`pip install midi2audio`"
            )

        # Clean up temp files
        try:
            os.unlink(midi_path)
            wav_path_check = midi_path.replace(".mid", ".wav")
            if os.path.exists(wav_path_check):
                os.unlink(wav_path_check)
        except Exception:
            pass

    except Exception as e:
        st.error(f"Failed to decode tokens to MIDI: {e}")

    st.success("✅ Generation complete!")

# ======================================================================
# Footer
# ======================================================================
st.divider()
st.caption(
    "Built with PyTorch & MidiTok  •  "
    "Generator: Autoregressive Transformer  •  "
    "Refiner: Levenshtein Transformer  •  "
    "Dataset: XMIDI (108K files)"
)

