import sys
import os
import base64
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import Config
from src.models.mood_generator import MoodModelGenerator
from src.models.refiner import ModelRefiner
from src.core.utils import get_tokenizer
from src.core.realtime_player import RealtimeMidiPlayer

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
    """Load the mood-only generator_2 checkpoint independently."""
    ckpt_path = Config.MODEL_CKPT_DIR / "generator_2" / "generator_2_best.pt"
    if not ckpt_path.exists():
        return None

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    vocab_size = checkpoint["model_state_dict"]["token_emb.weight"].shape[0]
    model = MoodModelGenerator(
        vocab_size=vocab_size,
        num_moods=Config.NUM_MOODS + 1,
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
    model = ModelRefiner(
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
# Streaming helpers
# ======================================================================
STREAM_INTERVAL = 128  # Decode + refresh the live player every N tokens


def _build_autoplay_player_html(midi_bytes: bytes) -> str:
    """Return a minimal html-midi-player snippet that autoplays the given MIDI bytes."""
    midi_b64 = base64.b64encode(midi_bytes).decode("utf-8")
    uri = f"data:audio/midi;base64,{midi_b64}"
    return f"""
    <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.77,npm/@magenta/music@1.23.1/es6/core.js,npm/html-midi-player@1.5.0"></script>
    <style>
      midi-player {{ display: block; width: 100%; }}
      midi-player::part(control-panel) {{ background: #1a1a2e; border: 1px solid #333; border-radius: 8px; }}
      midi-player::part(play-button) {{ color: #1db954; border: 2px solid #1db954; border-radius: 50%; }}
    </style>
    <midi-player src="{uri}" sound-font autoplay></midi-player>
    """


def render_partial_player(placeholder, token_ids: list, tokenizer, tokenizer_vocab_size: int) -> None:
    """
    Decode token_ids to MIDI and render an autoplaying preview inside *placeholder*.
    Silently skips if the partial sequence cannot produce a valid MIDI object.
    """
    safe_ids = [min(max(tid, 0), tokenizer_vocab_size - 1) for tid in token_ids]
    try:
        midi_obj = tokenizer(safe_ids)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            ppath = tmp.name
        midi_obj.dump_midi(ppath)
        with open(ppath, "rb") as f:
            midi_bytes = f.read()
        try:
            os.unlink(ppath)
        except Exception:
            pass
        if len(midi_bytes) < 20:          # empty / header-only MIDI — skip
            return
        player_html = _build_autoplay_player_html(midi_bytes)
        with placeholder:
            components.html(player_html, height=90)
    except Exception:
        pass  # partial sequence may not yet contain complete notes


# ======================================================================
# Sampling helpers
# ======================================================================
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

    # Get actual tokenizer vocabulary size
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = gen_model.vocab_size
    
    # Diagnostic: Check if there's a mismatch and explain why
    if model_vocab_size != tokenizer_vocab_size:
        st.warning(
            f"⚠️ Vocabulary size mismatch detected!\n\n"
            f"**Model vocab_size:** {model_vocab_size}\n"
            f"**Tokenizer vocab_size:** {tokenizer_vocab_size}\n\n"
            f"**Why this happens:**\n"
            f"- The model was trained with vocab_size={model_vocab_size} and outputs logits for indices [0, {model_vocab_size-1}]\n"
            f"- The tokenizer currently has {tokenizer_vocab_size} tokens\n"
            f"- During sampling, we might sample indices >= {tokenizer_vocab_size} which don't exist in the tokenizer\n\n"
            f"**Possible causes:**\n"
            f"1. Tokenizer was retrained/modified after model training\n"
            f"2. Different tokenizer file is being loaded than the one used during training\n"
            f"3. Tokenizer vocabulary changed (e.g., special tokens added/removed)\n\n"
            f"**Solution:** Clamping logits to tokenizer size to prevent decode errors."
        )
    else:
        # Even if sizes match, we should still validate token IDs are in valid range
        # because MidiTok might have gaps or non-contiguous token IDs
        st.info(f"✓ Vocabulary sizes match: {model_vocab_size} tokens")

    # ---- 1. GENERATE DRAFT (with real-time audio playback) ----
    st.subheader("1️⃣ Generating draft…")
    progress = st.progress(0)
    status_text = st.empty()

    st.caption("🎵 **Real-time playback** — notes play through your speakers as tokens are generated.")

    m_id = torch.tensor([mood_id], device=DEVICE)

    # Start with BOS token (id=1 in most tokenizers, fallback to 1)
    bos_id = tokenizer.special_tokens_ids[1] if hasattr(tokenizer, "special_tokens_ids") else 1
    bos_id = min(bos_id, tokenizer_vocab_size - 1)
    sequence = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)

    player = RealtimeMidiPlayer(tokenizer, bpm=120)

    with torch.no_grad():
        for i in range(generate_length):
            ctx = sequence[:, -Config.MAX_SEQ_LEN :]
            logits = gen_model(ctx, m_id)
            next_logits = logits[:, -1, :]
            next_token = top_k_top_p_sample(next_logits, top_k, top_p, temperature, tokenizer_vocab_size)
            next_token = torch.clamp(next_token, 0, tokenizer_vocab_size - 1)
            sequence = torch.cat([sequence, next_token], dim=1)

            player.feed_token(next_token.item())

            if (i + 1) % 10 == 0 or i == generate_length - 1:
                progress.progress((i + 1) / generate_length)
                status_text.text(f"Generated {i + 1}/{generate_length} tokens")

    player.close()

    draft_ids = sequence[0].cpu().tolist()
    # Validate all token IDs are in valid range
    draft_ids = [min(max(tid, 0), tokenizer_vocab_size - 1) for tid in draft_ids]
    status_text.text(f"Draft complete: {len(draft_ids)} tokens")

    # ---- 2. REFINE (optional) ----
    if use_refiner and ref_model is not None:
        st.subheader("2️⃣ Refining & polishing...")
        g_id = torch.tensor([genre_id], device=DEVICE)
        refined = sequence.clone()
        with torch.no_grad():
            for p in range(refiner_passes):
                del_logits, tok_logits = ref_model(refined, m_id, g_id)
                if tok_logits.size(-1) > tokenizer_vocab_size:
                    tok_logits = tok_logits[:, :, :tokenizer_vocab_size]
                refined = torch.argmax(tok_logits, dim=-1)
                refined = torch.clamp(refined, 0, tokenizer_vocab_size - 1)
                st.text(f"  Refiner pass {p + 1}/{refiner_passes} done")
        final_ids = refined[0].cpu().tolist()
        final_ids = [min(max(tid, 0), tokenizer_vocab_size - 1) for tid in final_ids]
    else:
        final_ids = draft_ids
        if use_refiner and ref_model is None:
            st.warning("Refiner checkpoint not found - skipping refinement.")

    # ---- 3. CONVERT TO MIDI & PLAY ----
    st.subheader("3️⃣ Your AI Composition")

    # Validate token IDs before decoding
    invalid_ids = [tid for tid in final_ids if tid < 0 or tid >= tokenizer_vocab_size]
    if invalid_ids:
        st.error(
            f"❌ Invalid token IDs detected: {len(invalid_ids)} out of {len(final_ids)} tokens "
            f"are outside valid range [0, {tokenizer_vocab_size - 1}]. "
            f"First few invalid IDs: {invalid_ids[:10]}"
        )
        safe_token = bos_id if 0 <= bos_id < tokenizer_vocab_size else 0
        final_ids = [tid if 0 <= tid < tokenizer_vocab_size else safe_token for tid in final_ids]
        st.warning(f"Replaced {len(invalid_ids)} invalid token IDs with safe token {safe_token}")

    # Decode tokens back to MIDI
    try:
        midi_obj = tokenizer(final_ids)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            midi_path = tmp.name
        midi_obj.dump_midi(midi_path)

        # Read MIDI bytes for both playback and download
        with open(midi_path, "rb") as f:
            midi_bytes = f.read()

        # Base64-encode MIDI for in-browser player
        midi_b64 = base64.b64encode(midi_bytes).decode("utf-8")
        midi_data_uri = f"data:audio/midi;base64,{midi_b64}"

        # Embed html-midi-player web component (Tone.js synth, zero deps)
        player_html = f"""
        <script
          src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.77,npm/@magenta/music@1.23.1/es6/core.js,npm/html-midi-player@1.5.0"
        ></script>
        <style>
          midi-player {{
            display: block;
            width: 100%;
            margin-bottom: 8px;
          }}
          midi-player::part(control-panel) {{
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 8px;
          }}
          midi-player::part(play-button) {{
            color: #1db954;
            border: 2px solid #1db954;
            border-radius: 50%;
            transition: all 0.2s;
          }}
          midi-player::part(play-button):hover {{
            background: #1db954;
            color: #fff;
          }}
          midi-player::part(time) {{
            color: #e0e0e0;
            font-family: 'Segoe UI', sans-serif;
          }}
          midi-visualizer {{
            display: block;
            width: 100%;
            overflow-x: auto;
          }}
          midi-visualizer .piano-roll-visualizer {{
            background: #0d1117;
            border: 1px solid #333;
            border-radius: 8px;
            overflow: hidden;
          }}
          midi-visualizer svg rect.note {{
            opacity: 0.85;
            rx: 2;
            ry: 2;
          }}
          midi-visualizer svg rect.note[data-is-active="true"] {{
            opacity: 1;
            stroke: #1db954;
            stroke-width: 1.5;
          }}
        </style>
        <midi-player
          src="{midi_data_uri}"
          sound-font
          autoplay
          visualizer="#myVisualizer"
        ></midi-player>
        <midi-visualizer
          type="piano-roll"
          id="myVisualizer"
          src="{midi_data_uri}"
        ></midi-visualizer>
        """
        components.html(player_html, height=480, scrolling=True)

        # Download button (fallback)
        st.download_button(
            label="⬇️ Download MIDI",
            data=midi_bytes,
            file_name=f"generated_{selected_mood}_{selected_genre}.mid",
            mime="audio/midi",
        )

        # Clean up temp file
        try:
            os.unlink(midi_path)
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

