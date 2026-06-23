import streamlit as st
import torch
import threading
import queue
import time

import sounddevice as sd
import numpy as np
import os

# Adjust these imports based on your exact project structure
from src.core.config import Config
from src.core.utils import get_tokenizer
from src.models.cached_transformer import KVCache
from src.models.general_model_handler import GeneralModelHandler
# Assuming your model code is in src/models/neg_cfg_generator.py
from src.models.neg_cfg_generator import NegCFGGenerator, NegCFGGeneratorHandler
from src.core.audio_engine import AudioEngine 

# ---------------------------------------------------------------------------
# 1. Resource Caching (Load Once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_system():
    """Loads the model, handler, and audio engine only once."""
    device = Config.DEVICE
    tokenizer = get_tokenizer()
    
    # Initialize Model
    model = NegCFGGenerator(vocab_size=tokenizer.vocab_size).to(device)
    
    # Initialize Dummy Optimizer/Scheduler/Criterion for the Handler
    # (Since we are only generating, we don't need real ones, but the Handler expects them)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    criterion = torch.nn.CrossEntropyLoss()
    
    handler = NegCFGGeneratorHandler(model, optimizer, criterion, scheduler)
    
    # Load your best checkpoint (adjust epoch/path if necessary)
    try:
        handler.load_checkpoint() 
    except Exception as e:
        st.warning(f"Could not load checkpoint automatically: {e}")

    handler.model.eval()

    # Initialize Audio Engine
    engine = AudioEngine()

    return tokenizer, handler, engine

# ---------------------------------------------------------------------------
# 2. The Background Generation Thread
# ---------------------------------------------------------------------------
def generation_loop(handler, engine, stop_event, mood_queue):
    device = Config.DEVICE
    
    # # --- DIAGNOSTIC 1: Check Soundfont ---
    # sf2_path = Config.RESOURCES_DIR / "FluidR3_GM.sf2"
    # print(f"\n🎹 [DIAGNOSTIC] Soundfont exists at {sf2_path}: {os.path.exists(sf2_path)}")
    
    # # --- DIAGNOSTIC 2: Check Hardware Audio Device ---
    # try:
    #     device_info = sd.query_devices(sd.default.device[1])
    #     print(f"🎧 [DIAGNOSTIC] Playing audio to hardware: {device_info['name']}")
    # except Exception as e:
    #     print(f"🎧 [DIAGNOSTIC] Could not query audio device: {e}")

    # # --- DIAGNOSTIC 3: Intercept the Audio Stream (Monkey-Patch) ---
    # original_write = engine.stream.write
    # def sniff_audio(audio_data):
    #     # Measure the maximum volume peak in the chunk
    #     max_vol = np.max(np.abs(audio_data))
    #     is_silent = "YES" if max_vol < 0.0001 else "NO"
    #     print(f"🔊 [AUDIO PLAYING] Sent {len(audio_data)} samples to speakers | Max Volume: {max_vol:.4f} | Is Silent? {is_silent}")
    #     original_write(audio_data)
    
    # # Override the engine's stream write method with our sniffer
    # engine.stream.write = sniff_audio

    # --- STANDARD GENERATION LOGIC ---
    current_tokens = torch.tensor([[1]], device=device) 
    target_mood_id = 0
    
    print("\n--- [THREAD STARTED] Waiting for UI mood... ---")
    initial_mood = mood_queue.get() 
    target_mood_id = Config.MOOD_TO_ID[initial_mood]
    current_moods = torch.tensor([[target_mood_id]], device=device)

    if Config.USE_KV_CACHE:
        cache = KVCache.from_model(handler.model, batch_size=Config.NUM_MOODS + 1)
    else:
        cache = None

    try:
        token_count = 0

        while not stop_event.is_set():
            try:
                new_mood = mood_queue.get_nowait()
                target_mood_id = Config.MOOD_TO_ID[new_mood]
            except queue.Empty:
                pass 

            while (engine.audio_queue.qsize() > 1):
                time.sleep(0.1)

            with torch.inference_mode():
                current_tokens, current_moods, next_token = handler.generate_single_step(
                    current_tokens, current_moods, target_mood_id, cache=cache
                )

            tok_val = next_token.item()
            engine.push_token(tok_val)
            token_count += 1
            
            if tok_val == 4:
                print(f"🎵 [BAR DETECTED] Token 4 at #{token_count} - Sent to FluidSynth!")
                
    finally:
        engine.push_token(4, stop=True) 
        engine.playback_done.wait()
        print("🎹 [THREAD EXITING] Generation loop has ended.")


# ---------------------------------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Live MIDI Generator", layout="centered")
st.title("🎹 Live Autoregressive MIDI Generator")

# Load resources
tokenizer, handler, engine = load_system()

# Initialize Session State variables for thread management
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "mood_queue" not in st.session_state:
    st.session_state.mood_queue = queue.Queue()

st.markdown("---")

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Get available moods dynamically from your Config
    selected_mood = st.selectbox(
        "Select Generative Mood", 
        list(Config.MOOD_TO_ID.keys()),
        index=0
    )
    # Push the currently selected mood to the queue so the thread picks it up
    st.session_state.mood_queue.put(selected_mood)

with col2:
    st.write("") # spacing
    st.write("") # spacing
    
    if not st.session_state.is_playing:
        if st.button("▶️ Play", use_container_width=True):
            st.session_state.is_playing = True
            st.session_state.stop_event = threading.Event()
            
            # Spin up the generator worker thread
            worker_thread = threading.Thread(
                target=generation_loop,
                args=(handler, engine, st.session_state.stop_event, st.session_state.mood_queue),
                daemon=True
            )
            worker_thread.start()
            st.rerun()
    else:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.is_playing = False
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.rerun()

# Status indicator
if st.session_state.is_playing:
    st.success(f"🎵 Currently Generating & Playing: **{selected_mood}**")
else:
    st.info("Status: Stopped. Click Play to start generating.")