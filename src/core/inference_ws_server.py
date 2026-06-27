"""
FastAPI WebSocket server for continuous MAESTRO affect inference & real-time Chrollo Music Generation.

Run:
    uvicorn src.core.inference_ws_server:app --host 0.0.0.0 --port 8000 --reload

WebSocket endpoints:
    ws://localhost:8000/ws/inference     (Physiological Affect Inference only)
    ws://localhost:8000/ws/mood_music  (Combined Affect Inference + Real-time MIDI generation)
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.core.simulate_inference import (
    DEFAULT_DATASET_PATH,
    LoadedModels,
    MIN_SIGNAL_SAMPLES,
    RECOMMENDED_SIGNAL_SAMPLES,
    create_pipeline,
    generate_dummy_signals,
    load_dataset_windows,
    load_lstm_models,
    normalize_signal_dict,
    run_inference,
)

# --- Added imports for Music Generation ---
from src.core.config import Config
from src.core.utils import get_tokenizer
from src.models.mood_classifier import MoodClassifier, MoodClassifierHandler
from src.models.chrollo import Chrollo, ChrolloHandler 

logger = logging.getLogger(__name__)

# Globals for models
_models: LoadedModels | None = None
_chrollo: Chrollo | None = None
_mood_classifier: MoodClassifier | None = None
_chrollo_handler: ChrolloHandler | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _models, _chrollo, _mood_classifier, _chrollo_handler
    
    # 1. Load LSTM Physiological Models
    logger.info("Loading LSTM models for WebSocket inference...")
    _models = load_lstm_models()
    logger.info("Models ready on %s", _models.device)
    
    # 2. Load Chrollo Music Models
    logger.info("Loading Chrollo and Classifier models for Music generation...")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    device = Config.DEVICE

    _mood_classifier = MoodClassifier(vocab_size=vocab_size).to(device)
    mc_opt = torch.optim.AdamW(_mood_classifier.parameters())
    mc_sch = torch.optim.lr_scheduler.CosineAnnealingLR(mc_opt, T_max=1)
    mc_criterion = nn.CrossEntropyLoss()
    mc_handler = MoodClassifierHandler(
        model=_mood_classifier, optimizer=mc_opt, scheduler=mc_sch, criterion=mc_criterion
    )
    mc_handler.load_checkpoint()  # Loads best epoch by default

    _chrollo = Chrollo(vocab_size=vocab_size).to(device)
    chr_opt = torch.optim.AdamW(_chrollo.parameters())
    chr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(chr_opt, T_max=1)
    chr_criterion = nn.CrossEntropyLoss(ignore_index=0)
    _chrollo_handler = ChrolloHandler(
        model=_chrollo, optimizer=chr_opt, scheduler=chr_sch, criterion=chr_criterion, classifier_handler=mc_handler
    )
    _chrollo_handler.load_checkpoint()

    _mood_classifier.eval()
    _chrollo.eval()
    logger.info("Music generation models loaded successfully.")

    yield
    
    # Cleanup
    _models = None
    _chrollo = None
    _mood_classifier = None
    _chrollo_handler = None


app = FastAPI(
    title="MAESTRO Inference & Music WebSocket",
    description="Stream physiological signals and receive continuous predictions + generated MIDI tokens.",
    lifespan=lifespan,
)


def _new_session_pipeline():
    if _models is None:
        raise RuntimeError("Models are not loaded yet.")
    return create_pipeline(_models)


def _error_message(exc: Exception) -> dict[str, Any]:
    return {"type": "error", "message": str(exc)}


def _ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "ok", **payload}


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload))


def _signals_to_lists(signals: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: np.asarray(v, dtype=np.float64).ravel().tolist() for k, v in signals.items()}


# ---------------------------------------------------------------------------
# Standard Inference Session (Physiological Only)
# ---------------------------------------------------------------------------
class InferenceSession:
    """Per-connection state: pipeline + optional chunk buffer for streaming."""

    def __init__(self) -> None:
        self.pipeline = _new_session_pipeline()
        self.calibrated = False
        self.sequence = 0
        self._chunk_buffers: dict[str, list[float]] = {"bvp": [], "gsr": [], "skt": []}

    def reset_chunks(self) -> None:
        for key in self._chunk_buffers:
            self._chunk_buffers[key].clear()

    def append_chunk(self, signals: dict[str, Any]) -> None:
        for key in ("bvp", "gsr", "skt"):
            if key not in signals:
                raise ValueError(f"chunk missing '{key}'")
            self._chunk_buffers[key].extend(np.asarray(signals[key], dtype=np.float64).ravel().tolist())

    def buffered_signals(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float64) for k, v in self._chunk_buffers.items()}

    def calibrate(self, signals: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_signal_dict(signals)
        self.pipeline.calibrate(normalized)
        self.calibrated = True
        return _ok(
            {
                "event": "calibrated",
                "samples_per_channel": len(normalized["bvp"]),
            }
        )

    def predict(self, signals: dict[str, Any]) -> dict[str, Any]:
        if not self.calibrated:
            raise RuntimeError("Send a 'calibrate' message before 'predict'.")
        result = run_inference(self.pipeline, signals)
        self.sequence += 1
        payload = result.to_dict()
        payload["event"] = "prediction"
        payload["sequence"] = self.sequence
        return _ok(payload)

    def predict_chunk(self, signals: dict[str, Any], min_samples: int, flush: bool) -> dict[str, Any] | None:
        self.append_chunk(signals)
        buffered = self.buffered_signals()
        n = len(buffered["bvp"])
        
        if n < min_samples and not flush:
            return _ok(
                {
                    "event": "buffering",
                    "buffered_samples": n,
                    "required_samples": min_samples,
                }
            )
        if n < MIN_SIGNAL_SAMPLES:
            raise ValueError(
                f"Buffered {n} samples but need at least {MIN_SIGNAL_SAMPLES} to predict."
            )
            
        if n > min_samples:
            window = {k: np.copy(v[-min_samples:]) for k, v in buffered.items()}
        else:
            window = {k: np.copy(v) for k, v in buffered.items()}
            
        self.reset_chunks()
        return self.predict(window)


# ---------------------------------------------------------------------------
# Integrated Session (Physiological + Music Generation)
# ---------------------------------------------------------------------------
class MoodMusicSession(InferenceSession):
    def __init__(self, ws: WebSocket):
        super().__init__()
        self.ws = ws
        self.active = True
        self.client_queue_size = 0
        self.task = None
        
        # Start with a neutral or default mood until the first prediction
        self.target_mood_id = 8  # Unconditional mood id fallback

    async def start_generator(self):
        self.task = asyncio.create_task(self.generation_loop())

    def stop(self):
        self.active = False
        if self.task:
            self.task.cancel()

    def predict(self, signals: dict[str, Any]) -> dict[str, Any]:
        result = super().predict(signals)
        
        # Intercept the prediction to update the current target mood
        if result.get("type") == "ok":
            mood_name = result.get("mood", {}).get("name")
            if mood_name and mood_name in Config.MOOD_TO_ID:
                self.target_mood_id = Config.MOOD_TO_ID[mood_name]
                
        return result

    async def generation_loop(self):
        """Background task generating MIDI tokens autonomously based on the current mood."""
        device = Config.DEVICE
        num_branches = Config.NUM_MOODS + 1
        
        if Config.USE_KV_CACHE:
            from src.models.cached_transformer import KVCache
            generator_cache = KVCache.from_model(_chrollo, batch_size=num_branches)
            classifier_cache = KVCache.from_model(_mood_classifier)
        else:
            generator_cache = None
            classifier_cache = None

        current_tokens = torch.tensor([[1]], device=device)
        current_moods = torch.tensor([[self.target_mood_id]], device=device)
        
        # Send Start token
        await _send_json(self.ws, _ok({"event": "music_token", "token": 1, "mood_id": self.target_mood_id}))

        while self.active:
            # Backpressure logic: Don't generate if client's audio buffer is full
            if self.client_queue_size > 1:
                await asyncio.sleep(0.1)
                continue

            try:
                # Run the model iteration thread-safely to not block the FastAPI loop
                current_tokens, current_moods, next_token = await asyncio.to_thread(
                    _chrollo_handler.generate_single_step,
                    current_tokens, 
                    current_moods, 
                    self.target_mood_id,
                    generator_cache=generator_cache,
                    classifier_cache=classifier_cache
                )

                token_val = next_token.item()
                await _send_json(self.ws, _ok({
                    "event": "music_token", 
                    "token": token_val,
                    "mood_id": self.target_mood_id
                }))
                
                # Tiny yield to let incoming data chunks process
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background generation loop encountered an error: {e}")
                break


# ---------------------------------------------------------------------------
# Direct Music Session (Manual Mood — no physiological inference)
# ---------------------------------------------------------------------------
class DirectMusicSession:
    """Generates MIDI tokens for a user-supplied mood with no physio inference."""

    def __init__(self, ws: WebSocket, initial_mood: str = "happy") -> None:
        self.ws = ws
        self.active = True
        self.client_queue_size = 0
        self.task = None
        mood_id = Config.MOOD_TO_ID.get(initial_mood, 4)  # default: happy
        self.target_mood_id = mood_id

    def set_mood(self, mood_name: str) -> dict[str, Any]:
        if mood_name not in Config.MOOD_TO_ID:
            raise ValueError(f"Unknown mood '{mood_name}'. Valid moods: {Config.MOODS}")
        self.target_mood_id = Config.MOOD_TO_ID[mood_name]
        return _ok({"event": "mood_set", "mood": mood_name, "mood_id": self.target_mood_id})

    async def start_generator(self):
        self.task = asyncio.create_task(self.generation_loop())

    def stop(self):
        self.active = False
        if self.task:
            self.task.cancel()

    async def generation_loop(self):
        """Background task generating MIDI tokens for the current manually-set mood."""
        device = Config.DEVICE
        num_branches = Config.NUM_MOODS + 1

        if Config.USE_KV_CACHE:
            from src.models.cached_transformer import KVCache
            generator_cache = KVCache.from_model(_chrollo, batch_size=num_branches)
            classifier_cache = KVCache.from_model(_mood_classifier)
        else:
            generator_cache = None
            classifier_cache = None

        current_tokens = torch.tensor([[1]], device=device)
        current_moods = torch.tensor([[self.target_mood_id]], device=device)

        # Send start token
        await _send_json(self.ws, _ok({
            "event": "music_token",
            "token": 1,
            "mood_id": self.target_mood_id,
        }))

        while self.active:
            if self.client_queue_size > 1:
                await asyncio.sleep(0.1)
                continue
            try:
                current_tokens, current_moods, next_token = await asyncio.to_thread(
                    _chrollo_handler.generate_single_step,
                    current_tokens,
                    current_moods,
                    self.target_mood_id,
                    generator_cache=generator_cache,
                    classifier_cache=classifier_cache,
                )
                token_val = next_token.item()
                await _send_json(self.ws, _ok({
                    "event": "music_token",
                    "token": token_val,
                    "mood_id": self.target_mood_id,
                }))
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DirectMusicSession generation error: {e}")
                break


# ---------------------------------------------------------------------------
# Direct Music Route  (/ws/direct_music)
# ---------------------------------------------------------------------------
@app.websocket("/ws/direct_music")
async def direct_music_ws(ws: WebSocket) -> None:
    """
    Manual mood → music endpoint.  No physiological inference.

    Client messages:
        { "type": "set_mood",      "mood": "<mood_name>" }
        { "type": "queue_status",  "qsize": <int> }
        { "type": "ping" }

    Server messages:
        { "type": "ok", "event": "connected",  "moods": [...] }
        { "type": "ok", "event": "mood_set",   "mood": "...", "mood_id": int }
        { "type": "ok", "event": "music_token","token": int, "mood_id": int }
        { "type": "ok", "event": "pong" }
        { "type": "error", "message": "..." }
    """
    await ws.accept()

    # Parse optional ?mood=<name> query parameter for initial mood
    initial_mood = ws.query_params.get("mood", "happy")
    if initial_mood not in Config.MOOD_TO_ID:
        initial_mood = "happy"

    session = DirectMusicSession(ws, initial_mood=initial_mood)
    await session.start_generator()

    await _send_json(ws, _ok({
        "event": "connected",
        "message": f"Direct Music endpoint ready. Current mood: {initial_mood}",
        "mood": initial_mood,
        "mood_id": session.target_mood_id,
        "moods": Config.MOODS,
    }))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                await _send_json(ws, _error_message(exc))
                continue

            msg_type = msg.get("type")

            if msg_type == "set_mood":
                try:
                    response = session.set_mood(msg.get("mood", ""))
                    await _send_json(ws, response)
                except ValueError as exc:
                    await _send_json(ws, _error_message(exc))

            elif msg_type == "queue_status":
                session.client_queue_size = msg.get("qsize", 0)

            elif msg_type == "ping":
                await _send_json(ws, _ok({"event": "pong"}))

    except WebSocketDisconnect:
        logger.info("Client disconnected from direct_music")
    finally:
        session.stop()


# ---------------------------------------------------------------------------
# Original Inference Route
# ---------------------------------------------------------------------------
@app.websocket("/ws/inference")
async def inference_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = InferenceSession()
    await _send_json(ws, _ok({"event": "connected"}))
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")
            
            if msg_type == "calibrate":
                await _send_json(ws, session.calibrate(msg.get("signals", msg)))
            elif msg_type == "predict_chunk":
                min_samples = int(msg.get("min_samples", RECOMMENDED_SIGNAL_SAMPLES))
                flush = bool(msg.get("flush", False))
                resp = session.predict_chunk(msg.get("signals", msg), min_samples=min_samples, flush=flush)
                if resp:
                    await _send_json(ws, resp)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Integrated Music + Inference Route
# ---------------------------------------------------------------------------
@app.websocket("/ws/mood_music")
async def mood_music_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = MoodMusicSession(ws)
    await session.start_generator()

    await _send_json(ws, _ok({
        "event": "connected", 
        "message": "Mood+Music Endpoint connected. Send 'calibrate' to begin."
    }))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                await _send_json(ws, _error_message(exc))
                continue

            msg_type = msg.get("type")

            # Client updates us on how many tokens it currently has buffered
            if msg_type == "queue_status":
                session.client_queue_size = msg.get("qsize", 0)
                continue
            
            # Standard Physiological Pipeline
            try:
                if msg_type == "calibrate":
                    response = session.calibrate(msg.get("signals", msg))
                    await _send_json(ws, response)

                elif msg_type == "predict_chunk":
                    signals = msg.get("signals", msg)
                    min_samples = int(msg.get("min_samples", RECOMMENDED_SIGNAL_SAMPLES))
                    flush = bool(msg.get("flush", False))
                    
                    response = session.predict_chunk(signals, min_samples=min_samples, flush=flush)
                    if response is not None:
                        await _send_json(ws, response)
                
                elif msg_type == "ping":
                    await _send_json(ws, _ok({"event": "pong"}))

            except Exception as exc:
                logger.exception("Inference error in mood_music endpoint")
                await _send_json(ws, _error_message(exc))

    except WebSocketDisconnect:
        logger.info("Client disconnected from mood_music")
    finally:
        session.stop()