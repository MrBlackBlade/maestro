"""
FastAPI WebSocket server for continuous MAESTRO affect inference.

Run:
    uvicorn src.core.inference_ws_server:app --host 0.0.0.0 --port 8000 --reload

WebSocket endpoint:
    ws://localhost:8000/ws/inference
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
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

logger = logging.getLogger(__name__)

_models: LoadedModels | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _models
    logger.info("Loading LSTM models for WebSocket inference...")
    _models = load_lstm_models()
    logger.info("Models ready on %s", _models.device)
    yield
    _models = None


app = FastAPI(
    title="MAESTRO Inference WebSocket",
    description="Stream physiological signals and receive continuous valence/arousal predictions.",
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
            
        # FIX: Explicitly slice and copy into a fresh array context 
        # so .clear() inside reset_chunks() doesn't wipe out the data being passed!
        if n > min_samples:
            window = {k: np.copy(v[-min_samples:]) for k, v in buffered.items()}
        else:
            window = {k: np.copy(v) for k, v in buffered.items()}
            
        # Now it is safe to empty the session accumulators
        self.reset_chunks()
        return self.predict(window)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "models_loaded": _models is not None,
            "device": str(_models.device) if _models else None,
            "websocket_path": "/ws/inference",
            "recommended_samples": RECOMMENDED_SIGNAL_SAMPLES,
            "min_samples": MIN_SIGNAL_SAMPLES,
        }
    )


@app.get("/schema")
def schema() -> JSONResponse:
    return JSONResponse(
        {
            "websocket_url": "ws://<host>:<port>/ws/inference",
            "client_to_server": {
                "ping": {"type": "ping"},
                "calibrate": {
                    "type": "calibrate",
                    "signals": {"bvp": "[float, ...]", "gsr": "[float, ...]", "skt": "[float, ...]"},
                    "note": f"Each array should have >= {MIN_SIGNAL_SAMPLES} samples (recommended {RECOMMENDED_SIGNAL_SAMPLES} at 1 kHz).",
                },
                "predict": {
                    "type": "predict",
                    "signals": {"bvp": "[float, ...]", "gsr": "[float, ...]", "skt": "[float, ...]"},
                },
                "predict_chunk": {
                    "type": "predict_chunk",
                    "signals": {"bvp": "[float, ...]", "gsr": "[float, ...]", "skt": "[float, ...]"},
                    "min_samples": RECOMMENDED_SIGNAL_SAMPLES,
                    "flush": "optional bool — predict even if buffer < min_samples",
                },
                "reset": {"type": "reset"},
                "use_dummy": {
                    "type": "use_dummy",
                    "source": "optional 'h5' | 'synthetic' (default: h5 if file exists else synthetic)",
                    "n_samples": RECOMMENDED_SIGNAL_SAMPLES,
                    "pred_idx": 10,
                },
            },
            "server_to_client": {
                "ok": "type=ok plus event-specific fields",
                "error": {"type": "error", "message": "human-readable reason"},
                "prediction_fields": [
                    "sequence",
                    "valence",
                    "arousal",
                    "valence_norm",
                    "arousal_norm",
                    "mood",
                    "music_params",
                ],
            },
        }
    )


@app.websocket("/ws/inference")
async def inference_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = InferenceSession()
    await _send_json(
        ws,
        _ok(
            {
                "event": "connected",
                "message": "Send 'calibrate' then 'predict' or 'predict_chunk'.",
                "min_samples": MIN_SIGNAL_SAMPLES,
                "recommended_samples": RECOMMENDED_SIGNAL_SAMPLES,
            }
        ),
    )

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                await _send_json(ws, _error_message(exc))
                continue

            msg_type = msg.get("type")
            try:
                if msg_type == "ping":
                    await _send_json(ws, _ok({"event": "pong"}))

                elif msg_type == "reset":
                    session = InferenceSession()
                    await _send_json(ws, _ok({"event": "reset"}))

                elif msg_type == "use_dummy":
                    source = msg.get("source")
                    n_samples = int(msg.get("n_samples", RECOMMENDED_SIGNAL_SAMPLES))
                    if source == "synthetic" or (
                        source is None and not DEFAULT_DATASET_PATH.exists()
                    ):
                        baseline = generate_dummy_signals(n_samples=n_samples, seed=42)
                        window = generate_dummy_signals(n_samples=n_samples, seed=99)
                    else:
                        pred_idx = int(msg.get("pred_idx", 10))
                        baseline, window, meta = load_dataset_windows(pred_idx=pred_idx)
                        await _send_json(
                            ws,
                            _ok({"event": "dummy_data", "source": "h5", "meta": meta}),
                        )
                        await _send_json(ws, _ok({"event": "dummy_baseline", "signals": _signals_to_lists(baseline)}))
                        await _send_json(ws, _ok({"event": "dummy_window", "signals": _signals_to_lists(window)}))
                        continue

                    await _send_json(ws, _ok({"event": "dummy_data", "source": "synthetic", "n_samples": n_samples}))
                    await _send_json(ws, _ok({"event": "dummy_baseline", "signals": _signals_to_lists(baseline)}))
                    await _send_json(ws, _ok({"event": "dummy_window", "signals": _signals_to_lists(window)}))

                elif msg_type == "calibrate":
                    signals = msg.get("signals", msg)
                    response = session.calibrate(signals)
                    await _send_json(ws, response)

                elif msg_type == "predict":
                    signals = msg.get("signals", msg)
                    response = session.predict(signals)
                    await _send_json(ws, response)

                elif msg_type == "predict_chunk":
                    signals = msg.get("signals", msg)
                    min_samples = int(msg.get("min_samples", RECOMMENDED_SIGNAL_SAMPLES))
                    flush = bool(msg.get("flush", False))
                    response = session.predict_chunk(signals, min_samples=min_samples, flush=flush)
                    if response is not None:
                        await _send_json(ws, response)

                else:
                    await _send_json(
                        ws,
                        _error_message(
                            ValueError(
                                f"Unknown type '{msg_type}'. "
                                "Use ping | calibrate | predict | predict_chunk | reset | use_dummy."
                            )
                        ),
                    )

            except Exception as exc:
                logger.exception("WebSocket handler error")
                await _send_json(ws, _error_message(exc))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
