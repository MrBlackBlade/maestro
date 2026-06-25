#!/usr/bin/env python
"""
Stream actual dataset files into the MAESTRO combined Inference+Music WebSocket.

Loads real physiological logs (BVP/GSR/SKT), runs a calibration sequence, and 
streams chunks sequentially. As the pipeline updates your predicted mood in real-time,
the server streams generated MIDI tokens which are played via the terminal audio engine
and saved to a file upon completion.

Prerequisites:
    pip install pandas openpyxl websockets
    uvicorn src.core.inference_ws_server:app --host 127.0.0.1 --port 8000

Usage:
    python scripts/stream_real_dataset_music.py --file path/to/your_data.csv --output my_song.mid
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import websockets
except ImportError as exc:
    raise SystemExit("Missing dependency 'websockets'. Install with: pip install websockets") from exc

from src.core.config import Config
from src.core.simulate_inference import MIN_SIGNAL_SAMPLES, RECOMMENDED_SIGNAL_SAMPLES
from src.core.audio_engine import AudioEngine
from src.core.utils import get_tokenizer, save_midi

FS = int(Config.FS_PHYSIO if hasattr(Config, 'FS_PHYSIO') else 1000) # Ensure sync with config


def load_dataset_file(file_path: Path) -> dict[str, np.ndarray]:
    print(f"[LOAD ] Reading dataset from {file_path.name}...")
    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
        
    df.columns = [col.lower().strip() for col in df.columns]
    
    required = ["bvp", "gsr", "skt"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing crucial physiological column mapping: '{col}'")
            
    return {
        "bvp": df["bvp"].to_numpy(dtype=np.float64),
        "gsr": df["gsr"].to_numpy(dtype=np.float64),
        "skt": df["skt"].to_numpy(dtype=np.float64),
    }


def _fmt_prediction(msg: dict) -> str:
    mood = msg.get("mood", {})
    return (
        f"[PRED #{msg.get('sequence', '?'):>3}] "
        f"valence={msg.get('valence', 0):6.3f}  "
        f"arousal={msg.get('arousal', 0):6.3f}  "
        f"mood={mood.get('name', '?'):<12}"
    )


async def _audio_queue_reporter(ws, audio_engine: AudioEngine):
    """Continuously reports the local audio buffer size to the server for backpressure."""
    try:
        while True:
            await asyncio.sleep(0.1)
            await ws.send(json.dumps({
                "type": "queue_status", 
                "qsize": audio_engine.audio_queue.qsize()
            }))
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Queue reporter error: {e}")


async def _ws_receiver(ws, audio_engine: AudioEngine, predictions: list, generated_tokens: list):
    """Listens continuously for both Prediction logs and Generated Music tokens."""
    try:
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            event = msg.get("event")
            
            if event == "music_token":
                token = msg["token"]
                audio_engine.push_token(token)
                generated_tokens.append(token)
            elif event == "prediction":
                predictions.append(msg)
                print(f"  {_fmt_prediction(msg)}")
            elif event == "calibrated":
                print("  [CALIB] Baseline calibration accepted!")
            elif event == "connected":
                print(f"  [CONN ] {msg.get('message')}")
            elif msg.get("type") == "error":
                print(f"  ERROR: {msg.get('message')}")

    except websockets.exceptions.ConnectionClosed:
        print("\n[WS] Connection closed.")
    except asyncio.CancelledError:
        pass


async def stream_real_data(
    url: str,
    file_path: Path,
    chunk_size: int,
    chunk_interval: float,
    output_path: str,
) -> None:
    
    signals = load_dataset_file(file_path)
    total_samples = len(signals["bvp"])
    
    if total_samples < RECOMMENDED_SIGNAL_SAMPLES:
        raise ValueError(f"Dataset needs minimum {RECOMMENDED_SIGNAL_SAMPLES} samples.")

    baseline = {
        "bvp": signals["bvp"][:RECOMMENDED_SIGNAL_SAMPLES],
        "gsr": signals["gsr"][:RECOMMENDED_SIGNAL_SAMPLES],
        "skt": signals["skt"][:RECOMMENDED_SIGNAL_SAMPLES],
    }
    
    stream_bvp = signals["bvp"][RECOMMENDED_SIGNAL_SAMPLES:]
    stream_gsr = signals["gsr"][RECOMMENDED_SIGNAL_SAMPLES:]
    stream_skt = signals["skt"][RECOMMENDED_SIGNAL_SAMPLES:]
    stream_samples = len(stream_bvp)
    
    n_chunks = (stream_samples + chunk_size - 1) // chunk_size

    print("=" * 72)
    print(" MAESTRO Affective Music Synthesizer")
    print("=" * 72)
    print(f"  Source File   : {file_path.name}")
    print(f"  WebSocket     : {url}")
    print(f"  Chunk Vector  : {chunk_size} samples")
    print("=" * 72)

    predictions: list[dict] = []
    generated_tokens: list[int] = []
    
    # Init Audio Engine
    audio_engine = AudioEngine()

    try:
        async with websockets.connect(url, max_size=50 * 1024 * 1024) as ws:
            # 1. Start concurrent listener & queue reporting loops
            recv_task = asyncio.create_task(_ws_receiver(ws, audio_engine, predictions, generated_tokens))
            reporter_task = asyncio.create_task(_audio_queue_reporter(ws, audio_engine))

            # 2. Setup Calibration
            print("\n[1/2] Submitting Calibration Block...")
            await ws.send(json.dumps({
                "type": "calibrate",
                "signals": {k: v.tolist() for k, v in baseline.items()},
            }))
            await asyncio.sleep(1.0) # Grace period for calibration processing

            # 3. Stream Main Data
            print(f"\n[2/2] Streaming {n_chunks} Real Data Chunks & Generating Music...\n")
            t0 = time.perf_counter()

            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, stream_samples)

                chunk = {
                    "bvp": stream_bvp[start:end].tolist(),
                    "gsr": stream_gsr[start:end].tolist(),
                    "skt": stream_skt[start:end].tolist(),
                }

                await ws.send(json.dumps({
                    "type": "predict_chunk",
                    "signals": chunk,
                    "min_samples": RECOMMENDED_SIGNAL_SAMPLES,
                    "flush": False,
                }))

                if chunk_idx < n_chunks - 1 and chunk_interval > 0:
                    await asyncio.sleep(chunk_interval)

            print("\n[FLUSH] Processing remaining pipeline buffers...")
            await ws.send(json.dumps({
                "type": "predict_chunk",
                "signals": {"bvp": [], "gsr": [], "skt": []},
                "min_samples": RECOMMENDED_SIGNAL_SAMPLES,
                "flush": True,
            }))
            
            # Allow final tokens to stream in
            await asyncio.sleep(2.0)
            
            # Clean up Tasks
            recv_task.cancel()
            reporter_task.cancel()
            
    finally:
        # Wrap up MIDI playback identically to chrollo.py
        print("\nStopping Audio Engine...")
        audio_engine.push_token(4, stop=True)
        audio_engine.playback_done.wait()

        if generated_tokens:
            try:
                tokenizer = get_tokenizer()
                save_midi(generated_tokens, tokenizer, output_path)
                print(f"Successfully saved {len(generated_tokens)} tokens to {output_path}")
            except Exception as e:
                print(f"Failed to save MIDI file: {e}")
        
    print("=" * 72)
    print(f" Session Complete — Captured {len(predictions)} inference events.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream physiological data & play generated affect music.")
    parser.add_argument("--file", type=Path, default=REPO_ROOT /"datasets"/ "sub_1.csv")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/mood_music")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--fast", action="store_true")
    # Added argument for saving the generated output
    parser.add_argument("--output", type=str, default="generated_affect_music.mid", help="Path to save the generated MIDI")
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(f"Target dataset not found: {args.file}")

    interval = 0.0 if args.fast else (args.interval if args.interval is not None else args.chunk_size / FS)

    try:
        asyncio.run(
            stream_real_data(
                url=args.url,
                file_path=args.file,
                chunk_size=args.chunk_size,
                chunk_interval=interval,
                output_path=args.output,
            )
        )
    except KeyboardInterrupt:
        print("\nStream canceled by operator.")

if __name__ == "__main__":
    main()