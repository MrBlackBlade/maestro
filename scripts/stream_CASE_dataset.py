#!/usr/bin/env python
"""
Stream actual dataset files into the MAESTRO inference WebSocket.

Loads real physiological logs (BVP/GSR/SKT), runs a calibration sequence 
using the initial samples, and streams chunks sequentially to verify model behavior.

Prerequisites:
    pip install pandas openpyxl websockets
    uvicorn src.core.inference_ws_server:app --host 127.0.0.1 --port 8000

Usage:
    python scripts/stream_real_dataset.py --file path/to/your_data.xlsx
    python scripts/stream_real_dataset.py --file path/to/your_data.csv --fast
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import websockets
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'websockets'. Install with: pip install -r env/requirements-ws.txt"
    ) from exc

from src.core.config import Config
from src.core.simulate_inference import (
    DEFAULT_CFG,
    MIN_SIGNAL_SAMPLES,
    RECOMMENDED_SIGNAL_SAMPLES,
)

FS = int(DEFAULT_CFG["fs_physio"])


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
            
    # RETURN RAW DATA ARRAYS WITHOUT ALTERING AMPLITUDES
    return {
        "bvp": df["bvp"].to_numpy(dtype=np.float64),
        "gsr": df["gsr"].to_numpy(dtype=np.float64),
        "skt": df["skt"].to_numpy(dtype=np.float64),
    }


def _fmt_prediction(msg: dict) -> str:
    mood = msg.get("mood", {})
    music = msg.get("music_params", {})
    return (
        f"[PRED #{msg.get('sequence', '?'):>3}] "
        f"valence={msg.get('valences', msg.get('valence', 0)):6.3f}  "
        f"arousal={msg.get('arousal', 0):6.3f}  "
        f"norm V={msg.get('valence_norm', 0):+.3f} A={msg.get('arousal_norm', 0):+.3f}  "
        f"mood={mood.get('name', '?'):<12}  "
        f"tempo={music.get('tempo_bpm', '?')} bpm"
    )


def _fmt_buffering(msg: dict) -> str:
    buf = msg.get("buffered_samples", 0)
    req = msg.get("required_samples", RECOMMENDED_SIGNAL_SAMPLES)
    pct = 100.0 * buf / req if req else 0.0
    bar_len = 30
    filled = int(bar_len * buf / req) if req else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    return f"[BUF  {buf:5d}/{req}  {pct:5.1f}%] |{bar}|"


async def _recv_and_print(ws) -> dict | None:
    raw = await ws.recv()
    msg = json.loads(raw)
    if msg.get("type") == "error":
        print(f"  ERROR: {msg.get('message')}")
        raise RuntimeError(msg.get("message", "server error"))

    event = msg.get("event")
    if event == "buffering":
        print(f"  {_fmt_buffering(msg)}")
    elif event == "prediction":
        print(f"  {_fmt_prediction(msg)}")
    elif event == "calibrated":
        print(f"  [CALIB] Baseline calibration accepted ({msg.get('samples_per_channel')} samples/channel)")
    elif event == "connected":
        print(f"  [CONN ] {msg.get('message')}")
    return msg


async def stream_real_data(
    url: str,
    file_path: Path,
    chunk_size: int,
    chunk_interval: float,
) -> None:
    # 1. Load data arrays
    signals = load_dataset_file(file_path)
    total_samples = len(signals["bvp"])
    
    if total_samples < RECOMMENDED_SIGNAL_SAMPLES:
        raise ValueError(f"Dataset file only has {total_samples} samples. Requires minimum {RECOMMENDED_SIGNAL_SAMPLES} for calibration.")

    # 2. Slice Calibration Window (Take the initial window block as baseline setup)
    baseline = {
        "bvp": signals["bvp"][:RECOMMENDED_SIGNAL_SAMPLES],
        "gsr": signals["gsr"][:RECOMMENDED_SIGNAL_SAMPLES],
        "skt": signals["skt"][:RECOMMENDED_SIGNAL_SAMPLES],
    }
    
    # 3. Stream Segment (Everything after calibration window)
    stream_bvp = signals["bvp"][RECOMMENDED_SIGNAL_SAMPLES:]
    stream_gsr = signals["gsr"][RECOMMENDED_SIGNAL_SAMPLES:]
    stream_skt = signals["skt"][RECOMMENDED_SIGNAL_SAMPLES:]
    stream_samples = len(stream_bvp)
    
    n_chunks = (stream_samples + chunk_size - 1) // chunk_size
    expected_predictions = max(0, stream_samples // RECOMMENDED_SIGNAL_SAMPLES)

    print("=" * 72)
    print(" MAESTRO Real Subject Data Streamer Engine")
    print("=" * 72)
    print(f"  Source File   : {file_path.name}")
    print(f"  WebSocket Target: {url}")
    print(f"  Sample Rate   : {FS} Hz")
    print(f"  Chunk Vector  : {chunk_size} samples ({chunk_size / FS:.2f} s)")
    print(f"  Calib Window  : {RECOMMENDED_SIGNAL_SAMPLES} samples ({RECOMMENDED_SIGNAL_SAMPLES / FS:.1f} s)")
    print(f"  Stream Buffer : {stream_samples} samples ({stream_samples / FS:.1f} s)")
    print(f"  Expect Preds  : ~{expected_predictions}")
    print("=" * 72)

    predictions: list[dict] = []
    t0 = time.perf_counter()

    async with websockets.connect(url, max_size=50 * 1024 * 1024) as ws:
        print("\n[1/3] Establishing Server Link...")
        await _recv_and_print(ws)

        print("\n[2/3] Submitting Real Baseline Block for Calibration...")
        await ws.send(
            json.dumps(
                {
                    "type": "calibrate",
                    "signals": {k: v.tolist() for k, v in baseline.items()},
                }
            )
        )
        cal = await _recv_and_print(ws)
        if cal is None or cal.get("event") != "calibrated":
            raise RuntimeError("Pipeline Calibration rejected.")

        print(f"\n[3/3] Streaming {n_chunks} Real Data Chunks...")
        print("-" * 72)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, stream_samples)
            elapsed = time.perf_counter() - t0

            chunk = {
                "bvp": stream_bvp[start:end].tolist(),
                "gsr": stream_gsr[start:end].tolist(),
                "skt": stream_skt[start:end].tolist(),
            }

            print(f"\n>> Chunk {chunk_idx + 1:>4}/{n_chunks} | Samples [{start:6d}:{end:6d}] | t={elapsed:6.2f}s")

            await ws.send(
                json.dumps(
                    {
                        "type": "predict_chunk",
                        "signals": chunk,
                        "min_samples": RECOMMENDED_SIGNAL_SAMPLES,
                        "flush": False,
                    }
                )
            )

            msg = await _recv_and_print(ws)
            if msg and msg.get("event") == "prediction":
                predictions.append(msg)

            if chunk_idx < n_chunks - 1 and chunk_interval > 0:
                await asyncio.sleep(chunk_interval)

        # Flush trailing samples
        remainder = stream_samples % RECOMMENDED_SIGNAL_SAMPLES
        if remainder >= MIN_SIGNAL_SAMPLES:
            print(f"\n>> [FLUSH] Processing remaining trailing pipeline buffers...")
            await ws.send(
                json.dumps(
                    {
                        "type": "predict_chunk",
                        "signals": {"bvp": [], "gsr": [], "skt": []},
                        "min_samples": RECOMMENDED_SIGNAL_SAMPLES,
                        "flush": True,
                    }
                )
            )
            msg = await _recv_and_print(ws)
            if msg and msg.get("event") == "prediction":
                predictions.append(msg)

    print("\n" + "=" * 72)
    print(f" Run Complete — Captured {len(predictions)} inference events.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream actual patient files over inference sockets.")
    
    parser.add_argument(
        "--file", 
        type=Path, 
        default=REPO_ROOT /"datasets"/ "sub_1.csv", 
        help="Path to raw XLSX or CSV file (default: sub_1.csv in repository root)."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/inference")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Samples sent per loop ticks.")
    parser.add_argument("--interval", type=float, default=None, help="Force override delay intervals.")
    parser.add_argument("--fast", action="store_true", help="Blast packets ignoring time delays.")
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(
            f"Target data filepath does not exist: {args.file}\n"
            f"Please verify 'sub_1.csv' is placed correctly in your project base folder."
        )

    interval = 0.0 if args.fast else (args.interval if args.interval is not None else args.chunk_size / FS)

    try:
        asyncio.run(
            stream_real_data(
                url=args.url,
                file_path=args.file,
                chunk_size=args.chunk_size,
                chunk_interval=interval,
            )
        )
    except KeyboardInterrupt:
        print("\nStream canceled by operator.")


if __name__ == "__main__":
    main()