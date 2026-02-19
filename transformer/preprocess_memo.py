"""
Preprocessing pipeline for the Memo2496 dataset.

Steps:
    1. Load Memo2496 annotations (valence, arousal)
    2. Transcribe audio files to MIDI using basic-pitch
    3. Tokenize each MIDI with the custom tokenizer
    4. Save (emotion, tokens) pairs to HDF5

Usage:
    python -m transformer.preprocess_memo [--skip-transcription]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    from .config import DEFAULT_CONFIG, PathConfig, TokenizerConfig
    from .tokenizer import MIDITokenizer
except ImportError:
    from config import DEFAULT_CONFIG, PathConfig, MaestroConfig, TokenizerConfig
    from tokenizer import MIDITokenizer
    DEFAULT_CONFIG = MaestroConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(annotations_dir: Path) -> Dict[str, Tuple[float, float]]:
    """
    Load Memo2496 valence-arousal annotations.

    Handles the Memo2496 time-series format where valence and arousal are
    stored as per-second samples in separate CSV files, plus a song info
    CSV that maps numeric song_id -> UUID-based file_name.

    Falls back to generic CSV/JSON loading if the Memo2496-specific files
    are not found.

    Returns a dict mapping song_id (file stem) -> (valence, arousal).
    """
    annotations = {}

    # ── Memo2496-specific time-series format ──────────────────────────────
    valence_csv = annotations_dir / "valence_all_average.csv"
    arousal_csv = annotations_dir / "arousal_all_average.csv"
    songs_info_csv = annotations_dir / "songs_info_all.csv"

    if valence_csv.exists() and arousal_csv.exists() and songs_info_csv.exists():
        print("  Detected Memo2496 time-series annotation format")

        # 1. Build mapping: numeric song_id -> UUID file stem
        id_to_stem: Dict[str, str] = {}
        with open(songs_info_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                numeric_id = row.get("song_id", "").strip()
                file_name = row.get("file_name", "").strip()
                if numeric_id and file_name:
                    id_to_stem[numeric_id] = Path(file_name).stem
        print(f"    Loaded {len(id_to_stem)} song ID -> file name mappings")

        # 2. Read time-series valence (mean across all time‐points per song)
        valence_means: Dict[str, float] = {}
        print(f"  Loading valence from: {valence_csv.name}")
        with open(valence_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                numeric_id = row.get("song_id", "").strip()
                if not numeric_id:
                    continue
                # Collect all sample_*ms values, skipping empty cells
                samples = []
                for key, value in row.items():
                    if key and key.strip().startswith("sample_"):
                        value = value.strip() if value else ""
                        if value:
                            try:
                                samples.append(float(value))
                            except ValueError:
                                pass
                if samples:
                    valence_means[numeric_id] = float(np.mean(samples))

        # 3. Read time-series arousal (mean across all time‐points per song)
        arousal_means: Dict[str, float] = {}
        print(f"  Loading arousal from: {arousal_csv.name}")
        with open(arousal_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                numeric_id = row.get("song_id", "").strip()
                if not numeric_id:
                    continue
                samples = []
                for key, value in row.items():
                    if key and key.strip().startswith("sample_"):
                        value = value.strip() if value else ""
                        if value:
                            try:
                                samples.append(float(value))
                            except ValueError:
                                pass
                if samples:
                    arousal_means[numeric_id] = float(np.mean(samples))

        # 4. Combine: map to file stems
        matched = 0
        for numeric_id in valence_means:
            if numeric_id in arousal_means and numeric_id in id_to_stem:
                stem = id_to_stem[numeric_id]
                annotations[stem] = (valence_means[numeric_id], arousal_means[numeric_id])
                matched += 1

        print(f"    Matched {matched} songs with both valence + arousal + file mapping")

        if annotations:
            return annotations
        else:
            print("    ⚠ No annotations matched — falling back to generic loader")

    # ── Generic CSV fallback ──────────────────────────────────────────────
    csv_files = list(annotations_dir.glob("*.csv"))
    if csv_files:
        for csv_file in csv_files:
            print(f"  Loading annotations from: {csv_file.name}")
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                headers = [h.strip().lower() for h in reader.fieldnames] if reader.fieldnames else []

                # Find valence/arousal column names (flexible matching)
                val_col = _find_column(headers, ["valence", "v", "val"])
                aro_col = _find_column(headers, ["arousal", "a", "aro"])
                id_col = _find_column(headers, ["id", "song_id", "track_id", "filename", "file", "name", "song"])

                if val_col is None or aro_col is None:
                    print(f"    ⚠ Could not find valence/arousal columns in {csv_file.name}")
                    print(f"    Available columns: {headers}")
                    continue

                for row in reader:
                    clean_row = {k.strip().lower(): v.strip() for k, v in row.items()}
                    song_id = clean_row.get(id_col, "")
                    valence = float(clean_row.get(val_col, 0))
                    arousal = float(clean_row.get(aro_col, 0))
                    if song_id:
                        # Remove file extension from song_id if present
                        song_id = Path(song_id).stem
                        annotations[song_id] = (valence, arousal)

        if annotations:
            return annotations

    # ── Generic JSON fallback ─────────────────────────────────────────────
    json_files = list(annotations_dir.glob("*.json"))
    for json_file in json_files:
        print(f"  Loading annotations from: {json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for entry in data:
                song_id = str(entry.get("id", entry.get("song_id", entry.get("filename", ""))))
                song_id = Path(song_id).stem
                valence = float(entry.get("valence", entry.get("v", 0)))
                arousal = float(entry.get("arousal", entry.get("a", 0)))
                if song_id:
                    annotations[song_id] = (valence, arousal)
        elif isinstance(data, dict):
            for song_id, vals in data.items():
                song_id = Path(song_id).stem
                if isinstance(vals, dict):
                    valence = float(vals.get("valence", vals.get("v", 0)))
                    arousal = float(vals.get("arousal", vals.get("a", 0)))
                elif isinstance(vals, (list, tuple)) and len(vals) >= 2:
                    valence, arousal = float(vals[0]), float(vals[1])
                else:
                    continue
                annotations[song_id] = (valence, arousal)

    if not annotations:
        # Last resort: list available files for debugging
        print("  ⚠ No standard annotation files found. Listing available files:")
        for f in annotations_dir.iterdir():
            print(f"    - {f.name} ({f.stat().st_size:,} bytes)")

    return annotations


def _find_column(headers: List[str], candidates: List[str]) -> Optional[str]:
    """Find first matching column name from a list of candidates."""
    for candidate in candidates:
        if candidate in headers:
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Audio → MIDI Transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio_to_midi(
    audio_dir: Path,
    output_dir: Path,
    song_ids: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Transcribe audio files to MIDI using basic-pitch.

    Args:
        audio_dir: Directory containing audio files.
        output_dir: Directory to save MIDI files.
        song_ids: Optional filter — only transcribe these song IDs.

    Returns:
        Dict mapping song_id → midi_path for successfully transcribed files.
    """
    try:
        from basic_pitch.inference import predict_and_save
        from basic_pitch import build_icassp_2022_model_path, FilenameSuffix
        # Force ONNX backend — TF is unreliable on Windows CUDA envs
        MODEL_PATH = build_icassp_2022_model_path(FilenameSuffix.onnx)
    except ImportError:
        print("ERROR: basic-pitch is not installed.")
        print("Install it with: pip install basic-pitch")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    if song_ids:
        song_id_set = set(song_ids)
        audio_files = [f for f in audio_files if f.stem in song_id_set]

    print(f"\n  Found {len(audio_files)} audio files to transcribe")

    transcribed = {}
    for i, audio_path in enumerate(sorted(audio_files)):
        midi_path = output_dir / f"{audio_path.stem}.mid"

        if midi_path.exists():
            print(f"  [{i+1}/{len(audio_files)}] Skipping (already exists): {audio_path.stem}")
            transcribed[audio_path.stem] = midi_path
            continue

        print(f"  [{i+1}/{len(audio_files)}] Transcribing: {audio_path.name} ...", end=" ", flush=True)

        try:
            predict_and_save(
                audio_path_list=[str(audio_path)],
                output_directory=str(output_dir),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=MODEL_PATH,
            )
            # basic-pitch saves with _basic_pitch suffix sometimes
            possible_names = [
                output_dir / f"{audio_path.stem}.mid",
                output_dir / f"{audio_path.stem}_basic_pitch.mid",
                output_dir / f"{audio_path.stem}_basic_pitch.midi",
            ]
            found = False
            for p in possible_names:
                if p.exists():
                    # Rename to consistent name
                    if p != midi_path:
                        p.rename(midi_path)
                    transcribed[audio_path.stem] = midi_path
                    found = True
                    break

            if found:
                print("✓")
            else:
                print("⚠ MIDI file not found after transcription")

        except Exception as e:
            print(f"✗ Error: {e}")

    return transcribed


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization & HDF5 Creation
# ─────────────────────────────────────────────────────────────────────────────

def build_processed_dataset(
    annotations: Dict[str, Tuple[float, float]],
    midi_dir: Path,
    output_path: Path,
    tokenizer: MIDITokenizer,
    max_seq_len: int = 2048,
) -> int:
    """
    Tokenize MIDI files and save (emotion, tokens) pairs to HDF5.

    Returns:
        Number of successfully processed samples.
    """
    print(f"\n  Tokenizing MIDI files and building HDF5...")

    emotions_list = []
    tokens_list = []
    lengths_list = []
    processed_ids = []

    midi_files = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.midi"))
    print(f"  Found {len(midi_files)} MIDI files")

    for i, midi_path in enumerate(midi_files):
        song_id = midi_path.stem

        # Check if we have annotations for this song
        if song_id not in annotations:
            continue

        valence, arousal = annotations[song_id]

        try:
            tokens = tokenizer.midi_to_tokens(str(midi_path))

            # Skip if too short (less than 10 meaningful tokens)
            if len(tokens) < 10:
                continue

            # Truncate if too long (keep BOS, truncate content, add EOS)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len - 1] + [tokenizer.eos_token_id]

            emotions_list.append([valence, arousal])
            tokens_list.append(np.array(tokens, dtype=np.int32))
            lengths_list.append(len(tokens))
            processed_ids.append(song_id)

            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{len(midi_files)} files, "
                      f"{len(processed_ids)} valid samples so far")

        except Exception as e:
            print(f"    ⚠ Error processing {song_id}: {e}")

    if not processed_ids:
        print("  ✗ No valid samples found!")
        return 0

    # Save to HDF5
    print(f"\n  Saving {len(processed_ids)} samples to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Emotions array
        f.create_dataset(
            "emotions",
            data=np.array(emotions_list, dtype=np.float32),
        )

        # Variable-length token sequences
        dt = h5py.vlen_dtype(np.int32)
        tokens_ds = f.create_dataset("tokens", shape=(len(tokens_list),), dtype=dt)
        for i, tok_seq in enumerate(tokens_list):
            tokens_ds[i] = tok_seq

        # Lengths
        f.create_dataset(
            "lengths",
            data=np.array(lengths_list, dtype=np.int32),
        )

        # Metadata
        f.attrs["num_samples"] = len(processed_ids)
        f.attrs["max_seq_len"] = max_seq_len
        f.attrs["vocab_size"] = tokenizer.vocab_size

    print(f"  ✓ Saved successfully!")
    print(f"    Samples: {len(processed_ids)}")
    print(f"    Avg token length: {np.mean(lengths_list):.0f}")
    print(f"    Max token length: {np.max(lengths_list)}")
    print(f"    Min token length: {np.min(lengths_list)}")

    return len(processed_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Memo2496 dataset for Emotion → MIDI Transformer"
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip audio-to-MIDI transcription (use existing MIDI files)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override path to Memo2496 dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output HDF5 file path",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum token sequence length (default: 2048)",
    )
    args = parser.parse_args()

    # Resolve paths
    paths = DEFAULT_CONFIG.paths
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        audio_dir = dataset_dir / "MusicRawData"
        annotations_dir = dataset_dir / "Annotations"
        midi_dir = dataset_dir / "transcribed_midi"
    else:
        dataset_dir = paths.dataset_dir
        audio_dir = paths.memo_audio_dir
        annotations_dir = paths.memo_annotations_dir
        midi_dir = paths.transcribed_midi_dir

    output_path = Path(args.output) if args.output else paths.processed_data_path

    print("=" * 60)
    print("Memo2496 Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Dataset dir:    {dataset_dir}")
    print(f"  Audio dir:      {audio_dir}")
    print(f"  Annotations:    {annotations_dir}")
    print(f"  MIDI output:    {midi_dir}")
    print(f"  HDF5 output:    {output_path}")
    print(f"  Max seq length: {args.max_seq_len}")

    # Step 1: Load annotations
    print("\n" + "─" * 60)
    print("Step 1: Loading annotations")
    print("─" * 60)

    if not annotations_dir.exists():
        print(f"  ✗ Annotations directory not found: {annotations_dir}")
        print(f"  Please download the Memo2496 dataset and place it at: {dataset_dir}")
        sys.exit(1)

    annotations = load_annotations(annotations_dir)
    print(f"  Loaded {len(annotations)} annotations")

    if not annotations:
        print("  ✗ No annotations loaded! Check the annotations directory.")
        sys.exit(1)

    # Show annotation statistics
    vals = [v for v, a in annotations.values()]
    aros = [a for v, a in annotations.values()]
    print(f"  Valence range: [{min(vals):.2f}, {max(vals):.2f}]")
    print(f"  Arousal range: [{min(aros):.2f}, {max(aros):.2f}]")

    # Step 2: Audio → MIDI transcription
    print("\n" + "─" * 60)
    print("Step 2: Audio → MIDI transcription")
    print("─" * 60)

    if args.skip_transcription:
        print("  Skipping transcription (--skip-transcription flag)")
        if not midi_dir.exists():
            print(f"  ✗ MIDI directory not found: {midi_dir}")
            sys.exit(1)
    else:
        if not audio_dir.exists():
            print(f"  ✗ Audio directory not found: {audio_dir}")
            sys.exit(1)

        transcribed = transcribe_audio_to_midi(
            audio_dir=audio_dir,
            output_dir=midi_dir,
            song_ids=list(annotations.keys()),
        )
        print(f"  Transcribed {len(transcribed)} audio files to MIDI")

    # Step 3: Tokenize and build HDF5
    print("\n" + "─" * 60)
    print("Step 3: Tokenize MIDI and build HDF5 dataset")
    print("─" * 60)

    tokenizer = MIDITokenizer()
    num_samples = build_processed_dataset(
        annotations=annotations,
        midi_dir=midi_dir,
        output_path=output_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"  Total valid samples: {num_samples}")
    print(f"  Output file: {output_path}")
    print(f"\n  Next step: Train the model with:")
    print(f"    python -m transformer.train --data {output_path}")


if __name__ == "__main__":
    main()
