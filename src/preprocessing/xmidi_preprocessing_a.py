import os
import sys
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from symusic import Score
import numpy as np
import pandas as pd

from src.core.config import Config
from src.core.utils import get_tokenizer, save_mappings

def parse_xmidi_filename(filename: str) -> tuple[str, str, str] | None:
    """
    Parse  XMIDI_<mood>_<genre>_<ID>.midi  and return (mood, genre, id).
    Returns None if the filename does not match the pattern.
    """
    stem = Path(filename).stem          # strip .midi
    parts = stem.split("_")
    # Expected: ["XMIDI", mood, genre, ID]
    if len(parts) < 4 or parts[0] != "XMIDI":
        return None
    mood = parts[1].lower()
    genre = parts[2].lower()
    file_id = "_".join(parts[3:])       # ID may itself contain underscores
    return mood, genre, file_id


def train_tokenizer():
    data_dir = Config.XMIDI_DATASET_DIR
    print(f"Scanning XMIDI dataset at: {data_dir}")

    if not data_dir.exists():
        print(f"ERROR: Dataset directory does not exist: {data_dir}")
        return

    midi_files = sorted(data_dir.glob("*.midi"))
    print(f"Found {len(midi_files)} .midi files")

    # ---- Parse filenames ----
    rows = []
    skipped = 0
    for fp in tqdm(midi_files, desc="Parsing filenames"):
        parsed = parse_xmidi_filename(fp.name)
        if parsed is None:
            skipped += 1
            continue
        mood, genre, fid = parsed
        rows.append({
            "filename": fp.name,
            "mood": mood,
            "genre": genre,
            "file_id": fid,
        })

    if skipped:
        print(f"Skipped {skipped} files that don't match XMIDI naming convention")

    df = pd.DataFrame(rows)

    # ---- Map labels to integer IDs (using the canonical order from Config) ----
    df["mood_id"] = df["mood"].map(Config.MOOD_TO_ID)
    df["genre_id"] = df["genre"].map(Config.GENRE_TO_ID)

    # Check for any unmapped labels
    unmapped_moods = df[df["mood_id"].isna()]["mood"].unique()
    unmapped_genres = df[df["genre_id"].isna()]["genre"].unique()
    if len(unmapped_moods):
        print(f"WARNING: Unknown moods found (not in Config.MOODS): {unmapped_moods}")
    if len(unmapped_genres):
        print(f"WARNING: Unknown genres found (not in Config.GENRES): {unmapped_genres}")

    # Drop rows with unknown labels
    df = df.dropna(subset=["mood_id", "genre_id"]).reset_index(drop=True)
    df["mood_id"] = df["mood_id"].astype(int)
    df["genre_id"] = df["genre_id"].astype(int)

    # ---- Save CSV ----
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(Config.METADATA_CSV, index=False)
    print(f"\nMetadata saved to {Config.METADATA_CSV}")
    print(f"  Total samples : {len(df)}")
    print(f"  Moods          : {dict(df['mood'].value_counts())}")
    print(f"  Genres         : {dict(df['genre'].value_counts())}")

    # ---- Save label mappings (for the Streamlit app) ----
    mappings_path = Config.DATA_DIR / "label_mappings.json"
    save_mappings(Config.MOOD_TO_ID, Config.GENRE_TO_ID, mappings_path)
    print(f"  Label mappings : {mappings_path}")

    # ---- Train & save tokenizer ----
    print("\nTraining REMI tokenizer on a subset of MIDI files...")
    tokenizer = get_tokenizer()

    # Train on up to 1000 files for a representative vocabulary
    sample_paths = [data_dir / row["filename"] for _, row in df.head(1000).iterrows()]
    tokenizer.train(vocab_size=1000, files_paths=sample_paths)
    tokenizer.save(Config.TOKENIZER_PARAMS_PATH)
    print(f"  Tokenizer saved to {Config.TOKENIZER_PARAMS_PATH}")
    print(f"  Vocabulary size: {len(tokenizer)}")

    print("\nDone! Next step: python 2_train_generator.py")


# ---------------------------------------------------------------------------
#  Single-file tokenization (called by worker processes)
# ---------------------------------------------------------------------------

# Module-level tokenizer cache — each worker process initialises its own copy
_worker_tokenizer = None

def _init_worker():
    """Called once per worker process to create a private tokenizer instance
    and silence C-level stdout/stderr (symusic debug prints)."""
    global _worker_tokenizer
    # Redirect C-level file descriptors to devnull so C++ debug prints
    # from symusic don't pollute the parent terminal.
    # This is safe because each process has its own fd table.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)  # silence stdout
    os.dup2(devnull_fd, 2)  # silence stderr
    os.close(devnull_fd)
    _worker_tokenizer = get_tokenizer()


def _tokenize_one(args: tuple) -> tuple[str, bool, str | None]:
    """
    Tokenize a single MIDI file and save the result as a .npy file.

    Accepts a tuple (midi_path_str, output_path_str) so it can be used
    with ProcessPoolExecutor (all args must be picklable).

    Returns (filename, success, error_message).
    """
    midi_path_str, output_path_str = args
    midi_path = Path(midi_path_str)
    output_path = Path(output_path_str)

    try:
        score = Score(str(midi_path))
        tok_result = _worker_tokenizer.encode(score)

        if isinstance(tok_result, list):
            token_ids = tok_result[0].ids
        else:
            token_ids = tok_result.ids

        np.save(output_path, np.array(token_ids, dtype=np.int32))
        return midi_path.name, True, None

    except Exception as e:
        return midi_path.name, False, str(e)


# ---------------------------------------------------------------------------
#  Main parallel preprocessing entry-point
# ---------------------------------------------------------------------------

def preprocess_all_tokens():
    """Tokenize all MIDI files in parallel and save as .npy files."""
    print("=" * 60)
    print("PRE-PROCESSING: Tokenizing all MIDI files (multi-process)")
    print("=" * 60)

    # Load metadata
    if not Config.METADATA_CSV.exists():
        print(f"ERROR: Metadata CSV not found: {Config.METADATA_CSV}")
        print("Run 'python 1_create_metadata.py' first!")
        return

    df = pd.read_csv(Config.METADATA_CSV)
    print(f"Found {len(df)} MIDI files to process")

    # Load tokenizer (main process — just to report vocab size)
    tokenizer = get_tokenizer()
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Create output directory
    tokenized_dir = Config.DATA_DIR / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {tokenized_dir}")

    # --- Build work list (skip already-processed) ---
    work_args: list[tuple[str, str]] = []
    already_done = 0

    for _, row in df.iterrows():
        midi_path = Config.XMIDI_DATASET_DIR / row["filename"]
        output_path = tokenized_dir / row["filename"].replace(".midi", ".npy")
        if output_path.exists():
            already_done += 1
        else:
            work_args.append((str(midi_path), str(output_path)))

    total = len(df)
    to_process = len(work_args)
    print(f"Already processed: {already_done}/{total}")
    print(f"To tokenize now:   {to_process}/{total}")

    if to_process == 0:
        print("Nothing to do – all files already tokenized.")
        return

    # --- Parallel tokenization (multi-process) ---
    num_workers = Config.TOKENIZE_NUM_WORKERS
    print(f"Using {num_workers} worker processes", flush=True)

    success_count = already_done
    fail_count = 0
    failed_files: list[tuple[str, str]] = []

    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
        ) as pool:
            futures = {
                pool.submit(_tokenize_one, args): args[0]
                for args in work_args
            }

            with tqdm(total=to_process, desc="Tokenizing") as pbar:
                for future in as_completed(futures):
                    filename, ok, err = future.result()
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                        failed_files.append((filename, err or "unknown error"))
                        if fail_count <= 10:
                            tqdm.write(f"  FAILED: {filename}: {err}")
                    pbar.update(1)
    except Exception:
        traceback.print_exc()
        return

    # --- Report ---
    print("\n" + "=" * 60)
    print("Pre-processing complete!")
    print(f"  Success: {success_count}/{total}")
    print(f"  Failed:  {fail_count}/{total}")
    if failed_files:
        fail_log = tokenized_dir / "failed_files.txt"
        print(f"\nFailed files saved to: {fail_log}")
        with open(fail_log, "w") as f:
            for filename, error in failed_files:
                f.write(f"{filename}: {error}\n")


if __name__ == "__main__":
    preprocess_all_tokens()