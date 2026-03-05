import sys
from pathlib import Path

from tqdm import tqdm
from symusic import Score
import numpy as np
import pandas as pd

# Ensure project root is importable
# sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.config import Config
from src.core.utils import get_tokenizer, save_mappings, suppress_stdout_stderr

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

def preprocess_all_tokens():
    """Tokenize all MIDI files and save as .npy files."""
    print("=" * 60)
    print("PRE-PROCESSING: Tokenizing all MIDI files")
    print("=" * 60)

    # Load metadata
    if not Config.METADATA_CSV.exists():
        print(f"ERROR: Metadata CSV not found: {Config.METADATA_CSV}")
        print("Run 'python 1_create_metadata.py' first!")
        return

    df = pd.read_csv(Config.METADATA_CSV)
    print(f"Found {len(df)} MIDI files to process")

    # Load tokenizer
    tokenizer = get_tokenizer()
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Create output directory
    tokenized_dir = Config.DATA_DIR / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {tokenized_dir}")

    # Process each file
    success_count = 0
    fail_count = 0
    failed_files = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        midi_path = Config.XMIDI_DATASET_DIR / row["filename"]
        output_path = tokenized_dir / f"{row['filename'].replace('.midi', '.npy')}"

        # Skip if already processed
        if output_path.exists():
            success_count += 1
            continue

        try:
            # Tokenize (suppress debug output)
            with suppress_stdout_stderr():
                score = Score(str(midi_path))
                tok_result = tokenizer.encode(score)

            if isinstance(tok_result, list):
                token_ids = tok_result[0].ids
            else:
                token_ids = tok_result.ids

            # Save as numpy array (much faster to load than re-tokenizing)
            np.save(output_path, np.array(token_ids, dtype=np.int32))
            success_count += 1

        except Exception as e:
            fail_count += 1
            failed_files.append((row["filename"], str(e)))
            if fail_count <= 10:  # Print first 10 errors
                print(f"\nFailed to tokenize {row['filename']}: {e}")

    print("\n" + "=" * 60)
    print(f"Pre-processing complete!")
    print(f"  Success: {success_count}/{len(df)}")
    print(f"  Failed:  {fail_count}/{len(df)}")
    if failed_files:
        print(f"\nFailed files saved to: {tokenized_dir / 'failed_files.txt'}")
        with open(tokenized_dir / "failed_files.txt", "w") as f:
            for filename, error in failed_files:
                f.write(f"{filename}: {error}\n")

if __name__ == "__main__":
    preprocess_all_tokens()