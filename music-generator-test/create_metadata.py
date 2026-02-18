"""
Step 1 – Create metadata.csv from XMIDI filenames.

XMIDI files follow the naming convention:
    XMIDI_<mood>_<genre>_<ID>.midi

This script scans the dataset directory, parses mood and genre from each
filename, assigns integer IDs, and saves the result to data/metadata.csv.

It also trains / saves the REMI tokenizer so that all downstream scripts
share the exact same vocabulary.

Usage
-----
    cd music-generator-test
    python 1_create_metadata.py
"""
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.utils import get_tokenizer, save_mappings


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


def main():
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


if __name__ == "__main__":
    main()

