"""
Step 0 – Pre-process and cache all tokenized MIDI files.

This script tokenizes all MIDI files from the XMIDI dataset and saves them
as numpy arrays. This eliminates the disk I/O bottleneck during training.

Usage
-----
    cd music-generator-test
    python 0_preprocess_tokens.py

This should be run BEFORE training. It will create:
    data/tokenized/  – directory with .npy files (one per MIDI file)
"""
import sys
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.utils import get_tokenizer
from src.dataset import suppress_stdout_stderr
from symusic import Score


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
    tokenizer = get_tokenizer(Config.TOKENIZER_PARAMS_PATH)
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
    print("=" * 60)
    print("\nNext step: python 2_train_generator.py")


if __name__ == "__main__":
    preprocess_all_tokens()

