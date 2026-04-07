import pretty_midi as pm
import numpy as np
import os 
from src.core.config import Config

def compute_pitch_class_histogram(midi_path: str) -> np.ndarray:
    """
    Compute the pitch class histogram for a MIDI file.
    
    Args:
        midi_path: Path to the MIDI file
        
    Returns:
        A numpy array of shape (12,) representing the histogram
    """
    # Load the MIDI file
    midi = pm.PrettyMIDI(midi_path)
    
    # Get the pitch class histogram
    histogram = midi.get_pitch_class_histogram()
    
    return histogram

if __name__ == "__main__":
  files = os.listdir(Config.MIDI_DIR)
  midi_files = []

  for file in files:
    if file.endswith(".mid") or file.endswith(".midi"):
      midi_files.append(file)

  print("--- PITCH CLASS HISTOGRAM ---")
  print("Available MIDI files:")
  for i, midi_file in enumerate(midi_files):
    print(f"\t{i+1}: {midi_file}")
  
  selected_file = input("Enter the number of the desired MIDI file: ")
  hist = compute_pitch_class_histogram(Config.MIDI_DIR / midi_files[int(selected_file) - 1])
  print(f"The Pitch Class Histogram for the file is: ")
  for i, val in enumerate(hist):
    print(f"\t{i}: {val:.4f}")
  print(f"The most common pitch class is: {np.argmax(hist)}")
