import pretty_midi as pm
import numpy as np
import os 
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from src.core.config import Config
from src.core.utils import get_tokenizer, save_midi
from src.models.mood_generator import MoodModelGenerator, MoodModelGeneratorHandler
from src.models.neg_cfg_generator import NegCFGGenerator, NegCFGGeneratorHandler
from src.models.cached_transformer import KVCache

MODEL_REGISTRY = {
    MoodModelGeneratorHandler.MODEL_NAME: (MoodModelGenerator, MoodModelGeneratorHandler),
    NegCFGGeneratorHandler.MODEL_NAME: (NegCFGGenerator, NegCFGGeneratorHandler),
}

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

def generate_and_evaluate_model(model_name: str, epoch: int | None = None, length: int = 1024, mood: str = "magnificent"):
    device = Config.DEVICE
    print(f"Loading {model_name} on {device}...")
    
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    ModelClass, HandlerClass = MODEL_REGISTRY[model_name]
    model = ModelClass(vocab_size=vocab_size).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    handler = HandlerClass(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion
    )
    handler.load_checkpoint(epoch=epoch)
    model.eval()
    
    target_mood_id = Config.MOOD_TO_ID[mood]
    current_tokens = torch.tensor([[1]], device=device)
    current_moods = torch.tensor([[target_mood_id]], device=device)
    
    if model_name == MoodModelGeneratorHandler.MODEL_NAME:
        if Config.USE_KV_CACHE:
            cond_cache = KVCache.from_model(model)
            uncond_cache = KVCache.from_model(model)
        else:
            cond_cache = uncond_cache = None
            
        for step in tqdm(range(length), desc=f"Generating {length} tokens from {model_name}"):
            current_tokens, current_moods, next_token = handler.generate_single_step(
                current_tokens, current_moods, target_mood_id,
                cond_cache=cond_cache, uncond_cache=uncond_cache,
            )
            
    elif model_name == NegCFGGeneratorHandler.MODEL_NAME:
        num_branches = Config.NUM_MOODS + 1
        if Config.USE_KV_CACHE:
            cache = KVCache.from_model(model, batch_size=num_branches)
        else:
            cache = None
            
        for step in tqdm(range(length), desc=f"Generating {length} tokens from {model_name}"):
            current_tokens, current_moods, next_token = handler.generate_single_step(
                current_tokens, current_moods, target_mood_id,
                cache=cache,
            )
            
    generated_tokens = current_tokens.squeeze(0).cpu().tolist()
    temp_midi = Config.MIDI_DIR / f"temp_{model_name}_pch.mid"
    save_midi(generated_tokens, tokenizer, str(temp_midi))
    print(f"Saved generated sample to {temp_midi}")
    
    return compute_pitch_class_histogram(str(temp_midi))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Pitch Class Histogram (PCH)")
    parser.add_argument("--model-name", type=str, choices=list(MODEL_REGISTRY.keys()),
                        help="Generate a new file using this model, then compute PCH.")
    parser.add_argument("--epoch", type=int, default=None,
                      help="Checkpoint epoch to load (default: best)")
    parser.add_argument("--length", type=int, default=1024,
                      help="Number of tokens to generate if using --model-name")
    parser.add_argument("--mood", type=str, default="magnificent", choices=Config.MOODS,
                      help="Mood to condition on when generating")
    args = parser.parse_args()

    if args.model_name:
        Config.MIDI_DIR.mkdir(parents=True, exist_ok=True)
        print(f"--- PITCH CLASS HISTOGRAM for model {args.model_name} ---")
        hist = generate_and_evaluate_model(args.model_name, args.epoch, args.length, args.mood)
    else:
        # Fall back to interactive mode if no model provided
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
        hist = compute_pitch_class_histogram(str(Config.MIDI_DIR / midi_files[int(selected_file) - 1]))

    print(f"The Pitch Class Histogram is: ")
    for i, val in enumerate(hist):
        print(f"\t{i}: {val:.4f}")
    if np.sum(hist) > 0:
        print(f"The most common pitch class is: {np.argmax(hist)}")
    else:
        print("No pitch classes detected (histogram is all zeros).")
