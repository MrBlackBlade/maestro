import pretty_midi as pm
import numpy as np
import os
import argparse
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from src.core.config import Config
from src.core.utils import get_tokenizer, save_midi
from src.models.mood_generator import MoodModelGenerator, MoodModelGeneratorHandler
from src.models.neg_cfg_generator import NegCFGGenerator, NegCFGGeneratorHandler
from src.models.mood_classifier import MoodClassifier, MoodClassifierHandler
from src.models.chrollo import Chrollo, ChrolloHandler
from src.models.cached_transformer import KVCache

MODEL_REGISTRY = {
    MoodModelGeneratorHandler.MODEL_NAME: (MoodModelGenerator, MoodModelGeneratorHandler),
    NegCFGGeneratorHandler.MODEL_NAME: (NegCFGGenerator, NegCFGGeneratorHandler),
    ChrolloHandler.MODEL_NAME: (Chrollo, ChrolloHandler),
}

def compute_harmonic_consistency(midi_path: str, window_size_sec: float = 2.0, fs: int = 10) -> float:
    """
    Compute harmonic consistency using Windowed Pitch Class Profile (PCP) Entropy.
    Lower entropy means clearer, more consistent chords/scales.
    
    Args:
        midi_path: Path to the MIDI file
        window_size_sec: The size of the sliding window in seconds
        fs: Sampling frequency for the chroma feature
        
    Returns:
        The average Shannon Entropy across all valid windows.
    """
    try:
        midi = pm.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {e}")
        return float('nan')
        
    # Get chroma matrix: 12 x M 
    # (12 pitch classes, M time steps where M = duration * fs)
    chroma = midi.get_chroma(fs=fs)
    
    if chroma.shape[1] == 0:
        print("MIDI has no tonal content.")
        return float('nan')
        
    window_frames = int(window_size_sec * fs)
    num_frames = chroma.shape[1]
    
    if num_frames < window_frames:
        # If the file is shorter than one window, just use the whole file
        window_frames = num_frames
        
    entropies = []
    
    # Non-overlapping windows (or could be 50% overlap if preferred)
    step_size = window_frames
    
    for start in range(0, num_frames, step_size):
        end = min(start + window_frames, num_frames)
        
        # Extract the window
        window = chroma[:, start:end]
        
        # Sum the energy for each pitch class within the window
        pcp = np.sum(window, axis=1)
        total_energy = np.sum(pcp)
        
        if total_energy == 0:
            continue # Silence, skip
            
        # Normalize to create a probability distribution
        p = pcp / total_energy
        
        # Calculate Shannon entropy: - sum(p * log2(p)) for p > 0
        p_nz = p[p > 0]
        entropy = -np.sum(p_nz * np.log2(p_nz))
        entropies.append(entropy)
        
    if not entropies:
        print("No valid windows with content found.")
        return float('nan')
        
    avg_entropy = np.mean(entropies)
    return float(avg_entropy)

def generate_and_evaluate_model(
    model_name: str, 
    epoch: int | None = None, 
    length: int = 2048, 
    mood: str = "magnificent",
    transition_mood: str | None = None,
    transition_step: int = 1024
):
    device = Config.DEVICE
    print(f"Loading {model_name} on {device}...")
    
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    ModelClass, HandlerClass = MODEL_REGISTRY[model_name]
    model = ModelClass(vocab_size=vocab_size).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    if ModelClass == Chrollo:
        # For Chrollo, we also need to initialize a mood classifier and its handler
        mood_classifier = MoodClassifier(vocab_size=vocab_size).to(device)
        mood_classifier_optimizer = torch.optim.AdamW(
            mood_classifier.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
        )
        mood_classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            mood_classifier_optimizer,
            T_max=1,
            eta_min=1e-6,
        )
        mood_classifier_criterion = nn.CrossEntropyLoss()
        mood_classifier_handler = MoodClassifierHandler(
            model=mood_classifier, 
            optimizer=mood_classifier_optimizer, 
            scheduler=mood_classifier_scheduler, 
            criterion=mood_classifier_criterion
        )
        
        handler = HandlerClass(
            model=model, optimizer=optimizer, scheduler=scheduler, 
            criterion=criterion, classifier_handler=mood_classifier_handler
        )
    else:
        handler = HandlerClass(
            model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion
        )
    handler.load_checkpoint(epoch=epoch)
    model.eval()
    
    target_mood_id = Config.MOOD_TO_ID[mood]
    if transition_mood:
        transition_mood_id = Config.MOOD_TO_ID[transition_mood]
    else:
        transition_mood_id = None
        
    current_tokens = torch.tensor([[1]], device=device)
    current_moods = torch.tensor([[target_mood_id]], device=device)
    
    if model_name == MoodModelGeneratorHandler.MODEL_NAME:
        if Config.USE_KV_CACHE:
            cond_cache = KVCache.from_model(model)
            uncond_cache = KVCache.from_model(model)
        else:
            cond_cache = uncond_cache = None
            
        for step in tqdm(range(length), desc=f"Generating {length} tokens from {model_name}"):
            if transition_mood_id is not None and step == transition_step:
                target_mood_id = transition_mood_id
                print(f"\n[Step {step}] Transitioning mood to {transition_mood}!")
                
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
            if transition_mood_id is not None and step == transition_step:
                target_mood_id = transition_mood_id
                print(f"\n[Step {step}] Transitioning mood to {transition_mood}!")
                
            current_tokens, current_moods, next_token = handler.generate_single_step(
                current_tokens, current_moods, target_mood_id,
                cache=cache,
            )
    
    elif model_name == ChrolloHandler.MODEL_NAME:
        if Config.USE_KV_CACHE:
            cond_cache = KVCache.from_model(model)
            uncond_cache = KVCache.from_model(model)
        else:
            cond_cache = uncond_cache = None
            
        for step in tqdm(range(length), desc=f"Generating {length} tokens from {model_name}"):
            if transition_mood_id is not None and step == transition_step:
                target_mood_id = transition_mood_id
                print(f"\n[Step {step}] Transitioning mood to {transition_mood}!")
                
            current_tokens, current_moods, next_token = handler.generate_single_step(
                current_tokens, current_moods, target_mood_id,
                cond_cache=cond_cache, uncond_cache=uncond_cache,
            )   
            
    generated_tokens = current_tokens.squeeze(0).cpu().tolist()
    temp_midi = Config.MIDI_DIR / f"temp_{model_name}_hc.mid"
    save_midi(generated_tokens, tokenizer, str(temp_midi))
    print(f"Saved generated sample to {temp_midi}")
    
    return compute_harmonic_consistency(str(temp_midi))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Harmonic Consistency (Windowed PCP Entropy)")
    parser.add_argument("--model-name", type=str, choices=list(MODEL_REGISTRY.keys()),
                        help="Generate a new file using this model, then evaluate.")
    parser.add_argument("--epoch", type=int, default=None,
                      help="Checkpoint epoch to load (default: best)")
    parser.add_argument("--length", type=int, default=2048,
                      help="Number of tokens to generate if using --model-name")
    parser.add_argument("--mood", type=str, default="magnificent", choices=Config.MOODS,
                      help="Starting mood when generating")
    parser.add_argument("--transition-mood", type=str, default=None, choices=Config.MOODS,
                      help="Mood to transition to mid-generation. If not provided, no transition occurs.")
    parser.add_argument("--transition-step", type=int, default=1024,
                      help="Token step to initiate the mood transition (if transition-mood is set).")
    args = parser.parse_args()

    if args.model_name:
        Config.MIDI_DIR.mkdir(parents=True, exist_ok=True)
        print(f"--- HARMONIC CONSISTENCY EVALUATION for model {args.model_name} ---")
        entropy = generate_and_evaluate_model(
            args.model_name, args.epoch, args.length, args.mood,
            transition_mood=args.transition_mood, transition_step=args.transition_step
        )
    else:
        # Fall back to interactive mode if no model provided
        files = os.listdir(Config.MIDI_DIR)
        midi_files = [f for f in files if f.endswith(".mid") or f.endswith(".midi")]

        if not midi_files:
            print(f"No MIDI files found in {Config.MIDI_DIR}.")
            exit(0)

        print("--- HARMONIC CONSISTENCY EVALUATION ---")
        print("Available MIDI files:")
        for i, midi_file in enumerate(midi_files):
            print(f"\t{i+1}: {midi_file}")
      
        try:
            selected_file = input("Enter the number of the desired MIDI file: ")
            file_idx = int(selected_file) - 1
            if file_idx < 0 or file_idx >= len(midi_files):
                raise ValueError("Selection out of bounds.")
        except Exception as e:
            print("Invalid selection.")
            exit(1)
            
        file_path = str(Config.MIDI_DIR / midi_files[file_idx])
        print(f"Evaluating {file_path}...")
        entropy = compute_harmonic_consistency(file_path)

    print("\n" + "="*45)
    print("HARMONIC CONSISTENCY RESULTS")
    print("="*45)
    print(f"Average PCP Entropy: {entropy:.4f} bits")
    print(f"(Scale: 0.0 is perfect consistency, ~3.58 is complete randomness)")
    print("="*45 + "\n")
