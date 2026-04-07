import argparse
import math
import torch
import torch.nn as nn
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import get_tokenizer
from src.dataloaders.mood_dataset_cached import get_mood_cached_dataloader
from src.models.neg_cfg_generator import NegCFGGenerator, NegCFGGeneratorHandler
from src.models.mood_generator import MoodModelGenerator, MoodModelGeneratorHandler

MODEL_REGISTRY = {
    MoodModelGeneratorHandler.MODEL_NAME: (MoodModelGenerator, MoodModelGeneratorHandler),
    NegCFGGeneratorHandler.MODEL_NAME: (NegCFGGenerator, NegCFGGeneratorHandler),
}

def evaluate_perplexity(args):
    device = Config.DEVICE
    print(f"Device: {device}")
    
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    print(f"Evaluating for model: {args.model_name} with sample factor: {args.sample_factor}")

    # Load model
    if args.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{args.model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    
    ModelClass, HandlerClass = MODEL_REGISTRY[args.model_name]
    model = ModelClass(vocab_size=vocab_size).to(device)
    
    # We create dummy optimizer/scheduler/criterion to initialize handler 
    # since we only need the handler to load checkpoints
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    # Using sum so we can compute exact per-token loss averaging
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    
    handler = HandlerClass(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion
    )
    
    # Load checkpoint
    handler.load_checkpoint(epoch=args.epoch)
    model.eval()

    # Load data
    dataloader = get_mood_cached_dataloader(
        batch_size=args.batch_size,
        shuffle=False, # Deterministic validation
        num_workers=Config.NUM_WORKERS,
        persistent_workers=False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        sample_factor=args.sample_factor
    )
    
    print(f"Evaluating on {len(dataloader.dataset)} samples ({len(dataloader)} batches).")
    
    total_nll = 0.0
    total_valid_tokens = 0
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating Perplexity"):
            tokens, moods, true_mood = batch
            
            tokens = tokens.to(device)
            moods = moods.to(device)
            
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]
            
            # Forward pass
            logits = model(inp, moods)
            
            # Compute NLL for the batch
            # Target is the next tokens
            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt.reshape(-1)
            )
            
            total_nll += loss.item()
            total_valid_tokens += (tgt != 0).sum().item()
            
    if total_valid_tokens == 0:
        print("No valid tokens found in dataset.")
        return

    avg_nll = total_nll / total_valid_tokens
    perplexity = math.exp(avg_nll)
    
    print()
    print("=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Valid Tokens: {total_valid_tokens:,}")
    print(f"Average NLL:        {avg_nll:.4f}")
    print(f"Perplexity:         {perplexity:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Perplexity of Models")
    parser.add_argument("--model-name", type=str, required=True, choices=list(MODEL_REGISTRY.keys()),
                        help="The identification name of the model to evaluate (e.g. generator_2, generator_3)")
    parser.add_argument("--epoch", type=int, default=None,
                      help="Checkpoint epoch to load (default: best)")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE,
                      help="Batch size for evaluation")
    parser.add_argument("--sample-factor", type=float, default=1.0,
                      help="Fraction of dataset to use for evaluation (0.0 to 1.0)")
                      
    args = parser.parse_args()
    evaluate_perplexity(args)
