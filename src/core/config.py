"""
Configuration file for the Music Generation project.
All hyperparameters and paths are centralized here.
"""
import torch
from pathlib import Path


class Config:
    # ========================
    # Paths
    # ========================
    # Project-relative paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    DATASETS_DIR = PROJECT_ROOT / "datasets"
    XMIDI_DATASET_DIR = DATASETS_DIR / "XMIDI_Dataset"
    
    DATA_DIR = PROJECT_ROOT / "data"
    METADATA_CSV = DATA_DIR / "metadata.csv"
    TOKENIZER_PARAMS_PATH = DATA_DIR / "tokenizer.json"
    TOKENIZED_DIR = DATA_DIR / "tokenized"  # Pre-processed tokenized sequences

    RESOURCES_DIR = PROJECT_ROOT / "res"

    MIDI_DIR = PROJECT_ROOT / "generated_samples"

    # Model checkpoint directories
    MODEL_CKPT_DIR = PROJECT_ROOT / "models"
    GENERATOR_CKPT_DIR = PROJECT_ROOT / "models" / "generator"
    REFINER_CKPT_DIR = PROJECT_ROOT / "models" / "refiner"

    # ========================
    # Dataset Labels (from XMIDI filenames)
    # ========================
    # 11 moods detected in the dataset
    MOODS = [
        "angry", "exciting", "fear", "funny", "happy",
        "lazy", "magnificent", "quiet", "romantic", "sad", "warm",
    ]
    MOOD_TO_ID = {m: i for i, m in enumerate(MOODS)}
    NUM_MOODS = len(MOODS)

    # 6 genres detected in the dataset
    GENRES = ["classical", "country", "jazz", "pop", "rock", "traditional"]
    GENRE_TO_ID = {g: i for i, g in enumerate(GENRES)}
    NUM_GENRES = len(GENRES)

    # Train Choices
    TRAIN_CHOICES = ["generator", "classifier"]
    DEFAULT_TRAIN_CHOICE = "generator"

    # ========================
    # Tokenizer
    # ========================
    PITCH_RANGE = (21, 108)             # Standard piano range
    NUM_VELOCITIES = 8                  # How many volume levels to track
    USE_CHORDS = True
    USE_PROGRAMS = True                 # Single-track MIDI, no program tokens needed

    # ========================
    # Model Hyperparameters
    # ========================
    D_MODEL = 512                    # Transformer hidden dimension
    NUM_HEADS = 8                    # Multi-head attention heads
    NUM_LAYERS = 6                   # Transformer decoder layers
    DIM_FEEDFORWARD = 2048           # FFN inner dimension
    DROPOUT = 0.1
    MAX_SEQ_LEN = 1024               # Maximum positional embedding length

    # ========================
    # Training Hyperparameters
    # ========================
    SEQ_LEN = 1024                   # Training chunk length
    BATCH_SIZE = 32                   # Increased for better GPU utilization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Set epochs to 5 for a quick test run; increase to 20 or more for better results
    EPOCHS = 50
    GRAD_CLIP = 1.0                  # Gradient clipping max norm
    
    # Data loading optimization
    NUM_WORKERS = 4                   # Parallel data loading workers
    PREFETCH_FACTOR = 2               # Prefetch batches ahead
    PERSISTENT_WORKERS = True         # Keep workers alive between epochs

    # ========================
    # Refiner-specific
    # ========================
    REFINER_NUM_LAYERS = 4           # Shallower than generator
    NOISE_LEVEL = 0.20               # Fraction of tokens to corrupt during refiner training

    # ========================
    # Mood Classifier
    # ========================
    CLASSIFIER_NUM_LAYERS = 4        # Number of layers in the mood classifier
    CLASSIFIER_WARMUP = 64           # Skip loss for first N positions (too little context to classify mood)
    MOOD_LOSS_WEIGHT = 0.1           # Weight of mood classifier CE relative to token CE
    MOOD_CLASSIFIER_DETACH = False   # If True, stop classifier gradients from flowing into transformer
    NEG_CFG_SCALE = 1.0              # Base negative guidance scale for penalty moods
    POS_CFG_SCALE = 2.0              # Positive guidance scale for target mood

    # ========================
    # Generation / Inference
    # ========================
    TEMPERATURE = 0.9
    TOP_K = 50
    TOP_P = 0.95                     # Nucleus sampling threshold
    GENERATE_LENGTH = 256            # Default number of tokens to generate
    USE_KV_CACHE = True              # KV caching for faster autoregressive generation
    CLASSIFIER_WINDOW = 128          # Sliding window size fed to the decoupled classifier at inference
    CLASSIFIER_EMA_ALPHA = 0.05      # EMA smoothing factor for classifier mood_probs (lower = smoother)

    # Dynamic Temperature / Entropy
    # Entropy -> How chaotic is the model?
    ENTROPY_SIGNIFICANCE_THRESH = 0.1       # Low enough to consider value? (not a periodic peak)
    ENTROPY_LOW = 0.2                     # Is it considered low enough to increment temperature?
    ENTROPY_HIGH = 0.5                     # Is it considered high enough to decrement temperature?

    D_TEMP_UP = 0.0005              # temperature increment magnitude
    D_TEMP_DOWN = 0.001              # temperature decrement magnitude
    D_TEMP_MIN = 1.15
    D_TEMP_MAX = 0.15                # temperature maximum offset

    # ========================
    # Device
    # ========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"

    # ========================
    # Preprocessing
    # ========================
    TOKENIZE_NUM_WORKERS = 8                    # Number of worker processes for parallel tokenization