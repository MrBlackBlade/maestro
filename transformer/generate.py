"""
MIDI generation from emotion input.

Loads a trained EmotionMusicTransformer checkpoint and generates
MIDI files conditioned on valence/arousal values.

Usage:
    python -m transformer.generate --valence 0.8 --arousal 0.6 --output happy.mid
    python -m transformer.generate --valence -0.5 --arousal -0.3 --output sad.mid
    python -m transformer.generate --preset happy --output happy.mid
"""

import argparse
from pathlib import Path
from typing import Optional

import torch

try:
    from .config import DEFAULT_CONFIG, MaestroConfig, ModelConfig
    from .model import EmotionMusicTransformer
    from .tokenizer import MIDITokenizer
except ImportError:
    from config import DEFAULT_CONFIG, MaestroConfig, ModelConfig
    from model import EmotionMusicTransformer
    from tokenizer import MIDITokenizer
    DEFAULT_CONFIG = MaestroConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Emotion presets (Russell's circumplex model)
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_PRESETS = {
    # High valence, high arousal
    "happy":       (0.8, 0.7),
    "excited":     (0.6, 0.9),
    "joyful":      (0.9, 0.6),
    "energetic":   (0.5, 0.95),

    # High valence, low arousal
    "calm":        (0.6, -0.5),
    "relaxed":     (0.7, -0.6),
    "peaceful":    (0.8, -0.7),
    "serene":      (0.9, -0.8),

    # Low valence, high arousal
    "angry":       (-0.7, 0.8),
    "tense":       (-0.5, 0.7),
    "anxious":     (-0.6, 0.6),
    "fearful":     (-0.8, 0.9),

    # Low valence, low arousal
    "sad":         (-0.7, -0.5),
    "depressed":   (-0.8, -0.7),
    "melancholic": (-0.6, -0.4),
    "gloomy":      (-0.5, -0.6),

    # Neutral
    "neutral":     (0.0, 0.0),
}


class MusicGenerator:
    """
    Generates MIDI music conditioned on emotion.

    Example:
        gen = MusicGenerator("checkpoints/best_model.pt")
        gen.generate(valence=0.8, arousal=0.6, output_path="happy.mid")
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        config: Optional[MaestroConfig] = None,
    ):
        self.cfg = config or DEFAULT_CONFIG
        self.device = torch.device(device or self.cfg.device)

        # Initialize tokenizer
        self.tokenizer = MIDITokenizer(self.cfg.tokenizer)

        # Load model from checkpoint
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        print(f"MusicGenerator ready on {self.device}")
        print(f"  Model parameters: {self.model.count_parameters():,}")

    def _load_model(self, checkpoint_path: str) -> EmotionMusicTransformer:
        """Load model from checkpoint."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Reconstruct model config from checkpoint
        saved_cfg = checkpoint.get("model_config", {})
        model_cfg = ModelConfig(
            d_model=saved_cfg.get("d_model", self.cfg.model.d_model),
            nhead=saved_cfg.get("nhead", self.cfg.model.nhead),
            num_layers=saved_cfg.get("num_layers", self.cfg.model.num_layers),
            dim_feedforward=saved_cfg.get("dim_feedforward", self.cfg.model.dim_feedforward),
            dropout=saved_cfg.get("dropout", self.cfg.model.dropout),
            max_seq_len=saved_cfg.get("max_seq_len", self.cfg.model.max_seq_len),
            emotion_dim=saved_cfg.get("emotion_dim", self.cfg.model.emotion_dim),
        )

        vocab_size = checkpoint.get("vocab_size", self.tokenizer.vocab_size)
        model = EmotionMusicTransformer(
            vocab_size=vocab_size,
            model_cfg=model_cfg,
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', '?')}")

        return model

    def generate(
        self,
        valence: float = 0.0,
        arousal: float = 0.0,
        output_path: str = "generated.mid",
        max_len: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        tempo: float = 120.0,
    ) -> Path:
        """
        Generate a MIDI file from emotion input.

        Args:
            valence: Valence value (typically -1 to 1).
            arousal: Arousal value (typically -1 to 1).
            output_path: Path to save the generated MIDI file.
            max_len: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling threshold.
            tempo: MIDI tempo in BPM.

        Returns:
            Path to the generated MIDI file.
        """
        gen_cfg = self.cfg.generate
        max_len = max_len or gen_cfg.max_gen_len
        temperature = temperature if temperature is not None else gen_cfg.temperature
        top_k = top_k if top_k is not None else gen_cfg.top_k
        top_p = top_p if top_p is not None else gen_cfg.top_p

        print(f"\nGenerating music...")
        print(f"  Emotion:     valence={valence:.2f}, arousal={arousal:.2f}")
        print(f"  Temperature: {temperature}")
        print(f"  Top-k:       {top_k}")
        print(f"  Top-p:       {top_p}")
        print(f"  Max tokens:  {max_len}")

        # Create emotion tensor
        emotion = torch.tensor([[valence, arousal]], dtype=torch.float32)

        # Generate tokens
        with torch.no_grad():
            generated = self.model.generate(
                emotion=emotion,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                bos_token=self.tokenizer.bos_token_id,
                eos_token=self.tokenizer.eos_token_id,
            )

        tokens = generated[0].cpu().tolist()
        print(f"  Generated {len(tokens)} tokens")

        # Decode to MIDI and save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer.save_midi(tokens, str(output_path), tempo=tempo)

        # Get some stats about the generated MIDI
        midi = self.tokenizer.tokens_to_midi(tokens, tempo)
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        num_instruments = len(midi.instruments)
        duration = midi.get_end_time()

        print(f"  ✓ MIDI saved to: {output_path}")
        print(f"    Notes:       {total_notes}")
        print(f"    Instruments: {num_instruments}")
        print(f"    Duration:    {duration:.1f}s")

        return output_path

    def generate_preset(
        self,
        preset: str,
        output_path: str = None,
        **kwargs,
    ) -> Path:
        """Generate using an emotion preset name."""
        if preset not in EMOTION_PRESETS:
            available = ", ".join(sorted(EMOTION_PRESETS.keys()))
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        valence, arousal = EMOTION_PRESETS[preset]

        if output_path is None:
            output_dir = self.cfg.paths.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{preset}.mid")

        print(f"Using preset '{preset}': valence={valence}, arousal={arousal}")
        return self.generate(
            valence=valence,
            arousal=arousal,
            output_path=output_path,
            **kwargs,
        )

    def generate_spectrum(
        self,
        output_dir: str = None,
        num_per_quadrant: int = 2,
        **kwargs,
    ) -> list:
        """
        Generate multiple MIDI files across the emotion spectrum.
        Useful for evaluating whether the model produces different music
        for different emotions.
        """
        if output_dir is None:
            output_dir = self.cfg.paths.output_dir / "spectrum"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        quadrants = {
            "positive_high": [(0.5, 0.5), (0.8, 0.8)],
            "positive_low":  [(0.5, -0.5), (0.8, -0.8)],
            "negative_high": [(-0.5, 0.5), (-0.8, 0.8)],
            "negative_low":  [(-0.5, -0.5), (-0.8, -0.8)],
        }

        generated = []
        for quadrant_name, emotions in quadrants.items():
            for i, (v, a) in enumerate(emotions[:num_per_quadrant]):
                fname = f"{quadrant_name}_{i+1}_v{v:.1f}_a{a:.1f}.mid"
                path = self.generate(
                    valence=v,
                    arousal=a,
                    output_path=str(output_dir / fname),
                    **kwargs,
                )
                generated.append(path)

        print(f"\n  Generated {len(generated)} files across the emotion spectrum")
        return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIDI music from emotion input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m transformer.generate --valence 0.8 --arousal 0.6 --output happy.mid
  python -m transformer.generate --preset sad --output sad_song.mid
  python -m transformer.generate --preset calm --temperature 0.8 --max-len 512

Available presets:
  """ + ", ".join(sorted(EMOTION_PRESETS.keys())),
    )

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: checkpoints/best_model.pt)")
    parser.add_argument("--valence", type=float, default=None, help="Valence value (-1 to 1)")
    parser.add_argument("--arousal", type=float, default=None, help="Arousal value (-1 to 1)")
    parser.add_argument("--preset", type=str, choices=list(EMOTION_PRESETS.keys()),
                        help="Use a named emotion preset")
    parser.add_argument("--output", "-o", type=str, default="generated.mid",
                        help="Output MIDI file path")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold")
    parser.add_argument("--max-len", type=int, default=None, help="Maximum tokens to generate")
    parser.add_argument("--tempo", type=float, default=120.0, help="MIDI tempo in BPM")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--spectrum", action="store_true",
                        help="Generate across the full emotion spectrum")

    args = parser.parse_args()

    # Determine checkpoint path
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = str(DEFAULT_CONFIG.paths.checkpoint_dir / "best_model.pt")

    if not Path(checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("Train the model first with: python -m transformer.train")
        return

    # Create generator
    config = MaestroConfig()
    if args.device:
        config.device = args.device

    generator = MusicGenerator(
        checkpoint_path=checkpoint,
        config=config,
    )

    gen_kwargs = {}
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.max_len is not None:
        gen_kwargs["max_len"] = args.max_len
    gen_kwargs["tempo"] = args.tempo

    if args.spectrum:
        generator.generate_spectrum(**gen_kwargs)
    elif args.preset:
        generator.generate_preset(
            preset=args.preset,
            output_path=args.output,
            **gen_kwargs,
        )
    elif args.valence is not None and args.arousal is not None:
        generator.generate(
            valence=args.valence,
            arousal=args.arousal,
            output_path=args.output,
            **gen_kwargs,
        )
    else:
        print("ERROR: Provide either --preset or both --valence and --arousal")
        parser.print_help()


if __name__ == "__main__":
    main()
