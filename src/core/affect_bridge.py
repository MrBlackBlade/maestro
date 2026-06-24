"""Utilities for mapping continuous affect values to discrete mood IDs.

The music generators in this repository condition on discrete mood IDs, while
the dual LSTM notebook predicts continuous valence/arousal values. These helpers
bridge that gap so downstream generation can stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from src.core.config import Config

# Mood prototypes are defined in the standard normalized circumplex space [-1.0, 1.0]
MOOD_PROTOTYPES: dict[str, tuple[float, float]] = {
    "angry": (-0.90, 0.95),
    "exciting": (0.80, 0.95),
    "fear": (-0.85, 0.85),
    "funny": (0.25, 0.55),
    "happy": (0.90, 0.70),
    "lazy": (-0.35, -0.80),
    "magnificent": (0.85, 0.10),
    "quiet": (-0.10, -0.55),
    "romantic": (0.75, 0.05),
    "sad": (-0.90, -0.70),
    "warm": (0.60, 0.25),
}

@dataclass(frozen=True)
class MoodMatch:
    """Result of mapping valence/arousal to a discrete mood."""
    mood_id: int
    mood_name: str
    circumplex_valence: float
    circumplex_arousal: float
    distance: float

def affect_to_mood_match(
    valence_joystick: float,
    arousal_joystick: float,
) -> MoodMatch:
    """
    Map continuous valence/arousal values (on the 1-9 joystick scale)
    to the nearest configured mood using cosine (angular) distance.
    """
    # 1. Map joystick [1, 9] back to normalized circumplex [-1.0, 1.0]
    v_circ = (valence_joystick - 5.0) / 4.0
    a_circ = (arousal_joystick - 5.0) / 4.0

    # Clip to bounds just in case of severe overshoot
    v_circ = max(-1.0, min(1.0, v_circ))
    a_circ = max(-1.0, min(1.0, a_circ))

    best_mood = None
    best_distance = float("inf")
    
    # Calculate magnitude of the prediction vector
    pred_mag = math.hypot(v_circ, a_circ)

    for mood_name in Config.MOODS:
        proto_v, proto_a = MOOD_PROTOTYPES[mood_name]
        
        # Calculate cosine distance (angular distance)
        # Cosine Similarity = (A.B) / (|A| |B|)
        proto_mag = math.hypot(proto_v, proto_a)
        
        if pred_mag < 1e-5 or proto_mag < 1e-5:
            # If extremely close to origin (neutral), fallback to Euclidean
            dist = math.hypot(v_circ - proto_v, a_circ - proto_a)
        else:
            dot_product = (v_circ * proto_v) + (a_circ * proto_a)
            cos_sim = dot_product / (pred_mag * proto_mag)
            # Clip cos_sim to [-1, 1] to avoid float precision issues with acos
            cos_sim = max(-1.0, min(1.0, cos_sim))
            # Angular distance [0, pi]
            dist = math.acos(cos_sim)
            
            # Combine with Euclidean to penalize if magnitude is way off
            # (e.g. angle is right but it's very weak)
            euclidean = math.hypot(v_circ - proto_v, a_circ - proto_a)
            dist = (dist * 0.7) + (euclidean * 0.3)

        if dist < best_distance:
            best_distance = dist
            best_mood = mood_name

    if best_mood is None:
        raise RuntimeError("Failed to resolve a mood from valence/arousal values.")

    return MoodMatch(
        mood_id=Config.MOOD_TO_ID[best_mood],
        mood_name=best_mood,
        circumplex_valence=v_circ,
        circumplex_arousal=a_circ,
        distance=best_distance,
    )
