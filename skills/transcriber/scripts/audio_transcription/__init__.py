"""Локальный конвейер расшифровки аудио."""

__version__ = "0.6.0"

from .models import Correction, Diarization, Hypothesis, ReviewItem, Segment, SpeakerTurn

__all__ = [
    "Correction",
    "Diarization",
    "Hypothesis",
    "ReviewItem",
    "Segment",
    "SpeakerTurn",
]
