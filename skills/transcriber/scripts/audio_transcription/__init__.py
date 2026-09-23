"""Локальный конвейер расшифровки аудио.

`__version__` — единственная точка правды о версии скилла. Отсюда её берут
`manifest.json` и `transcribe.py --version`, а `tests/test_docs_consistency.py`
следит, чтобы frontmatter SKILL.md и README называли то же число: версия в
манифесте — часть аудит-следа, и разойтись ей нельзя.
"""

__version__ = "0.15.1"

from .models import Correction, Diarization, Hypothesis, ReviewItem, Segment, SpeakerTurn

__all__ = [
    "Correction",
    "Diarization",
    "Hypothesis",
    "ReviewItem",
    "Segment",
    "SpeakerTurn",
]
