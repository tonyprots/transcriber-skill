from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None
    speakers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("Начало сегмента не может быть отрицательным")
        if self.end < self.start:
            raise ValueError("Конец сегмента не может быть раньше начала")
        if not self.text.strip():
            raise ValueError("Текст сегмента не может быть пустым")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence должен быть в диапазоне [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["speakers"] = list(self.speakers)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Segment":
        return cls(
            start=float(payload["start"]),
            end=float(payload["end"]),
            text=str(payload["text"]),
            confidence=(
                float(payload["confidence"])
                if payload.get("confidence") is not None
                else None
            ),
            speakers=tuple(str(item) for item in payload.get("speakers", [])),
        )


@dataclass
class Hypothesis:
    model: str
    language: str
    elapsed_seconds: float
    segments: list[Segment]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "language": self.language,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Hypothesis":
        return cls(
            model=str(payload["model"]),
            language=str(payload["language"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            segments=[Segment.from_dict(item) for item in payload.get("segments", [])],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ReviewItem:
    start: float
    end: float
    readable_text: str
    comparison_text: str
    similarity: float
    differing_tokens: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["differing_tokens"] = list(self.differing_tokens)
        return result


@dataclass(frozen=True)
class Correction:
    start: float
    end: float
    before: str
    after: str
    canonical: str
    rule: str
    automatic: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MediaInfo:
    source: Path
    duration_seconds: float | None
    codec: str | None
    sample_rate: int | None
    channels: int | None
    size_bytes: int | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "duration_seconds": self.duration_seconds,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AudioChunk:
    sequence: int
    path: Path
    start: float
    end: float
    speakers: tuple[str, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "speakers": list(self.speakers),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AudioChunk":
        return cls(
            sequence=int(payload["sequence"]),
            path=Path(payload["path"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            speakers=tuple(str(item) for item in payload.get("speakers", [])),
        )


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speakers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Некорректные границы реплики")
        if not self.speakers:
            raise ValueError("У реплики должен быть хотя бы один говорящий")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "speakers": list(self.speakers),
            "overlap": len(self.speakers) > 1,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpeakerTurn":
        return cls(
            float(payload["start"]),
            float(payload["end"]),
            tuple(str(item) for item in payload["speakers"]),
        )


@dataclass
class Diarization:
    model: str
    elapsed_seconds: float
    turns: list[SpeakerTurn]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(sorted({speaker for turn in self.turns for speaker in turn.speakers}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "speaker_count": len(self.speakers),
            "speakers": list(self.speakers),
            "turns": [turn.to_dict() for turn in self.turns],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Diarization":
        return cls(
            model=str(payload["model"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            turns=[SpeakerTurn.from_dict(item) for item in payload.get("turns", [])],
            metadata=dict(payload.get("metadata", {})),
        )
