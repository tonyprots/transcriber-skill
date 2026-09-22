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
    # Сколько слов разошлось: по нему очередь ранжируется. Одно слово в
    # двадцатисекундном окне и развалившаяся фраза — разная работа для человека.
    weight: int = 1
    # `substantive` — слушать, `spelling` — то же слово записано иначе
    # (кандидат в словарь), `window` — отметка на всё окно без точного места.
    kind: str = "substantive"

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
    # Ссылка, если файл скачан по ней: временный путь в манифесте ничего не
    # говорит о происхождении записи, а воспроизводимость держится на нём.
    origin: str | None = None
    # Расшифрован только кусок записи. Длительность тогда — длина куска, а
    # таймкоды результата сдвинуты к началу исходника, чтобы цитату можно
    # было сверить с субтитрами и плеером без пересчёта.
    section: Section | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "origin": self.origin,
            "section": self.section.to_dict() if self.section else None,
            "duration_seconds": self.duration_seconds,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class Section:
    """Кусок записи по времени исходника: `--section 00:10:50-00:11:30`."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"Фрагмент {self.start:g}–{self.end:g} с: конец должен быть позже начала"
            )

    @classmethod
    def parse(cls, text: str) -> "Section":
        """`1:02:03-1:04:00`, `10:50-11:30` или секунды `650-690`."""
        parts = text.strip().split("-")
        if len(parts) != 2:
            raise ValueError(f"Фрагмент задаётся как НАЧАЛО-КОНЕЦ, получено: {text!r}")
        return cls(_clock_seconds(parts[0]), _clock_seconds(parts[1]))

    @property
    def label(self) -> str:
        return f"{_clock(self.start)}–{_clock(self.end)}"

    @property
    def slug(self) -> str:
        return f"{_clock(self.start).replace(':', '')}-{_clock(self.end).replace(':', '')}"

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end}


def _clock_seconds(text: str) -> float:
    fields = text.strip().split(":")
    if not 1 <= len(fields) <= 3 or not all(fields):
        raise ValueError(f"Не время: {text!r}; ожидается ЧЧ:ММ:СС, ММ:СС или секунды")
    try:
        numbers = [float(field) for field in fields]
    except ValueError as error:
        raise ValueError(f"Не время: {text!r}") from error
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


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
