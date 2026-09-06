"""Упаковка окон для проверочной модели.

Whisper всегда доводит вход до 30 секунд, поэтому окно на 1 секунду стоит
почти столько же, сколько окно на 20 (замер: 2.87 с против 3.77 с). При
диаризации окна режутся по сменам говорящих, медиана выходит 2.4 секунды,
и проверка часовой записи превращается в 500+ почти пустых вызовов.

Проверочной модели метки говорящих не нужны — она выдаёт только текст для
сверки. Поэтому её окна можно склеивать в длинные, а результат раскладывать
обратно по исходным окнам через пословные таймкоды.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import AudioChunk, Hypothesis, Segment


@dataclass(frozen=True)
class PackedWindow:
    """Длинное окно и исходные окна, которые оно накрывает."""

    chunk: AudioChunk
    sources: tuple[AudioChunk, ...]


def plan_packing(
    chunks: list[AudioChunk],
    *,
    max_seconds: float = 24.0,
    max_gap_seconds: float = 3.0,
) -> list[tuple[AudioChunk, ...]]:
    """Группирует соседние окна, не выходя за max_seconds и не сшивая паузы."""
    groups: list[list[AudioChunk]] = []
    for chunk in sorted(chunks, key=lambda item: (item.start, item.end)):
        if groups:
            current = groups[-1]
            gap = chunk.start - current[-1].end
            span = chunk.end - current[0].start
            if gap <= max_gap_seconds and span <= max_seconds:
                current.append(chunk)
                continue
        groups.append([chunk])
    return [tuple(group) for group in groups]


def _read_frames(prepared: Path) -> tuple[wave._wave_params, bytes]:
    with wave.open(str(prepared), "rb") as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())
    return params, frames


def build_packed_windows(
    prepared: Path,
    chunks: list[AudioChunk],
    work_dir: Path,
    *,
    max_seconds: float = 24.0,
    max_gap_seconds: float = 3.0,
) -> list[PackedWindow]:
    """Режет из подготовленного WAV длинные окна под проверочную модель."""
    if not chunks:
        return []
    groups = plan_packing(chunks, max_seconds=max_seconds, max_gap_seconds=max_gap_seconds)
    params, frames = _read_frames(prepared)
    frame_size = params.nchannels * params.sampwidth
    rate = params.framerate
    total_frames = len(frames) // frame_size

    packed_dir = work_dir / "verifier-windows"
    packed_dir.mkdir(parents=True, exist_ok=True)

    windows: list[PackedWindow] = []
    for sequence, group in enumerate(groups):
        # Одиночное окно переупаковывать незачем — берём готовый файл, но
        # номер даём по группе: у исходного окна он может совпасть с номером
        # соседней склейки, и тогда unpack_hypothesis перепутает результаты.
        if len(group) == 1:
            single = group[0]
            windows.append(
                PackedWindow(
                    chunk=AudioChunk(sequence, single.path, single.start, single.end, single.speakers),
                    sources=group,
                )
            )
            continue
        start = group[0].start
        end = group[-1].end
        first = max(0, int(round(start * rate)))
        last = min(total_frames, int(round(end * rate)))
        path = packed_dir / f"packed-{sequence:04d}.wav"
        with wave.open(str(path), "wb") as target:
            target.setparams(params)
            target.writeframes(frames[first * frame_size : last * frame_size])
        windows.append(
            PackedWindow(
                chunk=AudioChunk(
                    sequence=sequence,
                    path=path,
                    start=first / rate,
                    end=last / rate,
                ),
                sources=group,
            )
        )
    return windows


def _words_of(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        word
        for word in result.get("words", []) or []
        if str(word.get("text", "")).strip()
    ]


def _text_for(
    source: AudioChunk,
    words: list[dict[str, Any]],
    *,
    tolerance: float = 0.35,
) -> tuple[str, float | None]:
    """Собирает текст исходного окна из слов, попавших в его границы.

    Основное правило — середина слова внутри окна. Если так не набралось
    ничего, окно осталось бы вовсе без проверки, поэтому берём слова,
    которые окно хотя бы задевает: у Whisper пословные таймкоды плывут на
    доли секунды, и на коротком окне этого хватает, чтобы промахнуться.
    """
    taken = [
        word
        for word in words
        if source.start <= (float(word.get("start", 0.0)) + float(word.get("end", 0.0))) / 2 < source.end
    ]
    if not taken:
        taken = [
            word
            for word in words
            if float(word.get("end", 0.0)) > source.start - tolerance
            and float(word.get("start", 0.0)) < source.end + tolerance
        ]
    text = " ".join(str(word["text"]).strip() for word in taken).strip()
    confidences = [
        float(word["confidence"]) for word in taken if word.get("confidence") is not None
    ]
    return text, (sum(confidences) / len(confidences) if confidences else None)


def unpack_hypothesis(
    hypothesis: Hypothesis, windows: list[PackedWindow]
) -> Hypothesis:
    """Раскладывает результат по длинным окнам обратно на исходные окна."""
    by_sequence = {
        int(item.get("sequence", -1)): item
        for item in hypothesis.metadata.get("chunks", [])
    }
    segments: list[Segment] = []
    chunk_results: list[dict[str, Any]] = []
    for window in windows:
        result = by_sequence.get(window.chunk.sequence)
        if result is None:
            continue
        if len(window.sources) == 1:
            source = window.sources[0]
            text = str(result.get("text", "")).strip()
            if text:
                segments.append(
                    Segment(
                        source.start,
                        source.end,
                        text,
                        result.get("confidence"),
                        source.speakers,
                    )
                )
            chunk_results.append({**result, **source.to_dict(), "text": text})
            continue

        words = _words_of(result)
        if not words:
            # Без пословных таймкодов разложить нечем: отдаём весь текст
            # первому окну, остальные останутся пустыми и попадут в review.
            leftover = str(result.get("text", "")).strip()
            for index, source in enumerate(window.sources):
                text = leftover if index == 0 else ""
                if text:
                    segments.append(
                        Segment(source.start, source.end, text, None, source.speakers)
                    )
                chunk_results.append({**source.to_dict(), "text": text, "status": "ok"})
            continue

        for source in window.sources:
            text, confidence = _text_for(source, words)
            if text:
                segments.append(
                    Segment(source.start, source.end, text, confidence, source.speakers)
                )
            chunk_results.append(
                {
                    **source.to_dict(),
                    "text": text,
                    "confidence": confidence,
                    "status": "ok" if text else "empty",
                    "packed_from": window.chunk.sequence,
                }
            )

    metadata = dict(hypothesis.metadata)
    metadata["chunks"] = sorted(chunk_results, key=lambda item: float(item["start"]))
    metadata["packing"] = {
        "windows": len(windows),
        "sources": sum(len(window.sources) for window in windows),
        "max_seconds": max(
            (window.chunk.duration for window in windows), default=0.0
        ),
    }
    return Hypothesis(
        model=hypothesis.model,
        language=hypothesis.language,
        elapsed_seconds=hypothesis.elapsed_seconds,
        segments=sorted(segments, key=lambda item: (item.start, item.end)),
        metadata=metadata,
    )
