from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .audio import SAMPLE_RATE, MediaToolError
from .models import Correction, Diarization, Hypothesis, MediaInfo, ReviewItem, Segment


def format_timestamp(seconds: float, *, srt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _speaker_name(identifier: str) -> str:
    if identifier == "unknown":
        return "Неизвестный спикер"
    if identifier.startswith("speaker_") and identifier[8:].isdigit():
        return f"Спикер {int(identifier[8:])}"
    return identifier


def _speaker_prefix(segment: Segment) -> str:
    if not segment.speakers:
        return ""
    names = " + ".join(_speaker_name(item) for item in segment.speakers)
    return f"{names}: " if len(segment.speakers) == 1 else f"Перекрытие ({names}): "


def _markdown(segments: list[Segment], title: str) -> str:
    lines = [f"# {title}", ""]
    for segment in segments:
        prefix = _speaker_prefix(segment)
        lines.append(f"[{format_timestamp(segment.start)}] {prefix}{segment.text.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _srt(segments: list[Segment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_timestamp(segment.start, srt=True)} --> {format_timestamp(segment.end, srt=True)}",
                    f"{_speaker_prefix(segment)}{segment.text.strip()}",
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _vtt(segments: list[Segment]) -> str:
    blocks = ["WEBVTT"]
    for segment in segments:
        blocks.append(
            "\n".join(
                [
                    f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}",
                    f"{_speaker_prefix(segment)}{segment.text.strip()}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def _review_markdown(
    items: list[ReviewItem], suggestions: list[dict[str, Any]]
) -> str:
    lines = ["# Требует проверки", ""]
    if not items and not suggestions:
        return "# Требует проверки\n\nРасхождений выше заданного порога не найдено.\n"
    for item in items:
        lines.extend(
            [
                f"## {format_timestamp(item.start)}–{format_timestamp(item.end)}",
                "",
                f"- Причина: {item.reason}; сходство {item.similarity:.1%}",
                f"- Читаемая гипотеза: {item.readable_text or '—'}",
                f"- Проверочная гипотеза: {item.comparison_text or '—'}",
                f"- Различия: {', '.join(item.differing_tokens) or '—'}",
                "",
            ]
        )
    if suggestions:
        lines.extend(["## Возможные термины из словаря", ""])
        for suggestion in suggestions:
            lines.append(
                f"- [{format_timestamp(float(suggestion['start']))}] "
                f"«{suggestion['heard']}» → «{suggestion['canonical']}» "
                f"(сходство {float(suggestion['similarity']):.1%})"
            )
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def ensure_output_available(output_dir: Path, *, overwrite: bool = False) -> Path:
    """Проверяет каталог результата до тяжёлой работы и возвращает нормальный путь.

    Проверка стоит и здесь, и в начале прогона: без ранней проверки отказ
    «каталог уже существует» приходил после всей расшифровки.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Путь результата уже занят файлом: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Каталог уже существует: {output_dir}. "
            "Укажите пустой каталог или добавьте --overwrite."
        )
    return output_dir


def ensure_disk_space(
    work_dir: Path,
    duration_seconds: float | None,
    *,
    headroom_bytes: int = 2 * 1024**3,
) -> None:
    """Ставит отказ до расшифровки, а не после часа работы.

    Прогон 2026-09-04 умер на 92-м окне из 536, когда на диске осталось 140 МБ:
    воркер убило молча, а текст ошибки пришёл через 13 минут уборки. Временные
    WAV занимают примерно три длительности записи при 16 кГц моно.
    """
    if duration_seconds is None:
        return
    needed = int(duration_seconds * SAMPLE_RATE * 2 * 3) + headroom_bytes
    free = shutil.disk_usage(work_dir).free
    if free < needed:
        raise MediaToolError(
            f"Мало места на диске: свободно {free / 1024**3:.1f} ГБ, "
            f"на запись длиной {duration_seconds / 60:.0f} мин нужно "
            f"{needed / 1024**3:.1f} ГБ вместе с запасом. "
            "Освободите место и повторите — кэш гипотез сохранится."
        )


def write_bundle(
    output_dir: Path,
    *,
    media: MediaInfo,
    hypotheses: list[Hypothesis],
    lexical_segments: list[Segment],
    readable_segments: list[Segment],
    review_items: list[ReviewItem],
    corrections: list[Correction],
    suggestions: list[dict[str, Any]],
    mode: str,
    offline: bool,
    route: dict[str, Any] | None = None,
    diarization: Diarization | None = None,
    warnings: list[str] | None = None,
    overwrite: bool = False,
    prepared_audio: Path | None = None,
    timings: dict[str, float] | None = None,
    generator: str | None = None,
    fillers_removed: int = 0,
) -> Path:
    output_dir = ensure_output_available(output_dir, overwrite=overwrite)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        raw = {
            "source": media.to_dict(),
            "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
            "review_items": [item.to_dict() for item in review_items],
            "corrections": [correction.to_dict() for correction in corrections],
            "glossary_suggestions": suggestions,
            "warnings": warnings or [],
            "language_route": route,
            "diarization": diarization.to_dict() if diarization else None,
        }
        files = [
            "manifest.json",
            "raw.json",
            "verbatim.md",
            "readable.md",
            "subtitles.srt",
            "subtitles.vtt",
            "segments.json",
            "glossary-audit.json",
            "review-needed.md",
        ]
        if prepared_audio is not None:
            files.append("prepared.wav")
        if diarization is not None:
            files.append("speakers.json")
        manifest = {
            "schema_version": 5,
            "generator": generator,
            "created_at": datetime.now().astimezone().isoformat(),
            "mode": mode,
            "offline": offline,
            "source": media.to_dict(),
            "models": [hypothesis.model for hypothesis in hypotheses],
            "language_route": route,
            "review_count": len(review_items),
            "automatic_correction_count": len(corrections),
            "fillers_removed": fillers_removed,
            "warnings": warnings or [],
            "diarization_model": diarization.model if diarization else None,
            "speaker_count": len(diarization.speakers) if diarization else None,
            "timings_seconds": timings or {},
            "files": files,
        }
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "raw.json", raw)
        (staging / "verbatim.md").write_text(
            _markdown(lexical_segments, "Дословная расшифровка"), encoding="utf-8"
        )
        (staging / "readable.md").write_text(
            _markdown(readable_segments, "Читаемая расшифровка"), encoding="utf-8"
        )
        (staging / "subtitles.srt").write_text(_srt(readable_segments), encoding="utf-8")
        (staging / "subtitles.vtt").write_text(_vtt(readable_segments), encoding="utf-8")
        _write_json(
            staging / "segments.json",
            {
                "readable": [segment.to_dict() for segment in readable_segments],
                "hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses],
                "review_items": [item.to_dict() for item in review_items],
            },
        )
        _write_json(
            staging / "glossary-audit.json",
            {
                "corrections": [correction.to_dict() for correction in corrections],
                "suggestions": suggestions,
            },
        )
        if diarization is not None:
            _write_json(staging / "speakers.json", diarization.to_dict())
        (staging / "review-needed.md").write_text(
            _review_markdown(review_items, suggestions), encoding="utf-8"
        )
        if prepared_audio is not None:
            shutil.copy2(prepared_audio, staging / "prepared.wav")
        if output_dir.exists() and any(output_dir.iterdir()):
            archive = output_dir.with_name(
                f"{output_dir.name}.archive-{datetime.now():%Y%m%d-%H%M%S-%f}"
            )
            os.replace(output_dir, archive)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir
