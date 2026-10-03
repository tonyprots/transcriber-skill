from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .audio import SAMPLE_RATE, MediaToolError
from .models import Correction, Diarization, Hypothesis, MediaInfo, ReviewItem, Section, Segment
from .reconcile import MINOR_KINDS


def format_timestamp(seconds: float, *, srt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


SpeakerNames = dict[str, str]


def parse_speaker_names(text: str) -> SpeakerNames:
    """`1=Антон,2=Мария` → {"speaker_1": "Антон", "speaker_2": "Мария"}.

    Номер тот же, что в «Спикер 1» готовой расшифровки: по нему имя и
    назначают, прочитав текст. `unknown=…` переименовывает неразмеченную речь.
    """
    names: SpeakerNames = {}
    for part in filter(None, (item.strip() for item in text.split(","))):
        key, separator, name = part.partition("=")
        key, name = key.strip(), name.strip()
        if not separator or not key or not name:
            raise ValueError(f"Имя спикера задаётся как НОМЕР=ИМЯ, а не «{part}»")
        if key.isdigit():
            key = f"speaker_{int(key)}"
        elif key.lower() in {"unknown", "неизвестный"}:
            key = "unknown"
        elif not (key.startswith("speaker_") and key[8:].isdigit()):
            raise ValueError(f"Непонятный номер спикера: «{key}»; ожидался 1, 2, … или unknown")
        names[key] = name
    if not names:
        raise ValueError("Не задано ни одного имени спикера")
    return names


def _speaker_name(identifier: str, names: SpeakerNames | None = None) -> str:
    if names and identifier in names:
        return names[identifier]
    if identifier == "unknown":
        return "Неизвестный спикер"
    if identifier.startswith("speaker_") and identifier[8:].isdigit():
        return f"Спикер {int(identifier[8:])}"
    return identifier


def _speaker_prefix(segment: Segment, names: SpeakerNames | None = None) -> str:
    if not segment.speakers:
        return ""
    label = " + ".join(_speaker_name(item, names) for item in segment.speakers)
    return f"{label}: " if len(segment.speakers) == 1 else f"Перекрытие ({label}): "


def _markdown(segments: list[Segment], title: str, names: SpeakerNames | None = None) -> str:
    lines = [f"# {title}", ""]
    for segment in segments:
        prefix = _speaker_prefix(segment, names)
        lines.append(f"[{format_timestamp(segment.start)}] {prefix}{segment.text.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _srt(segments: list[Segment], names: SpeakerNames | None = None) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_timestamp(segment.start, srt=True)} --> {format_timestamp(segment.end, srt=True)}",
                    f"{_speaker_prefix(segment, names)}{segment.text.strip()}",
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _vtt(segments: list[Segment], names: SpeakerNames | None = None) -> str:
    blocks = ["WEBVTT"]
    for segment in segments:
        blocks.append(
            "\n".join(
                [
                    f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}",
                    f"{_speaker_prefix(segment, names)}{segment.text.strip()}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


# Сколько строк раздела разворачивать. На диктовке в 8 минут было 67 мест
# «Слушать», на встрече в 15 минут — 247 (2026-10-01); дальше первых сорока
# строк их не читал ни человек, ни агент. Остальное не пропадает: все места
# лежат в `segments.json` → `review_items`.
REVIEW_SECTION_LIMIT = 25


def _limited(items: list[ReviewItem], heading: str) -> tuple[list[ReviewItem], str]:
    shown = items[:REVIEW_SECTION_LIMIT]
    if len(items) == len(shown):
        return shown, f"## {heading} — {len(items)}"
    return shown, f"## {heading} — {len(items)}, здесь первые {len(shown)}"


def _rest_line(total: int, shown: int) -> list[str]:
    if total <= shown:
        return []
    return ["", f"Ещё {total - shown} — в `segments.json` → `review_items`."]


def _review_sections(items: list[ReviewItem]) -> list[str]:
    """Очередь порядком важности, а не порядком записи.

    Элемент очереди — разошедшееся место, а не окно: слушать нужно секунду, а
    не двадцать. Сначала идёт то, где модели услышали разные слова, потом
    разное написание одного слова, и одной строкой — мелочь вроде «uh».
    """
    minor = [item for item in items if item.kind in MINOR_KINDS]
    spelling = [item for item in items if item.kind == "spelling"]
    listen = [item for item in items if item.kind in {"substantive", "window"}]
    listen.sort(key=lambda item: (-item.weight, item.start))
    lines: list[str] = []
    if listen:
        shown, heading = _limited(listen, "Слушать")
        lines.extend(
            [
                heading,
                "",
                "Сверху те места, где расхождение длиннее: одно слово может "
                "оказаться опечаткой, развалившаяся фраза — потерянной мыслью. "
                "Метка времени внутри окна оценена по позиции слова.",
                "",
            ]
        )
        lines.extend(_review_line(item) for item in shown)
        lines.extend(_rest_line(len(listen), len(shown)))
        lines.append("")
    if spelling:
        # Сначала самые весомые, показываются — порядком записи.
        shown, heading = _limited(
            sorted(spelling, key=lambda item: (-item.weight, item.start)),
            "То же слово записано иначе",
        )
        lines.extend(
            [
                heading,
                "",
                "Слушать нечего: модели согласны, что слово прозвучало, и "
                "расходятся в написании. Отсюда берутся записи словаря.",
                "",
            ]
        )
        lines.extend(_review_line(item) for item in sorted(shown, key=lambda i: i.start))
        lines.extend(_rest_line(len(spelling), len(shown)))
        lines.append("")
    if minor:
        lines.extend(
            [
                f"## Мелкие расхождения — {len(minor)}",
                "",
                "Слова-паразиты и разная запись чисел («Q4» против «Q four»). "
                "В отчёт не разворачиваются, но лежат целиком в `segments.json`.",
                "",
            ]
        )
    return lines


def _review_line(item: ReviewItem) -> str:
    difference = ", ".join(item.differing_tokens) or item.reason.split(": ", 1)[-1]
    context = item.readable_text.strip()
    line = f"- [{format_timestamp(item.start)}] {difference}"
    return f"{line} — в контексте: {context}" if context else line


def _review_markdown(
    items: list[ReviewItem],
    suggestions: list[dict[str, Any]],
    learned: dict[str, Any] | None = None,
    corrections: list[Correction] | None = None,
) -> str:
    lines = ["# Требует проверки", ""]
    taken = [item for item in corrections or [] if item.rule == "verifier_canonical"]
    # Включившаяся замена — первое, что человек должен увидеть: дальше она
    # будет молча применяться к каждой расшифровке, и «молча» здесь опаснее
    # любого расхождения в очереди.
    promoted = list((learned or {}).get("promoted", []))
    if promoted:
        lines.extend(
            [
                "## Скилл начал заменять",
                "",
                "Термины повторились в нескольких записях, и с этого прогона "
                "скилл правит их сам. Если написание неверно — поправьте запись "
                f"в {(learned or {}).get('path') or 'словаре скилла'}, ручная "
                "правка сильнее автоматики.",
                "",
            ]
        )
        lines.extend(f"- {term}" for term in promoted)
        lines.append("")
    if taken:
        # Правка текста по второй модели — то же «молча», что и включённая
        # замена, поэтому стоит наверху, а не теряется среди расхождений.
        lines.extend(
            [
                f"## Взято у проверяющей — {len(taken)}",
                "",
                "Модели разошлись, и проверяющая написала термин так, как он "
                "записан в словаре, а основная — похожее по звучанию. В тексте "
                "стоит термин; в очередь эти места не попали.",
                "",
            ]
        )
        lines.extend(
            f"- [{format_timestamp(item.start)}] «{item.before}» → {item.after}" for item in taken
        )
        lines.append("")
    dropped = sorted(
        (item for item in items if item.kind == "verifier_only"), key=lambda item: item.start
    )
    if dropped:
        # Текст здесь не правится (вставка у проверяющей на эталоне вредит),
        # но выпавшее название меняет смысл фразы — поэтому место наверху.
        lines.extend(
            [
                f"## Возможно, выпало из текста — {len(dropped)}",
                "",
                "Здесь у основной модели пусто, а проверяющая услышала название или "
                "термин словаря. В текст это не вставлено: проверяющая иногда "
                "дописывает то, чего не было. ∅ — место пропуска.",
                "",
            ]
        )
        lines.extend(
            f"- [{format_timestamp(item.start)}] «{item.verifier_span.strip()}» — в контексте: "
            f"{item.readable_text.strip()}"
            for item in dropped
        )
        lines.append("")
    items = [item for item in items if item.kind not in {"verifier_canonical", "verifier_only"}]
    if not items and not suggestions:
        if promoted or taken or dropped:
            return "\n".join(lines).rstrip() + "\n"
        return "# Требует проверки\n\nРасхождений выше заданного порога не найдено.\n"
    lines.extend(_review_sections(items))
    if suggestions:
        lines.extend(["## Возможные термины из словаря", ""])
        for suggestion in suggestions:
            lines.append(
                f"- [{format_timestamp(float(suggestion['start']))}] "
                f"«{suggestion['heard']}» → «{suggestion['canonical']}» "
                f"(сходство {float(suggestion['similarity']):.1%})"
            )
    waiting = list((learned or {}).get("pending", []))
    if waiting:
        lines.extend(
            [
                "",
                "## Правильное написание знает только человек",
                "",
                "Термин слышно, но правильно его не написала ни одна модель. "
                "Скилл запомнил варианты и ждёт: назовите написание — и он "
                "начнёт его держать.",
                "",
            ]
        )
        lines.extend(f"- {term}" for term in waiting)
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


def _write_texts(
    directory: Path,
    lexical_segments: list[Segment],
    readable_segments: list[Segment],
    *,
    scope: str,
    names: SpeakerNames | None,
) -> None:
    """Четыре файла, которые читает человек: в них и только в них живут имена."""
    # Про фрагмент говорит заголовок: без него таймкоды с 00:10:50
    # выглядят как потерянное начало записи.
    (directory / "verbatim.md").write_text(
        _markdown(lexical_segments, f"Дословная расшифровка{scope}", names), encoding="utf-8"
    )
    (directory / "readable.md").write_text(
        _markdown(readable_segments, f"Читаемая расшифровка{scope}", names), encoding="utf-8"
    )
    (directory / "subtitles.srt").write_text(_srt(readable_segments, names), encoding="utf-8")
    (directory / "subtitles.vtt").write_text(_vtt(readable_segments, names), encoding="utf-8")


def rename_speakers(output_dir: Path, names: SpeakerNames) -> list[str]:
    """Переписывает готовый результат с именами вместо «Спикер N».

    Имена обычно известны только после чтения текста, поэтому пересборка
    идёт из `segments.json`, без пересчёта звука. Машинные файлы (`raw.json`,
    `segments.json`, `speakers.json`) хранят исходные метки: имя — подпись для
    человека, а не данные. Возвращает метки, которых в записи нет.
    """
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Нет manifest.json: {output_dir} — это не результат transcriber")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored = json.loads((output_dir / "segments.json").read_text(encoding="utf-8"))
    readable = [Segment.from_dict(item) for item in stored["readable"]]
    # До схемы 7 гипотезы лежали и в `segments.json`; теперь только в `raw.json`.
    hypotheses = stored.get("hypotheses") or json.loads(
        (output_dir / "raw.json").read_text(encoding="utf-8")
    )["hypotheses"]
    lexical = [Segment.from_dict(item) for item in hypotheses[0]["segments"]]
    present = {speaker for segment in readable + lexical for speaker in segment.speakers}
    if not present:
        raise ValueError("В этой расшифровке нет разметки по говорящим: запускали без --diarize")
    merged = {**(manifest.get("speaker_names") or {}), **names}
    section = (manifest.get("source") or {}).get("section")
    scope = (
        f" (фрагмент {Section(float(section['start']), float(section['end'])).label})"
        if isinstance(section, dict)
        else ""
    )
    _write_texts(output_dir, lexical, readable, scope=scope, names=merged)
    manifest["speaker_names"] = merged
    _write_json(manifest_path, manifest)
    return sorted(set(names) - present)


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
    learned: dict[str, Any] | None = None,
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
    model_revisions: dict[str, str | None] | None = None,
    speaker_names: SpeakerNames | None = None,
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
            "schema_version": 7,
            "generator": generator,
            "created_at": datetime.now().astimezone().isoformat(),
            "mode": mode,
            "offline": offline,
            "source": media.to_dict(),
            "models": [hypothesis.model for hypothesis in hypotheses],
            # Имя модели не определяет веса: репозиторий на хабе могут
            # перезалить, и тот же прогон даст другой текст. Ревизия из
            # локального кэша — единственное, по чему потом видно, на чём
            # именно получен этот результат.
            "model_revisions": model_revisions or {},
            "language_route": route,
            "review_count": sum(item.kind != "verifier_canonical" for item in review_items),
            "automatic_correction_count": len(corrections),
            "fillers_removed": fillers_removed,
            "warnings": warnings or [],
            "diarization_model": diarization.model if diarization else None,
            "speaker_count": len(diarization.speakers) if diarization else None,
            "speaker_names": speaker_names or {},
            "timings_seconds": timings or {},
            "files": files,
        }
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "raw.json", raw)
        # Про фрагмент говорит заголовок: без него таймкоды с 00:10:50
        # выглядят как потерянное начало записи.
        _write_texts(
            staging,
            lexical_segments,
            readable_segments,
            scope=f" (фрагмент {media.section.label})" if media.section else "",
            names=speaker_names,
        )
        _write_json(
            staging / "segments.json",
            # Гипотезы моделей — только в `raw.json`: до схемы 7 здесь лежала
            # их полная копия, половина веса файла (180 КБ на 9 минут).
            {
                "readable": [segment.to_dict() for segment in readable_segments],
                "review_items": [item.to_dict() for item in review_items],
            },
        )
        _write_json(
            staging / "glossary-audit.json",
            {
                "corrections": [correction.to_dict() for correction in corrections],
                "suggestions": suggestions,
                # Что скилл дописал в свой словарь на этой записи: без этого
                # блока автозамена, включившаяся сама, выглядела бы в отчёте
                # правкой без причины.
                "learned": learned,
            },
        )
        if diarization is not None:
            _write_json(staging / "speakers.json", diarization.to_dict())
        (staging / "review-needed.md").write_text(
            _review_markdown(review_items, suggestions, learned, corrections), encoding="utf-8"
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
