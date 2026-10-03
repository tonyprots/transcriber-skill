from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Diarization, Hypothesis
from .retry import crash_error


CACHE_SCHEMA = 1

# Гипотеза в кэше — это текст записи. Он переживал удаление самой расшифровки
# и копился годами: у автора к 2026-10-03 — 602 гипотезы на 70 МБ. Срок
# считается от последнего обращения: запись, которую перепрогоняют, живёт.
CACHE_MAX_AGE_DAYS = 90
STAGES = ("hypotheses", "diarizations")
# Чьё это: SHA-256 исходника (у фрагмента — с границами). По нему
# `forget` находит записи одной расшифровки; ключ — хеш, по нему не найти.
SOURCE_FIELD = "cache_source"


def cache_key(stage: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"schema": CACHE_SCHEMA, "stage": stage, **payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def crashed_windows(hypothesis: Hypothesis) -> list[str]:
    """Ошибки окон, упавших на исключении, а не на пустом ответе модели."""
    chunks = hypothesis.metadata.get("chunks") or []
    return [
        error
        for chunk in chunks
        if isinstance(chunk, dict) and (error := crash_error(chunk))
    ]


def load_hypothesis(cache_dir: Path, key: str) -> Hypothesis | None:
    path = cache_dir / "hypotheses" / f"{key}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hypothesis = Hypothesis.from_dict(payload)
        _touch(path)
        # Записи, сохранённые до этой проверки, могли запомнить сбой среды
        # как результат. Такую запись не отдаём: окна посчитаются заново.
        if crashed_windows(hypothesis):
            return None
        metadata = dict(hypothesis.metadata)
        metadata["cache_hit"] = True
        metadata["cached_elapsed_seconds"] = payload.get("elapsed_seconds")
        return Hypothesis(
            model=hypothesis.model,
            language=hypothesis.language,
            elapsed_seconds=0.0,
            segments=hypothesis.segments,
            metadata=metadata,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_hypothesis(
    cache_dir: Path, key: str, hypothesis: Hypothesis, source: str | None = None
) -> bool:
    """Сохраняет гипотезу, если в ней нет окон, упавших на исключении.

    Иначе сбой одного прогона стал бы ответом всех следующих: 2026-09-23
    прогон, прерванный выключением системы, запомнил 608 пустых окон, и
    повтор за 0,24 с выдал пустую расшифровку. Досчитывать только упавшие
    окна не стали: у сбоя среды обычно падает всё, и выгода была бы мнимой.
    """
    if crashed_windows(hypothesis):
        return False
    _save_json(cache_dir / "hypotheses", key, _tagged(hypothesis.to_dict(), source))
    return True


def load_diarization(cache_dir: Path, key: str) -> Diarization | None:
    path = cache_dir / "diarizations" / f"{key}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        diarization = Diarization.from_dict(payload)
        _touch(path)
        metadata = dict(diarization.metadata)
        metadata["cache_hit"] = True
        metadata["cached_elapsed_seconds"] = payload.get("elapsed_seconds")
        return Diarization(
            diarization.model,
            0.0,
            diarization.turns,
            metadata,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_diarization(
    cache_dir: Path, key: str, diarization: Diarization, source: str | None = None
) -> None:
    _save_json(cache_dir / "diarizations", key, _tagged(diarization.to_dict(), source))


def _tagged(payload: dict[str, Any], source: str | None) -> dict[str, Any]:
    return payload | {SOURCE_FIELD: source} if source else payload


def _touch(path: Path) -> None:
    try:
        os.utime(path)
    except OSError:
        pass


@dataclass(frozen=True)
class Swept:
    files: int = 0
    bytes: int = 0

    def __add__(self, other: "Swept") -> "Swept":
        return Swept(self.files + other.files, self.bytes + other.bytes)


def _entries(cache_dir: Path) -> list[Path]:
    # Временные файлы начинаются с точки: их пишет _save_json прямо сейчас.
    return [
        path
        for stage in STAGES
        for path in sorted((cache_dir / stage).glob("*.json"))
        if not path.name.startswith(".")
    ]


def _remove(paths: list[Path]) -> Swept:
    swept = Swept()
    for path in paths:
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        swept += Swept(1, size)
    return swept


def cache_stats(cache_dir: Path, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    entries = _entries(cache_dir)
    sizes: list[int] = []
    ages: list[float] = []
    by_stage = {stage: 0 for stage in STAGES}
    for path in entries:
        try:
            stat = path.stat()
        except OSError:
            continue
        sizes.append(stat.st_size)
        ages.append((now - stat.st_mtime) / 86400)
        by_stage[path.parent.name] = by_stage.get(path.parent.name, 0) + 1
    return {
        "path": str(cache_dir),
        "files": len(sizes),
        "bytes": sum(sizes),
        "by_stage": by_stage,
        "oldest_days": round(max(ages), 1) if ages else None,
        "older_than_max_age": sum(age > CACHE_MAX_AGE_DAYS for age in ages),
        "max_age_days": CACHE_MAX_AGE_DAYS,
    }


def prune(cache_dir: Path, older_than_days: float, now: float | None = None) -> Swept:
    """Удаляет записи, к которым не обращались дольше срока."""
    limit = (time.time() if now is None else now) - older_than_days * 86400
    stale = []
    for path in _entries(cache_dir):
        try:
            if path.stat().st_mtime < limit:
                stale.append(path)
        except OSError:
            continue
    return _remove(stale)


def forget(cache_dir: Path, source_sha256: str) -> Swept:
    """Удаляет всё, что кэш помнит об одной записи, включая её фрагменты.

    Записи, сохранённые до 0.19, метки не несут и так не находятся: их
    уберёт `prune` по сроку или `clear`.
    """
    wanted = source_sha256.strip().lower()
    found = []
    for path in _entries(cache_dir):
        try:
            source = json.loads(path.read_text(encoding="utf-8")).get(SOURCE_FIELD)
        except (OSError, ValueError):
            continue
        if isinstance(source, str) and (source == wanted or source.startswith(wanted + "@")):
            found.append(path)
    return _remove(found)


def clear(cache_dir: Path) -> Swept:
    return _remove(_entries(cache_dir))


def _save_json(directory: Path, key: str, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{key}.json"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{key}-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
