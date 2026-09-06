from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import Diarization, Hypothesis


CACHE_SCHEMA = 1


def cache_key(stage: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"schema": CACHE_SCHEMA, "stage": stage, **payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_hypothesis(cache_dir: Path, key: str) -> Hypothesis | None:
    path = cache_dir / "hypotheses" / f"{key}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hypothesis = Hypothesis.from_dict(payload)
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


def save_hypothesis(cache_dir: Path, key: str, hypothesis: Hypothesis) -> None:
    _save_json(cache_dir / "hypotheses", key, hypothesis.to_dict())


def load_diarization(cache_dir: Path, key: str) -> Diarization | None:
    path = cache_dir / "diarizations" / f"{key}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        diarization = Diarization.from_dict(payload)
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


def save_diarization(cache_dir: Path, key: str, diarization: Diarization) -> None:
    _save_json(cache_dir / "diarizations", key, diarization.to_dict())


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
