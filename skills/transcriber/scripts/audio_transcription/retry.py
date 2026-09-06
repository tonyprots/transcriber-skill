from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .audio import split_audio_chunk
from .models import AudioChunk


Recognition = dict[str, Any]


def _failure_reason(error: Exception | None, text: str) -> str:
    return str(error) if error is not None else "модель вернула пустой текст"


def _merge_children(
    chunk: AudioChunk,
    children: list[Recognition],
    *,
    reason: str,
) -> Recognition:
    text = " ".join(str(child.get("text", "")).strip() for child in children).strip()
    incomplete = sum(child.get("status") in {"failed", "partial"} for child in children)
    result: Recognition = {
        **chunk.to_dict(),
        "text": text,
        "status": "failed" if not text else ("partial" if incomplete else "recovered"),
        "retry_attempts": 1 + sum(int(child.get("retry_attempts", 1)) for child in children),
        "retry_depth": 1 + max(int(child.get("retry_depth", 0)) for child in children),
        "retry_reason": reason,
        "subchunks": children,
    }
    for key in ("words", "segment_metrics"):
        values = [item for child in children for item in child.get(key, [])]
        if values:
            result[key] = values
    languages = [str(child["language"]) for child in children if child.get("language")]
    if languages:
        result["language"] = languages[-1]
    return result


def recognize_with_retry(
    chunk: AudioChunk,
    recognize: Callable[[AudioChunk], Recognition],
    retry_dir: Path,
    *,
    max_depth: int = 2,
    min_part_seconds: float = 2.0,
    _depth: int = 0,
    _branch: str = "root",
) -> Recognition:
    """Повторяет только сбойное окно, рекурсивно деля его на равные части."""
    error: Exception | None = None
    try:
        result = dict(recognize(chunk))
        text = str(result.get("text", "")).strip()
        if text:
            return {
                **chunk.to_dict(),
                **result,
                "text": text,
                "status": "ok",
                "retry_attempts": 1,
                "retry_depth": 0,
            }
    except Exception as caught:  # разные ASR-runtime используют разные ошибки
        error = caught
        result = {}
        text = ""

    reason = _failure_reason(error, text)
    can_split = _depth < max_depth and chunk.duration >= 2 * min_part_seconds
    if not can_split:
        return {
            **chunk.to_dict(),
            **result,
            "text": "",
            "status": "failed",
            "retry_attempts": 1,
            "retry_depth": 0,
            "error": reason,
        }

    left, right = split_audio_chunk(
        chunk,
        retry_dir,
        label=f"d{_depth + 1}-{_branch}",
    )
    children = [
        recognize_with_retry(
            child,
            recognize,
            retry_dir,
            max_depth=max_depth,
            min_part_seconds=min_part_seconds,
            _depth=_depth + 1,
            _branch=f"{_branch}{side}",
        )
        for child, side in ((left, "l"), (right, "r"))
    ]
    return _merge_children(chunk, children, reason=reason)
