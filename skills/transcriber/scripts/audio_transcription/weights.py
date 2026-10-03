"""Веса моделей по закреплённой ревизии — единственный путь их скачать и найти.

Раньше каждая библиотека качала свою модель сама и брала `main` репозитория:
имя модели одно, а веса через месяц могли быть другими, и замеры из
references/quality.md переставали описывать прогон молча. Теперь веса берутся
по коммиту из каталога (`ModelEntry.revision`), библиотеке отдаётся локальный
путь, а фактическая ревизия возвращается в метаданные гипотезы.

Коммит неизменяем, поэтому `huggingface_hub` не спрашивает Hub, что сейчас в
`main`: при скачанных весах сеть нужна разве что один раз — за списком файлов
коммита, который он кэширует на диске.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import ModelEntry, entry_for_model, unpinned

# Тот же набор, что берёт `faster_whisper.download_model`: функция не отдаёт
# его наружу, а качать репозиторий целиком незачем.
FASTER_WHISPER_PATTERNS = (
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
)


@dataclass(frozen=True)
class Weights:
    path: Path
    revision: str | None
    pinned: bool

    def to_dict(self) -> dict[str, Any]:
        return {"revision": self.revision, "pinned": self.pinned}


def _onnx_patterns(entry: ModelEntry) -> list[str]:
    """Файлы, которые onnx-asr качает для этой модели, — его же правилом.

    Репозиторий GigaAM v3 держит все варианты сразу (CTC, RNNT, E2E, int8), и
    снапшот целиком весил бы в разы больше нужного.
    """
    from onnx_asr.loader import create_asr_resolver, create_vad_resolver

    resolver = (
        create_vad_resolver(entry.model)
        if entry.family == "vad"
        else create_asr_resolver(entry.model)
    )
    files = list(resolver.model_type._get_model_files(entry.quantization).values())
    files += [name.removeprefix("**/") for name in files if name.startswith("**/")]
    return [
        "config.json",
        "config.yaml",
        *files,
        *(str(Path(name).with_suffix(".onnx?data")) for name in files if name.endswith(".onnx")),
    ]


def allow_patterns(entry: ModelEntry) -> list[str] | None:
    if entry.family in {"vad", "gigaam", "onnx"}:
        return _onnx_patterns(entry)
    if entry.family == "whisper-cpu":
        return list(FASTER_WHISPER_PATTERNS)
    if entry.family in {"whisper", "diarization"}:
        from mlx_audio.utils import DEFAULT_ALLOW_PATTERNS

        return list(DEFAULT_ALLOW_PATTERNS)
    return None


def _warn(text: str) -> None:
    print(f"Внимание: {text}", file=sys.stderr, flush=True)


def _silent_tqdm() -> type:
    from tqdm.auto import tqdm

    class Silent(tqdm):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    return Silent


def fetch(
    entry: ModelEntry,
    *,
    download: Callable[..., str] | None = None,
    warn: Callable[[str], None] = _warn,
    quiet: bool = True,
) -> Weights:
    """Локальный путь к весам `entry` по закреплённой ревизии, со скачиванием.

    Офлайн (`HF_HUB_OFFLINE`, его ставит `load_with_hub_fallback`) — только из
    кэша. Пропавший с Hub коммит (force-push у автора) не роняет установку:
    сначала ищем его в кэше, потом берём `main` и громко говорим, что веса уже
    не те, на которых мерили.
    """
    if not entry.repo:
        raise ValueError(f"У модели {entry.label} нет репозитория на Hub")
    if download is None:
        from huggingface_hub import snapshot_download

        # Полоса «Fetching 5 files» на каждом прогоне — шум в stderr, где идёт
        # ход работы. При первой установке (prefetch) полоса нужна: там гигабайты.
        extra = {"tqdm_class": _silent_tqdm()} if quiet else {}

        def download(*args: Any, **kwargs: Any) -> str:
            return snapshot_download(*args, **kwargs, **extra)
    from huggingface_hub.errors import RevisionNotFoundError

    patterns = allow_patterns(entry)
    wanted = None if unpinned() else entry.revision
    try:
        path = Path(download(entry.repo, revision=wanted, allow_patterns=patterns))
    except RevisionNotFoundError:
        try:
            path = Path(
                download(entry.repo, revision=wanted, allow_patterns=patterns, local_files_only=True)
            )
        except Exception:  # noqa: BLE001 — любой промах кэша ведёт к main
            warn(
                f"{entry.label}: закреплённой ревизии {str(wanted)[:12]} на Hugging Face "
                "больше нет, беру текущую. Замеры качества к ней не относятся."
            )
            path = Path(download(entry.repo, allow_patterns=patterns))
    revision = path.name if path.parent.name == "snapshots" else None
    return Weights(path, revision, bool(entry.revision) and revision == entry.revision)


def weights_for(model: str, quantization: str | None = None) -> Weights | None:
    """Веса модели из каталога или None — тогда библиотека грузит её сама по имени."""
    entry = entry_for_model(model, quantization)
    if entry is None or not entry.repo or not entry.in_hf_cache:
        return None
    return fetch(entry)
