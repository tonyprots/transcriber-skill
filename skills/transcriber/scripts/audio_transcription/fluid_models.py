"""CoreML-модели FluidAudio: где лежат, полный ли набор, докачка закреплённой ревизии.

fluidaudiocli ищет модели в каталоге Application Support. Сам он скачал бы
последний коммит репозитория, поэтому ставим их мы — с ревизии, на которой
мерили (catalog.FLUIDAUDIO.revision).
"""
from __future__ import annotations

from pathlib import Path

from .catalog import FLUIDAUDIO

REQUIRED = (
    "pyannote_segmentation.mlmodelc/model.mil",
    "wespeaker_v2.mlmodelc/model.mil",
    "plda-parameters.json",
)


class ModelsUnavailable(RuntimeError):
    """Моделей нет, а скачать их нельзя или не удалось."""


def default_target() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "FluidAudio"
        / "Models"
        / "speaker-diarization-coreml"
    )


def ready(target: Path | None = None) -> bool:
    target = target or default_target()
    return all((target / relative).is_file() for relative in REQUIRED)


def ensure(target: Path | None = None, *, offline: bool = False) -> str:
    """Возвращает "ready" или "downloaded"; иначе ModelsUnavailable."""
    target = (target or default_target()).expanduser()
    if ready(target):
        return "ready"
    if target.exists() and any(target.iterdir()):
        raise ModelsUnavailable(
            f"Каталог моделей FluidAudio неполон: {target}. "
            "Переместите его в архив и повторите запуск — набор скачается заново"
        )
    if offline:
        raise ModelsUnavailable(
            "Нет CoreML-моделей FluidAudio (35 МБ), а прогон офлайн. Запустите "
            "без --offline или scripts/setup_fluidaudio_models.py"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ModelsUnavailable("Не установлен huggingface_hub") from error
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=FLUIDAUDIO.repo,
        revision=FLUIDAUDIO.revision,
        local_dir=target,
        allow_patterns=["*.json", "*.mlmodelc/**"],
    )
    if not ready(target):
        raise ModelsUnavailable(f"После загрузки набор моделей FluidAudio неполон: {target}")
    return "downloaded"
