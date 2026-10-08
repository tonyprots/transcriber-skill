"""FluidAudio: бинарник и CoreML-модели — где лежат, докачка с проверкой.

fluidaudiocli ищет модели в каталоге Application Support. Сам он скачал бы
последний коммит репозитория, поэтому ставим их мы — с ревизии, на которой
мерили (catalog.FLUIDAUDIO.revision).

Бинарник собирает CI репозитория и кладёт в Release; адрес и SHA-256 лежат в
`bin/macos-arm64/`. Раньше его ставил только setup.sh, и установка через
`/plugin`, `npx skills` или корпоративный стор оставалась без диаризации.
Теперь он докачивается при первом `--diarize`, как модели.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from .catalog import FLUIDAUDIO

REQUIRED = (
    "pyannote_segmentation.mlmodelc/model.mil",
    "wespeaker_v2.mlmodelc/model.mil",
    "plda-parameters.json",
)
# Только то, что берёт офлайновый режим: репозиторий держит ещё mlpackage-
# исходники, int8- и PLDA-варианты, и широкий фильтр качал 70 МБ вместо 34.
MODELS = (
    "Segmentation", "Embedding", "FBank", "PldaRho",
    "pyannote_segmentation", "wespeaker_v2",
)
FILES = ("config.json", "plda-parameters.json", "xvector-transform.json")


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
        allow_patterns=[*FILES, *(f"{name}.mlmodelc/**" for name in MODELS)],
    )
    if not ready(target):
        raise ModelsUnavailable(f"После загрузки набор моделей FluidAudio неполон: {target}")
    return "downloaded"


def bundled_binary(skill_dir: Path | None = None) -> Path:
    skill_dir = skill_dir or Path(__file__).resolve().parents[2]
    return skill_dir / "bin" / "macos-arm64" / "fluidaudiocli"


def ensure_binary(skill_dir: Path | None = None, *, offline: bool = False, opener=urllib.request.urlopen) -> Path:
    """Поставочный fluidaudiocli: уже на месте или скачанный и сверенный по SHA-256."""
    target = bundled_binary(skill_dir)
    if target.is_file() and os.access(target, os.X_OK):
        return target
    try:
        url = target.with_name("fluidaudiocli.url").read_text(encoding="utf-8").strip()
        expected = target.with_name("fluidaudiocli.sha256").read_text(encoding="utf-8").split()[0].lower()
    except (OSError, IndexError) as error:
        raise ModelsUnavailable(f"Нет адреса или суммы fluidaudiocli рядом с {target}") from error
    if offline:
        raise ModelsUnavailable(
            "Нет fluidaudiocli (около 20 МБ), а прогон офлайн. Запустите без --offline "
            "или prefetch_models.py --diarize"
        )
    part = target.with_name("fluidaudiocli.part")
    digest = hashlib.sha256()
    try:
        with opener(url, timeout=60) as response, part.open("wb") as handle:
            for block in iter(lambda: response.read(1 << 20), b""):
                digest.update(block)
                handle.write(block)
    except OSError as error:
        part.unlink(missing_ok=True)
        raise ModelsUnavailable(f"fluidaudiocli не скачался: {error}") from error
    if digest.hexdigest() != expected:
        part.unlink(missing_ok=True)
        raise ModelsUnavailable("Скачанный fluidaudiocli не совпал с опубликованной суммой")
    part.chmod(0o755)
    part.replace(target)
    return target
