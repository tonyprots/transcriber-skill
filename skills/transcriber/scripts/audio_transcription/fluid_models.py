"""FluidAudio: бинарник и CoreML-модели — где лежат, докачка с проверкой.

`fluidaudiocli process --mode offline` берёт модели из
`Application Support/FluidAudio/Models/speaker-diarization/`. Если набора там
нет, он сам качает последний коммит репозитория, поэтому ставим их мы — с
ревизии, на которой мерили (catalog.FLUIDAUDIO.revision), — а бинарнику
закрываем сеть (`no_network_command`, `OFFLINE_ENV`).

По 0.24.1 включительно скилл ставил модели в соседний `speaker-diarization-coreml`, который
бинарник не читает: он молча докачивал свой набор с `main`, и `--offline` сеть
диаризации не запрещал. Проба 2026-10-16 на поставочном бинарнике под
`sandbox-exec` без сети: с одним `-coreml` он идёт в сеть и падает, с набором
ниже в `speaker-diarization` отрабатывает.

Бинарник собирает CI репозитория и кладёт в Release; адрес и SHA-256 лежат в
`bin/macos-arm64/`. Раньше его ставил только setup.sh, и установка через
`/plugin`, `npx skills` или корпоративный стор оставалась без диаризации.
Теперь он докачивается при первом `--diarize`, как модели.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from .catalog import FLUIDAUDIO, choose_fluidaudio_binary

# Ровно то, что читает офлайновый режим бинарника: репозиторий держит ещё
# mlpackage-исходники, int8-, PLDA- и pyannote-варианты, и широкий фильтр качал
# 70 МБ вместо 34.
MODELS = ("Segmentation", "Embedding", "FBank", "PldaRho")
FILES = ("config.json", "plda-parameters.json", "xvector-transform.json")
# Файл на месте — значит докачан: huggingface_hub переносит его из `.incomplete`
# целиком. coremldata.bin проверяет и сам бинарник, веса — самый большой файл.
REQUIRED = (
    *(f"{name}.mlmodelc/{part}" for name in MODELS for part in ("coremldata.bin", "model.mil", "weights/weight.bin")),
    *FILES,
)

# Бинарник берёт адрес реестра моделей из этих переменных. Закрытый локальный
# порт (discard): если до загрузки всё же дойдёт, она упадёт сразу и наружу не
# уйдёт — проба 2026-10-16 показала запрос на lo0.
CLOSED_REGISTRY = "http://127.0.0.1:9"
OFFLINE_ENV = {"MODEL_REGISTRY_URL": CLOSED_REGISTRY, "REGISTRY_URL": CLOSED_REGISTRY}
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
NO_NETWORK_PROFILE = "(version 1)(allow default)(deny network*)"


class ModelsUnavailable(RuntimeError):
    """Моделей нет, а скачать их нельзя или не удалось."""


def default_target() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "FluidAudio"
        / "Models"
        / "speaker-diarization"
    )


def ready(target: Path | None = None) -> bool:
    target = target or default_target()
    return all((target / relative).is_file() for relative in REQUIRED)


def installed_revision(target: Path | None = None) -> str | None:
    """Коммит репозитория, с которого лежат модели, или None, если не узнать.

    Наша загрузка оставляет его в служебных файлах huggingface_hub
    (`.cache/huggingface/download/*.metadata`, первая строка). Набор, который
    бинарник скачал себе сам по 0.24.1, несёт `provenance.json` репозитория с
    `artifact_snapshot`.
    """
    target = target or default_target()
    commits = set()
    for relative in FILES:
        try:
            with (target / ".cache" / "huggingface" / "download" / f"{relative}.metadata").open(encoding="utf-8") as handle:
                commits.add(handle.readline().strip())
        except OSError:
            continue
    commits.discard("")
    if len(commits) == 1:
        return commits.pop()
    if commits:
        return None  # файлы с разных коммитов: ревизии у набора нет
    try:
        snapshot = json.loads((target / "provenance.json").read_text(encoding="utf-8")).get("artifact_snapshot")
    except (OSError, ValueError, AttributeError):
        return None
    return str(snapshot) if snapshot else None


def weights_metadata(target: Path | None = None) -> dict[str, object]:
    """Те же поля, что у весов из Hub (`weights.Weights.to_dict`): манифест и кэш
    читают их одинаково для любой модели.

    Ревизию не узнать — так и пишем: манифест предупредит, что замеры к прогону
    не относятся, а в кэш под закреплённым ключом результат не ляжет.
    """
    revision = installed_revision(target)
    return {"revision": revision or "unknown", "pinned": revision == FLUIDAUDIO.revision}


def offline_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Окружение бинарника: реестр моделей — закрытый порт."""
    environment = dict(os.environ if base is None else base)
    environment.update(OFFLINE_ENV)
    return environment


def no_network_command(command: list[str]) -> list[str]:
    """Команда под `sandbox-exec` без сети, если песочница здесь работает.

    Вложенную песочницу macOS запрещает: агент, запустивший скилл из своей
    песочницы, получил бы отказ `sandbox_apply` вместо диаризации. Поэтому
    сначала пустая проба; не вышло — остаётся закрытый реестр.
    """
    if not os.access(SANDBOX_EXEC, os.X_OK):
        return list(command)
    prefix = [SANDBOX_EXEC, "-p", NO_NETWORK_PROFILE]
    try:
        probe = subprocess.run([*prefix, "/usr/bin/true"], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return list(command)
    return [*prefix, *command] if probe.returncode == 0 else list(command)


def _foreign(target: Path) -> list[str]:
    """Что в каталоге не из нашего набора: такое не перезаписываем.

    Набор и служебный `.cache` huggingface_hub — следы прерванной загрузки,
    которую snapshot_download сам докачает и сверит. `.DS_Store` кладёт Finder.
    """
    if not target.is_dir():
        return []
    known = {*FILES, *(f"{name}.mlmodelc" for name in MODELS), ".cache", ".DS_Store"}
    return sorted(path.name for path in target.iterdir() if path.name not in known)


def preflight(explicit: str | Path | None = None, *, offline: bool = False) -> str | None:
    """Почему диаризация FluidAudio не запустится, или None.

    Проверяется до подготовки звука: раньше бинарник искали в воркере — после
    SHA-256 источника, VAD и выбора языка, — и неверный путь или офлайн без
    моделей всплывали через минуты. Онлайн недостающее докачает воркер.
    """
    chosen = choose_fluidaudio_binary(explicit, offline=offline)
    if chosen.problem:
        return chosen.problem
    if offline and not ready():
        return (
            f"Нет CoreML-моделей FluidAudio (35 МБ) в {default_target()}, а прогон офлайн. "
            "Запустите без --offline или поставьте заранее: prefetch_models.py --diarize"
        )
    return None


def ensure(target: Path | None = None, *, offline: bool = False, quiet: bool = False) -> str:
    """Возвращает "ready" или "downloaded"; иначе ModelsUnavailable.

    `quiet` — для воркера: полоса загрузки молчит, а байты идут слушателю
    `weights.download_listener` (сторож зависаний). prefetch и setup её
    показывают.
    """
    target = (target or default_target()).expanduser()
    if ready(target):
        return "ready"
    foreign = _foreign(target)
    if foreign:
        raise ModelsUnavailable(
            f"Каталог моделей FluidAudio неполон, и в нём чужие файлы ({', '.join(foreign[:3])}): {target}. "
            "Переместите его в архив и повторите запуск — набор скачается заново"
        )
    if offline:
        raise ModelsUnavailable(
            f"Нет CoreML-моделей FluidAudio (35 МБ) в {target}, а прогон офлайн. Запустите "
            "без --offline или prefetch_models.py --diarize"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ModelsUnavailable("Не установлен huggingface_hub") from error
    target.mkdir(parents=True, exist_ok=True)
    # Ошибки Hub бывают любого вида (OSError, httpx, свои классы по версиям);
    # для прогона все они значат одно — моделей нет, и это понятный отказ, а
    # не «воркер упал». Повторный запуск докачает с места обрыва.
    extra: dict[str, object] = {}
    if quiet:
        from .weights import _silent_tqdm

        extra["tqdm_class"] = _silent_tqdm()
    try:
        snapshot_download(
            repo_id=FLUIDAUDIO.repo,
            revision=FLUIDAUDIO.revision,
            local_dir=target,
            allow_patterns=[*FILES, *(f"{name}.mlmodelc/**" for name in MODELS)],
            **extra,
        )
    except Exception as error:
        raise ModelsUnavailable(f"CoreML-модели FluidAudio не скачались: {error}") from error
    if not ready(target):
        raise ModelsUnavailable(f"После загрузки набор моделей FluidAudio неполон: {target}")
    return "downloaded"


def bundled_binary(skill_dir: Path | None = None) -> Path:
    skill_dir = skill_dir or Path(__file__).resolve().parents[2]
    return skill_dir / "bin" / "macos-arm64" / "fluidaudiocli"


def ensure_binary(
    skill_dir: Path | None = None,
    *,
    offline: bool = False,
    opener=urllib.request.urlopen,
    progress=None,
) -> Path:
    """Поставочный fluidaudiocli: уже на месте или скачанный и сверенный по SHA-256.

    `progress(байты)` зовётся на каждый пришедший блок: воркер пишет по нему
    признак жизни для сторожа зависаний.
    """
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
                if progress is not None:
                    progress(len(block))
    except OSError as error:
        part.unlink(missing_ok=True)
        raise ModelsUnavailable(f"fluidaudiocli не скачался: {error}") from error
    if digest.hexdigest() != expected:
        part.unlink(missing_ok=True)
        raise ModelsUnavailable("Скачанный fluidaudiocli не совпал с опубликованной суммой")
    part.chmod(0o755)
    part.replace(target)
    return target
