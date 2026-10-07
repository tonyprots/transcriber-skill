"""Каталог моделей: единственное место, где записаны их идентификаторы.

Здесь же дата последнего локального замера качества (`calibrated`). Скилл не
обновляет модели сам и не должен: выбор ASR держится на замерах WER, а не на
номере версии. Но молча работать на устаревшей модели он тоже не должен,
поэтому `scripts/doctor.py` сверяет этот каталог с датой и с тем, что знает
установленная библиотека, и говорит пользователю, когда пора пересверить.

Меняете модель — правьте запись здесь, а вместе с ней `calibrated` и цифры в
`references/models.md` и `references/quality.md`. Новая дата без нового замера
превращает отчёт doctor в ложное успокоение.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Через сколько дней без замера маршрут считается непроверенным. Полгода —
# это примерно один релизный цикл заметной ASR-модели.
CALIBRATION_SHELF_LIFE_DAYS = 180

# Дата последнего полного прогона eval из references/quality.md.
LAST_CALIBRATION = date(2026, 9, 6)

_VERSION_IN_NAME = re.compile(r"(?<![a-z0-9])v(\d+)(?![a-z0-9])")


@dataclass(frozen=True)
class BackendSpec:
    family: str
    model: str
    quantization: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "family": self.family,
            "model": self.model,
            "quantization": self.quantization,
        }


@dataclass(frozen=True)
class ModelEntry:
    """Одна модель: чем её грузят, откуда качают и когда последний раз мерили."""

    key: str
    label: str
    role: str
    family: str
    model: str
    repo: str | None
    download_gb: float
    upstream: str
    calibrated: date | None = None
    calibration_note: str = ""
    quantization: str | None = None
    apple_only: bool = False
    in_hf_cache: bool = True
    # Коммит репозитория, на котором сделан замер. Имя модели веса не
    # определяет: репозиторий на хабе могут перезалить. Меняется только вместе
    # с новым замером — процедура в references/maintenance.md.
    revision: str | None = None

    @property
    def spec(self) -> BackendSpec:
        return BackendSpec(self.family, self.model, self.quantization)

    @property
    def cache_dir(self) -> str | None:
        """Каталог в кэше Hugging Face, который появляется после загрузки."""
        if not self.repo or not self.in_hf_cache:
            return None
        return "models--" + self.repo.replace("/", "--")

    def age_days(self, today: date | None = None) -> int | None:
        if self.calibrated is None:
            return None
        return ((today or date.today()) - self.calibrated).days

    def is_stale(self, today: date | None = None) -> bool:
        age = self.age_days(today)
        return age is not None and age > CALIBRATION_SHELF_LIFE_DAYS

    def newer_variants(self, candidates: object) -> list[str]:
        """Имена из `candidates` вида «то же самое, но версией выше».

        Для `gigaam-v3-e2e-rnnt` вернёт `gigaam-v4-e2e-rnnt`, если такое имя
        встретилось, и ничего — на `gigaam-v3-e2e-ctc` или `gigaam-v2-rnnt`.
        """
        return _newer_variants(self.model, candidates)

    def newer_repos(self, candidates: object) -> list[str]:
        if not self.repo:
            return []
        return _newer_variants(self.repo, candidates)


def _newer_variants(name: str, candidates: object) -> list[str]:
    match = _VERSION_IN_NAME.search(name)
    if match is None:
        return []
    current = int(match.group(1))
    pattern = re.compile(
        re.escape(name[: match.start()]) + r"v(\d+)" + re.escape(name[match.end() :]) + r"\Z"
    )
    found: dict[str, int] = {}
    for candidate in candidates or ():
        text = str(candidate)
        hit = pattern.match(text)
        if hit and int(hit.group(1)) > current:
            found[text] = int(hit.group(1))
    # Самая свежая версия первой: «v10» лексикографически меньше «v4».
    return sorted(found, key=lambda name: -found[name])


def search_term(repo: str) -> str:
    """Запрос к Hugging Face, покрывающий соседние версии того же репозитория."""
    name = repo.split("/")[-1]
    match = _VERSION_IN_NAME.search(name)
    if match is None:
        return name
    return name[: match.start()].rstrip("-_") or name


SILERO_VAD = ModelEntry(
    key="silero-vad",
    label="Silero VAD",
    role="общая нарезка окон для обеих ASR",
    family="vad",
    model="silero",
    repo="istupakov/silero-vad-onnx",
    revision="b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
    download_gb=0.01,
    upstream="Silero Team; ONNX-сборка Ильи Ступакова (istupakov), автора onnx-asr",
    calibration_note="границы окон общие для всех маршрутов, отдельного WER у VAD нет",
)

GIGAAM_RUSSIAN = ModelEntry(
    key="gigaam-ru",
    label="GigaAM v3 E2E RNNT",
    role="основная ASR для русского",
    family="gigaam",
    model="gigaam-v3-e2e-rnnt",
    repo="istupakov/gigaam-v3-onnx",
    revision="322c3b29492673eb7d0b434bfa9dfb8653e34d02",
    download_gb=0.9,
    upstream="GigaAM v3 — Сбер (SberDevices); ONNX-конвертация Ильи Ступакова (istupakov)",
    calibrated=LAST_CALIBRATION,
    calibration_note="WER 6,8% на десяти личных русских голосовых",
)

GIGAAM_MULTILINGUAL_FAST = ModelEntry(
    key="gigaam-multilingual",
    label="GigaAM Multilingual CTC int8",
    role="в маршрутах не участвует с 2026-09-20",
    family="gigaam",
    model="gigaam-multilingual-ctc",
    quantization="int8",
    repo="istupakov/gigaam-multilingual-ctc-onnx",
    revision="458860e1983aef670dd9795fb6af603c82767d5d",
    download_gb=0.2,
    upstream="GigaAM Multilingual — Сбер (SberDevices); ONNX-конвертация Ильи Ступакова",
    calibrated=date(2026, 9, 20),
    calibration_note=(
        "WER 29,2% на английском bake-off против 18,5% у основной: проверка, "
        "которая ошибается в полтора раза чаще проверяемого, отправляла в "
        "очередь 98% окон — заменена на Parakeet TDT v3"
    ),
)

PARAKEET_ENGLISH = ModelEntry(
    key="parakeet-en",
    label="Parakeet TDT 0.6B v3",
    role="независимая проверка для английского",
    family="onnx",
    model="nemo-parakeet-tdt-0.6b-v3",
    repo="istupakov/parakeet-tdt-0.6b-v3-onnx",
    revision="8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
    download_gb=2.4,
    upstream="Parakeet TDT 0.6B v3 — NVIDIA NeMo; ONNX-конвертация Ильи Ступакова (istupakov)",
    calibrated=date(2026, 9, 20),
    calibration_note=(
        "WER 18,2% на английском bake-off против 18,5% у основной — вровень; "
        "трансдьюсер против attention encoder-decoder у Whisper, поэтому "
        "ошибки независимы"
    ),
)

CANARY_ENGLISH = ModelEntry(
    key="canary-en",
    label="Canary 1B v2",
    role="взвешен как проверяющая 2026-09-20, не выбран",
    family="onnx",
    model="nemo-canary-1b-v2",
    repo="istupakov/canary-1b-v2-onnx",
    revision="5ebc1520cef7b6b318b3526ad17adbfe00bc1bfc",
    download_gb=3.7,
    upstream="Canary 1B v2 — NVIDIA NeMo; ONNX-конвертация Ильи Ступакова (istupakov)",
    calibrated=date(2026, 9, 20),
    calibration_note=(
        "WER 20,5% на английском bake-off — хуже Parakeet на 29 ошибок, вчетверо "
        "медленнее и той же архитектуры AED, что у Whisper: ошибки коррелируют "
        "с проверяемым. Очередь на часовом интервью 197 окон из 272 при проверке "
        "1597 с против 115 с у прежней"
    ),
)

WHISPER_TURBO = ModelEntry(
    key="whisper-turbo",
    label="Whisper large-v3-turbo (MLX)",
    role="основная ASR для английского, проверка для русского",
    family="whisper",
    model="mlx-community/whisper-large-v3-turbo-asr-fp16",
    repo="mlx-community/whisper-large-v3-turbo-asr-fp16",
    revision="624c19c9af5603fa73b83bce14d4aeea96156d18",
    download_gb=1.5,
    upstream="Whisper large-v3-turbo — OpenAI; сборка под MLX — mlx-community",
    calibrated=LAST_CALIBRATION,
    calibration_note=(
        "WER 14,5% на русских голосовых; на английском 18,5% "
        "(bake-off 2026-09-20), 28,3% на MINDS-14"
    ),
    apple_only=True,
)

VOSK_RUSSIAN = ModelEntry(
    key="vosk-ru",
    label="Vosk ru (alphacep)",
    role="лёгкая проверка для русского в режиме fast",
    family="onnx",
    model="alphacep/vosk-model-ru",
    repo="alphacep/vosk-model-ru",
    revision="df6a54a4d8e5d43e82675e4f5dba2d507731a0d1",
    download_gb=0.25,
    upstream="Alpha Cephei; запускается через onnx-asr",
    calibrated=date(2026, 9, 12),
    calibration_note=(
        "очередь на eval40: отмечено 37/57 окон, точность 70%, покрытие 98%, "
        "11 с против 234 с у Whisper; собственный WER 10,7% — только сигнал, не текст"
    ),
)

FASTER_WHISPER_FALLBACK = ModelEntry(
    key="faster-whisper-medium",
    label="faster-whisper medium (CPU)",
    role="запасной Whisper вне Apple Silicon",
    family="whisper-cpu",
    model="medium",
    repo="Systran/faster-whisper-medium",
    revision="08e178d48790749d25932bbc082711ddcfdfbc4f",
    download_gb=1.5,
    upstream="Whisper medium — OpenAI; сборка CTranslate2 — Systran",
    calibrated=LAST_CALIBRATION,
    calibration_note="WER 13,3% на русских голосовых; запасной путь, в маршрутах не участвует",
)

SORTFORMER = ModelEntry(
    key="sortformer",
    label="Sortformer 4spk (диаризация по явному флагу)",
    role="диаризация до четырёх голосов, только --diarization-backend sortformer",
    family="diarization",
    model="mlx-community/diar_sortformer_4spk-v1-fp16",
    repo="mlx-community/diar_sortformer_4spk-v1-fp16",
    revision="5233f9c657bf5e5535b7b79de040dab3ff9b905f",
    download_gb=0.2,
    upstream="Sortformer — NVIDIA NeMo; сборка под MLX — mlx-community",
    calibrated=LAST_CALIBRATION,
    calibration_note="DER 21,4% на 44 записях VoxConverse вдвоём — путает голоса",
    apple_only=True,
)

FLUIDAUDIO = ModelEntry(
    key="fluidaudio",
    label="FluidAudio Offline Community-1 (диаризация)",
    role="диаризация по умолчанию, любое число голосов",
    family="diarization",
    model="community-1",
    repo="FluidInference/speaker-diarization-coreml",
    revision="1ed7a662fdc7109e36d822db793ee6eebdaf8594",
    download_gb=0.034,
    upstream="FluidInference; CoreML-модели ставятся при первом --diarize",
    calibrated=LAST_CALIBRATION,
    calibration_note="DER 4,6% на 44 записях VoxConverse вдвоём, 8,8% на русских интервью",
    apple_only=True,
    in_hf_cache=False,
)

CATALOG: tuple[ModelEntry, ...] = (
    SILERO_VAD,
    GIGAAM_RUSSIAN,
    WHISPER_TURBO,
    GIGAAM_MULTILINGUAL_FAST,
    PARAKEET_ENGLISH,
    CANARY_ENGLISH,
    VOSK_RUSSIAN,
    FASTER_WHISPER_FALLBACK,
    SORTFORMER,
    FLUIDAUDIO,
)

BY_KEY: dict[str, ModelEntry] = {entry.key: entry for entry in CATALOG}


def entry_for_model(model: str, quantization: str | None = None) -> ModelEntry | None:
    for entry in CATALOG:
        if entry.model == model and entry.quantization == quantization:
            return entry
    return None


def route_entries(route: str, *, apple: bool = True, diarize: bool = False) -> tuple[ModelEntry, ...]:
    """Модели, которые качает маршрут. Порядок тот же, что в prefetch_models.py.

    Это же перечисление отвечает на вопрос «сколько весит установка», поэтому
    оно живёт здесь, а не в трёх текстах, которые разъезжаются по одному.
    """
    if route not in {"ru", "en", "all"}:
        raise ValueError(f"Неизвестный маршрут: {route}")
    entries = [SILERO_VAD]
    if route in ("ru", "all"):
        entries += [GIGAAM_RUSSIAN, VOSK_RUSSIAN]
    entries.append(WHISPER_TURBO if apple else FASTER_WHISPER_FALLBACK)
    if route in ("en", "all"):
        entries.append(PARAKEET_ENGLISH)
    if diarize and apple:
        entries.append(FLUIDAUDIO)
    return tuple(entries)


def route_download_gb(route: str, *, apple: bool = True, diarize: bool = False) -> float:
    return sum(entry.download_gb for entry in route_entries(route, apple=apple, diarize=diarize))


def format_gb(value: float) -> str:
    """«2,7 ГБ» — русской запятой, потому что число идёт в текст для пользователя.

    Всё, что меньше сотни мегабайт, печатается в мегабайтах: «0,0 ГБ» рядом с
    моделью читается как «размера нет», хотя это 34 МБ загрузки.
    """
    if value < 0.1:
        return f"{round(value * 1024):.0f} МБ"
    return f"{value:.1f} ГБ".replace(".", ",")


def hf_cache_dir() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 — huggingface_hub может быть не установлен
        return Path.home() / ".cache" / "huggingface" / "hub"


def fluidaudio_binary_path(
    explicit: str | Path | None = None,
    skill_dir: Path | None = None,
) -> Path | None:
    """Где лежит fluidaudiocli, или None. Единственный порядок поиска на скилл.

    Порядок: явный путь (`--fluidaudio-bin`) → `TRANSCRIBER_FLUIDAUDIO_BIN` →
    `PATH` → папка внутри скилла. Раньше поиск был написан дважды — в
    `diarization_worker` и в `doctor`, — и реализации разошлись: доктор смотрел
    только в папку скилла и печатал «✗» там, где диаризация работала.
    """
    if skill_dir is None:
        skill_dir = Path(__file__).resolve().parents[2]
    candidates = [
        explicit,
        os.environ.get("TRANSCRIBER_FLUIDAUDIO_BIN"),
        shutil.which("fluidaudiocli"),
        skill_dir / "bin" / "macos-arm64" / "fluidaudiocli",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def bundled_binary_problem(path: Path, skill_dir: Path | None = None) -> str | None:
    """Что не так с поставочным fluidaudiocli, или None.

    Бинарник лежит в репозитории готовым, а setup.sh снимает с него карантин
    macOS. Делать это вслепую нельзя: сначала сверяем SHA-256 с опубликованным
    в `fluidaudiocli.sha256` (тот же хеш — в third-party/fluidaudio/NOTICE.md).
    Сумма в том же репозитории защищает от порчи и подмены файла, но не от
    подмены репозитория: от этого — только своя сборка (`--fluidaudio-bin`).
    Свой путь не проверяется: его пользователь собрал или выбрал сам.
    """
    if skill_dir is None:
        skill_dir = Path(__file__).resolve().parents[2]
    bundled = skill_dir / "bin" / "macos-arm64" / "fluidaudiocli"
    if not bundled.is_file() or Path(path).resolve() != bundled.resolve():
        return None
    try:
        expected = bundled.with_name("fluidaudiocli.sha256").read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError):
        return "нет опубликованной контрольной суммы fluidaudiocli.sha256 рядом с бинарником"
    digest = hashlib.sha256()
    with bundled.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected.lower():
        return (
            f"поставочный fluidaudiocli не совпал с опубликованной суммой "
            f"({digest.hexdigest()[:12]} вместо {expected[:12]}); соберите свой по "
            "third-party/fluidaudio/NOTICE.md и передайте --fluidaudio-bin"
        )
    return None


UNPIN_ENV = "TRANSCRIBER_UNPIN"


def unpinned() -> bool:
    """`TRANSCRIBER_UNPIN=1` — грузить свежий `main` вместо закреплённых весов.

    Выход для экспериментов: попробовать перезалитую модель без релиза. Манифест
    такого прогона помечает веса `pinned: false`, а цифры из quality.md к нему
    уже не относятся.
    """
    return os.environ.get(UNPIN_ENV, "").strip() not in ("", "0")


def local_revision(entry: ModelEntry, cache: Path | None = None) -> str | None:
    """Последняя скачанная ревизия: коммит из `refs/main` локального кэша.

    С закреплением это уже не «веса прогона»: снапшот по коммиту `refs/main`
    не двигает. Нужна для doctor и для режима `TRANSCRIBER_UNPIN`.
    """
    cache_dir = entry.cache_dir
    if not cache_dir:
        return None
    ref = (cache or hf_cache_dir()) / cache_dir / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def expected_revision(entry: ModelEntry, cache: Path | None = None) -> str | None:
    """Ревизия, которой будет посчитан прогон. Она же идёт в ключ кэша.

    Закреплённая — всегда, кроме `TRANSCRIBER_UNPIN`: тогда то, что лежит в
    кэше как `main`, иначе гипотеза старых весов вернулась бы из кэша под
    видом новых.
    """
    if entry.revision and not unpinned():
        return entry.revision
    return local_revision(entry, cache)


def revision_for_model(model: str, quantization: str | None = None) -> str | None:
    entry = entry_for_model(model, quantization)
    return expected_revision(entry) if entry is not None else None


def revisions_for_models(models: Iterable[str], cache: Path | None = None) -> dict[str, str | None]:
    """{идентификатор модели: ревизия} для моделей, занятых в прогоне."""
    found: dict[str, str | None] = {}
    for model in models:
        entry = entry_for_model(model)
        if entry is not None:
            found[model] = expected_revision(entry, cache)
    return found


def staleness_warnings(models: Iterable[str], today: date | None = None) -> list[str]:
    """Офлайн-предупреждения об устаревании для моделей, занятых в прогоне.

    Сети не требует и потому вызывается на каждом запуске: иначе сигнал живёт
    только в doctor.py, который после установки никто не запускает. Полную
    картину, включая хаб, по-прежнему даёт `doctor.py --check-updates`.
    """
    today = today or date.today()
    entries = [entry for entry in (entry_for_model(model) for model in models) if entry]
    warnings: list[str] = []

    stale = [entry for entry in entries if entry.is_stale(today)]
    if stale:
        oldest = max(entry.age_days(today) or 0 for entry in stale)
        warnings.append(
            f"качество моделей не сверяли {oldest} дн. ({', '.join(entry.label for entry in stale)}); "
            "проверить, не вышло ли версии новее: doctor.py --check-updates"
        )

    names = known_asr_names()
    for entry in entries:
        newer = entry.newer_variants(names)
        if newer:
            warnings.append(
                f"установленная onnx-asr знает {newer[0]} — версия новее, чем {entry.model} "
                "в маршруте; замена требует своего замера WER"
            )
    return warnings


def known_asr_names() -> tuple[str, ...]:
    """Имена ASR-моделей, которые знает установленная onnx-asr.

    Сигнал об устаревании, не требующий сети: если библиотеку обновили и в ней
    появилась `gigaam-v4-*`, значит маршрут пора пересверять. Приватного API
    здесь нет, но структура чужая, поэтому любая ошибка означает «не знаю».
    """
    try:
        import typing

        from onnx_asr import loader

        names = getattr(loader, "AsrNames", None)
        return tuple(str(item) for item in typing.get_args(names)) if names else ()
    except Exception:  # noqa: BLE001 — библиотеки может не быть или её могли перестроить
        return ()
