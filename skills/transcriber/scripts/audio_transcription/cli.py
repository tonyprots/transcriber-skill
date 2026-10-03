from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import yaml

from . import __version__
from .audio import MediaToolError, prepare_audio, split_speech_windows
from .fetching import (
    FetchError,
    RemoteMedia,
    expand_playlist,
    fetch_audio,
    looks_like_url,
    probe_remote,
    public_url,
)
from .machine_lock import machine_slot
from .backends import BackendMissing, BackendUnavailable
from .catalog import (
    FASTER_WHISPER_FALLBACK,
    FLUIDAUDIO,
    SILERO_VAD,
    SORTFORMER,
    WHISPER_TURBO,
    revision_for_model,
    revisions_for_models,
    staleness_warnings,
)
from .cache import (
    CACHE_MAX_AGE_DAYS,
    crashed_windows,
    cache_key,
    load_diarization,
    load_hypothesis,
    save_diarization,
    save_hypothesis,
    prune as prune_cache,
)
from .diarization import split_chunks_by_diarization
from .glossary import (
    GlossaryEntry,
    apply_glossary,
    load_glossary,
    mark_verifier_only_terms,
    merge_glossaries,
    suggest_glossary_matches,
    take_verifier_canonicals,
)
from .glossary_store import (
    entries_of,
    load_document,
    profile_path,
    record_fingerprint,
    store_path,
    update_from_run,
)
from .fillers import strip_filler_segments
from .mining import undisputed_words
from .models import AudioChunk, Diarization, Hypothesis, ReviewItem, Section
from .packing import PackedWindow, build_packed_windows, unpack_hypothesis
from .isolated import run_isolated_backend, run_isolated_diarization
from .reconcile import find_review_items, normalize_text, segments_for_windows
from .render import ensure_disk_space, ensure_output_available, parse_speaker_names, write_bundle
from .routing import (
    AUTO_LANGUAGE,
    BackendSpec,
    diarization_backend,
    language_route,
    normalize_language,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Локальная расшифровка аудио с независимой сверкой гипотез.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"transcriber {__version__}",
        help="Версия скилла — та же, что уходит в manifest.json",
    )
    parser.add_argument(
        "input",
        nargs="+",
        help=(
            "Аудио- или видеофайл либо ссылка (видео — через yt-dlp, выпуск "
            "подкаста Яндекс Музыки — напрямую). Несколько — пакет: по очереди, "
            "каждый в свой каталог внутри --output"
        ),
    )
    parser.add_argument(
        "--playlist",
        action="store_true",
        help=(
            "Ссылка на плейлист или канал разворачивается в ролики, и они идут "
            "пакетом. Без флага берётся только сам ролик"
        ),
    )
    parser.add_argument(
        "--playlist-limit",
        type=int,
        metavar="N",
        help="С --playlist: только первые N роликов (у канала их бывают тысячи)",
    )
    parser.add_argument(
        "--section",
        action="append",
        type=_section_argument,
        metavar="НАЧАЛО-КОНЕЦ",
        help=(
            "Расшифровать только кусок: 00:10:50-00:11:30, 10:50-11:30 или "
            "650-690. Можно повторить — запись скачается один раз. Таймкоды "
            "результата — по исходнику"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "max"),
        default="max",
        help=(
            "max (по умолчанию) — обе модели маршрута на каждом окне; "
            "fast — одна основная, на русском с лёгкой проверяющей"
        ),
    )
    parser.add_argument(
        "--language",
        default="auto",
        help=(
            "Код языка (ru, en, …) или auto (по умолчанию): у ссылки язык берётся "
            "из её описания, у файла Whisper определяет его по трём окнам речи"
        ),
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        help="Свой YAML-словарь терминов: читается дополнительно к тому, что скилл ведёт сам",
    )
    parser.add_argument(
        "--glossary-store",
        type=Path,
        help=(
            "Где лежит словарь, который скилл ведёт сам "
            "(по умолчанию ~/.transcriber/glossary.yaml, можно задать "
            "переменной TRANSCRIBER_GLOSSARY_STORE)"
        ),
    )
    parser.add_argument(
        "--glossary-profile",
        help=(
            "Именованный свой словарь: ~/.transcriber/glossaries/ИМЯ.yaml вместо общего. "
            "Чтобы рабочие и личные записи не учили друг друга "
            "(можно задать переменной TRANSCRIBER_GLOSSARY_PROFILE)"
        ),
    )
    parser.add_argument(
        "--no-learn",
        action="store_true",
        help="Не пополнять свой словарь по итогам прогона (читать — по-прежнему)",
    )
    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="Прогон вообще без словарей: ни читать, ни пополнять",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Каталог результата; создаётся скриптом, непустой требует --overwrite. "
            "В пакете — общий родитель, по умолчанию текущий каталог"
        ),
    )
    parser.add_argument(
        "--speech-window-seconds",
        type=float,
        default=20.0,
        help=(
            "Потолок общего речевого окна: по нему режут обе модели маршрута. "
            "Длиннее — больше контекста у Whisper, но и больше цена ошибки в окне"
        ),
    )
    parser.add_argument(
        "--verifier-window-seconds",
        type=float,
        default=24.0,
        help=(
            "Длина окна проверочной модели: короткие окна склеиваются, "
            "потому что Whisper платит за вызов, а не за секунду. 0 — не склеивать."
        ),
    )
    parser.add_argument(
        "--whisper-backend",
        choices=("auto", "mlx", "faster"),
        default="auto",
        help="auto выбирает MLX на Apple Silicon и CPU-fallback иначе",
    )
    parser.add_argument("--whisper-model", help="Переопределить модель Whisper")
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.82,
        help=(
            "Сходство гипотез, ниже которого окно уходит в очередь проверки "
            "(по умолчанию 0.82). Числа, имена и термины попадают туда в любом случае"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Не ходить на Hugging Face: работать только на том, что уже в кэше",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Разрешить непустой каталог результата; прежний уезжает в архивный рядом",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help=(
            "Пропустить запись, у которой каталог результата уже содержит "
            "manifest.json: так продолжают прерванную пачку, не скачивая готовое заново"
        ),
    )
    parser.add_argument(
        "--early-text",
        type=Path,
        help=(
            "Записать сюда читаемый текст сразу после основной модели, не дожидаясь "
            "проверяющей: она текст не меняет и досчитывается ради очереди и словаря"
        ),
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Оставить подготовленный prepared.wav в каталоге результата (плюс размер записи)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "transcriber",
        help="Каталог кэша гипотез",
    )
    parser.add_argument(
        "--cache-max-age-days",
        type=float,
        default=CACHE_MAX_AGE_DAYS,
        help=(
            "Перед прогоном убрать из кэша записи, к которым не обращались дольше "
            f"стольких дней (по умолчанию {CACHE_MAX_AGE_DAYS}; 0 — не убирать)"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Считать заново, не читая и не записывая кэш гипотез: для контрольного "
            "прогона и для записей, текст которых не должен оставаться на диске"
        ),
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Определить говорящих и разрезать общие VAD-окна по сменам спикера",
    )
    parser.add_argument(
        "--speaker-names",
        type=_speaker_names_argument,
        metavar="1=ИМЯ,2=ИМЯ",
        help=(
            "Имена вместо «Спикер N» в текстах и субтитрах; требует --diarize. "
            "Готовый результат переименовывает scripts/rename_speakers.py"
        ),
    )
    parser.add_argument(
        "--diarization-model",
        default=SORTFORMER.model,
        help="Модель диаризации Sortformer; FluidAudio выбирается отдельным флагом",
    )
    parser.add_argument(
        "--diarization-backend",
        choices=("auto", "sortformer", "fluidaudio"),
        default="auto",
        help="auto выбирает FluidAudio только для ожидаемых 5+ голосов",
    )
    parser.add_argument(
        "--expected-speakers",
        type=int,
        help="Ожидаемое число участников; включает безопасный выбор backend",
    )
    parser.add_argument(
        "--fluidaudio-bin",
        type=Path,
        help="Путь к fluidaudiocli для диаризации больших встреч",
    )
    parser.add_argument(
        "--diarization-threshold",
        type=float,
        help="Порог backend: по умолчанию 0.4 для Sortformer и 0.8 для FluidAudio",
    )
    parser.add_argument(
        "--diarization-chunk-seconds",
        type=float,
        default=30.0,
        help="Длина куска, которым запись подаётся в диаризацию (по умолчанию 30)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Не печатать ход работы в stderr",
    )
    parser.add_argument(
        "--backend-stall-timeout",
        type=float,
        default=300.0,
        help="Остановить ASR, если он не завершил ни одного окна за это число секунд",
    )
    return parser


def _speaker_names_argument(text: str) -> dict[str, str]:
    try:
        return parse_speaker_names(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _section_argument(text: str) -> Section:
    try:
        return Section.parse(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


# Отказы, которые означают «эта запись не получилась», а не ошибку в коде.
# В пакете они записываются в результат и не останавливают остальные записи.
FAILURES = (
    BackendUnavailable,
    FetchError,
    FileNotFoundError,
    FileExistsError,
    MediaToolError,
    ValueError,
    OSError,
    yaml.YAMLError,
)


class ProgressReporter:
    """Признаки жизни в stderr: без них фоновый прогон молчит до самого конца.

    stdout остаётся чистым — там только итоговый JSON с путём результата.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        stream: Any = None,
        interval_seconds: float = 20.0,
    ) -> None:
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.interval_seconds = interval_seconds

    def _emit(self, text: str) -> None:
        if not self.enabled:
            return
        print(f"[{time.strftime('%H:%M:%S')}] {text}", file=self.stream, flush=True)

    def stage(self, text: str) -> None:
        self._emit(text)

    @contextmanager
    def waiting(self, label: str, *, every_seconds: float = 60.0) -> Iterator[None]:
        """Признак жизни на этапе, у которого нет прогресса по окнам.

        Загрузка модели и нарезка окон по говорящим идут единым куском:
        на 42-минутной встрече нарезка молчит 7–8 минут. Такая тишина
        неотличима от зависания — 2026-09-17 её разбирали через `ps`, хотя
        прогон был жив. Теперь этап сам говорит, сколько уже идёт.
        """
        if not self.enabled:
            yield
            return
        done = threading.Event()
        started = time.monotonic()

        def tick() -> None:
            while not done.wait(every_seconds):
                self._emit(f"{label}: идёт {time.monotonic() - started:.0f} с")

        watcher = threading.Thread(target=tick, name=f"waiting:{label}", daemon=True)
        watcher.start()
        try:
            yield
        finally:
            done.set()
            watcher.join(timeout=1.0)

    def windows(self, label: str) -> Callable[[dict[str, Any]], None] | None:
        """Колбэк прогресса по окнам: первое, последнее и не чаще интервала."""
        if not self.enabled:
            return None
        last = [0.0]

        def report(payload: dict[str, Any]) -> None:
            completed = payload.get("completed")
            total = payload.get("total")
            now = time.monotonic()
            final = isinstance(completed, int) and completed == total
            if last[0] and not final and now - last[0] < self.interval_seconds:
                return
            last[0] = now
            if isinstance(total, int) and total:
                self._emit(f"{label}: окно {completed}/{total}")
            else:
                self._emit(f"{label}: {payload.get('stage', 'работает')}")

        return report


def run_whisper(
    args: argparse.Namespace,
    chunks: list[AudioChunk],
    hotwords: list[str],
    work_dir: Path,
    default_model: str,
    language: str,
    report: ProgressReporter,
) -> Hypothesis:
    mlx_model = args.whisper_model or default_model
    if args.whisper_backend in {"auto", "mlx"}:
        try:
            return run_isolated_backend(
                "mlx-whisper",
                chunks,
                language,
                work_dir,
                config={
                    "model_name": mlx_model,
                    "offline": args.offline,
                    "hotwords": hotwords,
                },
                stall_timeout_seconds=args.backend_stall_timeout,
                on_progress=report.windows(f"Whisper {mlx_model}"),
            )
        except BackendMissing:
            # Нет mlx-audio или не Apple Silicon — единственный повод идти на CPU.
            # Сбой в работе или watchdog пробрасываются: молча повторять часы
            # распознавания на медленной модели нельзя.
            if args.whisper_backend == "mlx":
                raise
            report.stage("Whisper Turbo MLX недоступен, переход на faster-whisper medium (CPU, медленнее)")
    # Переопределение модели относится к MLX-репозиторию; CTranslate2-модель у fallback своя.
    fallback = FASTER_WHISPER_FALLBACK.model
    faster_model = fallback if args.whisper_backend == "auto" else (args.whisper_model or fallback)
    return run_isolated_backend(
        "faster-whisper",
        chunks,
        language,
        work_dir,
        config={
            "model_name": faster_model,
            "offline": args.offline,
            "hotwords": hotwords,
        },
        stall_timeout_seconds=args.backend_stall_timeout,
        on_progress=report.windows(f"faster-whisper {faster_model}"),
    )


def weights_signature(
    args: argparse.Namespace, spec: BackendSpec | None = None
) -> dict[str, str | None]:
    """Ревизии весов, от которых зависит гипотеза, — для ключа кэша.

    Без них перезалитая модель возвращала из кэша гипотезу старых весов, а
    манифест при этом называл новые. VAD входит всегда: от него зависят окна.
    Для Whisper заранее неизвестно, поднимется MLX или CPU-fallback, поэтому
    в подпись идут обе модели.
    """
    signature = {SILERO_VAD.model: revision_for_model(SILERO_VAD.model)}
    if spec is None or spec.family == "whisper":
        models = [args.whisper_model or (spec.model if spec else WHISPER_TURBO.model)]
        models.append(FASTER_WHISPER_FALLBACK.model)
        signature.update({model: revision_for_model(model) for model in models})
    else:
        signature[spec.model] = revision_for_model(spec.model, spec.quantization)
    return signature


def weights_as_keyed(metadata: dict[str, Any], signature: dict[str, str | None]) -> bool:
    """Посчитано ли теми весами, что записаны в ключ. Нет — в кэш не кладём.

    Расходятся они, когда закреплённый коммит пропал с Hub и `weights.fetch`
    взял `main`: такая гипотеза под ключом закреплённых весов вернулась бы
    потом под чужим именем.
    """
    weights = metadata.get("weights")
    return not weights or weights.get("revision") in signature.values()


def loaded_revisions(
    expected: dict[str, str | None],
    results: list[Any],
    warnings: list[str],
) -> dict[str, str | None]:
    """Ревизии для манифеста: какими весами посчитано на самом деле.

    Ожидаемые берутся из каталога, но воркер сообщает фактические — они
    расходятся при `TRANSCRIBER_UNPIN` и при пропавшем с Hub коммите. Такой
    прогон в манифесте помечается предупреждением: цифры quality.md к нему не
    относятся.
    """
    found = dict(expected)
    for result in results:
        weights = (getattr(result, "metadata", None) or {}).get("weights")
        if not weights or not weights.get("revision"):
            continue
        found[result.model] = weights["revision"]
        if not weights.get("pinned"):
            text = (
                f"{result.model}: веса не закреплённые ({str(weights['revision'])[:12]}); "
                "замеры качества из references/quality.md к этому прогону не относятся"
            )
            if text not in warnings:
                warnings.append(text)
    return found


def run_route_backend(
    args: argparse.Namespace,
    spec: BackendSpec,
    chunks: list[AudioChunk],
    hotwords: list[str],
    work_dir: Path,
    *,
    role: str,
    language: str,
    report: ProgressReporter,
) -> Hypothesis:
    if spec.family == "whisper":
        hypothesis = run_whisper(
            args,
            chunks,
            hotwords,
            work_dir,
            spec.model,
            language,
            report,
        )
    elif spec.family in {"gigaam", "onnx"}:
        # Один и тот же воркер: GigaAM и Vosk обе поднимаются через onnx-asr,
        # различает их только имя модели. Семейства разведены, чтобы в отчёте
        # и в каталоге не называть Vosk гигаамом.
        hypothesis = run_isolated_backend(
            "gigaam",
            chunks,
            language,
            work_dir,
            config={
                "model_name": spec.model,
                "quantization": spec.quantization,
                "offline": args.offline,
            },
            stall_timeout_seconds=args.backend_stall_timeout,
            on_progress=report.windows(
                f"GigaAM {spec.model}" if spec.family == "gigaam" else spec.model
            ),
        )
    else:
        raise ValueError(f"Неизвестное семейство ASR: {spec.family}")
    hypothesis.metadata["role"] = role
    return hypothesis


def canonicalized(hypothesis: Hypothesis, glossary: list[GlossaryEntry]) -> Hypothesis:
    """Копия гипотезы с применёнными автозаменами — только для сверки.

    В выдачу и в аудит-след идёт исходный текст модели: `hypotheses` остаётся
    тем, что она сказала на самом деле. Здесь нужно другое — чтобы термин,
    который словарь и так исправляет автоматически, не выглядел расхождением
    моделей. На eval40 состав очереди от этого не меняется (замер 2026-09-12),
    зато вторая версия перестаёт показывать заведомо чужое написание.
    """
    segments, _ = apply_glossary(hypothesis.segments, glossary)
    return Hypothesis(
        model=hypothesis.model,
        language=hypothesis.language,
        elapsed_seconds=hypothesis.elapsed_seconds,
        segments=segments,
        metadata=hypothesis.metadata,
    )


def verifier_failure_items(
    chunks: list[AudioChunk], primary: Hypothesis
) -> list[ReviewItem]:
    chunk_text = {
        int(item["sequence"]): str(item.get("text", ""))
        for item in primary.metadata.get("chunks", [])
    }
    return [
        ReviewItem(
            start=chunk.start,
            end=chunk.end,
            readable_text=chunk_text.get(chunk.sequence, ""),
            comparison_text="",
            similarity=0.0,
            differing_tokens=(),
            reason="Независимый verifier недоступен",
        )
        for chunk in chunks
    ]


def primary_retry_items(primary: Hypothesis) -> list[ReviewItem]:
    items = []
    for chunk in primary.metadata.get("chunks", []):
        status = str(chunk.get("status", "ok"))
        if status not in {"failed", "partial"}:
            continue
        reason = (
            f"Основная модель {primary.model} не распознала окно после retry-split"
            if status == "failed"
            else f"Основная модель {primary.model} распознала окно после retry-split только частично"
        )
        items.append(
            ReviewItem(
                start=float(chunk["start"]),
                end=float(chunk["end"]),
                readable_text=str(chunk.get("text", "")),
                comparison_text="",
                similarity=0.0,
                differing_tokens=(),
                reason=reason,
            )
        )
    return items


# Доля окон, которую основная модель может не распознать без тревоги. На 388
# прошлых прогонах пустой ответ давал до 10,3 % окон (шум на встрече), поэтому
# порог вдвое выше. Падение на исключении тревожно при любой доле: в тех же
# прогонах его не было ни разу.
FAILED_SHARE_WARNING = 0.2


def primary_failure_warning(primary: Hypothesis) -> str | None:
    """Предупреждение, если основная модель провалила заметную часть окон.

    Без него провал выглядел как успех: 2026-09-23 все 608 окон упали, а
    `warnings` был пуст, и о пустой расшифровке говорили только 608 пунктов
    очереди.
    """
    chunks = [chunk for chunk in primary.metadata.get("chunks", []) if isinstance(chunk, dict)]
    if not chunks:
        return None
    failed = sum(chunk.get("status") == "failed" for chunk in chunks)
    crashes = crashed_windows(primary)
    if not crashes and failed / len(chunks) < FAILED_SHARE_WARNING:
        return None
    text = (
        f"Основная модель {primary.model} не распознала {failed} из {len(chunks)} "
        f"окон ({failed / len(chunks):.0%}); расшифровка неполна"
    )
    if crashes:
        text += (
            f". Окон со сбоем среды: {len(crashes)}, первая ошибка: {crashes[0]}. "
            "В кэш результат не сохранён — повторный прогон посчитает их заново"
        )
    return text


# Из скольких окон речи судить о языке: начало, середина и конец. Заставка
# или вступление на другом языке не должны решать за всю запись.
LANGUAGE_SAMPLE_WINDOWS = 3
# Ниже этой доли у первого языка решение ненадёжно — о нём надо сказать.
LANGUAGE_CONFIDENCE_WARNING = 0.6


def language_sample(chunks: list[AudioChunk], count: int = LANGUAGE_SAMPLE_WINDOWS) -> list[AudioChunk]:
    if len(chunks) <= count:
        return list(chunks)
    step = (len(chunks) - 1) / (count - 1)
    return [chunks[round(index * step)] for index in range(count)]


def resolve_language(
    args: argparse.Namespace,
    *,
    remote: RemoteMedia | None,
    chunks: list[AudioChunk],
    work_dir: Path,
    source_identity: str,
    report: ProgressReporter,
) -> tuple[str, dict[str, Any]]:
    """Язык записи и то, откуда он известен.

    До 0.16 язык по умолчанию был русским, и английская запись без
    `--language en` молча уходила в GigaAM, которая отдавала мусор.
    """
    requested = normalize_language(args.language)
    if requested != AUTO_LANGUAGE:
        return requested, {"method": "argument", "language": requested}
    if remote is not None and remote.language:
        language = normalize_language(remote.language)
        if language != AUTO_LANGUAGE:
            report.stage(f"Язык из описания ссылки: {language}")
            return language, {"method": "source-metadata", "language": language}
    if not chunks:
        return "ru", {"method": "default", "language": "ru"}
    sample = language_sample(chunks)
    key = cache_key(
        "language",
        {
            "pipeline": "1",
            "source_sha256": source_identity,
            "whisper_backend": args.whisper_backend,
            "weights": weights_signature(args),
            "sample": [[chunk.sequence, round(chunk.start, 6), round(chunk.end, 6)] for chunk in sample],
        },
    )
    found = None if args.no_cache else load_hypothesis(args.cache_dir, key)
    if found is None:
        report.stage(f"Определяю язык, окон речи для пробы: {len(sample)}")
        try:
            found = _run_language_id(args, sample, work_dir, report)
        except BackendUnavailable as error:
            return "ru", {
                "method": "default",
                "language": "ru",
                "warning": (
                    f"Язык определить не удалось ({error}); взят русский. "
                    "Если запись на другом языке, повторите с --language"
                ),
            }
        if not args.no_cache and weights_as_keyed(found.metadata, weights_signature(args)):
            save_hypothesis(args.cache_dir, key, found, source_identity)
    language = normalize_language(found.language)
    if language in {AUTO_LANGUAGE, "und"}:
        return "ru", {
            "method": "default",
            "language": "ru",
            "warning": "Whisper не назвал язык; взят русский. Если запись на другом языке, повторите с --language",
        }
    probability = float(found.metadata.get("language_probability") or 0.0)
    detection: dict[str, Any] = {
        "method": "whisper-language-id",
        "language": language,
        "probability": probability,
        "model": found.model,
        "top_languages": found.metadata.get("top_languages", []),
        "windows": found.metadata.get("windows"),
    }
    report.stage(f"Язык: {language} ({probability:.0%})")
    if probability < LANGUAGE_CONFIDENCE_WARNING:
        others = ", ".join(
            f"{code} {share:.0%}" for code, share in detection["top_languages"][:3]
        )
        detection["warning"] = (
            f"Язык определён неуверенно ({others}); если выбран не тот, "
            "повторите с --language"
        )
    return language, detection


def _run_language_id(
    args: argparse.Namespace,
    sample: list[AudioChunk],
    work_dir: Path,
    report: ProgressReporter,
) -> Hypothesis:
    """Тот же выбор, что у `run_whisper`: MLX на Apple Silicon, иначе CPU."""
    if args.whisper_backend in {"auto", "mlx"}:
        try:
            return run_isolated_backend(
                "mlx-whisper-lid",
                sample,
                AUTO_LANGUAGE,
                work_dir,
                config={"model_name": args.whisper_model or WHISPER_TURBO.model, "offline": args.offline},
                stall_timeout_seconds=args.backend_stall_timeout,
                on_progress=report.windows("Определение языка"),
            )
        except BackendMissing:
            if args.whisper_backend == "mlx":
                raise
    fallback = FASTER_WHISPER_FALLBACK.model
    return run_isolated_backend(
        "faster-whisper-lid",
        sample,
        AUTO_LANGUAGE,
        work_dir,
        config={
            "model_name": fallback if args.whisper_backend == "auto" else (args.whisper_model or fallback),
            "offline": args.offline,
        },
        stall_timeout_seconds=args.backend_stall_timeout,
        on_progress=report.windows("Определение языка"),
    )


def _slug(title: str) -> str:
    """Имя каталога из заголовка видео: только то, что безопасно в пути."""
    cleaned = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip().lower()
    return re.sub(r"[\s_-]+", "-", cleaned)[:80] or "video"


def _spoken_duration(remote: RemoteMedia) -> str:
    if not remote.duration_seconds:
        return "длительность неизвестна"
    return f"{remote.duration_seconds / 60:.0f} мин"


def _validate(args: argparse.Namespace) -> None:
    if args.early_text is not None and len(args.input) * len(_sections(args)) > 1:
        raise ValueError("--early-text пишет один файл и годится только для одной записи")
    if args.glossary_profile:
        profile_path(args.glossary_profile)  # плохое имя — до часа расшифровки, а не после
    if not 0 <= args.review_threshold <= 1:
        raise ValueError("--review-threshold должен быть в диапазоне [0, 1]")
    if args.verifier_window_seconds < 0:
        raise ValueError("--verifier-window-seconds не может быть отрицательным")
    if args.backend_stall_timeout <= 0:
        raise ValueError("--backend-stall-timeout должен быть положительным")
    if (
        args.diarization_threshold is not None
        and not 0 <= args.diarization_threshold <= 1
    ):
        raise ValueError("--diarization-threshold должен быть в диапазоне [0, 1]")
    if args.diarization_chunk_seconds <= 0:
        raise ValueError("--diarization-chunk-seconds должен быть положительным")
    if args.expected_speakers is not None and not args.diarize:
        raise ValueError("--expected-speakers требует --diarize")
    if args.speaker_names and not args.diarize:
        raise ValueError("--speaker-names требует --diarize: без него говорящих не различить")
    if args.diarize and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        raise ValueError(
            "--diarize работает только на macOS Apple Silicon: Sortformer требует MLX, "
            "FluidAudio собран для macOS arm64. Запустите без --diarize."
        )


def _sections(args: argparse.Namespace) -> list[Section | None]:
    return list(args.section or []) or [None]


def _resolve_input(raw: str) -> tuple[RemoteMedia | None, Path | None]:
    """Ссылку опрашиваем до скачивания: занятый каталог результата или
    неверный язык должны падать раньше, чем уедут мегабайты."""
    raw = str(raw).strip()
    if looks_like_url(raw):
        return probe_remote(raw), None
    return None, Path(raw).expanduser().resolve()


def _output_stem(remote: RemoteMedia | None, source: Path | None, section: Section | None) -> str:
    stem = _slug(remote.title) if remote else source.stem  # type: ignore[union-attr]
    return f"{stem}-{section.slug}" if section else stem


def _download(remote: RemoteMedia, dest: Path, report: "ProgressReporter") -> tuple[Path, float]:
    report.stage(f"Скачиваю аудио: {remote.title} ({_spoken_duration(remote)})")
    started = time.monotonic()
    path = fetch_audio(remote.url, dest)
    return path, round(time.monotonic() - started, 2)


def run(args: argparse.Namespace) -> Path:
    """Одна запись, один результат. Пакет — `run_batch`."""
    sections = _sections(args)
    if len(args.input) != 1 or len(sections) != 1:
        raise ValueError("Несколько записей или фрагментов обрабатывает run_batch()")
    _validate(args)
    remote, source = _resolve_input(args.input[0])
    section = sections[0]
    if args.output:
        output = args.output.expanduser().resolve()
    elif remote:
        output = Path(f"{_output_stem(remote, None, section)}.transcript").resolve()
    else:
        output = source.with_name(f"{_output_stem(None, source, section)}.transcript")  # type: ignore[union-attr]
    report = ProgressReporter(enabled=not args.quiet)
    if args.skip_done and _is_done(output):
        report.stage(f"Уже расшифровано, пропускаю: {output}")
        return output
    output = ensure_output_available(output, overwrite=args.overwrite)
    with scratch_dir("transcriber-download-") as downloads:
        download_seconds = None
        if remote is not None:
            source, download_seconds = _download(remote, downloads, report)
        return _transcribe(
            args,
            source=source,  # type: ignore[arg-type]
            remote=remote,
            section=section,
            output=output,
            report=report,
            download_seconds=download_seconds,
        )


def run_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Несколько записей и/или фрагментов подряд, каждая в свой каталог.

    2026-09-22 очередь из десятка роликов две сессии собирали вручную:
    `nohup`, `run.sh`, файлы-флаги и `sleep` в цикле. Здесь то же самое без
    обвязки: сначала опрашиваются все ссылки и проверяются все каталоги —
    конфликт имён виден до первого скачанного байта, — затем каждая запись
    скачивается один раз и расшифровывается по каждому фрагменту. Отказ одной
    записи попадает в результат и не останавливает остальные.
    """
    _validate(args)
    sections = _sections(args)
    base = (args.output or Path.cwd()).expanduser().resolve()
    report = ProgressReporter(enabled=not args.quiet)
    results: list[dict[str, Any]] = []
    plan: list[tuple[str, RemoteMedia | None, Path | None, list[tuple[Section | None, Path]]]] = []
    taken: set[Path] = set()
    for raw in args.input:
        try:
            remote, source = _resolve_input(raw)
        except FAILURES as error:
            report.stage(f"Пропускаю {raw}: {error}")
            results.extend(_failure(raw, section, error) for section in sections)
            continue
        jobs = []
        for section in sections:
            name = _output_stem(remote, source, section)
            candidate, attempt = base / name, 2
            while candidate in taken:
                candidate, attempt = base / f"{name}-{attempt}", attempt + 1
            taken.add(candidate)
            if args.skip_done and _is_done(candidate):
                report.stage(f"Уже расшифровано, пропускаю: {candidate.name}")
                results.append({"input": raw, "skipped": True, **summarize(candidate)})
                continue
            # Занятый каталог — отказ одной записи, а не всей пачки: раньше
            # один такой каталог ронял пакет до первого скачанного байта.
            try:
                jobs.append((section, ensure_output_available(candidate, overwrite=args.overwrite)))
            except FAILURES as error:
                report.stage(f"Пропускаю {candidate.name}: {error}")
                results.append(_failure(raw, section, error))
        if jobs:
            plan.append((raw, remote, source, jobs))
    total = sum(len(jobs) for *_, jobs in plan)
    done = 0
    for raw, remote, source, jobs in plan:
        with scratch_dir("transcriber-download-") as downloads:
            download_seconds = None
            if remote is not None:
                try:
                    source, download_seconds = _download(remote, downloads, report)
                except FAILURES as error:
                    report.stage(f"Не скачалось {raw}: {error}")
                    results.extend(_failure(raw, section, error) for section, _ in jobs)
                    done += len(jobs)
                    continue
            for section, output in jobs:
                done += 1
                report.stage(f"Пакет {done}/{total}: {output.name}")
                try:
                    path = _transcribe(
                        args,
                        source=source,  # type: ignore[arg-type]
                        remote=remote,
                        section=section,
                        output=output,
                        report=report,
                        # Скачивание одно на все фрагменты — числится за первым.
                        download_seconds=download_seconds,
                    )
                except FAILURES as error:
                    report.stage(f"Ошибка на {output.name}: {error}")
                    results.append(_failure(raw, section, error))
                else:
                    results.append({"input": raw, **summarize(path)})
                download_seconds = None
    return results


def _is_done(output: Path) -> bool:
    """Готов тот каталог, где есть манифест: он пишется последним."""
    return (output.expanduser() / "manifest.json").is_file()


def _failure(raw: str, section: Section | None, error: BaseException) -> dict[str, Any]:
    return {
        "input": raw,
        "section": section.to_dict() if section else None,
        "error": str(error),
    }


def summarize(output: Path) -> dict[str, Any]:
    """Что печатается в stdout: пути к файлам и главные счётчики.

    Одного каталога мало: 2026-09-22 агент сам угадывал разметку
    `readable.md`, резал её по разделителю, которого там нет, получал пустоту
    и перезапускал расшифровку. Пути и счётчики снимают догадки.
    """
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    source = manifest.get("source") or {}
    return {
        "output": str(output),
        "readable": str(output / "readable.md"),
        "verbatim": str(output / "verbatim.md"),
        "review": str(output / "review-needed.md"),
        "manifest": str(output / "manifest.json"),
        "section": source.get("section"),
        "duration_seconds": source.get("duration_seconds"),
        "review_count": manifest.get("review_count"),
        "speaker_count": manifest.get("speaker_count"),
        "warnings": manifest.get("warnings") or [],
    }


@contextmanager
def scratch_dir(prefix: str) -> Iterator[Path]:
    """Временный каталог, который убирается не на глазах у вызывающего.

    Антивирус проверяет каждое удаление, и уборка сотен WAV растягивалась до
    десяти минут после готового результата (`.memory/defender-slow-temp-cleanup.md`).
    Всё это время stdout молчал, Handy не получал текст, а следующая запись
    пачки ждала. Удаление уходит отдельному процессу вне нашей сессии: он
    переживает выход интерпретатора и никого не задерживает.
    """
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        try:
            subprocess.Popen(
                ["rm", "-rf", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            shutil.rmtree(path, ignore_errors=True)


def write_plain_text(path: Path, segments: list[Any]) -> None:
    """Читаемый текст без заголовков и таймкодов — для буфера обмена и агента.

    Сегмент — окно речи, и окно режет фразу где придётся, поэтому абзац
    продолжается, пока предыдущий кусок не кончится концом предложения.
    """
    text = ""
    for segment in segments:
        chunk = segment.text.strip()
        if not chunk:
            continue
        if not text:
            text = chunk
        elif text.endswith((".", "!", "?", "…")):
            text += "\n\n" + chunk
        else:
            text += " " + chunk
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def disputed_primary_words(review_items: list[ReviewItem]) -> set[str]:
    """Слова основной модели, на которых проверяющая с ней разошлась."""
    words: set[str] = set()
    for item in review_items:
        for pair in item.differing_tokens:
            words.update(normalize_text(pair.split(" / ", 1)[0]).split())
    return words


def _transcribe(
    args: argparse.Namespace,
    *,
    source: Path,
    remote: RemoteMedia | None,
    section: Section | None,
    output: Path,
    report: "ProgressReporter",
    download_seconds: float | None,
) -> Path:
    diarization_kind = (
        diarization_backend(args.diarization_backend, args.expected_speakers)
        if args.diarize
        else None
    )
    diarization_threshold = args.diarization_threshold
    if diarization_threshold is None:
        diarization_threshold = 0.8 if diarization_kind == "fluidaudio" else 0.4

    if not args.no_cache and args.cache_max_age_days > 0:
        # Уборка — побочное дело прогона: упасть из-за неё он не должен.
        try:
            swept = prune_cache(args.cache_dir, args.cache_max_age_days)
        except OSError:
            swept = None
        if swept and swept.files:
            report.stage(
                f"Кэш: убрано записей старше {args.cache_max_age_days:g} дн.: {swept.files}"
            )

    # Скачивание в total не входит: это сеть, а не расшифровка, и по total
    # сравнивают скорость прогонов между собой.
    timings: dict[str, float] = {}
    if download_seconds is not None:
        timings["download"] = download_seconds
    started_at = time.monotonic()

    def timed(stage: str, since: float) -> float:
        timings[stage] = round(time.monotonic() - since, 2)
        return time.monotonic()

    with scratch_dir("transcriber-asr-") as work_dir, ExitStack() as held:
        scope = f", фрагмент {section.label}" if section else ""
        report.stage(f"Подготовка аудио: {source.name}{scope}")
        prepared, media = prepare_audio(source, work_dir, section)
        if remote is not None:
            media = replace(media, origin=public_url(remote.url))
        # Кэш гипотез ключуется исходником. У фрагмента тот же SHA-256, что у
        # всей записи, и без границ в ключе два куска одного ролика получили
        # бы общий кэш.
        source_identity = (
            media.sha256
            if media.section is None
            else f"{media.sha256}@{media.section.start:.3f}-{media.section.end:.3f}"
        )
        mark = timed("prepare_audio", started_at)
        duration = (
            f"{media.duration_seconds:.0f} с"
            if media.duration_seconds is not None
            else "неизвестна"
        )
        ensure_disk_space(work_dir, media.duration_seconds)
        # Очередь занимаем после скачивания и ffmpeg — им делить машину
        # незачем, — а отпускаем до уборки временных WAV: её растягивает
        # антивирус, и следующий прогон ждать её не должен.
        timings["queue"] = round(
            held.enter_context(
                machine_slot(
                    output.name,
                    on_wait=lambda holder, waited: report.stage(
                        f"Жду очереди ({waited:.0f} с): машину занимает {holder}"
                    ),
                )
            ),
            2,
        )
        mark = time.monotonic()
        report.stage(f"Длительность {duration}, режим {args.mode}; ищу речевые окна")
        # Загрузку модели называем отдельной строкой: без неё тишина в логе на
        # неотвечающем Hub неотличима от долгой работы, и 2026-09-17 причину
        # зависания пришлось искать через `ps`.
        report.stage("Silero VAD: загрузка")
        with report.waiting("Silero VAD"):
            base_chunks = split_speech_windows(
                prepared,
                work_dir,
                max_seconds=args.speech_window_seconds,
                pack=not args.diarize,
                offline=args.offline,
            )
        mark = timed("vad", mark)
        report.stage(f"Речевых окон: {len(base_chunks)}")
        warnings: list[str] = []
        language, detection = resolve_language(
            args,
            remote=remote,
            chunks=base_chunks,
            work_dir=work_dir,
            source_identity=source_identity,
            report=report,
        )
        if detection.get("warning"):
            report.stage(f"Внимание: {detection['warning']}")
            warnings.append(str(detection["warning"]))
        if detection["method"] == "whisper-language-id":
            mark = timed("language_id", mark)
        route = language_route(language)
        if route.warning:
            warnings.append(route.warning)
        # Устаревание моделей проверяется на каждом прогоне: doctor.py после
        # установки никто не запускает, и сигнал бы туда не дошёл.
        verifier_spec = route.verifier_for(args.mode)
        used_models = [route.primary.model] + ([verifier_spec.model] if verifier_spec else [])
        if args.diarize:
            used_models.append(args.diarization_model)
        for text in staleness_warnings(used_models):
            report.stage(f"Внимание: {text}")
            warnings.append(text)
        if not base_chunks:
            warnings.append("Silero VAD не нашёл речи: результат пустой")
        diarization: Diarization | None = None
        if args.diarize:
            assert diarization_kind is not None
            diarization_model = (
                "FluidAudio Community-1 offline"
                if diarization_kind == "fluidaudio"
                else args.diarization_model
            )
            diarization_key = cache_key(
                "diarization",
                {
                    "pipeline": "0.5",
                    "source_sha256": source_identity,
                    "backend": diarization_kind,
                    "model": diarization_model,
                    "revision": (
                        FLUIDAUDIO.revision
                        if diarization_kind == "fluidaudio"
                        else revision_for_model(args.diarization_model)
                    ),
                    "threshold": diarization_threshold,
                    "chunk_seconds": args.diarization_chunk_seconds,
                    "expected_speakers": args.expected_speakers,
                },
            )
            diarization = (
                None
                if args.no_cache
                else load_diarization(args.cache_dir, diarization_key)
            )
            if diarization is None:
                report.stage(f"Диаризация: {diarization_kind}")
                diarization = run_isolated_diarization(
                    diarization_kind,
                    prepared,
                    work_dir,
                    config={
                        "model_name": args.diarization_model,
                        "threshold": diarization_threshold,
                        "chunk_seconds": args.diarization_chunk_seconds,
                        "binary": (
                            str(args.fluidaudio_bin.expanduser().resolve())
                            if args.fluidaudio_bin
                            else None
                        ),
                        "offline": args.offline,
                    },
                    stall_timeout_seconds=args.backend_stall_timeout,
                    on_progress=report.windows("Диаризация"),
                )
                if not args.no_cache and weights_as_keyed(
                    diarization.metadata,
                    {args.diarization_model: revision_for_model(args.diarization_model)},
                ):
                    save_diarization(args.cache_dir, diarization_key, diarization, source_identity)
            # Нарезка режет каждое окно по границам реплик и пишет их на диск:
            # на 42-минутной встрече это 7–8 минут без единой строки в логе.
            report.stage(f"Нарезка окон по говорящим: {len(base_chunks)}")
            with report.waiting("Нарезка окон"):
                chunks = split_chunks_by_diarization(
                    prepared,
                    base_chunks,
                    diarization.turns,
                    work_dir,
                )
            mark = timed("diarization", mark)
            if not diarization.speakers and base_chunks:
                warnings.append(
                    "Диаризация не нашла говорящих; речевые окна помечены как unknown"
                )
            if diarization_kind == "sortformer" and len(diarization.speakers) >= 4:
                warnings.append(
                    "Sortformer достиг лимита 4 спикера; в записи могут быть дополнительные участники"
                )
            if (
                args.expected_speakers is not None
                and len(diarization.speakers) != args.expected_speakers
            ):
                warnings.append(
                    f"Диаризация нашла {len(diarization.speakers)} голосов из "
                    f"ожидаемых {args.expected_speakers}; метки требуют проверки"
                )
        else:
            chunks = base_chunks
        curated = [] if args.no_glossary else load_glossary(args.glossary)
        learned_path = (
            None
            if args.no_glossary
            else store_path(args.glossary_store, args.glossary_profile)
        )
        glossary = merge_glossaries(
            curated, [] if learned_path is None else entries_of(load_document(learned_path))
        )
        # Подсказки модели берём только из выверенного словаря пользователя.
        # Свой словарь скилл пополняет после каждого прогона, и если пускать
        # его сюда, ключ кэша менялся бы вместе с ним: обещание «повторный
        # прогон за секунды» перестало бы работать ровно тогда, когда словарь
        # растёт быстрее всего.
        hotwords = [entry.canonical for entry in curated]
        chunk_signature = [
            [
                chunk.sequence,
                round(chunk.start, 6),
                round(chunk.end, 6),
                list(chunk.speakers),
            ]
            for chunk in chunks
        ]
        primary_key = cache_key(
            "primary",
            {
                "pipeline": "0.5",
                "source_sha256": source_identity,
                "language": route.language,
                "route": route.identifier,
                "spec": route.primary.to_dict(),
                "whisper_backend": args.whisper_backend,
                "whisper_model_override": args.whisper_model,
                "weights": weights_signature(args, route.primary),
                # GigaAM словарь не принимает, Whisper — принимает: для него
                # смена словаря должна менять ключ.
                "hotwords": hotwords if route.primary.family == "whisper" else None,
                "chunks": chunk_signature,
            },
        )
        readable = None if args.no_cache else load_hypothesis(args.cache_dir, primary_key)
        if readable is None:
            report.stage(f"Основная модель: {route.primary.model}, окон {len(chunks)}")
            readable = run_route_backend(
                args,
                route.primary,
                chunks,
                hotwords,
                work_dir,
                role="primary",
                language=route.language,
                report=report,
            )
            if not args.no_cache and weights_as_keyed(
                readable.metadata, weights_signature(args, route.primary)
            ):
                save_hypothesis(args.cache_dir, primary_key, readable, source_identity)
        else:
            report.stage(f"Основная модель {route.primary.model}: результат из кэша")
        mark = timed("primary", mark)
        failure = primary_failure_warning(readable)
        if failure:
            # Громче остальных: провал модели нельзя спрятать за `--quiet`,
            # иначе пустой результат снова выглядит успешным.
            print(f"Внимание: {failure}", file=sys.stderr, flush=True)
            warnings.append(failure)
        hypotheses = [readable]
        review_items: list[ReviewItem] = primary_retry_items(readable)
        if args.early_text is not None:
            early_segments, _ = apply_glossary(
                strip_filler_segments(readable.segments)[0], glossary
            )
            write_plain_text(args.early_text, early_segments)
            # Отдельной строкой и мимо `--quiet`: по ней вызывающий узнаёт, что
            # текст можно брать, не дожидаясь проверки.
            print(f"Текст основной модели готов: {args.early_text}", file=sys.stderr, flush=True)
        verifier_ran = False

        # `fast` раньше шёл вообще без проверки, и пользователь не получал ни
        # одного указания, где текст мог поехать. Русскому маршруту добавлена
        # лёгкая проверяющая: Vosk даёт ту же очередь за 11 с против 234 с у
        # Whisper (замер 2026-09-12, experiments/verifier-bakeoff/). Текст она
        # не правит и не должна: подстановка её варианта на расхождении подняла
        # WER с 5,9% до 10,7%.
        if verifier_spec is not None:
            # Проверяются все окна: отбор «рискованных» убран 2026-09-06 —
            # он накрывал 78–90 % ошибок вместо 98–100 % и почти не экономил
            # время, потому что Whisper платит за упакованный пакет, а полный
            # набор окон пакуется плотнее выборки.
            if chunks:
                # Склейка окупается только у Whisper: он доводит любой вход до
                # 30 секунд и потому платит за вызов, а не за длину. GigaAM
                # считает пропорционально длине, склеивать ей нечего.
                # Таймкоды у GigaAM при этом есть (`with_timestamps()` отдаёт
                # по-токенные, шаг 0,08 с) — просто здесь они не нужны.
                pack_windows = (
                    args.verifier_window_seconds > 0 and verifier_spec.family == "whisper"
                )
                packed_windows = (
                    build_packed_windows(
                        prepared,
                        chunks,
                        work_dir,
                        max_seconds=args.verifier_window_seconds,
                    )
                    if pack_windows
                    else [PackedWindow(chunk=chunk, sources=(chunk,)) for chunk in chunks]
                )
                packed_chunks = [window.chunk for window in packed_windows]
                try:
                    verifier_key = cache_key(
                        "verifier",
                        {
                            "pipeline": "0.5",
                            "source_sha256": source_identity,
                            "language": route.language,
                            "route": route.identifier,
                            "spec": verifier_spec.to_dict(),
                            "whisper_backend": args.whisper_backend,
                            "whisper_model_override": args.whisper_model,
                            "weights": weights_signature(args, verifier_spec),
                            # Словаря в ключе нет намеренно: проверяющая его не
                            # получает, поэтому пополнение словаря больше не
                            # обесценивает кэш. Раньше один новый термин
                            # пересчитывал всю проверку — на часовой встрече
                            # тринадцать минут за каждую итерацию цикла.
                            "packing": [
                                "packed-hypothesis",
                                args.verifier_window_seconds if pack_windows else 0,
                            ],
                            "chunks": [
                                [
                                    chunk.sequence,
                                    round(chunk.start, 6),
                                    round(chunk.end, 6),
                                    list(chunk.speakers),
                                ]
                                for chunk in chunks
                            ],
                        },
                    )
                    packed_hypothesis = (
                        None
                        if args.no_cache
                        else load_hypothesis(args.cache_dir, verifier_key)
                    )
                    if packed_hypothesis is None:
                        report.stage(
                            f"Независимая проверка: {verifier_spec.model}, "
                            f"окон {len(chunks)}"
                            + (
                                f"; упакованы в {len(packed_chunks)}"
                                if len(packed_chunks) < len(chunks)
                                else ""
                            )
                        )
                        packed_hypothesis = run_route_backend(
                            args,
                            verifier_spec,
                            packed_chunks,
                            # Проверяющая идёт без словаря. `hotwords` — это
                            # канонические формы уже известных словарю
                            # терминов, поэтому незнакомому термину они не
                            # помогают, а знакомый и так канонизируется
                            # словарём при сверке, без пересчёта модели.
                            [],
                            work_dir,
                            role="independent_verifier",
                            language=route.language,
                            report=report,
                        )
                        if not args.no_cache and weights_as_keyed(
                            packed_hypothesis.metadata,
                            weights_signature(args, verifier_spec),
                        ):
                            save_hypothesis(
                                args.cache_dir,
                                verifier_key,
                                packed_hypothesis,
                                source_identity,
                            )
                    # Кэшируем сырой ответ модели, а не разложенный по окнам
                    # результат: разбор дешёвый, и его правки не должны
                    # заставлять пересчитывать час распознавания.
                    verifier = unpack_hypothesis(packed_hypothesis, packed_windows)
                    hypotheses.append(verifier)
                    verifier_ran = True
                    review_items.extend(
                        find_review_items(
                            # Окна без длительности сверять не с чем: их
                            # отсекает фильтр по пересечению с окнами VAD.
                            canonicalized(
                                replace(
                                    readable,
                                    segments=segments_for_windows(
                                        readable.segments,
                                        [(chunk.start, chunk.end) for chunk in chunks],
                                    ),
                                ),
                                glossary,
                            ),
                            canonicalized(verifier, glossary),
                            threshold=args.review_threshold,
                            comparison_label=f"Независимый {verifier.model}",
                        )
                    )
                except BackendUnavailable as error:
                    warnings.append(str(error))
                    review_items.extend(verifier_failure_items(chunks, readable))
            mark = timed("verifier", mark)

        report.stage(f"Сборка результата: {output}")
        review_items.sort(key=lambda item: (item.start, item.end, item.reason))
        readable_input, fillers_removed = strip_filler_segments(readable.segments)
        learned_report = None
        # Отбор кандидатов ищет латиницу против кириллицы, то есть устроен под
        # русскую речь. На английской записи латиница с обеих сторон, и в
        # словарь шли служебные слова («the / tha», «and / an»). Читать словарь
        # чужой маршрут по-прежнему может, пополнять — нет.
        learns_from_route = route.language == "ru"
        if learned_path is not None and not args.no_learn and not learns_from_route:
            report.stage(
                "Словарь не пополняется: отбор настроен на русский, "
                f"а маршрут {route.language}"
            )
        if learned_path is not None and not args.no_learn and learns_from_route:
            try:
                learned_entries, learned_report = update_from_run(
                    learned_path,
                    [item.to_dict() for item in review_items],
                    blocked=undisputed_words(readable_input, review_items),
                    curated={
                        normalize_text(term)
                        for entry in curated
                        for term in (entry.canonical, *entry.aliases)
                    },
                    # Отпечаток записи, а не запуска: перепрогон того же файла
                    # не должен считаться вторым доказательством.
                    # Отпечаток считается по исходному адресу: это хеш, токен из
                    # него не прочесть, а очищенный адрес склеил бы разные
                    # записи с идентификатором в query.
                    source=record_fingerprint(
                        media.sha256, remote.url if remote is not None else None
                    ),
                )
                # Порядок важен: словарь пополняется до применения, поэтому
                # термин, добравший порог на этой записи, правит уже её текст,
                # а не только следующую.
                glossary = merge_glossaries(curated, learned_entries)
            except OSError as error:
                # Недоступный словарь не повод терять расшифровку: она уже
                # посчитана, и без пополнения останется только следующий раз.
                warnings.append(f"Словарь не пополнен: {error}")
            else:
                summary = learned_report.summary()
                if summary:
                    report.stage(summary)
        readable_segments, corrections = apply_glossary(readable_input, glossary)
        if verifier_ran:
            # После пополнения словаря: обучение видит место таким, каким его
            # услышали модели, а не уже закрытым.
            readable_segments, review_items, taken = take_verifier_canonicals(
                readable_segments, review_items, glossary
            )
            corrections = sorted([*corrections, *taken], key=lambda item: (item.start, item.end))
            review_items = mark_verifier_only_terms(
                review_items, glossary, latin_is_signal=route.language == "ru"
            )
        suggestions = suggest_glossary_matches(
            readable_segments,
            glossary,
            only_words=disputed_primary_words(review_items) if verifier_ran else None,
        )
        timed("glossary", mark)
        # Итог считаем после словаря: раньше он стоял до пополнения и подсказок,
        # и 10–30 секунд хвоста в ориентиры скорости не попадали.
        timings["total"] = round(time.monotonic() - started_at, 2)
        diarization_output = diarization
        if media.section is not None:
            # Время фрагмента переводим во время исходника в самом конце:
            # словарь и очередь выше работали с WAV куска, а читать результат
            # будут рядом с субтитрами и плеером всей записи.
            offset = media.section.start
            hypotheses = [_shift_hypothesis(item, offset) for item in hypotheses]
            readable = hypotheses[0]
            readable_segments = _shift_all(readable_segments, offset)
            review_items = _shift_all(review_items, offset)
            corrections = _shift_all(corrections, offset)
            suggestions = [_shift_mapping(item, offset) for item in suggestions]
            if diarization is not None:
                diarization_output = replace(
                    diarization, turns=_shift_all(diarization.turns, offset)
                )
        result = write_bundle(
            output,
            media=media,
            hypotheses=hypotheses,
            lexical_segments=readable.segments,
            readable_segments=readable_segments,
            review_items=review_items,
            corrections=corrections,
            suggestions=suggestions,
            learned=learned_report.to_dict() if learned_report else None,
            mode=args.mode,
            offline=args.offline,
            # Маршрут знает обе проверяющие, но запускалась одна. Без явного
            # `verifier_used` манифест в режиме fast называл бы Whisper,
            # который не работал.
            route=route.to_dict()
            | {
                "verifier_used": verifier_spec.to_dict() if verifier_spec else None,
                "language_detection": detection,
            },
            diarization=diarization_output,
            warnings=warnings,
            overwrite=args.overwrite,
            prepared_audio=prepared if args.keep_wav else None,
            timings=timings,
            generator=f"transcriber {__version__}",
            fillers_removed=fillers_removed,
            glossary_sources={
                "curated": str(args.glossary) if args.glossary and not args.no_glossary else None,
                "store": str(learned_path) if learned_path else None,
                "learning": learned_path is not None and not args.no_learn,
            },
            model_revisions=loaded_revisions(
                revisions_for_models([SILERO_VAD.model, *used_models]),
                [*hypotheses, *([diarization] if diarization is not None else [])],
                warnings,
            ),
            speaker_names=args.speaker_names,
        )
        # Результат уже на диске. Уборка временных WAV идёт отдельной стадией,
        # потому что антивирус, проверяющий каждую файловую операцию,
        # растягивает её на минуты; без этой строки в логе прогон выглядит
        # зависшим после готового каталога.
        report.stage(f"Результат записан: {output}")
        return result


def _shift_all(items: list[Any], offset: float) -> list[Any]:
    return [replace(item, start=item.start + offset, end=item.end + offset) for item in items]


def _shift_mapping(item: dict[str, Any], offset: float) -> dict[str, Any]:
    return item | {
        key: item[key] + offset
        for key in ("start", "end")
        if isinstance(item.get(key), (int, float))
    }


def _shift_hypothesis(hypothesis: Hypothesis, offset: float) -> Hypothesis:
    metadata = dict(hypothesis.metadata)
    chunks = metadata.get("chunks")
    if isinstance(chunks, list):
        metadata["chunks"] = [
            _shift_mapping(chunk, offset) if isinstance(chunk, dict) else chunk
            for chunk in chunks
        ]
    return replace(
        hypothesis,
        segments=_shift_all(hypothesis.segments, offset),
        metadata=metadata,
    )


def expand_inputs(args: argparse.Namespace) -> list[str]:
    """Ссылки-плейлисты превращаются в ролики; файлы и прочие ссылки — как есть."""
    if args.playlist_limit is not None and args.playlist_limit < 1:
        raise ValueError("--playlist-limit должен быть положительным")
    if not args.playlist:
        if args.playlist_limit is not None:
            raise ValueError("--playlist-limit требует --playlist")
        return list(args.input)
    expanded: list[str] = []
    for raw in args.input:
        if looks_like_url(raw):
            expanded.extend(expand_playlist(raw, limit=args.playlist_limit))
        else:
            expanded.append(raw)
    return expanded


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.input = expand_inputs(args)
    except FAILURES as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    batch = len(args.input) * len(_sections(args)) > 1
    try:
        if not batch:
            payload: dict[str, Any] = summarize(run(args))
        else:
            results = run_batch(args)
            payload = {"results": results}
    except FAILURES as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False))
    if batch and any("error" in item for item in payload["results"]):
        return 2
    return 0
