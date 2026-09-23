from __future__ import annotations

import argparse
import json
import platform
import re
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
from .fetching import FetchError, RemoteMedia, fetch_audio, looks_like_url, probe_remote
from .machine_lock import machine_slot
from .backends import BackendMissing, BackendUnavailable
from .catalog import (
    FASTER_WHISPER_FALLBACK,
    SILERO_VAD,
    SORTFORMER,
    revisions_for_models,
    staleness_warnings,
)
from .cache import (
    cache_key,
    load_diarization,
    load_hypothesis,
    save_diarization,
    save_hypothesis,
)
from .diarization import split_chunks_by_diarization
from .glossary import (
    GlossaryEntry,
    apply_glossary,
    load_glossary,
    merge_glossaries,
    suggest_glossary_matches,
)
from .glossary_store import (
    entries_of,
    load_document,
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
from .render import ensure_disk_space, ensure_output_available, write_bundle
from .routing import BackendSpec, diarization_backend, language_route


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
    parser.add_argument("--language", default="ru", help="Код языка, по умолчанию ru")
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
        "--no-cache",
        action="store_true",
        help="Считать заново, не читая и не записывая кэш гипотез: для контрольного прогона",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Определить говорящих и разрезать общие VAD-окна по сменам спикера",
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


def _slug(title: str) -> str:
    """Имя каталога из заголовка видео: только то, что безопасно в пути."""
    cleaned = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip().lower()
    return re.sub(r"[\s_-]+", "-", cleaned)[:80] or "video"


def _spoken_duration(remote: RemoteMedia) -> str:
    if not remote.duration_seconds:
        return "длительность неизвестна"
    return f"{remote.duration_seconds / 60:.0f} мин"


def _validate(args: argparse.Namespace) -> None:
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
    output = ensure_output_available(output, overwrite=args.overwrite)
    report = ProgressReporter(enabled=not args.quiet)
    with tempfile.TemporaryDirectory(prefix="transcriber-download-") as downloads:
        download_seconds = None
        if remote is not None:
            source, download_seconds = _download(remote, Path(downloads), report)
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
            jobs.append((section, ensure_output_available(candidate, overwrite=args.overwrite)))
        plan.append((raw, remote, source, jobs))
    total = sum(len(jobs) for *_, jobs in plan)
    done = 0
    for raw, remote, source, jobs in plan:
        with tempfile.TemporaryDirectory(prefix="transcriber-download-") as downloads:
            download_seconds = None
            if remote is not None:
                try:
                    source, download_seconds = _download(remote, Path(downloads), report)
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
    route = language_route(args.language)
    diarization_kind = (
        diarization_backend(args.diarization_backend, args.expected_speakers)
        if args.diarize
        else None
    )
    diarization_threshold = args.diarization_threshold
    if diarization_threshold is None:
        diarization_threshold = 0.8 if diarization_kind == "fluidaudio" else 0.4

    # Скачивание в total не входит: это сеть, а не расшифровка, и по total
    # сравнивают скорость прогонов между собой.
    timings: dict[str, float] = {}
    if download_seconds is not None:
        timings["download"] = download_seconds
    started_at = time.monotonic()

    def timed(stage: str, since: float) -> float:
        timings[stage] = round(time.monotonic() - since, 2)
        return time.monotonic()

    with tempfile.TemporaryDirectory(prefix="transcriber-asr-") as temporary, ExitStack() as held:
        work_dir = Path(temporary)
        scope = f", фрагмент {section.label}" if section else ""
        report.stage(f"Подготовка аудио: {source.name}{scope}")
        prepared, media = prepare_audio(source, work_dir, section)
        if remote is not None:
            media = replace(media, origin=remote.url)
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
        warnings: list[str] = [route.warning] if route.warning else []
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
                if not args.no_cache:
                    save_diarization(args.cache_dir, diarization_key, diarization)
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
        learned_path = None if args.no_glossary else store_path(args.glossary_store)
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
            if not args.no_cache:
                save_hypothesis(args.cache_dir, primary_key, readable)
        else:
            report.stage(f"Основная модель {route.primary.model}: результат из кэша")
        mark = timed("primary", mark)
        hypotheses = [readable]
        review_items: list[ReviewItem] = primary_retry_items(readable)

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
            verifier_chunks = chunks
            if verifier_chunks:
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
                        verifier_chunks,
                        work_dir,
                        max_seconds=args.verifier_window_seconds,
                    )
                    if pack_windows
                    else [PackedWindow(chunk=chunk, sources=(chunk,)) for chunk in verifier_chunks]
                )
                packed_chunks = [window.chunk for window in packed_windows]
                windows = [(chunk.start, chunk.end) for chunk in verifier_chunks]
                selected_primary = Hypothesis(
                    model=readable.model,
                    language=readable.language,
                    elapsed_seconds=readable.elapsed_seconds,
                    segments=segments_for_windows(readable.segments, windows),
                )
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
                                for chunk in verifier_chunks
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
                            f"окон {len(verifier_chunks)}"
                            + (
                                f"; упакованы в {len(packed_chunks)}"
                                if len(packed_chunks) < len(verifier_chunks)
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
                        if not args.no_cache:
                            save_hypothesis(
                                args.cache_dir, verifier_key, packed_hypothesis
                            )
                    # Кэшируем сырой ответ модели, а не разложенный по окнам
                    # результат: разбор дешёвый, и его правки не должны
                    # заставлять пересчитывать час распознавания.
                    verifier = unpack_hypothesis(packed_hypothesis, packed_windows)
                    hypotheses.append(verifier)
                    review_items.extend(
                        find_review_items(
                            canonicalized(selected_primary, glossary),
                            canonicalized(verifier, glossary),
                            threshold=args.review_threshold,
                            comparison_label=f"Независимый {verifier.model}",
                        )
                    )
                except BackendUnavailable as error:
                    warnings.append(str(error))
                    review_items.extend(verifier_failure_items(verifier_chunks, readable))
            timed("verifier", mark)

        report.stage(f"Сборка результата: {output}")
        timings["total"] = round(time.monotonic() - started_at, 2)
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
                    source=record_fingerprint(media.sha256, media.origin),
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
        suggestions = suggest_glossary_matches(readable_segments, glossary)
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
            | {"verifier_used": verifier_spec.to_dict() if verifier_spec else None},
            diarization=diarization_output,
            warnings=warnings,
            overwrite=args.overwrite,
            prepared_audio=prepared if args.keep_wav else None,
            timings=timings,
            generator=f"transcriber {__version__}",
            fillers_removed=fillers_removed,
            model_revisions=revisions_for_models([SILERO_VAD.model, *used_models]),
        )
        # Результат уже на диске. Уборка временных WAV идёт отдельной стадией,
        # потому что антивирус, проверяющий каждую файловую операцию,
        # растягивает её на минуты; без этой строки в логе прогон выглядит
        # зависшим после готового каталога.
        report.stage(
            f"Результат записан: {output}. Уборка временных файлов "
            f"({sum(1 for _ in work_dir.rglob('*.wav'))} WAV)"
        )
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
