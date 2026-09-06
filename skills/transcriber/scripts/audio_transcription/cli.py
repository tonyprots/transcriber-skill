from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from . import __version__
from .audio import MediaToolError, prepare_audio, split_speech_windows
from .backends import BackendMissing, BackendUnavailable
from .cache import (
    cache_key,
    load_diarization,
    load_hypothesis,
    save_diarization,
    save_hypothesis,
)
from .diarization import split_chunks_by_diarization
from .glossary import apply_glossary, load_glossary, suggest_glossary_matches
from .fillers import strip_filler_segments
from .models import AudioChunk, Diarization, Hypothesis, ReviewItem
from .packing import PackedWindow, build_packed_windows, unpack_hypothesis
from .isolated import run_isolated_backend, run_isolated_diarization
from .reconcile import find_review_items, segments_for_windows
from .render import ensure_disk_space, ensure_output_available, write_bundle
from .routing import BackendSpec, diarization_backend, language_route


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Локальная расшифровка аудио с независимой сверкой гипотез.",
    )
    parser.add_argument("input", type=Path, help="Аудио- или видеофайл")
    parser.add_argument("--mode", choices=("fast", "max"), default="max")
    parser.add_argument("--language", default="ru", help="Код языка, по умолчанию ru")
    parser.add_argument("--glossary", type=Path, help="YAML-словарь терминов")
    parser.add_argument("--output", type=Path, help="Каталог результата")
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
    parser.add_argument("--review-threshold", type=float, default=0.82)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-wav", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "transcriber",
        help="Каталог кэша гипотез",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Определить говорящих и разрезать общие VAD-окна по сменам спикера",
    )
    parser.add_argument(
        "--diarization-model",
        default="mlx-community/diar_sortformer_4spk-v1-fp16",
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
    parser.add_argument("--diarization-chunk-seconds", type=float, default=30.0)
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
    faster_model = "medium" if args.whisper_backend == "auto" else (args.whisper_model or "medium")
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
    elif spec.family == "gigaam":
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
            on_progress=report.windows(f"GigaAM {spec.model}"),
        )
    else:
        raise ValueError(f"Неизвестное семейство ASR: {spec.family}")
    hypothesis.metadata["role"] = role
    return hypothesis


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


def run(args: argparse.Namespace) -> Path:
    source = args.input.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else source.with_name(f"{source.stem}.transcript")
    )
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
    output = ensure_output_available(output, overwrite=args.overwrite)
    report = ProgressReporter(enabled=not args.quiet)

    route = language_route(args.language)
    diarization_kind = (
        diarization_backend(args.diarization_backend, args.expected_speakers)
        if args.diarize
        else None
    )
    diarization_threshold = args.diarization_threshold
    if diarization_threshold is None:
        diarization_threshold = 0.8 if diarization_kind == "fluidaudio" else 0.4

    timings: dict[str, float] = {}
    started_at = time.monotonic()

    def timed(stage: str, since: float) -> float:
        timings[stage] = round(time.monotonic() - since, 2)
        return time.monotonic()

    with tempfile.TemporaryDirectory(prefix="transcriber-asr-") as temporary:
        work_dir = Path(temporary)
        report.stage(f"Подготовка аудио: {source.name}")
        prepared, media = prepare_audio(source, work_dir)
        mark = timed("prepare_audio", started_at)
        duration = (
            f"{media.duration_seconds:.0f} с"
            if media.duration_seconds is not None
            else "неизвестна"
        )
        ensure_disk_space(work_dir, media.duration_seconds)
        report.stage(f"Длительность {duration}, режим {args.mode}; ищу речевые окна")
        base_chunks = split_speech_windows(prepared, work_dir, pack=not args.diarize)
        mark = timed("vad", mark)
        report.stage(f"Речевых окон: {len(base_chunks)}")
        warnings: list[str] = [route.warning] if route.warning else []
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
                    "source_sha256": media.sha256,
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
        glossary = load_glossary(args.glossary)
        hotwords = [entry.canonical for entry in glossary]
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
                "source_sha256": media.sha256,
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

        if args.mode == "max" and route.verifier is not None:
            # Проверяются все окна: отбор «рискованных» убран 2026-09-06 —
            # он накрывал 78–90 % ошибок вместо 98–100 % и почти не экономил
            # время, потому что Whisper платит за упакованный пакет, а полный
            # набор окон пакуется плотнее выборки.
            verifier_chunks = chunks
            if verifier_chunks:
                # Склейка окупается только у Whisper (платит за вызов) и требует
                # пословных таймкодов, которых у GigaAM нет: её текст нечем
                # разложить обратно по окнам.
                pack_windows = (
                    args.verifier_window_seconds > 0 and route.verifier.family == "whisper"
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
                            "source_sha256": media.sha256,
                            "language": route.language,
                            "route": route.identifier,
                            "spec": route.verifier.to_dict(),
                            "whisper_backend": args.whisper_backend,
                            "whisper_model_override": args.whisper_model,
                            "hotwords": hotwords,
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
                            f"Независимая проверка: {route.verifier.model}, "
                            f"окон {len(verifier_chunks)}"
                            + (
                                f"; упакованы в {len(packed_chunks)}"
                                if len(packed_chunks) < len(verifier_chunks)
                                else ""
                            )
                        )
                        packed_hypothesis = run_route_backend(
                            args,
                            route.verifier,
                            packed_chunks,
                            hotwords,
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
                            selected_primary,
                            verifier,
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
        readable_segments, corrections = apply_glossary(readable_input, glossary)
        suggestions = suggest_glossary_matches(readable_segments, glossary)
        result = write_bundle(
            output,
            media=media,
            hypotheses=hypotheses,
            lexical_segments=readable.segments,
            readable_segments=readable_segments,
            review_items=review_items,
            corrections=corrections,
            suggestions=suggestions,
            mode=args.mode,
            offline=args.offline,
            route=route.to_dict(),
            diarization=diarization,
            warnings=warnings,
            overwrite=args.overwrite,
            prepared_audio=prepared if args.keep_wav else None,
            timings=timings,
            generator=f"transcriber {__version__}",
            fillers_removed=fillers_removed,
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except (
        BackendUnavailable,
        FileNotFoundError,
        FileExistsError,
        MediaToolError,
        ValueError,
        OSError,
        yaml.YAMLError,
    ) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output)}, ensure_ascii=False))
    return 0
