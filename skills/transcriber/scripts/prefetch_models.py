#!/usr/bin/env python3
"""Скачивает модели заранее, чтобы первая расшифровка не ждала загрузки.

Запуск: <python из .venv> scripts/prefetch_models.py [ru|en|all] [--diarize]

ru  — Silero VAD, GigaAM v3 E2E, Vosk ru, Whisper Turbo;
en  — Silero VAD, Whisper Turbo, Parakeet TDT 0.6B v3;
all — всё вместе. --diarize добавляет Sortformer (только Apple Silicon).
На Linux и Intel вместо Whisper MLX скачивается faster-whisper medium.
Повторный запуск ничего не качает: всё уже в ~/.cache/huggingface.

Сколько это весит, печатает `--print-size ru|en|all`: размеры считаются по
scripts/audio_transcription/catalog.py, чтобы число в подсказках установщика
не разъезжалось с каталогом. Там же — какие именно это модели.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.catalog import (  # noqa: E402
    FASTER_WHISPER_FALLBACK,
    GIGAAM_RUSSIAN,
    PARAKEET_ENGLISH,
    SILERO_VAD,
    SORTFORMER,
    VOSK_RUSSIAN,
    WHISPER_TURBO,
    format_gb,
    route_download_gb,
)


def _apple() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _label(entry) -> str:
    if entry.download_gb < 0.05:
        return entry.label
    return f"{entry.label} ({entry.download_gb:.1f} ГБ)".replace(".", ",")


def _step(label: str, action) -> bool:
    started = time.monotonic()
    print(f"→ {label}", flush=True)
    try:
        action()
    except Exception as exc:  # noqa: BLE001 — сообщаем и идём дальше
        print(f"  не удалось: {exc}", file=sys.stderr)
        return False
    print(f"  готово за {time.monotonic() - started:.0f} с", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("route", nargs="?", choices=("ru", "en", "all"), default="ru")
    parser.add_argument("--diarize", action="store_true", help="добавить модель диаризации Sortformer")
    parser.add_argument(
        "--print-size",
        action="store_true",
        help="напечатать размер загрузки маршрута и выйти, ничего не качая",
    )
    args = parser.parse_args(argv)

    if args.print_size:
        print(format_gb(route_download_gb(args.route, apple=_apple(), diarize=args.diarize)))
        return 0

    import onnx_asr

    def load_onnx(entry):
        kwargs = {"providers": ["CPUExecutionProvider"]}
        if entry.quantization:
            kwargs["quantization"] = entry.quantization
        return lambda: onnx_asr.load_model(entry.model, **kwargs)

    steps: list[tuple[str, object]] = [
        (SILERO_VAD.label, lambda: onnx_asr.load_vad(SILERO_VAD.model, providers=["CPUExecutionProvider"]))
    ]
    if args.route in ("ru", "all"):
        steps.append((_label(GIGAAM_RUSSIAN), load_onnx(GIGAAM_RUSSIAN)))
        # Проверяющая для режима fast. Без неё fast работает, но молча:
        # очередь проверки просто не соберётся.
        steps.append((_label(VOSK_RUSSIAN), load_onnx(VOSK_RUSSIAN)))
    if _apple():
        from huggingface_hub import snapshot_download

        steps.append((_label(WHISPER_TURBO), lambda: snapshot_download(WHISPER_TURBO.repo)))
    else:
        from faster_whisper import download_model

        steps.append((_label(FASTER_WHISPER_FALLBACK), lambda: download_model(FASTER_WHISPER_FALLBACK.model)))
    if args.route in ("en", "all"):
        steps.append((_label(PARAKEET_ENGLISH), load_onnx(PARAKEET_ENGLISH)))
    if args.diarize:
        if _apple():
            from huggingface_hub import snapshot_download

            steps.append((_label(SORTFORMER), lambda: snapshot_download(SORTFORMER.repo)))
        else:
            print("Диаризация доступна только на macOS Apple Silicon: Sortformer пропущен", file=sys.stderr)

    failed = [label for label, action in steps if not _step(label, action)]
    if failed:
        print(f"Не скачалось: {', '.join(failed)}. Проверьте сеть или доступ к Hugging Face.", file=sys.stderr)
        return 1
    print("Все модели на месте.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
