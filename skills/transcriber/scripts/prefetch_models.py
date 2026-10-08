#!/usr/bin/env python3
"""Скачивает модели заранее, чтобы первая расшифровка не ждала загрузки.

Запуск: <python из .venv> scripts/prefetch_models.py [ru|en|all] [--diarize]

ru  — Silero VAD, GigaAM v3 E2E, Vosk ru, Whisper Turbo;
en  — Silero VAD, Whisper Turbo, Parakeet TDT 0.6B v3;
all — всё вместе. --diarize добавляет CoreML-модели FluidAudio (только Apple Silicon).
На Linux и Intel вместо Whisper MLX скачивается faster-whisper medium.
Повторный запуск ничего не качает: всё уже в ~/.cache/huggingface.
Качается закреплённая ревизия каждой модели (`revision` в каталоге), а не `main`.

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

from audio_transcription.fluid_models import ensure as ensure_fluid_models, ensure_binary  # noqa: E402
from audio_transcription.catalog import (  # noqa: E402
    FASTER_WHISPER_FALLBACK,
    GIGAAM_RUSSIAN,
    PARAKEET_ENGLISH,
    SILERO_VAD,
    FLUIDAUDIO,
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
    parser.add_argument("--diarize", action="store_true", help="добавить модели диаризации FluidAudio")
    parser.add_argument(
        "--print-size",
        action="store_true",
        help="напечатать размер загрузки маршрута и выйти, ничего не качая",
    )
    args = parser.parse_args(argv)

    if args.print_size:
        print(format_gb(route_download_gb(args.route, apple=_apple(), diarize=args.diarize)))
        return 0

    from audio_transcription.weights import fetch

    def pinned(entry):
        # Те же коммиты и те же файлы, что потом грузит прогон: иначе первая
        # расшифровка докачивала бы закреплённую ревизию поверх свежего main.
        return lambda: fetch(entry, quiet=False)

    steps: list[tuple[str, object]] = [(SILERO_VAD.label, pinned(SILERO_VAD))]
    if args.route in ("ru", "all"):
        steps.append((_label(GIGAAM_RUSSIAN), pinned(GIGAAM_RUSSIAN)))
        # Проверяющая для режима fast. Без неё fast работает, но молча:
        # очередь проверки просто не соберётся.
        steps.append((_label(VOSK_RUSSIAN), pinned(VOSK_RUSSIAN)))
    whisper = WHISPER_TURBO if _apple() else FASTER_WHISPER_FALLBACK
    steps.append((_label(whisper), pinned(whisper)))
    if args.route in ("en", "all"):
        steps.append((_label(PARAKEET_ENGLISH), pinned(PARAKEET_ENGLISH)))
    if args.diarize:
        if _apple():
            steps.append(("fluidaudiocli (около 20 МБ)", ensure_binary))
            steps.append((_label(FLUIDAUDIO), ensure_fluid_models))
        else:
            print("Диаризация доступна только на macOS Apple Silicon: FluidAudio пропущен", file=sys.stderr)

    failed = [label for label, action in steps if not _step(label, action)]
    if failed:
        print(f"Не скачалось: {', '.join(failed)}. Проверьте сеть или доступ к Hugging Face.", file=sys.stderr)
        return 1
    print("Все модели на месте.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
