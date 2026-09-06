#!/usr/bin/env python3
"""Скачивает модели заранее, чтобы первая расшифровка не ждала загрузки.

Запуск: <python из .venv> scripts/prefetch_models.py [ru|en|all] [--diarize]

ru  — Silero VAD, GigaAM v3 E2E, Whisper Turbo (около 2,5 ГБ);
en  — Silero VAD, Whisper Turbo, GigaAM Multilingual CTC int8 (около 1,8 ГБ);
all — всё вместе. --diarize добавляет Sortformer (0,2 ГБ, только Apple Silicon).
На Linux и Intel вместо Whisper MLX скачивается faster-whisper medium.
Повторный запуск ничего не качает: всё уже в ~/.cache/huggingface.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

WHISPER_MLX = "mlx-community/whisper-large-v3-turbo-asr-fp16"
SORTFORMER = "mlx-community/diar_sortformer_4spk-v1-fp16"


def _apple() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


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
    args = parser.parse_args(argv)

    import onnx_asr

    steps: list[tuple[str, object]] = [
        ("Silero VAD", lambda: onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"]))
    ]
    if args.route in ("ru", "all"):
        steps.append(
            ("GigaAM v3 E2E (0,9 ГБ)", lambda: onnx_asr.load_model("gigaam-v3-e2e-rnnt", providers=["CPUExecutionProvider"]))
        )
    if _apple():
        from huggingface_hub import snapshot_download

        steps.append(("Whisper Turbo MLX (1,5 ГБ)", lambda: snapshot_download(WHISPER_MLX)))
    else:
        from faster_whisper import download_model

        steps.append(("faster-whisper medium (1,5 ГБ, CPU)", lambda: download_model("medium")))
    if args.route in ("en", "all"):
        steps.append(
            (
                "GigaAM Multilingual CTC int8 (0,2 ГБ)",
                lambda: onnx_asr.load_model(
                    "gigaam-multilingual-ctc", quantization="int8", providers=["CPUExecutionProvider"]
                ),
            )
        )
    if args.diarize:
        if _apple():
            from huggingface_hub import snapshot_download

            steps.append(("Sortformer (0,2 ГБ)", lambda: snapshot_download(SORTFORMER)))
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
