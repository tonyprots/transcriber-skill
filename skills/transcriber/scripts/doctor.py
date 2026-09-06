#!/usr/bin/env python3
"""Проверка окружения перед первым запуском: что есть, чего не хватает.

Запуск: <python из .venv> scripts/doctor.py [--json]
Код возврата 0 — можно запускать transcribe.py; 1 — не хватает обязательного.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path

REQUIRED_MODULES = {
    "onnx_asr": "onnx-asr[cpu,hub]",
    "faster_whisper": "faster-whisper",
    "yaml": "PyYAML",
    "huggingface_hub": "huggingface_hub (входит в onnx-asr[hub])",
}
APPLE_MODULES = {"mlx_audio": "mlx-audio (Whisper Turbo MLX и диаризация Sortformer)"}

# Каталоги в кэше Hugging Face, которые появляются после первой загрузки.
MODELS = {
    "gigaam-v3 (ru, основная)": "models--istupakov--gigaam-v3-onnx",
    "silero VAD": "models--istupakov--silero-vad-onnx",
    "whisper-large-v3-turbo MLX": "models--mlx-community--whisper-large-v3-turbo-asr-fp16",
    "gigaam-multilingual (en, проверка)": "models--istupakov--gigaam-multilingual-ctc-onnx",
    "sortformer (диаризация 1–4)": "models--mlx-community--diar_sortformer_4spk-v1-fp16",
    "faster-whisper medium (CPU fallback)": "models--Systran--faster-whisper-medium",
}


def _hf_hub_dir() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 — huggingface_hub может быть не установлен
        return Path.home() / ".cache" / "huggingface" / "hub"


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 — ошибка импорта = модуля нет
        return None
    return str(getattr(module, "__version__", "ok"))


def collect() -> dict:
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    skill_dir = Path(__file__).resolve().parents[1]
    hub = _hf_hub_dir()
    fluid_models = (
        Path.home()
        / "Library"
        / "Application Support"
        / "FluidAudio"
        / "Models"
        / "speaker-diarization-coreml"
    )
    report = {
        "python": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 10),
        "platform": f"{platform.system()} {platform.machine()}",
        "apple_silicon": apple,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "modules": {name: _module_version(name) for name in REQUIRED_MODULES},
        "apple_modules": {name: _module_version(name) for name in APPLE_MODULES},
        "models_cached": {label: (hub / folder).is_dir() for label, folder in MODELS.items()},
        "fluidaudio_binary": (skill_dir / "bin" / "macos-arm64" / "fluidaudiocli").is_file(),
        "fluidaudio_models": (fluid_models / "plda-parameters.json").is_file(),
        "free_gb": round(shutil.disk_usage(Path.home()).free / 1024**3, 1),
    }
    problems = []
    if not report["python_ok"]:
        problems.append("нужен Python 3.10 или новее")
    if not report["ffmpeg"] or not report["ffprobe"]:
        problems.append("не найдены ffmpeg и ffprobe (brew install ffmpeg / apt install ffmpeg)")
    for name, package in REQUIRED_MODULES.items():
        if report["modules"][name] is None:
            problems.append(f"не установлен пакет {package}")
    report["problems"] = problems
    report["warnings"] = []
    if apple and report["apple_modules"]["mlx_audio"] is None:
        report["warnings"].append(
            "нет mlx-audio: Whisper пойдёт через медленный CPU fallback, диаризация недоступна"
        )
    if not apple:
        report["warnings"].append(
            "не Apple Silicon: диаризация и Whisper Turbo MLX недоступны, Whisper работает на CPU"
        )
    if report["free_gb"] < 8:
        report["warnings"].append("меньше 8 ГБ свободно: модели занимают около 4 ГБ, временные WAV — ещё гигабайты")
    return report


def render(report: dict) -> str:
    mark = lambda ok: "✓" if ok else "✗"  # noqa: E731
    lines = [
        f"Python {report['python']} {mark(report['python_ok'])}   {report['platform']}",
        f"ffmpeg {mark(report['ffmpeg'])}   ffprobe {mark(report['ffprobe'])}   свободно {report['free_gb']} ГБ",
        "Пакеты:",
    ]
    for name, version in {**report["modules"], **report["apple_modules"]}.items():
        lines.append(f"  {mark(version)} {name} {version or ''}".rstrip())
    lines.append("Модели в кэше (загружаются при первом запуске, около 1–2 ГБ каждая):")
    for label, present in report["models_cached"].items():
        lines.append(f"  {mark(present)} {label}")
    lines.append(
        f"FluidAudio (5+ голосов): бинарник {mark(report['fluidaudio_binary'])}, "
        f"CoreML-модели {mark(report['fluidaudio_models'])}"
    )
    for text in report["warnings"]:
        lines.append(f"! {text}")
    for text in report["problems"]:
        lines.append(f"✗ {text}")
    lines.append("Готово к запуску" if not report["problems"] else "Запуск невозможен, см. ✗")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения transcriber")
    parser.add_argument("--json", action="store_true", help="Вывести отчёт в JSON")
    args = parser.parse_args()
    report = collect()
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
