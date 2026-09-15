#!/usr/bin/env python3
"""Проверка окружения перед первым запуском: что есть, чего не хватает.

Запуск: <python из .venv> scripts/doctor.py [--json] [--check-updates]

Код возврата 0 — можно запускать transcribe.py; 1 — не хватает обязательного.

Кроме готовности окружения отчёт показывает, не устарели ли модели. Без сети
проверяется возраст локального замера качества и список моделей, которые знает
установленная onnx-asr. С `--check-updates` скилл дополнительно спрашивает
Hugging Face, не появилось ли на хабе версии новее. Ни то ни другое ничего не
переключает: выбор модели держится на замерах WER, поэтому решение остаётся за
пользователем.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.catalog import (  # noqa: E402
    CALIBRATION_SHELF_LIFE_DAYS,
    CATALOG,
    format_gb,
    hf_cache_dir,
    known_asr_names,
    local_revision,
    route_download_gb,
    search_term,
)

REQUIRED_MODULES = {
    "onnx_asr": "onnx-asr[cpu,hub]",
    "faster_whisper": "faster-whisper",
    "yaml": "PyYAML",
    "huggingface_hub": "huggingface_hub (входит в onnx-asr[hub])",
}
APPLE_MODULES = {"mlx_audio": "mlx-audio (Whisper Turbo MLX и диаризация Sortformer)"}

HUB_TIMEOUT_SECONDS = 10
HUB_SEARCH_LIMIT = 100


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:  # noqa: BLE001 — ошибка импорта = модуля нет
        return None
    return str(getattr(module, "__version__", "ok"))


def _fluid_models_dir() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "FluidAudio"
        / "Models"
        / "speaker-diarization-coreml"
    )


def check_catalog(today: date | None = None) -> list[dict]:
    """Состояние каждой модели: скачана ли, когда её последний раз мерили."""
    today = today or date.today()
    hub = hf_cache_dir()
    fluid_ready = (_fluid_models_dir() / "plda-parameters.json").is_file()
    rows = []
    for entry in CATALOG:
        cache_dir = entry.cache_dir
        if cache_dir:
            cached = (hub / cache_dir).is_dir()
        else:
            cached = fluid_ready if entry.key == "fluidaudio" else False
        rows.append(
            {
                "key": entry.key,
                "label": entry.label,
                "role": entry.role,
                "model": entry.model,
                "repo": entry.repo,
                "upstream": entry.upstream,
                "cached": cached,
                "download_gb": entry.download_gb,
                # Та же ревизия уходит в manifest.json каждого прогона: по ней
                # потом видно, на тех ли весах получен старый результат.
                "revision": local_revision(entry, hub),
                "apple_only": entry.apple_only,
                "calibrated": entry.calibrated.isoformat() if entry.calibrated else None,
                "calibration_note": entry.calibration_note,
                "age_days": entry.age_days(today),
                "stale": entry.is_stale(today),
            }
        )
    return rows


def check_library_catalog() -> dict:
    """Знает ли установленная onnx-asr модель новее той, что зашита в маршрут.

    Сети не требует: это сравнение с тем, что уже лежит в site-packages.
    """
    names = known_asr_names()
    if not names:
        return {"available": False, "findings": []}
    findings = []
    for entry in CATALOG:
        for newer in entry.newer_variants(names):
            findings.append(
                {
                    "key": entry.key,
                    "text": f"onnx-asr знает {newer} — версия новее, чем {entry.model} в маршруте",
                }
            )
    return {"available": True, "findings": findings}


def check_hub_updates(today: date | None = None) -> dict:
    """Спросить Hugging Face про новые версии. Требует сети, вызывается по флагу."""
    today = today or date.today()
    try:
        from huggingface_hub import HfApi
    except Exception as error:  # noqa: BLE001 — пакета может не быть
        return {"checked": False, "error": f"нет huggingface_hub: {error}", "findings": []}

    api = HfApi()
    findings: list[dict] = []
    errors: list[str] = []
    for entry in CATALOG:
        if not entry.repo:
            continue
        try:
            info = api.model_info(entry.repo, timeout=HUB_TIMEOUT_SECONDS)
        except Exception as error:  # noqa: BLE001 — сеть, 404, приватный репозиторий
            errors.append(f"{entry.repo}: {error}")
            continue
        modified = getattr(info, "last_modified", None)
        if modified and entry.calibrated and modified.date() > entry.calibrated:
            findings.append(
                {
                    "key": entry.key,
                    "text": (
                        f"{entry.repo} обновлялся {modified.date().isoformat()}, "
                        f"уже после замера {entry.calibrated.isoformat()}"
                    ),
                }
            )
        owner = entry.repo.split("/")[0]
        try:
            # Свежие сверху: у крупных организаций одна страница выдачи не
            # вмещает все сборки, а новая версия всегда среди недавних.
            siblings = [
                model.id
                for model in api.list_models(
                    author=owner,
                    search=search_term(entry.repo),
                    sort="lastModified",
                    limit=HUB_SEARCH_LIMIT,
                )
            ]
        except Exception as error:  # noqa: BLE001 — поиск не критичен
            errors.append(f"поиск по {owner}: {error}")
            continue
        for newer in entry.newer_repos(siblings):
            findings.append(
                {
                    "key": entry.key,
                    "text": f"на хабе есть {newer} — версия новее, чем {entry.repo}",
                }
            )
    return {"checked": True, "error": "; ".join(errors) if errors else None, "findings": findings}


def collect(check_updates: bool = False) -> dict:
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    skill_dir = Path(__file__).resolve().parents[1]
    models = check_catalog()
    report = {
        "python": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 10),
        "platform": f"{platform.system()} {platform.machine()}",
        "apple_silicon": apple,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "modules": {name: _module_version(name) for name in REQUIRED_MODULES},
        "apple_modules": {name: _module_version(name) for name in APPLE_MODULES},
        "models": models,
        "models_cached": {row["label"]: row["cached"] for row in models},
        "library_catalog": check_library_catalog(),
        "updates": check_hub_updates() if check_updates else {"checked": False, "findings": []},
        "fluidaudio_binary": (skill_dir / "bin" / "macos-arm64" / "fluidaudiocli").is_file(),
        "fluidaudio_models": (_fluid_models_dir() / "plda-parameters.json").is_file(),
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

    warnings = []
    if apple and report["apple_modules"]["mlx_audio"] is None:
        warnings.append("нет mlx-audio: Whisper пойдёт через медленный CPU fallback, диаризация недоступна")
    if not apple:
        warnings.append("не Apple Silicon: диаризация и Whisper Turbo MLX недоступны, Whisper работает на CPU")
    if report["free_gb"] < 8:
        warnings.append("меньше 8 ГБ свободно: модели занимают около 4 ГБ, временные WAV — ещё гигабайты")

    stale = [row for row in models if row["stale"]]
    if stale:
        oldest = max(row["age_days"] for row in stale)
        warnings.append(
            f"замер качества старше {CALIBRATION_SHELF_LIFE_DAYS} дней ({oldest} дн.), "
            f"пора пересверить маршрут: {', '.join(row['label'] for row in stale)}. "
            "Что вышло нового, покажет doctor.py --check-updates"
        )
    for finding in report["library_catalog"]["findings"] + report["updates"]["findings"]:
        warnings.append(finding["text"])
    if report["updates"]["findings"] or stale:
        warnings.append(
            "новая версия не значит «лучше на вашей речи»: перед заменой сверьте WER "
            "по references/quality.md, иначе цифры в отчётах перестанут описывать результат"
        )
    report["warnings"] = warnings
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

    lines.append(
        f"Модели (русский маршрут — {format_gb(route_download_gb('ru', apple=report['apple_silicon']))}, "
        f"английский — {format_gb(route_download_gb('en', apple=report['apple_silicon']))}):"
    )
    for row in report["models"]:
        age = row["age_days"]
        if age is None:
            calibration = "качество отдельно не мерится"
        elif row["stale"]:
            calibration = f"замер {row['calibrated']}, {age} дн. назад — пора пересверить"
        else:
            calibration = f"замер {row['calibrated']}, {age} дн. назад"
        revision = f" · веса {row['revision'][:12]}" if row["revision"] else ""
        lines.append(
            f"  {mark(row['cached'])} {row['label']} ({format_gb(row['download_gb'])}) — {row['role']}"
        )
        lines.append(f"      {row['model']} · {calibration}{revision}")

    lines.append(
        f"FluidAudio (5+ голосов): бинарник {mark(report['fluidaudio_binary'])}, "
        f"CoreML-модели {mark(report['fluidaudio_models'])}"
    )

    updates = report["updates"]
    if updates["checked"]:
        if updates["findings"]:
            lines.append("Обновления на Hugging Face:")
            for finding in updates["findings"]:
                lines.append(f"  ! {finding['text']}")
        else:
            lines.append("Обновления на Hugging Face: новых версий не нашлось")
        if updates.get("error"):
            lines.append(f"  (часть проверок не прошла: {updates['error']})")
    else:
        lines.append("Обновления на Hugging Face не проверялись: запустите с --check-updates")

    for text in report["warnings"]:
        lines.append(f"! {text}")
    for text in report["problems"]:
        lines.append(f"✗ {text}")
    lines.append("Готово к запуску" if not report["problems"] else "Запуск невозможен, см. ✗")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения transcriber")
    parser.add_argument("--json", action="store_true", help="Вывести отчёт в JSON")
    parser.add_argument(
        "--check-updates",
        action="store_true",
        help="Спросить Hugging Face о новых версиях моделей (нужна сеть)",
    )
    args = parser.parse_args()
    report = collect(check_updates=args.check_updates)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
