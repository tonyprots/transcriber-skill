#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


REPO_ID = "FluidInference/speaker-diarization-coreml"
REQUIRED = (
    "pyannote_segmentation.mlmodelc/model.mil",
    "wespeaker_v2.mlmodelc/model.mil",
    "plda-parameters.json",
)


def default_target() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "FluidAudio"
        / "Models"
        / "speaker-diarization-coreml"
    )


def ready(target: Path) -> bool:
    return all((target / relative).is_file() for relative in REQUIRED)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Загрузить локальные CoreML-модели для FluidAudio."
    )
    parser.add_argument("--target", type=Path, default=default_target())
    args = parser.parse_args()
    target = args.target.expanduser().resolve()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("FluidAudio в этом скилле поддерживается только на macOS arm64")
    if ready(target):
        print(json.dumps({"status": "ready", "target": str(target)}, ensure_ascii=False))
        return 0
    if target.exists() and any(target.iterdir()):
        raise SystemExit(
            f"Каталог содержит неполный набор моделей: {target}. "
            "Переместите его в архив и повторите запуск."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit("Не установлен huggingface_hub") from error
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=target,
        allow_patterns=["*.json", "*.mlmodelc/**"],
    )
    if not ready(target):
        raise SystemExit(f"После загрузки набор моделей неполон: {target}")
    print(json.dumps({"status": "downloaded", "target": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
