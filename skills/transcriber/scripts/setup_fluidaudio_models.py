#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.fluid_models import ModelsUnavailable, default_target, ensure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Загрузить локальные CoreML-модели для FluidAudio. Обычно не нужно: "
            "первый прогон с --diarize ставит их сам"
        )
    )
    parser.add_argument("--target", type=Path, default=default_target())
    args = parser.parse_args()
    target = args.target.expanduser().resolve()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("FluidAudio в этом скилле поддерживается только на macOS arm64")
    try:
        status = ensure(target)
    except ModelsUnavailable as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"status": status, "target": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
