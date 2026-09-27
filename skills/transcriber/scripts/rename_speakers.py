#!/usr/bin/env python3
"""Имена вместо «Спикер N» в готовой расшифровке — без пересчёта звука.

Запуск: <python из .venv> scripts/rename_speakers.py КАТАЛОГ 1=Антон,2=Мария

Переписывает readable.md, verbatim.md и субтитры, имена запоминает в
manifest.json (`speaker_names`). Метки в raw.json, segments.json и
speakers.json не трогаются: это аудит-след, а имя — подпись для человека.
Повторный запуск добавляет и заменяет имена, прежние остаются.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.exiting import exit_after_flush  # noqa: E402
from audio_transcription.render import parse_speaker_names, rename_speakers  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Назвать говорящих в готовой расшифровке")
    parser.add_argument("output", type=Path, help="Каталог результата transcribe.py")
    parser.add_argument("names", help="Имена по номерам из текста: 1=Антон,2=Мария; unknown=… для неразмеченной речи")
    args = parser.parse_args(argv)
    try:
        names = parse_speaker_names(args.names)
        missing = rename_speakers(args.output, names)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    if missing:
        print(f"Внимание: в записи нет меток {', '.join(missing)} — имя для них не понадобилось", file=sys.stderr)
    print(json.dumps({"output": str(args.output.expanduser().resolve()), "speaker_names": names}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    exit_after_flush(main())
