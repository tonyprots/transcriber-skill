#!/usr/bin/env python3
"""Уборка словаря, который скилл ведёт сам.

Запуск: <python из .venv> scripts/glossary_prune.py [--store ПУТЬ] [--dry-run]

Снимает выученные записи, которые никогда не станут термином («E ← и»,
«Zon ← там»), мусорные варианты у живых записей и выученные дубли ручных
записей; вариант, который числится за несколькими записями, оставляет одной.
Ручные и уже включённые в автозамену записи не трогает. Снятое уезжает в
раздел `pruned` с причиной и больше не заводится заново.

Та же уборка идёт после каждого прогона сама; скрипт нужен, чтобы увидеть
её итог или прибрать словарь сразу. По умолчанию применяет, `--dry-run` —
только показать.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.exiting import exit_after_flush  # noqa: E402
from audio_transcription.glossary_store import (  # noqa: E402
    load_document,
    prune,
    save_document,
    store_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Убрать мусор из словаря скилла")
    parser.add_argument("--store", type=Path, help="Путь к словарю (по умолчанию ~/.transcriber/glossary.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Показать, что будет снято, и ничего не менять")
    args = parser.parse_args(argv)
    path = store_path(args.store)
    try:
        document = load_document(path)
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    before = len(document["entries"])
    cleaned, report = prune(document)
    if not args.dry_run and (report.removed or report.trimmed):
        save_document(path, cleaned)
    print(
        json.dumps(
            {
                "store": str(path),
                "applied": not args.dry_run,
                "entries_before": before,
                "entries_after": len(cleaned["entries"]),
                **report.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    exit_after_flush(main())
