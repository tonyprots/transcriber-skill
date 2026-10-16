#!/usr/bin/env python3
"""Правка своего словаря без ручной правки YAML.

Запуск: <python из .venv> scripts/glossary_edit.py КОМАНДА [--language ru]
        [--store ПУТЬ | --profile ИМЯ] [--force] [--dry-run]

  remember "клод → Claude" ["Битрикс24" …]   человек назвал написание
  forget Zon [Ещё …]                        термин не нужен и не должен вернуться
  add-lines < термины.txt                   пачка строк «слышится → пишется»

`remember` пишет ручную запись с `auto_apply: true` в файл нужного языка
(`--language en` → `glossary.en.yaml`). Голый термин без стрелки — запись без
вариантов: она встанет в текст, когда проверяющая напишет термин верно.
Вариант, занятый другой записью, переезжает к новой. Пара, которую
литеральная замена испортит (падеж вместо ослышки, короткий или непохожий
вариант, перепутанные стороны), не пишется без `--force`.

`forget` переносит запись в раздел `pruned`: просто удалённая вернулась бы из
следующего расхождения.

`add-lines` читает строки из stdin, пустые и начинающиеся с `#` пропускает.
Разделитель — `→`, `->` или `=>`.

Код возврата: 0 — всё записано; 1 — часть пар отклонена (остальное записано);
2 — словарь не читается.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.exiting import exit_after_flush  # noqa: E402
from audio_transcription.glossary_store import (  # noqa: E402
    forget,
    load_document,
    parse_pair,
    remember,
    remember_problem,
    save_document,
    store_path,
)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", type=Path, help="Путь к словарю (по умолчанию ~/.transcriber/glossary.yaml)")
    common.add_argument("--profile", help="Именованный словарь: ~/.transcriber/glossaries/ИМЯ.yaml")
    common.add_argument(
        "--language",
        default="ru",
        help="Язык записей: у каждого свой файл (en → glossary.en.yaml), по умолчанию ru",
    )
    common.add_argument("--force", action="store_true", help="Записать пару, даже если страж против")
    common.add_argument("--dry-run", action="store_true", help="Показать изменения и ничего не писать")
    parser = argparse.ArgumentParser(description="Правка своего словаря transcriber")
    commands = parser.add_subparsers(dest="command", required=True)
    remember_cmd = commands.add_parser("remember", parents=[common], help="Записать написание термина")
    remember_cmd.add_argument("pairs", nargs="+", help="«слышится → пишется» или просто «Термин»")
    forget_cmd = commands.add_parser("forget", parents=[common], help="Убрать термин навсегда")
    forget_cmd.add_argument("canonicals", nargs="+", help="Канон записи")
    commands.add_parser("add-lines", parents=[common], help="Пачка строк из stdin")
    return parser


def main(argv: list[str] | None = None, stdin=None) -> int:
    args = _parser().parse_args(argv)
    try:
        path = store_path(args.store, args.profile, args.language)
        document = load_document(path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2

    results: list[dict] = []
    changed = False
    if args.command == "forget":
        for canonical in args.canonicals:
            updated, notes = forget(document, canonical)
            changed = changed or updated != document
            document = updated
            results.append({"line": canonical, "status": "ok", "notes": notes})
    else:
        if args.command == "remember":
            lines = args.pairs
        else:
            lines = [
                line.strip()
                for line in (stdin or sys.stdin).read().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        for line in lines:
            try:
                heard, written = parse_pair(line)
            except ValueError as error:
                results.append({"line": line, "status": "rejected", "reason": str(error)})
                continue
            problem = remember_problem(document, heard, written, args.language)
            if problem and not args.force:
                results.append({"line": line, "status": "rejected", "reason": problem})
                continue
            updated, notes = remember(document, heard, written, language=args.language)
            if updated == document:
                notes = ["уже записано так"]
            changed = changed or updated != document
            document = updated
            entry = {"line": line, "status": "ok", "notes": notes}
            if problem:
                entry["forced_over"] = problem
            results.append(entry)

    if changed and not args.dry_run:
        save_document(path, document)
    print(
        json.dumps(
            {"store": str(path), "applied": changed and not args.dry_run, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if any(item["status"] == "rejected" for item in results) else 0


if __name__ == "__main__":
    exit_after_flush(main())
