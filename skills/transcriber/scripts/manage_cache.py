#!/usr/bin/env python3
"""Кэш гипотез: сколько весит, убрать старое, забыть одну запись, очистить.

Запуск: <python из .venv> scripts/manage_cache.py stats|prune|forget|clear

В кэше лежит текст записей: гипотезы моделей переживают удаление самой
расшифровки. Прогон сам убирает записи старше 90 дней (`--cache-max-age-days`
у transcribe.py), а здесь — ручное управление:

  stats                    сколько записей, сколько места, самая старая;
  prune --older-than 30    убрать то, к чему не обращались 30 дней;
  forget <файл|sha256>     забыть всё про одну запись и её фрагменты;
  clear                    очистить кэш целиком.

Прогон с `--no-cache` в кэш не пишет вовсе — для записей, текст которых не
должен оставаться на диске.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.cache import cache_stats, clear, forget, prune  # noqa: E402

DEFAULT_CACHE = Path.home() / ".cache" / "transcriber"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def _megabytes(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} МБ".replace(".", ",")


def _source_sha256(value: str) -> str:
    if _SHA256.fullmatch(value):
        return value.lower()
    path = Path(value).expanduser()
    if not path.is_file():
        raise SystemExit(f"Нет такого файла и это не SHA-256: {value}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--json", action="store_true", help="ответ в JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stats", help="размер и возраст кэша")
    pruning = commands.add_parser("prune", help="убрать записи старше срока")
    pruning.add_argument("--older-than", type=float, required=True, metavar="ДНЕЙ")
    forgetting = commands.add_parser("forget", help="забыть одну запись")
    forgetting.add_argument("source", help="исходный файл записи или его SHA-256")
    commands.add_parser("clear", help="очистить кэш целиком")
    args = parser.parse_args(argv)
    cache_dir = args.cache_dir.expanduser()

    if args.command == "stats":
        stats = cache_stats(cache_dir)
        if args.json:
            print(json.dumps(stats, ensure_ascii=False, indent=2))
        else:
            days = stats["oldest_days"]
            oldest = f", самой старой {f'{days:g}'.replace('.', ',')} дн." if days is not None else ""
            print(f"{stats['path']}: записей {stats['files']}, {_megabytes(stats['bytes'])}{oldest}")
            if stats["older_than_max_age"]:
                print(
                    f"Старше {stats['max_age_days']} дн.: {stats['older_than_max_age']} — "
                    "уйдут при следующем прогоне"
                )
        return 0

    if args.command == "prune":
        swept = prune(cache_dir, args.older_than)
        what = f"старше {args.older_than:g} дн."
    elif args.command == "forget":
        sha256 = _source_sha256(args.source)
        swept = forget(cache_dir, sha256)
        what = f"записи {sha256[:12]}"
    else:
        swept = clear(cache_dir)
        what = "всего кэша"
    if args.json:
        print(json.dumps({"removed_files": swept.files, "removed_bytes": swept.bytes}))
    else:
        print(f"Убрано {what}: записей {swept.files}, {_megabytes(swept.bytes)}")
        if args.command == "forget" and not swept.files:
            print(
                "Записи до версии 0.19 не помечены исходником: их уберёт prune по сроку или clear",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
