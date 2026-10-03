#!/usr/bin/env python3
"""Ищет ютьюб-видео, пригодные в эталон: с ручными субтитрами и разговором.

Смысл бенчмарка. Сравнивать нашу расшифровку с автосубтитрами можно только
относительно третьего текста, который писал человек. Такой текст на ютьюбе
есть — дорожка субтитров, загруженная автором, а не собранная машиной; в
`yt-dlp` они лежат в разных полях, и наш `fetching.probe_remote` их уже
различает.

Что отбираем: разговорные жанры (интервью, дебаты, подкаст), 5–25 минут,
ручная дорожка на нужном языке. Один голос в кадре — редкость для ютьюба,
поэтому запросы нарочно про диалог.

Скрипт ничего не качает, кроме метаданных: аудио и субтитры забирает уже
`build_corpus.py` по отобранному списку.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills" / "transcriber" / "scripts"))

from audio_transcription.fetching import probe_remote  # noqa: E402

QUERIES = {
    "en": [
        "english conversation between two people subtitles",
        "interview with subtitles CC full episode",
        "panel discussion subtitles closed captions",
        "podcast episode english subtitles conversation",
    ],
    "ru": [
        "разговорный русский диалог с субтитрами",
        "russian conversation with subtitles two speakers",
        "интервью на русском с субтитрами",
        "подкаст на русском с субтитрами разговор",
        "russian podcast subtitles interview",
        "лекция на русском с субтитрами",
        "дебаты на русском субтитры",
        "TEDx на русском языке субтитры",
        "круглый стол дискуссия субтитры русский",
        "разговор двух человек русский субтитры автор",
    ],
}

MIN_SECONDS = 300
MAX_SECONDS = 1500


def search(query: str, limit: int) -> list[dict]:
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--dump-single-json", f"ytsearch{limit}:{query}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"  поиск не удался: {result.stderr.strip()[-200:]}", file=sys.stderr)
        return []
    return json.loads(result.stdout or "{}").get("entries", []) or []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", action="append", choices=("ru", "en"))
    parser.add_argument("--per-query", type=int, default=8)
    parser.add_argument("--need", type=int, default=3, help="сколько видео на язык")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "candidates.json")
    args = parser.parse_args()

    languages = args.language or ["en", "ru"]
    found: dict[str, list[dict]] = {language: [] for language in languages}
    for language in languages:
        for query in QUERIES[language]:
            if len(found[language]) >= args.need:
                break
            print(f"[{language}] {query}", flush=True)
            for entry in search(query, args.per_query):
                duration = entry.get("duration") or 0
                if not MIN_SECONDS <= duration <= MAX_SECONDS:
                    continue
                url = f"https://www.youtube.com/watch?v={entry['id']}"
                try:
                    media = probe_remote(url)
                except Exception as error:  # noqa: BLE001 — недоступное видео пропускаем
                    print(f"    {entry['id']}: {type(error).__name__}", flush=True)
                    continue
                manual = [track for track in media.tracks if not track.automatic]
                match = [
                    track for track in manual if track.language.lower().startswith(language)
                ]
                if not match:
                    continue
                found[language].append(
                    {
                        "id": entry["id"],
                        "url": url,
                        "title": media.title,
                        "language": language,
                        "duration_seconds": media.duration_seconds,
                        "manual_track": match[0].language,
                        "auto_tracks": sum(1 for track in media.tracks if track.automatic),
                    }
                )
                print(
                    f"    ✓ {entry['id']} {media.duration_seconds:.0f} с — "
                    f"ручная дорожка {match[0].language}",
                    flush=True,
                )
                if len(found[language]) >= args.need:
                    break

    args.output.write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({lang: len(items) for lang, items in found.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
