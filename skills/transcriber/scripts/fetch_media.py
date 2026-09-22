#!/usr/bin/env python3
"""Забрать у ссылки то, что нужно, не запуская расшифровку.

Скрипт печатает в stdout компактный JSON с путями и счётчиками и никогда —
содержимое субтитров: текст читают из файла, и только когда он действительно
нужен. Сырой VTT часового ролика весит под мегабайт, и тащить его в контекст
целиком незачем.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.exiting import exit_after_flush
from audio_transcription.fetching import (
    FetchError,
    fetch_audio,
    fetch_subtitles,
    probe_remote,
    read_cues,
    render_captions,
)
from audio_transcription.mirror import ANCHOR_SECONDS, ANCHORS, find_mirror

TRANSCRIBE = Path(__file__).resolve().parent / "transcribe.py"
# Пауза между ссылками пакета: 2026-09-22 два десятка запросов подряд к
# YouTube упёрлись в HTTP 429, и соседняя сессия обходила его циклом со sleep.
BATCH_PAUSE_SECONDS = 3.0
_sleep = time.sleep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch-media",
        description="Опросить ссылку, забрать субтитры или скачать аудио.",
    )
    parser.add_argument(
        "url",
        nargs="+",
        help="Ссылка на видео; несколько — пакетом, с паузой между запросами",
    )
    what = parser.add_mutually_exclusive_group()
    what.add_argument(
        "--probe",
        action="store_true",
        help="Что за видео и какие дорожки есть (по умолчанию)",
    )
    what.add_argument(
        "--subtitles",
        action="store_true",
        help="Забрать готовые субтитры источника — чужая гипотеза, зато секунды",
    )
    what.add_argument(
        "--audio",
        action="store_true",
        help="Только скачать звук; расшифровку запускает transcribe.py",
    )
    parser.add_argument(
        "--language",
        action="append",
        metavar="КОД",
        help="Предпочитаемый язык дорожки, можно повторить: --language ru --language en",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Только авторские субтитры: автоматические не брать",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help=(
            "Со --subtitles: если у ролика нет субтитров, искать ту же запись на "
            "YouTube и брать её субтитры — только после сверки по звуку в двух точках"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Каталог результата, по умолчанию ./media; в пакете у каждой ссылки "
            "свой подкаталог по названию"
        ),
    )
    return parser


def run(args: argparse.Namespace, url: str, destination: Path) -> dict[str, object]:
    languages = tuple(args.language or ())
    media = probe_remote(url)
    if not (args.subtitles or args.audio):
        picked = media.track_for(languages, allow_automatic=not args.no_auto)
        return {
            **media.summary(languages),
            "picked": _describe(media, picked),
        }

    if len(args.url) > 1:
        destination = destination / _slug(media.title)
    if args.audio:
        path = fetch_audio(media.url, destination)
        return {
            "audio": str(path),
            "size_bytes": path.stat().st_size,
            "duration_seconds": media.duration_seconds,
        }

    own = media.track_for(languages, allow_automatic=not args.no_auto)
    if args.mirror and own is None and media.extractor != "Youtube":
        return _from_mirror(media, languages, destination)
    raw, track = fetch_subtitles(
        media,
        destination,
        languages=languages,
        allow_automatic=not args.no_auto,
    )
    cues = read_cues(raw)
    captions = raw.with_name("captions.md")
    rendered = render_captions(media, track, cues)
    captions.write_text(rendered, encoding="utf-8")
    return {
        "captions": str(captions),
        "raw": str(raw),
        "track": _describe(media, track),
        "cues": len(cues),
        "characters": len(rendered),
    }


def _from_mirror(media, languages: tuple[str, ...], destination: Path) -> dict[str, object]:
    """Субтитры с копии на YouTube или отказ с причинами по каждому кандидату."""
    language = (languages or (media.language or "ru",))[0]
    found, rejected = find_mirror(
        media, language, destination, _anchor_transcriber(media, language)
    )
    if found is None:
        raise FetchError(
            "Своих субтитров нет, и проверенной копии на YouTube не нашлось "
            f"({json.dumps(rejected, ensure_ascii=False)}). Расшифруйте звук: transcribe.py."
        )
    verdict = found.verdict.to_dict()
    captions = destination / "captions.md"
    rendered = render_captions(media, found.track, found.cues, mirror=verdict)
    destination.mkdir(parents=True, exist_ok=True)
    captions.write_text(rendered, encoding="utf-8")
    return {
        "captions": str(captions),
        "raw": str(found.raw),
        "track": found.track.to_dict(),
        "mirror": verdict,
        "rejected": rejected,
        "cues": len(found.cues),
        "characters": len(rendered),
    }


def _anchor_transcriber(media, language: str):
    """Расшифровка опорных точек оригинала своим конвейером, один раз на все копии.

    Через `transcribe.py`, а не импортом: так точки проходят общую очередь
    машины и кэш, а процесс опроса не тянет в память модели.
    """
    heard: dict[int, str] = {}

    def transcribe(start: float, end: float) -> str:
        if not heard:
            heard.update(_transcribe_anchors(media, language))
        return heard.get(round(start), "")

    return transcribe


def _transcribe_anchors(media, language: str) -> dict[int, str]:
    starts = [round(media.duration_seconds * point) for point in ANCHORS]
    with tempfile.TemporaryDirectory(prefix="transcriber-mirror-") as work:
        command = [
            sys.executable, str(TRANSCRIBE), media.url, "--mode", "fast",
            "--no-learn", "--language", language, "--output", work,
        ]
        for start in starts:
            command += ["--section", f"{start}-{start + round(ANCHOR_SECONDS)}"]
        # Прогресс и ожидание очереди идут в stderr как есть: без них пять минут
        # в очереди за чужим прогоном неотличимы от зависания.
        done = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        try:
            results = json.loads(done.stdout)["results"]
        except (json.JSONDecodeError, KeyError) as error:
            raise FetchError(
                "Не удалось расшифровать точки сверки — причина в stderr выше"
            ) from error
        heard: dict[int, str] = {}
        for result in results:
            if not result.get("output") or not result.get("section"):
                continue
            segments = json.loads(
                Path(result["output"], "segments.json").read_text(encoding="utf-8")
            )
            heard[round(result["section"]["start"])] = " ".join(
                item["text"] for item in segments["readable"]
            )
        return heard


def _describe(media, track) -> dict[str, object] | None:
    """Пометка о переводе нужна до скачивания: она меняет решение, брать ли дорожку."""
    if track is None:
        return None
    return {**track.to_dict(), "translated": media.is_translation(track)}


def _slug(title: str) -> str:
    return re.sub(r"[^\w]+", "-", title.lower()).strip("-")[:60] or "video"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = (args.output or Path("media")).expanduser().resolve()
    if len(args.url) == 1:
        try:
            payload = run(args, args.url[0], destination)
        except FetchError as error:
            print(f"Ошибка: {error}", file=sys.stderr)
            return 1
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    # Пакет: отказ одной ссылки не останавливает остальные, а ролик без
    # субтитров не теряется — он получает `error` и идёт в transcribe.py.
    results: list[dict[str, object]] = []
    for index, url in enumerate(args.url):
        if index:
            _sleep(BATCH_PAUSE_SECONDS)
        print(f"Пакет {index + 1}/{len(args.url)}: {url}", file=sys.stderr)
        try:
            results.append({"input": url, **run(args, url, destination)})
        except FetchError as error:
            results.append({"input": url, "error": str(error)})
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 2 if any("error" in item for item in results) else 0


if __name__ == "__main__":
    exit_after_flush(main())
