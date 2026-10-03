#!/usr/bin/env python3
"""Корпус для английского bake-off: 30 фрагментов Earnings-22 и 10 AMI.

Почему эти два. Earnings-22 — корпоративные созвоны: имена компаний, цифры,
термины и неродные акценты, то есть ровно то, на чём разваливаются
автосубтитры. AMI — спонтанная речь встреч с перебиваниями. TED-LIUM в
открытом виде отдаётся только целыми докладами (`LIUM/tedlium` закрыт
аутентификацией), а резать его без выравнивания эталона нельзя.

Короткие сегменты отбрасываются намеренно. В обоих корпусах медиана реплики —
пара секунд («YEAH»), а скилл работает окнами до 20 секунд, и спор идёт про
имена собственные в контексте. Поэтому берём только длинные фрагменты.

Разнообразие записей держит `MAX_PER_RECORDING = 1`: один фрагмент на созвон,
тридцать фрагментов — тридцать разных созвонов. Потолок в три давал десять
записей, потому что отбор останавливается на нужном числе и успевает пройти
только первые страницы. Поле `recording` в манифесте нужно, чтобы это было
видно, а не предполагалось.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
SIZE_API = "https://datasets-server.huggingface.co/size"
PAGE = 100
# Страницы берутся вразнобой по всему сплиту, а не подряд: соседние строки —
# это соседние реплики одного созвона, и сорок страниц подряд упираются в
# лимит «не больше трёх из записи», покрыв 7% корпуса.
MAX_PAGES = 60
# datasets-server отдаёт 429 уже на втором десятке страниц подряд, поэтому
# между страницами держим паузу, а на отказ отвечаем ожиданием, а не падением.
PAGE_PAUSE_SECONDS = 1.5
RETRY_DELAYS = (5, 15, 45, 90)
MIN_SECONDS = 8.0
MIN_WORDS = 15
MAX_PER_RECORDING = 1

SOURCES = (
    {
        "name": "earnings22",
        "dataset": "distil-whisper/earnings22",
        "config": "chunked",
        "split": "test",
        "take": 30,
        "text": "transcription",
        "group": "file_id",
        "span": ("start_ts", "end_ts"),
        "license": "CC BY-SA 4.0",
    },
    {
        "name": "ami",
        "dataset": "edinburghcstr/ami",
        "config": "ihm",
        "split": "test",
        "take": 10,
        "text": "text",
        "group": "meeting_id",
        "span": ("begin_time", "end_time"),
        "license": "CC BY 4.0",
    },
)


def fetch_page(source: dict, offset: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": source["dataset"],
            "config": source["config"],
            "split": source["split"],
            "offset": offset,
            "length": PAGE,
        }
    )
    request = urllib.request.Request(
        f"{API}?{query}", headers={"User-Agent": "transcriber-bakeoff/1"}
    )
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response).get("rows", [])
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or delay is None:
                raise
            wait = float(error.headers.get("Retry-After") or delay)
            print(f"    HTTP {error.code}, жду {wait:.0f} с (попытка {attempt + 1})", flush=True)
            time.sleep(wait)
    return []


def split_rows(source: dict) -> int:
    query = urllib.parse.urlencode(
        {"dataset": source["dataset"], "config": source["config"], "split": source["split"]}
    )
    request = urllib.request.Request(
        f"{SIZE_API}?{query}", headers={"User-Agent": "transcriber-bakeoff/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        splits = json.load(response)["size"]["splits"]
    return next(item["num_rows"] for item in splits if item["split"] == source["split"])


def offsets(total: int) -> list[int]:
    step = max(PAGE, total // MAX_PAGES)
    return [offset for offset in range(0, max(total - PAGE, 0) + 1, step)][:MAX_PAGES]


def select(source: dict) -> list[dict]:
    chosen: list[dict] = []
    per_recording: dict[str, int] = {}
    total = split_rows(source)
    print(f"{source['name']}: {total} строк в сплите", flush=True)
    for page, offset in enumerate(offsets(total)):
        if page:
            time.sleep(PAGE_PAUSE_SECONDS)
        rows = fetch_page(source, offset)
        if not rows:
            break
        for item in rows:
            row = item["row"]
            start, end = (float(row[key]) for key in source["span"])
            reference = str(row[source["text"]]).strip()
            recording = str(row[source["group"]])
            if end - start < MIN_SECONDS or len(reference.split()) < MIN_WORDS:
                continue
            if per_recording.get(recording, 0) >= MAX_PER_RECORDING:
                continue
            per_recording[recording] = per_recording.get(recording, 0) + 1
            chosen.append(
                {
                    "row": row,
                    "row_idx": item["row_idx"],
                    "reference": reference,
                    "recording": recording,
                }
            )
            if len(chosen) == source["take"]:
                return chosen
    raise RuntimeError(
        f"{source['name']}: набрано {len(chosen)} из {source['take']} "
        f"за {MAX_PAGES} страниц по всему сплиту — ослабьте фильтры"
    )


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "transcriber-bakeoff/1"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)


def normalize(source_path: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_path),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(target),
        ],
        check=True,
    )


def duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "corpus")
    args = parser.parse_args()
    output = args.output.resolve()
    raw_dir, audio_dir = output / "raw", output / "audio"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for source in SOURCES:
        for position, picked in enumerate(select(source)):
            identifier = f"{source['name']}-{position:03d}"
            # Имя файла — по строке датасета, а не по позиции в корпусе:
            # позиционное имя при смене отбора молча оставляло старое аудио
            # рядом с новым эталоном, и замер мерил рассинхрон, а не модель.
            cache_name = f"{source['name']}-r{int(picked['row_idx']):06d}.wav"
            raw_path, audio_path = raw_dir / cache_name, audio_dir / cache_name
            if not raw_path.exists():
                download(str(picked["row"]["audio"][0]["src"]), raw_path)
            if not audio_path.exists():
                normalize(raw_path, audio_path)
            manifest.append(
                {
                    "id": identifier,
                    "suite": ["quick", "full"] if position < 5 else ["full"],
                    "audio": str(audio_path),
                    "duration_seconds": duration(audio_path),
                    "reference": picked["reference"],
                    "language": "en",
                    "source": source["dataset"],
                    "source_config": source["config"],
                    "source_row": int(picked["row_idx"]),
                    "recording": picked["recording"],
                    "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                    "license": source["license"],
                }
            )
            print(f"  {identifier}: {manifest[-1]['duration_seconds']:.1f} с", flush=True)

    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifest),
        encoding="utf-8",
    )
    total = sum(item["duration_seconds"] for item in manifest)
    print(
        json.dumps(
            {"manifest": str(manifest_path), "items": len(manifest), "seconds": round(total)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
