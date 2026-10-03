#!/usr/bin/env python3
"""Прогон кандидатов английского маршрута по корпусу bake-off.

Модели берутся теми же бэкендами, что и в самом скилле: замер должен описывать
скилл, а не отдельную установку библиотеки. Каждая запись корпуса подаётся
одним окном — нарезку VAD здесь не воспроизводим, эталоны у корпуса на целый
фрагмент.

Окнам раздаются разнесённые метки времени: бэкенды привязывают сегменты к
`chunk.start`, и уникальный старт — единственный способ надёжно вернуть
гипотезу к её записи, когда модель промолчала на части окон.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills" / "transcriber" / "scripts"))

from audio_transcription.backends import GigaAMBackend, MLXWhisperBackend  # noqa: E402
from audio_transcription.models import AudioChunk  # noqa: E402

SCHEMA_VERSION = 1
SLOT_SECONDS = 10_000.0


def build_backend(spec: dict):
    """GigaAMBackend назван по первому жильцу, но это общий бэкенд onnx-asr:

    имя модели и квантование приходят параметрами, поэтому Parakeet и Canary
    идут через него же без единой строки нового кода.
    """
    family = spec["family"]
    if family == "onnx_asr":
        return GigaAMBackend(spec["model"], quantization=spec.get("quantization"))
    if family == "mlx_whisper":
        return MLXWhisperBackend(spec["model"])
    raise ValueError(f"Неизвестное семейство бэкенда: {family}")


def load_corpus(path: Path, suite: str) -> list[dict]:
    """Манифест читается вместе со сверкой sha256 каждого файла.

    Один раз корпус уже пересобрался в ту же папку с прежними именами, аудио
    осталось от старого отбора, и замер честно посчитал WER рассинхрона —
    106% у всех четырёх кандидатов. Несовпадение суммы должно ронять прогон,
    а не превращаться в правдоподобную таблицу.
    """
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    items = [item for item in items if suite in item["suite"]]
    for item in items:
        digest = hashlib.sha256(Path(item["audio"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise SystemExit(
                f"{item['id']}: аудио не совпадает с манифестом — пересоберите корпус"
            )
    return items


def transcribe(spec: dict, items: list[dict]) -> tuple[list[dict], float]:
    backend = build_backend(spec)
    chunks = [
        AudioChunk(
            sequence=index,
            path=Path(item["audio"]),
            start=index * SLOT_SECONDS,
            end=index * SLOT_SECONDS + item["duration_seconds"],
        )
        for index, item in enumerate(items)
    ]
    started = time.monotonic()
    hypothesis = backend.transcribe_chunks(chunks, language="en")
    wall = time.monotonic() - started

    results = []
    for chunk, item in zip(chunks, items):
        spoken = " ".join(
            segment.text
            for segment in hypothesis.segments
            if chunk.start <= segment.start < chunk.end
        ).strip()
        results.append(
            {"id": item["id"], "text": spoken, "reference": item["reference"]}
        )
    return results, wall


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=here / "corpus" / "manifest.jsonl")
    parser.add_argument("--models", type=Path, default=here / "models")
    parser.add_argument("--runs", type=Path, default=here / "runs")
    parser.add_argument("--suite", default="full", choices=("quick", "full"))
    parser.add_argument("--only", action="append", help="Гонять только эти id, можно повторить")
    args = parser.parse_args()

    items = load_corpus(args.corpus, args.suite)
    if not items:
        raise SystemExit(f"В {args.corpus} нет записей набора {args.suite}")
    args.runs.mkdir(parents=True, exist_ok=True)

    specs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.models.glob("*.json"))]
    if args.only:
        specs = [spec for spec in specs if spec["id"] in set(args.only)]

    for spec in specs:
        print(f"→ {spec['id']} ({len(items)} записей)", flush=True)
        try:
            results, wall = transcribe(spec, items)
        except Exception as error:  # noqa: BLE001 — падение одного кандидата не роняет замер
            print(f"  ✗ {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            continue
        payload = {
            "schema_version": SCHEMA_VERSION,
            "model": spec,
            "suite": args.suite,
            "python": sys.version,
            "platform": platform.platform(),
            "runtime": {"note": "Первый элемент включает загрузку модели"},
            "wall_seconds": wall,
            "audio_seconds": sum(item["duration_seconds"] for item in items),
            "items": results,
        }
        target = args.runs / f"{spec['id']}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ {wall:.1f} с → {target.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
