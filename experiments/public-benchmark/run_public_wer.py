#!/usr/bin/env python3
"""WER основной и проверочной модели на публичных русских наборах.

Запуск: .venv/bin/python experiments/public-benchmark/run_public_wer.py \
        --dataset fleurs|cv [--limit N] [--seed 7] [--data ~/.cache/transcriber-eval]

Зачем: замеры в README сделаны на речи одного автора. Здесь то же сравнение
на чужих голосах — FLEURS ru (test, чтение Википедии) и Common Voice 17 ru
(test, чтение предложений, много микрофонов). Условия как в конвейере: каждый
файл режет тот же Silero VAD на окна до 20 с, обе модели получают одинаковые
окна. WER — после `normalize_text` (регистр, пунктуация, ё/е).

Числа — главный источник ложных ошибок на публичных наборах: эталон пишет
«1990», модель — «тысяча девятьсот девяностом», или наоборот. Поэтому отчёт
считает WER ещё и на подмножестве фраз, где цифр нет ни в эталоне, ни в
гипотезах обеих моделей. 95 % интервалы — бутстрэп по фразам.

Данные в репозиторий не входят: FLEURS — google/fleurs, Common Voice —
зеркало fsicoli/common_voice_17_0 (официальный набор Mozilla убран с HF).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills" / "transcriber" / "scripts"))

from audio_transcription.audio import prepare_audio, split_speech_windows  # noqa: E402
from audio_transcription.backends import GigaAMBackend, MLXWhisperBackend  # noqa: E402
from audio_transcription.fillers import strip_fillers  # noqa: E402
from audio_transcription.models import AudioChunk  # noqa: E402
from audio_transcription.reconcile import normalize_text  # noqa: E402

STRIDE = 10_000.0  # окна разных файлов разводятся по времени, чтобы не слиплись
DIGIT = re.compile(r"\d")
csv.field_size_limit(10**7)


def edit_distance(hypothesis: list[str], reference: list[str]) -> int:
    previous = list(range(len(reference) + 1))
    for i, left in enumerate(hypothesis, start=1):
        current = [i]
        for j, right in enumerate(reference, start=1):
            current.append(
                previous[j - 1]
                if left == right
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def load_items(dataset: str, data: Path, limit: int | None, seed: int) -> list[dict]:
    if dataset == "fleurs":
        rows = list(csv.reader((data / "fleurs/test.tsv").open(encoding="utf-8"), delimiter="\t"))
        items = [{"id": row[1].removesuffix(".wav"), "member": f"test/{row[1]}", "gold": row[2]} for row in rows]
        archive = data / "fleurs/test.tar.gz"
    else:
        rows = list(csv.DictReader((data / "cv/test.tsv").open(encoding="utf-8"), delimiter="\t"))
        items = [
            {"id": row["path"].removesuffix(".mp3"), "member": f"ru_test_0/{row['path']}", "gold": row["sentence"]}
            for row in rows
        ]
        archive = data / "cv/ru_test_0.tar"
    if limit and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    for item in items:
        item["archive"] = archive
    return items


def extract(items: list[dict], target: Path) -> None:
    wanted = {item["member"]: item for item in items if not (target / Path(item["member"]).name).exists()}
    if wanted:
        archives = {item["archive"] for item in wanted.values()}
        for archive in archives:
            with tarfile.open(archive) as tar:
                for member in tar:
                    if member.name in wanted and member.isfile():
                        source = tar.extractfile(member)
                        (target / Path(member.name).name).write_bytes(source.read())
    for item in items:
        item["audio"] = target / Path(item["member"]).name


def bootstrap(rows: list[dict], key: str, seed: int, rounds: int = 2000) -> tuple[float, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        words = sum(row["words"] for row in sample)
        values.append(sum(row[key] for row in sample) / words if words else 0.0)
    values.sort()
    return values[int(0.025 * rounds)], values[int(0.975 * rounds)]


def ratio_interval(rows: list[dict], seed: int, rounds: int = 2000) -> tuple[float, float, float]:
    rng = random.Random(seed + 1)
    point = sum(r["whisper_errors"] for r in rows) / max(1, sum(r["gigaam_errors"] for r in rows))
    values = []
    for _ in range(rounds):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        values.append(sum(r["whisper_errors"] for r in sample) / max(1, sum(r["gigaam_errors"] for r in sample)))
    values.sort()
    return point, values[int(0.025 * rounds)], values[int(0.975 * rounds)]


def summarize(rows: list[dict], seed: int) -> dict:
    words = sum(row["words"] for row in rows)
    result = {"utterances": len(rows), "reference_words": words}
    for key in ("gigaam_errors", "gigaam_nofill_errors", "whisper_errors"):
        errors = sum(row[key] for row in rows)
        low, high = bootstrap(rows, key, seed)
        result[key.removesuffix("_errors")] = {"errors": errors, "wer": errors / words, "ci95": [low, high]}
    point, low, high = ratio_interval(rows, seed)
    result["whisper_to_gigaam_error_ratio"] = {"point": point, "ci95": [low, high]}
    result["gigaam_better_utterances"] = sum(r["gigaam_errors"] < r["whisper_errors"] for r in rows)
    result["whisper_better_utterances"] = sum(r["whisper_errors"] < r["gigaam_errors"] for r in rows)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=["fleurs", "cv"], required=True)
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/transcriber-eval")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args(argv)

    work = args.data / f"work-{args.dataset}"
    audio_dir = work / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(args.dataset, args.data, args.limit, args.seed)
    extract(items, audio_dir)
    print(f"{args.dataset}: фраз {len(items)}", file=sys.stderr)

    chunks: list[AudioChunk] = []
    owner: list[int] = []
    for index, item in enumerate(items):
        item_dir = work / "windows" / item["id"]
        prepared, _ = prepare_audio(item["audio"], item_dir)
        for chunk in split_speech_windows(prepared, item_dir, offline=True):
            start = index * STRIDE + chunk.start
            chunks.append(AudioChunk(len(chunks), chunk.path, start, start + chunk.duration))
            owner.append(index)
    print(f"окон: {len(chunks)}", file=sys.stderr)

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"  {done}/{total}", file=sys.stderr, flush=True)

    hypotheses = {}
    for label, backend in (
        ("gigaam", GigaAMBackend("gigaam-v3-e2e-rnnt", offline=True)),
        ("whisper", MLXWhisperBackend(offline=True)),
    ):
        print(f"{label}…", file=sys.stderr)
        hypothesis = backend.transcribe_chunks(chunks, "ru", progress=progress)
        texts: dict[int, list[tuple[float, str]]] = {}
        for segment in hypothesis.segments:
            texts.setdefault(int(segment.start // STRIDE), []).append((segment.start, segment.text))
        hypotheses[label] = {k: " ".join(t for _, t in sorted(v)) for k, v in texts.items()}

    rows = []
    for index, item in enumerate(items):
        reference = normalize_text(item["gold"]).split()
        giga = hypotheses["gigaam"].get(index, "")
        whisper = hypotheses["whisper"].get(index, "")
        nofill, _ = strip_fillers(giga)
        rows.append(
            {
                "id": item["id"],
                "gold": item["gold"],
                "gigaam": giga,
                "whisper": whisper,
                "words": len(reference),
                "gigaam_errors": edit_distance(normalize_text(giga).split(), reference),
                "gigaam_nofill_errors": edit_distance(normalize_text(nofill).split(), reference),
                "whisper_errors": edit_distance(normalize_text(whisper).split(), reference),
                "has_digits": bool(DIGIT.search(item["gold"] + giga + whisper)),
            }
        )

    report = {
        "dataset": args.dataset,
        "seed": args.seed,
        "limit": args.limit,
        "all": summarize(rows, args.seed),
        "no_digits": summarize([r for r in rows if not r["has_digits"]], args.seed),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.dataset}-utterances.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output / f"{args.dataset}-summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
