#!/usr/bin/env python3
"""Сводка английского bake-off: WER и CER по кандидатам и по срезам корпуса.

Скорер берётся из `multilingual-bakeoff`: нормализация и подсчёт расстояния
там уже покрыты тестами, а две копии одной метрики разойдутся молча и сделают
замеры несравнимыми.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "multilingual-bakeoff"))

from score import error_rate  # noqa: E402


def score_run(payload: dict) -> dict:
    slices: dict[str, dict[str, int]] = {}
    for item in payload["items"]:
        source = item["id"].rsplit("-", 1)[0]
        for name, characters in (("word", False), ("char", True)):
            errors, total = error_rate(item["reference"], item["text"], characters=characters)
            bucket = slices.setdefault(source, {})
            bucket[f"{name}_errors"] = bucket.get(f"{name}_errors", 0) + errors
            bucket[f"{name}_total"] = bucket.get(f"{name}_total", 0) + total

    overall = {"word_errors": 0, "word_total": 0, "char_errors": 0, "char_total": 0}
    for bucket in slices.values():
        for key in overall:
            overall[key] += bucket.get(key, 0)

    audio = payload.get("audio_seconds") or 0.0
    wall = payload.get("wall_seconds") or 0.0
    return {
        "model": payload["model"]["id"],
        "role": payload["model"].get("role", ""),
        "status": payload["model"].get("status", ""),
        "word_errors": overall["word_errors"],
        "word_total": overall["word_total"],
        "wer": _rate(overall["word_errors"], overall["word_total"]),
        "cer": _rate(overall["char_errors"], overall["char_total"]),
        "rtfx": round(audio / wall, 1) if wall else None,
        "wall_seconds": round(wall, 1),
        "slices": {
            name: _rate(bucket["word_errors"], bucket["word_total"])
            for name, bucket in sorted(slices.items())
        },
    }


def _rate(errors: int, total: int) -> float | None:
    return round(errors / total, 4) if total else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--output", type=Path, default=HERE / "report.md")
    parser.add_argument("--scores", type=Path, default=HERE / "scores.json")
    args = parser.parse_args()

    scores = [score_run(json.loads(path.read_text(encoding="utf-8")))
              for path in sorted(args.runs.glob("*.json"))]
    if not scores:
        raise SystemExit(f"В {args.runs} нет прогонов")
    scores.sort(key=lambda item: (item["wer"] is None, item["wer"]))
    args.scores.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    slice_names = sorted({name for item in scores for name in item["slices"]})
    lines = [
        "# Английский bake-off",
        "",
        "Корпус: 30 фрагментов Earnings-22 (корпоративные созвоны) и 10 AMI "
        "(встречи), только сегменты длиннее 8 секунд и от 15 слов — короткие "
        "реплики не проверяют то, ради чего затевался замер.",
        "",
        "| Модель | Роль | WER | CER | Ошибок / слов | "
        + " | ".join(f"WER {name}" for name in slice_names)
        + " | RTFx |",
        "|---|---|---|---|---|" + "---|" * (len(slice_names) + 1),
    ]
    for item in scores:
        cells = [
            f"`{item['model']}`",
            item["role"],
            _percent(item["wer"]),
            _percent(item["cer"]),
            f"{item['word_errors']} / {item['word_total']}",
            *[_percent(item["slices"].get(name)) for name in slice_names],
            f"{item['rtfx']:.0f}×" if item["rtfx"] else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "RTFx — во сколько раз быстрее реального времени, вместе с загрузкой "
        "модели: на корпусе в несколько минут она заметно занижает результат.",
        "",
        "Колонка «ошибок / слов» важнее процентов: корпус маленький, и разница "
        "в единицы ошибок — это шум, а не преимущество модели. Срез AMI тем "
        "более: десять фрагментов, полторы минуты речи.",
        "",
        f"Сырые прогоны — `runs/`, построчные метрики — `{args.scores.name}`.",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "models": len(scores)}, ensure_ascii=False))
    return 0


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


if __name__ == "__main__":
    raise SystemExit(main())
