#!/usr/bin/env python3
"""ROVER: голосование нескольких гипотез вместо выбора одной.

Классика NIST: гипотезы выравниваются в сеть переходов, в каждой точке
ветвления голосуют системы, побеждает частотный вариант. Работает, только
если системы ошибаются по-разному, — поэтому здесь Whisper (attention
encoder-decoder), Parakeet (трансдьюсер) и Canary (AED другой школы), а не
три родственные модели.

Скрипт ничего не распознаёт: он читает готовые прогоны `runs/*.json`, поэтому
стоит нуля вычислений и повторяется на любой машине.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "multilingual-bakeoff"))

from score import error_rate, normalize  # noqa: E402

NULL = ""


def vote(slot: list[str], priority: dict[str, int]) -> str:
    """Побеждает частотный вариант; ничью решает более точная система.

    С тремя системами ничьи — обычное дело, и разрешать их случайным порядком
    нельзя: тогда результат зависит от того, в каком порядке лежат файлы.
    """
    counts = Counter(slot)
    best = max(counts.values())
    tied = [word for word, count in counts.items() if count == best]
    if len(tied) == 1:
        return tied[0]
    for index, word in enumerate(slot):
        if word in tied and index == min(
            position for position, value in enumerate(slot) if value in tied
        ):
            return word
    return tied[0]


def merge(network: list[list[str]], words: list[str], voters: int) -> list[list[str]]:
    """Добавляет очередную гипотезу в сеть переходов."""
    reference = [Counter(slot).most_common(1)[0][0] for slot in network]
    merged: list[list[str]] = []
    matcher = SequenceMatcher(None, reference, words, autojunk=False)
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(left_end - left_start):
                merged.append(network[left_start + offset] + [words[right_start + offset]])
            continue
        left = list(range(left_start, left_end))
        right = list(range(right_start, right_end))
        # Замены складываются позиция в позицию, хвост дополняется пустыми
        # голосами: пропуск слова — это тоже голос, и он должен считаться.
        for left_index, right_index in itertools.zip_longest(left, right):
            slot = network[left_index] if left_index is not None else [NULL] * voters
            word = words[right_index] if right_index is not None else NULL
            merged.append(slot + [word])
    return merged


def combine(hypotheses: list[list[str]], priority: dict[str, int] | None = None) -> list[str]:
    if not hypotheses:
        return []
    network: list[list[str]] = [[word] for word in hypotheses[0]]
    for index, words in enumerate(hypotheses[1:], start=1):
        network = merge(network, words, voters=index)
    return [word for word in (vote(slot, priority or {}) for slot in network) if word]


def load_runs(runs: Path) -> dict[str, dict[str, dict]]:
    return {
        path.stem: {item["id"]: item for item in json.loads(path.read_text(encoding="utf-8"))["items"]}
        for path in sorted(runs.glob("*.json"))
    }


def score(pairs: list[tuple[str, str]]) -> float:
    errors = total = 0
    for reference, hypothesis in pairs:
        item_errors, item_total = error_rate(reference, hypothesis)
        errors += item_errors
        total += item_total
    return errors / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--members",
        action="append",
        help="Идентификаторы моделей в порядке приоритета; можно повторить",
    )
    parser.add_argument("--output", type=Path, default=HERE / "rover.json")
    args = parser.parse_args()

    data = load_runs(args.runs)
    members = args.members or [
        "whisper-large-v3-turbo-mlx-fp16",
        "parakeet-tdt-0.6b-v3",
        "canary-1b-v2",
    ]
    missing = [name for name in members if name not in data]
    if missing:
        raise SystemExit(f"Нет прогонов: {', '.join(missing)}")

    ids = list(data[members[0]])
    singles = {
        name: score([(data[name][key]["reference"], data[name][key]["text"]) for key in ids])
        for name in members
    }
    fused_pairs: list[tuple[str, str]] = []
    oracle_pairs: list[tuple[str, str]] = []
    for key in ids:
        reference = data[members[0]][key]["reference"]
        hypotheses = [normalize(data[name][key]["text"]).split() for name in members]
        fused_pairs.append((reference, " ".join(combine(hypotheses))))
        # Оракул выбора системы: нижняя граница для «просто выбрать лучшую
        # модель на каждом фрагменте». Если фьюжн её не бьёт, голосование
        # не даёт ничего сверх удачного выбора.
        best = min(
            (data[name][key]["text"] for name in members),
            key=lambda text: error_rate(reference, text)[0],
        )
        oracle_pairs.append((reference, best))

    result = {
        "members": members,
        "singles": {name: round(value, 4) for name, value in singles.items()},
        "rover": round(score(fused_pairs), 4),
        "oracle_pick": round(score(oracle_pairs), 4),
        "items": len(ids),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    best_single = min(singles.values())
    gain = (best_single - result["rover"]) / best_single if best_single else 0.0
    print(json.dumps({**result, "relative_gain": round(gain, 3)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
