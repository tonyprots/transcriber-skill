#!/usr/bin/env python3
"""Кандидаты в словарь из очереди проверки готового прогона.

Запуск: <python из .venv> scripts/glossary_candidates.py OUTPUT_DIR [OUTPUT_DIR ...]
        [--glossary СЛОВАРЬ.yaml] [--min-count N] [--output КАНДИДАТЫ.yaml]

Берёт `segments.json` каждого каталога результата и смотрит на расхождения
моделей. Кандидат — пара «услышала основная / услышала проверяющая», где
проверяющая написала латиницей, а основная — кириллицей (так обычно
выглядят названия продуктов и англицизмы), либо одно и то же кириллическое
расхождение повторилось в нескольких окнах (по умолчанию от трёх). Уже известные записи словаря пропускаются.

Печатает YAML в формате словаря. Все записи выходят с `auto_apply: false`:
включать замену можно только после прослушивания окна или подтверждения
пользователя.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.glossary import load_glossary  # noqa: E402
from audio_transcription.reconcile import normalize_text  # noqa: E402

_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")


def _original_case(normalized: str, text: str) -> str:
    """Возвращает фрагмент text в исходном регистре, соответствующий normalized."""
    words = normalized.split()
    pattern = r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(0) if match else normalized


def collect_candidates(
    segments_payloads: list[dict], known: set[str], min_count: int = 3
) -> list[dict]:
    seen: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
    for payload in segments_payloads:
        for item in payload.get("review_items", []):
            for pair in item.get("differing_tokens", []):
                if " / " not in pair:
                    continue
                heard, verifier = (part.strip() for part in pair.split(" / ", 1))
                if not heard or not verifier:
                    continue
                if heard in known or verifier in known:
                    continue
                if not (_LETTER.search(heard) and _LETTER.search(verifier)):
                    continue
                if re.sub(r"[\s-]", "", heard) == re.sub(r"[\s-]", "", verifier):
                    continue
                latin_side = bool(_LATIN.search(verifier)) and not _CYRILLIC.search(verifier)
                key = (heard, verifier)
                entry = seen.setdefault(
                    key,
                    {
                        "heard": heard,
                        "verifier": verifier,
                        "canonical": _original_case(verifier, item.get("comparison_text", "")),
                        "latin": latin_side and bool(_CYRILLIC.search(heard)),
                        "count": 0,
                        "first_at": float(item.get("start", 0.0)),
                    },
                )
                entry["count"] += 1
    result = []
    for entry in seen.values():
        if entry["latin"] or entry["count"] >= min_count:
            result.append(entry)
    result.sort(key=lambda e: (not e["latin"], -e["count"], e["first_at"]))
    return result


def render_yaml(candidates: list[dict]) -> str:
    lines = [
        "# Кандидаты из очереди проверки. Проверь каждый по аудио, лишнее удали,",
        "# у подтверждённых поставь auto_apply: true и перенеси в рабочий словарь.",
        "version: 1",
        "entries:",
    ]
    if not candidates:
        lines.append("  []")
    for entry in candidates:
        why = "латиница у проверяющей модели" if entry["latin"] else f"повторилось в {entry['count']} окнах"
        lines.extend(
            [
                f"  - canonical: {json.dumps(entry['canonical'], ensure_ascii=False)}",
                f"    aliases: [{json.dumps(entry['heard'], ensure_ascii=False)}]",
                f"    context: {json.dumps(f'кандидат: {why}, первое окно {entry['first_at']:.1f} с', ensure_ascii=False)}",
                "    auto_apply: false",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dirs", nargs="+", type=Path, help="каталоги результата transcribe.py")
    parser.add_argument("--glossary", type=Path, help="уже существующий словарь: его записи пропускаются")
    parser.add_argument("--min-count", type=int, default=3, help="сколько окон нужно повторяющемуся кириллическому расхождению (по умолчанию 3)")
    parser.add_argument("--output", type=Path, help="куда записать YAML вместо stdout")
    args = parser.parse_args(argv)

    payloads = []
    for directory in args.output_dirs:
        path = directory / "segments.json"
        if not path.exists():
            print(f"Ошибка: нет {path}", file=sys.stderr)
            return 2
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    known: set[str] = set()
    for entry in load_glossary(args.glossary):
        known.add(normalize_text(entry.canonical))
        known.update(normalize_text(alias) for alias in entry.aliases)
    text = render_yaml(collect_candidates(payloads, known, args.min_count))
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Кандидатов записано: {text.count('- canonical:')} → {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
