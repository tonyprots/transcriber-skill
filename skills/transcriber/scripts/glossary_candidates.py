#!/usr/bin/env python3
"""Кандидаты в словарь из очереди проверки готового прогона.

Запуск: <python из .venv> scripts/glossary_candidates.py OUTPUT_DIR [OUTPUT_DIR ...]
        [--glossary СЛОВАРЬ.yaml] [--min-count N] [--output КАНДИДАТЫ.yaml]

Ручной цикл: показать кандидатов человеку и дать решить. Скиллу для своего
словаря этот скрипт не нужен — он снимает кандидатов сам после каждого прогона
(`audio_transcription/glossary_store.py`). Здесь тот же отбор, но результат
идёт в файл на подпись, а не в рабочий словарь.

Берёт `segments.json` каждого каталога результата и смотрит на расхождения
моделей. Кандидат — пара «услышала основная / услышала проверяющая», где одна
сторона написала латиницей, а другая кириллицей (так обычно выглядят названия
продуктов и англицизмы), либо одно и то же кириллическое расхождение
повторилось в нескольких окнах (по умолчанию от трёх). Уже известные записи
словаря пропускаются.

Печатает YAML в формате словаря двумя частями. В `entries` — готовые записи:
каноническое написание нашлось у той стороны, что написала латиницей. Ниже
закомментированными строками — места, где термин виден, но правильного
написания нет ни у одной модели; их называет пользователь.

Все записи выходят с `auto_apply: false`: включать замену можно только после
прослушивания окна или подтверждения пользователя.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.glossary import load_glossary  # noqa: E402
from audio_transcription.mining import collect_candidates  # noqa: E402
from audio_transcription.reconcile import normalize_text  # noqa: E402


def render_yaml(candidates: list[dict]) -> str:
    # Готовая запись получается, только когда каноническое написание кто-то
    # действительно произнёс правильно. Если обе модели перевирают термин,
    # запись собрать не из чего — такие места идут отдельным списком, и их
    # называет пользователь. Класть их в entries нельзя: пустой canonical
    # роняет load_glossary.
    ready = [entry for entry in candidates if entry["canonical"]]
    unnamed = [entry for entry in candidates if not entry["canonical"]]

    lines = [
        "# Кандидаты из очереди проверки. Проверь каждый по аудио, лишнее удали,",
        "# у подтверждённых поставь auto_apply: true и перенеси в рабочий словарь.",
        "version: 1",
        "entries:",
    ]
    if not ready:
        lines.append("  []")
    for entry in ready:
        why = (
            "латиница у проверяющей модели"
            if entry["latin"]
            else f"повторилось в {entry['count']} окнах"
        )
        aliases = entry.get("aliases") or [entry["heard"]]
        rendered = ", ".join(json.dumps(alias, ensure_ascii=False) for alias in aliases)
        lines.extend(
            [
                f"  - canonical: {json.dumps(entry['canonical'], ensure_ascii=False)}",
                f"    aliases: [{rendered}]",
                f"    context: {json.dumps(f'кандидат: {why}, первое окно {entry['first_at']:.1f} с', ensure_ascii=False)}",
                "    auto_apply: false",
            ]
        )

    if unnamed:
        lines += [
            "",
            "# Похоже на термины, но правильного написания нет ни у одной модели.",
            "# Послушай окно, впиши каноническое написание и перенеси в entries:",
        ]
        for entry in unnamed:
            aliases = entry.get("aliases") or [entry["heard"], entry["verifier"]]
            rendered = ", ".join(json.dumps(alias, ensure_ascii=False) for alias in aliases)
            lines.append(
                f"#   - canonical: \"?\"   # основная: {entry['heard']}, "
                f"проверяющая: {entry['verifier']}, окно {entry['first_at']:.1f} с"
            )
            lines.append(f"#     aliases: [{rendered}]")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dirs", nargs="+", type=Path, help="каталоги результата transcribe.py")
    parser.add_argument("--glossary", type=Path, help="уже существующий словарь: его записи пропускаются")
    parser.add_argument("--min-count", type=int, default=3, help="сколько окон нужно повторяющемуся кириллическому расхождению (по умолчанию 3)")
    parser.add_argument("--output", type=Path, help="куда записать YAML вместо stdout")
    args = parser.parse_args(argv)

    review_items = []
    for directory in args.output_dirs:
        path = directory / "segments.json"
        if not path.exists():
            print(f"Ошибка: нет {path}", file=sys.stderr)
            return 2
        payload = json.loads(path.read_text(encoding="utf-8"))
        review_items.extend(payload.get("review_items", []))
    known: set[str] = set()
    for entry in load_glossary(args.glossary):
        known.add(normalize_text(entry.canonical))
        known.update(normalize_text(alias) for alias in entry.aliases)
    text = render_yaml(collect_candidates(review_items, known, args.min_count))
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        # Закомментированные места тоже содержат «- canonical:», но записями
        # ещё не являются — считаем их отдельно.
        ready = sum(1 for line in text.splitlines() if line.startswith("  - canonical:"))
        unnamed = sum(1 for line in text.splitlines() if line.startswith("#   - canonical:"))
        print(
            f"Готовых записей: {ready}; мест, которые надо назвать: {unnamed} → {args.output}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
