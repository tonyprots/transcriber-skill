#!/usr/bin/env python3
"""Кандидаты в словарь из очереди проверки готового прогона.

Запуск: <python из .venv> scripts/glossary_candidates.py OUTPUT_DIR [OUTPUT_DIR ...]
        [--glossary СЛОВАРЬ.yaml] [--min-count N] [--output КАНДИДАТЫ.yaml]

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
import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_transcription.glossary import load_glossary  # noqa: E402
from audio_transcription.phonetic import phonetic_similarity  # noqa: E402
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


def _aligned_words(heard: str, verifier: str) -> list[tuple[str, str]]:
    """Пары «слово против слова» из одного расхождения.

    Расхождение приходит целым куском: «трец когда / threads» — два слова
    против одного. Выравнивания в таком куске нет, и если взять его как есть,
    алиасом станет «трец когда», то есть замена сработает только рядом с этим
    соседом. Поэтому берём только куски одинаковой длины и режем их пословно.
    """
    left, right = heard.split(), verifier.split()
    return list(zip(left, right)) if len(left) == len(right) else []


def _term_shaped(heard: str, verifier: str) -> bool:
    """Похоже ли расхождение на термин, а не на обычную ослышку.

    Признак — латиница с одной стороны и кириллица с другой, причём в любом
    порядке. Только у проверяющей латиницу искать мало: «Treds / трэдс» —
    такой же термин, просто перевирают обе модели, каждая по-своему.
    """
    heard_latin = bool(_LATIN.search(heard)) and not _CYRILLIC.search(heard)
    verifier_latin = bool(_LATIN.search(verifier)) and not _CYRILLIC.search(verifier)
    return (verifier_latin and bool(_CYRILLIC.search(heard))) or (
        heard_latin and bool(_CYRILLIC.search(verifier))
    )


def collect_candidates(
    segments_payloads: list[dict], known: set[str], min_count: int = 3
) -> list[dict]:
    seen: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
    for payload in segments_payloads:
        for item in payload.get("review_items", []):
            for pair in item.get("differing_tokens", []):
                if " / " not in pair:
                    continue
                left, right = (part.strip() for part in pair.split(" / ", 1))
                if not left or not right:
                    continue
                for heard, verifier in _aligned_words(left, right):
                    if heard in known or verifier in known:
                        continue
                    if not (_LETTER.search(heard) and _LETTER.search(verifier)):
                        continue
                    if re.sub(r"[\s-]", "", heard) == re.sub(r"[\s-]", "", verifier):
                        continue
                    term = _term_shaped(heard, verifier)
                    verifier_latin = bool(_LATIN.search(verifier)) and not _CYRILLIC.search(
                        verifier
                    )
                    key = (heard, verifier)
                    entry = seen.setdefault(
                        key,
                        {
                            "heard": heard,
                            "verifier": verifier,
                            # Каноническое написание берём у той стороны, что
                            # написала латиницей. Если латиницу написала сама
                            # основная модель, правильного написания нет ни у
                            # кого — тогда это не готовая запись, а место,
                            # которое должен назвать пользователь.
                            "canonical": (
                                _original_case(verifier, item.get("comparison_text", ""))
                                if verifier_latin
                                else ""
                            ),
                            "latin": term,
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
    return merge_by_sound(result)


# Строже порога подсказок в очереди: там ошибка стоит одной лишней строки, а
# здесь склейка двух разных терминов испортит готовую запись словаря.
MERGE_THRESHOLD = 0.8


def merge_by_sound(candidates: list[dict]) -> list[dict]:
    """Собирает варианты одного термина в одну запись.

    Одно и то же слово приходит несколькими расхождениями: «трэц / Threads»,
    «treds / трэдс», «treds / трэдсе». Без склейки пользователь получает три
    записи на один термин и должен сам догадаться, что это одно слово —
    причём догадаться по написаниям, у которых нет общих букв.

    Якорями становятся записи с уже известным каноническим написанием: к ним
    притягиваются безымянные варианты, звучащие так же. Безымянные остатки
    группируются между собой, но канон для них по-прежнему называет человек.
    """
    anchors = [entry for entry in candidates if entry["canonical"]]
    rest = [entry for entry in candidates if not entry["canonical"]]
    for entry in anchors:
        entry.setdefault("aliases", [entry["heard"]])

    leftovers: list[dict] = []
    for entry in rest:
        variants = [entry["heard"], entry["verifier"]]
        anchor = next(
            (
                candidate
                for candidate in anchors
                if any(
                    phonetic_similarity(variant, candidate["canonical"]) >= MERGE_THRESHOLD
                    for variant in variants
                )
            ),
            None,
        )
        if anchor is None:
            leftovers.append(entry)
            continue
        for variant in variants:
            if variant.casefold() != anchor["canonical"].casefold():
                anchor.setdefault("aliases", []).append(variant)
        anchor["count"] += entry["count"]

    merged_rest: list[dict] = []
    for entry in leftovers:
        group = next(
            (
                candidate
                for candidate in merged_rest
                if phonetic_similarity(entry["heard"], candidate["heard"]) >= MERGE_THRESHOLD
            ),
            None,
        )
        if group is None:
            entry.setdefault("aliases", [entry["heard"], entry["verifier"]])
            merged_rest.append(entry)
            continue
        for variant in (entry["heard"], entry["verifier"]):
            if variant not in group["aliases"]:
                group["aliases"].append(variant)
        group["count"] += entry["count"]

    for entry in anchors + merged_rest:
        seen_alias: dict[str, str] = {}
        for alias in entry.get("aliases", []):
            seen_alias.setdefault(alias.casefold(), alias)
        entry["aliases"] = list(seen_alias.values())
    return anchors + merged_rest


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
