"""Отбор кандидатов в словарь из очереди проверки.

Кандидат — расхождение двух моделей, похожее на термин, а не на случайную
ослышку. Признаков два: латиница с одной стороны и кириллица с другой (так
выглядят названия продуктов и англицизмы) либо одно и то же кириллическое
расхождение в нескольких окнах.

Модуль общий для двух потребителей: `scripts/glossary_candidates.py` (ручной
цикл — показать человеку и дать решить) и `glossary_store` (автоматическое
пополнение словаря после прогона). Раньше логика жила только в скрипте, и
пайплайну пришлось бы написать её заново — ровно так расходятся реализации.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

from .models import ReviewItem, Segment
from .phonetic import phonetic_similarity
from .reconcile import normalize_text

_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")

# Строже порога подсказок в очереди: там ошибка стоит одной лишней строки, а
# здесь склейка двух разных терминов испортит готовую запись словаря.
MERGE_THRESHOLD = 0.8


def original_case(normalized: str, text: str) -> str:
    """Возвращает фрагмент text в исходном регистре, соответствующий normalized."""
    words = normalized.split()
    pattern = r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(0) if match else normalized


def aligned_words(heard: str, verifier: str) -> list[tuple[str, str]]:
    """Пары «слово против слова» из одного расхождения.

    Расхождение приходит целым куском: «трец когда / threads» — два слова
    против одного. Выравнивания в таком куске нет, и если взять его как есть,
    алиасом станет «трец когда», то есть замена сработает только рядом с этим
    соседом. Поэтому берём только куски одинаковой длины и режем их пословно.
    """
    left, right = heard.split(), verifier.split()
    return list(zip(left, right)) if len(left) == len(right) else []


def term_shaped(heard: str, verifier: str) -> bool:
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
    review_items: list[dict[str, Any]], known: set[str], min_count: int = 3
) -> list[dict]:
    """Кандидаты из очереди проверки одного или нескольких прогонов.

    `review_items` — элементы `review_items` из `segments.json` (или их же
    словари прямо из пайплайна), `known` — нормализованные канон и алиасы уже
    известных записей: их незачем предлагать заново.
    """
    seen: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
    for item in review_items:
        for pair in item.get("differing_tokens", []):
            if " / " not in pair:
                continue
            left, right = (part.strip() for part in pair.split(" / ", 1))
            if not left or not right:
                continue
            for heard, verifier in aligned_words(left, right):
                if heard in known or verifier in known:
                    continue
                if not (_LETTER.search(heard) and _LETTER.search(verifier)):
                    continue
                if re.sub(r"[\s-]", "", heard) == re.sub(r"[\s-]", "", verifier):
                    continue
                term = term_shaped(heard, verifier)
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
                            original_case(verifier, item.get("comparison_text", ""))
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


def undisputed_words(
    segments: list[Segment], review_items: list[ReviewItem]
) -> set[str]:
    """Слова, о которых моделям спорить было не о чем.

    Такое слово — обычное слово языка, а не ослышка: обе модели услышали его
    одинаково. Автозамену по нему включать нельзя, иначе «трек» в «фитнес-трек»
    начнёт превращаться в «Трекер» во всех будущих расшифровках, и заметить это
    можно будет только сверкой с raw.json.

    Считаем пословно, а не по окнам: окно тянется на десяток секунд, спорным
    его делает одно слово, и по окнам из полусотни слов в зачёт не шло ни
    одного. На замере по двенадцати записям пословный счёт дал 2516 бесспорных
    слов против 1617 «по окнам».
    """
    disputed: set[str] = set()
    for item in review_items:
        for pair in item.differing_tokens:
            for side in pair.split(" / "):
                disputed.update(normalize_text(side).split())
    words: set[str] = set()
    for segment in segments:
        words.update(normalize_text(segment.text).split())
    return words - disputed


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
