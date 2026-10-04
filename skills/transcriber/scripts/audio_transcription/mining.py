"""Отбор кандидатов в словарь из очереди проверки.

Кандидат — расхождение двух моделей, похожее на термин, а не на случайную
ослышку. На русском признаков два: латиница с одной стороны и кириллица с
другой (так выглядят названия продуктов и англицизмы) либо одно и то же
кириллическое расхождение в нескольких окнах. На английском латиница с обеих
сторон, и признак — форма слова у проверяющей (`latin_term_shape`).

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


# Края слова, которые принадлежат фразе, а не термину. `+` и `#` не трогаем:
# без них «C++» и «C#» станут «C».
_EDGE_PUNCTUATION = re.compile(r"^[^\w+#]+|[^\w+#]+$")


def clean_token(token: str) -> str:
    """Слово из расхождения без пунктуации фразы; служебный токен — пустая строка.

    Расхождение режется по пробелам вместе со знаками: «ВКС, / VKS,» давало
    канон `VKS,`, и замена срабатывала только перед запятой. `<unk>` — не
    слово, а метка модели «здесь было что-то непонятное».
    """
    if "<" in token or ">" in token:
        return ""
    return _EDGE_PUNCTUATION.sub("", token)


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


# Языки, для которых отбор умеет отличать термин от ослышки. Русский ищет
# латиницу против кириллицы; английский — форму термина (`latin_term_shape`).
LEARNING_LANGUAGES = frozenset({"ru", "en"})

# Насколько ослышка обязана звучать как термин, чтобы стать кандидатом на
# английском маршруте. На русском такой отсечки нет: там пару делает
# кандидатом уже разница алфавитов. Здесь латиница с обеих сторон, и без
# звучания в словарь шли «that / Yeah» и «What / But» (0,33), а нужные пары
# лежат от 0,60 («opening» → OpenAI, «Chatsubiti» → ChatGPT) до 1,0.
# Та же граница, что у включения автозамены (`PROMOTE_MIN_PHONETIC`).
LATIN_MIN_PHONETIC = 0.6

_UPPER_INSIDE = re.compile(r"\w[A-Z]")
_DIGIT = re.compile(r"\d")


def latin_term_shape(word: str) -> bool:
    """Записано ли слово как название, а не как обычное слово языка.

    Признак на английском — форма: заглавная внутри слова или аббревиатура
    (`ChatGPT`, `OpenAI`, `SaaS`, `SCIM`), буквы вперемешку с цифрами
    (`GPT4`). Заглавная в начале не в счёт — так пишется любое слово в начале
    фразы («What / But»). Модели пишут название в такой форме, только когда
    узнали его; ослышка того же места выглядит обычным словом
    («Chatsubiti», «opening»).
    """
    letters = sum(char.isalpha() for char in word)
    if letters < 2:
        return False
    return bool(_UPPER_INSIDE.search(word) or _DIGIT.search(word))


def _latin_candidate(heard: str, verifier: str) -> tuple[bool, bool]:
    """Английская пара: (кандидат ли, есть ли у неё готовое написание).

    Написание берётся только у проверяющей, как и на русском: основная и
    есть текст, и если термин написала она, исправлять в этом месте нечего.
    Обе стороны в форме термина («ChatGPT / ChatGBT») — спор о написании, где
    правоту не установить; такая пара не берётся вовсе. Обе с заглавной, но
    без формы термина («Debo / Deebo») — похоже на имя: кандидат без
    написания, его назовёт человек, если место повторится.
    """
    if _CYRILLIC.search(heard) or _CYRILLIC.search(verifier):
        return False, False
    if phonetic_similarity(heard, verifier) < LATIN_MIN_PHONETIC:
        return False, False
    heard_term, verifier_term = latin_term_shape(heard), latin_term_shape(verifier)
    if verifier_term and not heard_term:
        return True, True
    if heard_term or verifier_term:
        return False, False
    return heard[:1].isupper() and verifier[:1].isupper(), False


def collect_candidates(
    review_items: list[dict[str, Any]],
    known: set[str],
    min_count: int = 3,
    language: str = "ru",
) -> list[dict]:
    """Кандидаты из очереди проверки одного или нескольких прогонов.

    `review_items` — элементы `review_items` из `segments.json` (или их же
    словари прямо из пайплайна), `known` — нормализованные канон и алиасы уже
    известных записей: их незачем предлагать заново. `language` выбирает
    признак термина: латиница против кириллицы на русском, форма слова на
    английском (`_latin_candidate`).
    """
    if language not in LEARNING_LANGUAGES:
        return []
    latin_route = language != "ru"
    seen: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
    for item in review_items:
        for pair in item.get("differing_tokens", []):
            if " / " not in pair:
                continue
            left, right = (part.strip() for part in pair.split(" / ", 1))
            if not left or not right:
                continue
            for heard, verifier in aligned_words(left, right):
                heard, verifier = clean_token(heard), clean_token(verifier)
                if not heard or not verifier:
                    continue
                if heard in known or verifier in known:
                    continue
                if not (_LETTER.search(heard) and _LETTER.search(verifier)):
                    continue
                if re.sub(r"[\s-]", "", heard) == re.sub(r"[\s-]", "", verifier):
                    continue
                if latin_route:
                    if re.sub(r"\W", "", heard).casefold() == re.sub(r"\W", "", verifier).casefold():
                        continue  # «U.S / US», «a.m / AM» — запись, а не слово
                    candidate, term = _latin_candidate(heard, verifier)
                    if not candidate:
                        continue
                    verifier_latin = term
                else:
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
