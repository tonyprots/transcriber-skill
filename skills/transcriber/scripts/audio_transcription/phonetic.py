"""Фонетический код, общий для кириллицы и латиницы.

Задача не в том, чтобы построить точную транскрипцию, а в том, чтобы «трэц» и
«threads» получили одинаковый код. Русская ASR слышит английский термин и
записывает его кириллицей — буквенное сравнение тут даёт ноль, потому что у
строк нет ни одной общей буквы.

Три шага:

1. Латиница разворачивается в кириллицу по чтению. Для аббревиатур (tgstat,
   CFO, SMM) работает чтение по названиям английских букв: «си-эф-оу».
2. Обе строки сводятся к огрублённому фонетическому скелету: оглушение,
   свёртка шипящих, редукция безударных гласных.
3. Коды сравниваются расстоянием Левенштейна, нормированным на длину.
"""
from __future__ import annotations

import re
import unicodedata

# Английские буквы так, как их произносит русскоговорящий, читая аббревиатуру.
LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф",
    "g": "джи", "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл",
    "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл ю", "x": "экс",
    "y": "уай", "z": "зед",
}

# Чтение латиницы как слова. Порядок важен: длинные сочетания раньше коротких.
DIGRAPHS = (
    ("tch", "ч"), ("sch", "ш"), ("tsch", "ч"),
    ("th", "т"), ("ch", "ч"), ("sh", "ш"), ("ph", "ф"), ("wh", "в"),
    ("ck", "к"), ("ng", "нг"), ("qu", "кв"), ("kh", "х"), ("zh", "ж"),
    ("ee", "и"), ("oo", "у"), ("ea", "и"), ("ou", "ау"), ("ai", "эй"),
    ("ay", "эй"), ("ey", "эй"), ("oa", "оу"), ("igh", "ай"),
)

SINGLES = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
}

# Свёртка к скелету: пары, которые ASR путает, склеиваются в один символ.
FOLD = {
    # звонкие и глухие неразличимы на слух в конце слова и рядом с глухими
    "б": "п", "в": "ф", "г": "к", "д": "т", "ж": "ш", "з": "с",
    # шипящие и свистящие сводятся грубо
    "щ": "ш", "ч": "ш", "ц": "с",
    # гласные редуцируются до трёх классов
    "о": "а", "ы": "и", "е": "и", "э": "и", "я": "а", "ю": "у", "ё": "а",
    "й": "и",
}
VOWELS = set("аиу")


def _latin_word_to_cyrillic(text: str) -> str:
    out = text
    for source, target in DIGRAPHS:
        out = out.replace(source, target)
    return "".join(SINGLES.get(char, char) for char in out)


def _latin_letters_to_cyrillic(text: str) -> str:
    return "".join(LETTER_NAMES.get(char, char) for char in text)


def _looks_like_abbreviation(token: str) -> bool:
    """Аббревиатура читается по буквам: мало гласных или всё заглавными."""
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return False
    if len(letters) <= 5 and all(c.isupper() for c in letters):
        return True
    vowels = sum(1 for c in token.lower() if c in "aeiouy")
    return vowels == 0 and len(letters) >= 2


def _fold(spoken: str) -> str:
    joined = spoken.replace("ъ", "").replace("ь", "")
    folded = "".join(FOLD.get(char, char) for char in joined)
    # Двойные согласные и подряд идущие одинаковые звуки — один звук.
    return re.sub(r"(.)\1+", r"\1", folded)


def phonetic_codes(text: str) -> tuple[str, ...]:
    """Все правдоподобные чтения строки.

    Латинское название можно прочитать и как слово, и по именам букв, и мы не
    знаем, как его произнёс говорящий: «TGStat» звучит и «тэгэстат», и
    «ти-джи-стат», причём в одной записи встречались оба варианта. Поэтому
    коды считаются для обоих чтений, а сравнение берёт лучшее совпадение.
    """
    text = unicodedata.normalize("NFKC", text)
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text)
    as_word: list[str] = []
    as_letters: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[A-Za-z]+", token):
            as_word.append(_latin_word_to_cyrillic(token.lower()))
            as_letters.append(_latin_letters_to_cyrillic(token.lower()))
        else:
            as_word.append(token.lower())
            as_letters.append(token.lower())
    variants = {_fold("".join(as_word)), _fold("".join(as_letters))}
    return tuple(sorted(variant for variant in variants if variant))


def phonetic_code(text: str) -> str:
    """Основное чтение строки: латиница как слово, если это не аббревиатура."""
    text = unicodedata.normalize("NFKC", text)
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text)
    pieces: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[A-Za-z]+", token):
            spoken = (
                _latin_letters_to_cyrillic(token.lower())
                if _looks_like_abbreviation(token)
                else _latin_word_to_cyrillic(token.lower())
            )
        else:
            spoken = token.lower()
        pieces.append(spoken)
    return _fold("".join(pieces))


def _skeleton(code: str) -> str:
    skeleton = "".join(
        char for index, char in enumerate(code) if char not in VOWELS or index == 0
    )
    return skeleton or code


def consonant_skeleton(text: str) -> str:
    """Костяк согласных: ASR путает безударные гласные чаще всего.

    Именно он сводит «тегостат» и «ти-джи-стат» к одному виду. Но на коротких
    словах костяк вырождается в две буквы и начинает совпадать со случайными
    словами, поэтому одного его мало — см. `phonetic_similarity`.
    """
    return _skeleton(phonetic_code(text))


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, start=1):
        current = [i]
        for j, rchar in enumerate(right, start=1):
            current.append(
                previous[j - 1]
                if lchar == rchar
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return 1.0 - _levenshtein(left, right) / max(len(left), len(right))


def phonetic_similarity(heard: str, canonical: str) -> float:
    """Близость произношений от 0 до 1.

    Внутри одной пары чтений берётся строгая из двух оценок: костяк согласных
    нужен, потому что ASR теряет безударные гласные, но на коротких словах он
    вырождается — у «свои» и «CFO» костяк одинаковый («сф»), хотя это разные
    слова. Полный код с гласными их разводит, поэтому обе оценки должны быть
    высокими одновременно.

    Между разными чтениями берётся лучшее: достаточно, чтобы услышанное
    совпало с одним правдоподобным произношением термина.
    """
    return max(
        min(_ratio(_skeleton(left), _skeleton(right)), _ratio(left, right))
        for left in phonetic_codes(heard)
        for right in phonetic_codes(canonical)
    )
