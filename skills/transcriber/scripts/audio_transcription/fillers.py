"""Удаление филлеров из читаемой расшифровки.

Дословная гипотеза остаётся как есть; из readable убираются «э-э-э», «а-а»,
«хмм», «ам», «um» и подобные звуки, которые модель честно распознала, но
которые в читаемом тексте только мешают. Слово после убранного филлера в
начале предложения получает заглавную букву. Заодно расставляются пробелы,
которые GigaAM теряет у тире и кавычек.
"""
from __future__ import annotations

import re

from .models import Segment

_TOKEN = re.compile(
    r"(?<![\w-])(?:э(?:-э)*|а(?:-а)+|и(?:-и)+|м(?:-м)+|м{2,}|х+м+|ам|эм+|uh+|um+|erm|hmm+)(?![\w-])",
    re.IGNORECASE,
)
_SENTENCE_END = ".!?…"
# Закрывающая кавычка сама предложение не кончает: «да» э, потом — середина
# фразы, а «Да!» э, потом — уже новое предложение.
_CLOSING_QUOTES = '»"”'
_OWN_SENTENCE = re.compile(r"[.!?…]+\s*")
_TAIL = re.compile(r"[,;:]?\s*")

# GigaAM E2E приклеивает тире и кавычки к слову: «пункт— про», «тракет—2»,
# «блока«Путешествия»» — 55 мест в 18 диктовках Handy (2026-10-04). Между
# двумя цифрами тире остаётся: «30—50%» — диапазон.
_DASH = re.compile("—")
_GLUED_QUOTES = (
    (re.compile(r"(?<=\w)«"), " «"),
    (re.compile(r"»(?=\w)"), "» "),
)


def _space_dash(match: re.Match[str]) -> str:
    text, at = match.string, match.start()
    left = text[at - 1] if at > 0 else ""
    right = text[at + 1] if at + 1 < len(text) else ""
    if left.isdigit() and right.isdigit():
        return "—"
    return f"{' ' if left.isalnum() else ''}—{' ' if right.isalnum() else ''}"


def fix_punctuation_spacing(text: str) -> str:
    """Отделяет пробелами тире и кавычки-ёлочки, приклеенные к слову."""
    out = _DASH.sub(_space_dash, text)
    for pattern, replacement in _GLUED_QUOTES:
        out = pattern.sub(replacement, out)
    return out


def _ends_sentence(text: str) -> bool:
    core = text.rstrip(_CLOSING_QUOTES)
    return bool(core) and core[-1] in _SENTENCE_END


def strip_fillers(text: str) -> tuple[str, int]:
    """Возвращает текст без филлеров и число убранных."""
    out = text
    removed = 0
    position = 0
    while True:
        match = _TOKEN.search(out, position)
        if match is None:
            break
        removed += 1
        start, end = match.span()
        before = out[:start].rstrip()
        sentence_end = _OWN_SENTENCE.match(out, end)
        at_sentence_start = not before or _ends_sentence(before)
        if sentence_end and not at_sentence_start:
            # «Мы пошли, э. Потом» — точка кончает предложение до филлера, и
            # уходит только сам филлер с висячей запятой перед ним.
            before = before.rstrip(",;:—– ") + sentence_end.group().strip()
            after = out[sentence_end.end():]
        else:
            after = out[(sentence_end or _TAIL.match(out, end)).end():]
            if at_sentence_start and after and after[0].islower():
                after = after[0].upper() + after[1:]
        out = f"{before} {after}" if before and after else before + after
        position = len(before)
    return out.strip(), removed


def strip_filler_segments(segments: list[Segment]) -> tuple[list[Segment], int]:
    """Убирает филлеры из сегментов; окно, состоявшее из одних филлеров, выпадает.

    Пробелы у тире и кавычек чинятся здесь же: это единственная точка, через
    которую идут и ранний текст для буфера, и readable.md.
    """
    result: list[Segment] = []
    total = 0
    for segment in segments:
        text, removed = strip_fillers(segment.text)
        total += removed
        text = fix_punctuation_spacing(text)
        if not text:
            continue
        result.append(
            Segment(segment.start, segment.end, text, segment.confidence, segment.speakers)
            if text != segment.text
            else segment
        )
    return result, total
