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
_SENTENCE_END = '.!?…»"'

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


def strip_fillers(text: str) -> tuple[str, int]:
    """Возвращает текст без филлеров и число убранных."""
    out = text
    removed = 0
    while True:
        match = _TOKEN.search(out)
        if match is None:
            break
        removed += 1
        start, end = match.span()
        own_sentence = re.match(r"[.!?…]\s*", out[end:])
        tail = own_sentence or re.match(r"[,;:]?\s*", out[end:])
        before = out[:start].rstrip()
        after = out[end + tail.end():]
        at_sentence_start = not before or before[-1] in _SENTENCE_END
        if at_sentence_start and after and after[0].islower():
            after = after[0].upper() + after[1:]
        out = f"{before} {after}" if before and after else before + after
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
