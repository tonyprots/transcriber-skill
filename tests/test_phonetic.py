"""Фонетическое сопоставление: ловит ли оно ослышки, на которых буквы бессильны.

Все пары ниже — настоящие расхождения из прогонов на личных записях и из
словаря, а не придуманные. Смысл модуля в одном: у «тэгэстат» и «TGStat» нет
ни одной общей буквы, поэтому буквенное сравнение даёт ноль, и очередь
проверки молчит ровно там, где должна сработать.
"""
from __future__ import annotations

from difflib import SequenceMatcher

import pytest

from audio_transcription.glossary import GlossaryEntry, suggest_glossary_matches
from audio_transcription.models import Segment
from audio_transcription.phonetic import (
    consonant_skeleton,
    phonetic_code,
    phonetic_similarity,
)

# Ослышки, у которых с термином нет общих букв: буквенное сравнение тут слепо.
CROSS_ALPHABET = [
    ("Тэгэстат", "TGStat"),
    ("Tegostat", "TGStat"),
    ("тегостат", "TGStat"),
    ("трец", "Threads"),
    ("трэц", "Threads"),
    ("эсэмэмщика", "SMM-щика"),
    ("клоуд-кода", "Claude Code"),
    ("серемка", "CRM-ка"),
    ("постгрес", "PostgreSQL"),
]

# Ослышки внутри одного алфавита: их ловят и буквы, фонетика не должна терять.
SAME_ALPHABET = [
    ("Treds", "Threads"),
    ("Interprice", "Enterprise"),
    ("страцессия", "стратсессия"),
    ("фимпланирование", "финпланирование"),
]

HEARD_AS_TERM = CROSS_ALPHABET + SAME_ALPHABET

# Обычные слова, которые нельзя принимать за термины.
NOT_A_TERM = [
    ("свою", "CFO"),
    ("ему", "M&A"),
    ("SMS", "SMM"),
    ("самом", "SMM"),
    ("подумаем", "Премиум"),
]


@pytest.mark.parametrize(("heard", "term"), HEARD_AS_TERM)
def test_phonetics_finds_what_letters_cannot(heard: str, term: str) -> None:
    assert phonetic_similarity(heard, term) >= 0.75


@pytest.mark.parametrize(("heard", "term"), CROSS_ALPHABET)
def test_these_pairs_are_invisible_to_letter_comparison(heard: str, term: str) -> None:
    """Страховка от подмены смысла: если буквы справляются сами, пара бесполезна.

    Порог 0.86 — тот же, что у буквенного поиска в `suggest_glossary_matches`.
    Без этой проверки список выше может незаметно выродиться в набор опечаток,
    на которых и старый механизм работал.
    """
    assert SequenceMatcher(None, heard.casefold(), term.casefold()).ratio() < 0.86


@pytest.mark.parametrize(("word", "term"), NOT_A_TERM)
def test_short_abbreviations_do_not_swallow_ordinary_words(word: str, term: str) -> None:
    """Короткие термины отсекаются длиной, а не порогом: их код вырождается."""
    segments = [Segment(0.0, 1.0, f"и тут {word} дальше")]
    entry = GlossaryEntry(canonical=term, aliases=(), auto_apply=False)
    assert suggest_glossary_matches(segments, [entry]) == []


def test_latin_abbreviation_is_read_by_letter_names() -> None:
    """«TGStat» звучит как «ти-джи-стат», а не как «тгстат»."""
    assert phonetic_similarity("тиджистат", "TGStat") >= 0.75


def test_voicing_and_unstressed_vowels_are_folded() -> None:
    """Оглушение и редукция — главные источники расхождений у русской ASR."""
    assert phonetic_code("код") == phonetic_code("кот")
    assert consonant_skeleton("молоко") == consonant_skeleton("малако")


def test_suggestion_reports_which_rule_fired() -> None:
    """В очереди должно быть видно, фонетика сработала или буквы."""
    segments = [Segment(0.0, 2.0, "открыл Тэгэстат вчера")]
    entry = GlossaryEntry(canonical="TGStat", aliases=(), auto_apply=False)
    (suggestion,) = suggest_glossary_matches(segments, [entry])
    assert suggestion["rule"] == "phonetic"
    assert suggestion["heard"] == "Тэгэстат"
    assert suggestion["canonical"] == "TGStat"


def test_exact_canonical_in_text_is_not_suggested() -> None:
    """Термин уже написан правильно — предлагать нечего."""
    segments = [Segment(0.0, 2.0, "открыл TGStat вчера")]
    entry = GlossaryEntry(canonical="TGStat", aliases=(), auto_apply=False)
    assert suggest_glossary_matches(segments, [entry]) == []


def test_trailing_punctuation_does_not_hide_a_term() -> None:
    """«трец.» с точкой — то же слово, что «трец»."""
    segments = [Segment(0.0, 2.0, "посмотрел трец.")]
    entry = GlossaryEntry(canonical="Threads", aliases=(), auto_apply=False)
    assert suggest_glossary_matches(segments, [entry])


def test_phonetics_never_edits_the_text() -> None:
    """Фонетика работает только на предложения — автозамена осталась точной.

    Иначе «свои» и «CFO» начали бы подменять друг друга в готовом тексте.
    """
    from audio_transcription.glossary import apply_glossary

    segments = [Segment(0.0, 2.0, "открыл Тэгэстат вчера")]
    entry = GlossaryEntry(canonical="TGStat", aliases=(), auto_apply=True)
    corrected, corrections = apply_glossary(segments, [entry])
    assert corrected[0].text == "открыл Тэгэстат вчера"
    assert corrections == []
