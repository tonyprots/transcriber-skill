from audio_transcription.glossary import GlossaryEntry, apply_glossary, suggest_glossary_matches
from audio_transcription.models import Segment


def test_only_auto_apply_exact_aliases_are_replaced() -> None:
    entries = [
        GlossaryEntry("Windows 11", ("виндовс одиннадцать",), auto_apply=True),
        GlossaryEntry("PostgreSQL", ("постгрес",), auto_apply=False),
    ]
    segments, corrections = apply_glossary(
        [Segment(0, 5, "Про виндовс одиннадцать и постгрес")], entries
    )
    assert segments[0].text == "Про Windows 11 и постгрес"
    assert len(corrections) == 1
    assert corrections[0].before == "виндовс одиннадцать"


def test_replacements_do_not_match_inside_words_or_cascade() -> None:
    entries = [
        GlossaryEntry("Бета", ("альфа",), auto_apply=True),
        GlossaryEntry("Гамма", ("бета",), auto_apply=True),
    ]
    segments, corrections = apply_glossary(
        [Segment(0, 1, "алфавит альфа")], entries
    )
    assert segments[0].text == "алфавит Бета"
    assert len(corrections) == 1


def test_canonical_term_is_not_suggested_again() -> None:
    entry = GlossaryEntry("GenAI Platform", ("Gen AI Platform",), auto_apply=True)
    assert not suggest_glossary_matches(
        [Segment(0, 1, "GenAI Platform и Apps")], [entry]
    )
