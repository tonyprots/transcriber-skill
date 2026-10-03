"""Канон словаря у проверяющей закрывает спорное место — но только узко.

Примеры — из настоящих диктовок 2026-09/10: верные замены и ложные, которые
правило обязано пропустить.
"""

from audio_transcription.glossary import GlossaryEntry, take_verifier_canonicals
from audio_transcription.models import Correction, Hypothesis, Segment
from audio_transcription.reconcile import find_review_items
from audio_transcription.render import _review_markdown

ENTRIES = [
    GlossaryEntry("Microsoft", ("майкрософт",), auto_apply=True),
    GlossaryEntry("MCP", ("эмсипи",), auto_apply=True),
    GlossaryEntry("Copilot", ("копилот",), auto_apply=True),
    GlossaryEntry("Docs", ("докс",), auto_apply=True),
    GlossaryEntry("SMM", ("эсэмэм",), auto_apply=True),
    GlossaryEntry("SMS", ("эсэмэс",), auto_apply=True),
    GlossaryEntry("Астра", ("астра",), auto_apply=True),
    GlossaryEntry("Obsidian", ("обсидиан",), auto_apply=False),
]


def run(primary: str, verifier: str, entries=ENTRIES):
    segments = [Segment(0.0, 10.0, primary)]
    items = find_review_items(
        Hypothesis("gigaam", "ru", 0.0, segments),
        Hypothesis("whisper", "ru", 0.0, [Segment(0.0, 10.0, verifier)]),
    )
    return take_verifier_canonicals(segments, items, entries)


def test_misheard_term_takes_verifier_canonical():
    segments, items, taken = run(
        "Так, коротко. Макросов собрал чат подписку.",
        "Так, коротко. Microsoft собрал чат подписку.",
    )
    assert segments[0].text == "Так, коротко. Microsoft собрал чат подписку."
    assert [(c.before, c.after, c.rule) for c in taken] == [
        ("Макросов", "Microsoft", "verifier_canonical")
    ]
    assert [item.kind for item in items] == ["verifier_canonical"]


def test_each_occurrence_is_fixed_once_and_punctuation_kept():
    segments, _, taken = run(
        "Открыть MC к нашим сервисам, и для MC плагинов.",
        "Открыть MCP к нашим сервисам, и для MCP плагинов.",
    )
    assert segments[0].text == "Открыть MCP к нашим сервисам, и для MCP плагинов."
    assert len(taken) == 2


def test_wider_verifier_span_still_replaces_only_the_word():
    segments, _, _ = run(
        "сентябрьское обновление Capile звучит плохо",
        "сентябрьское обновление Copilot в офис звучит плохо",
    )
    assert segments[0].text == "сентябрьское обновление Copilot звучит плохо"


def test_wide_primary_span_is_left_alone():
    # Права основная: OpenAI Dots — продукт, Whisper услышал «Docs».
    segments, items, taken = run(
        "Манус больше относится к истории к Dots, вот этим всем.",
        "Манус больше относится к Docs и вот этим всем.",
    )
    assert taken == []
    assert "Dots" in segments[0].text
    assert all(item.kind != "verifier_canonical" for item in items)


def test_dissimilar_sound_is_left_alone():
    _, _, taken = run("Конкуренция Bita 4 СберАстра.", "Конкуренция Bita 4 Сбер Астра.")
    assert taken == []


def test_untrusted_entry_is_not_taken():
    _, _, taken = run("к Apsidian как бэкенд", "к Obsidian как бэкенд")
    assert taken == []


def test_primary_word_that_is_another_term_is_kept():
    # «SMM» и «SMS» на слух одно: правая сторона не доказательство.
    segments, _, taken = run("бюджет на SMM вырос", "бюджет на SMS вырос")
    assert taken == []
    assert segments[0].text == "бюджет на SMM вырос"


def test_report_lists_taken_terms_and_hides_them_from_queue():
    _, items, taken = run(
        "Так, коротко. Макросов собрал чат подписку.",
        "Так, коротко. Microsoft собрал чат подписку.",
    )
    text = _review_markdown(items, [], None, taken)
    assert "## Взято у проверяющей — 1" in text
    assert "«Макросов» → Microsoft" in text
    assert "## Слушать" not in text and "## То же слово" not in text


def test_plain_corrections_do_not_appear_in_taken_section():
    plain = Correction(0.0, 1.0, "майкрософт", "Microsoft", "Microsoft", "exact_alias", True)
    assert "Взято у проверяющей" not in _review_markdown([], [], None, [plain])


def test_preposition_next_to_the_mishearing_survives():
    segments, _, taken = run("добавь, как в ВКрософт борется", "добавь, как Microsoft борется")
    assert [c.before for c in taken] == ["ВКрософт"]
    assert segments[0].text == "добавь, как в Microsoft борется"
