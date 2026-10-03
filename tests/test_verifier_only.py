"""Название, которое основная модель уронила, а проверяющая услышала.

Текст не меняется (на эталоне вставки вредят), место поднимается в отдельный
раздел отчёта. Примеры — из диктовок Handy 2026-10-02.
"""

from audio_transcription.glossary import GlossaryEntry, mark_verifier_only_terms
from audio_transcription.models import Hypothesis, ReviewItem, Segment
from audio_transcription.reconcile import find_review_items
from audio_transcription.render import REVIEW_SECTION_LIMIT, _review_markdown

ENTRIES = [GlossaryEntry("Яндекс 360", ("яндекс триста шестьдесят",), auto_apply=True)]


def items(primary: str, verifier: str, *, latin: bool = True):
    found = find_review_items(
        Hypothesis("gigaam", "ru", 0.0, [Segment(0.0, 10.0, primary)]),
        Hypothesis("whisper", "ru", 0.0, [Segment(0.0, 10.0, verifier)]),
    )
    return mark_verifier_only_terms(found, ENTRIES, latin_is_signal=latin)


def kinds(found):
    return [(item.kind, item.verifier_span) for item in found if item.kind != "filler"]


def test_dropped_latin_name_is_marked():
    found = items(
        "И в том числе из-за этого понижает лимиты для платных пользователей.",
        "И в том числе из-за этого OpenAI понижает лимиты для платных пользователей.",
    )
    assert ("verifier_only", "OpenAI") in kinds(found)


def test_dropped_glossary_term_is_marked_without_latin():
    found = items(
        "Вот заголовок, что это значит для нас.",
        "Вот заголовок, что это значит для нас в Яндекс 360.",
        latin=False,
    )
    assert any(kind == "verifier_only" for kind, _ in kinds(found))


def test_ordinary_dropped_word_stays_substantive():
    found = items("Я вроде уже понял.", "Я вроде уже не понял.")
    assert ("substantive", "не") in kinds(found)


def test_latin_is_not_a_signal_on_english_route():
    found = items("we raised prices", "we raised OpenAI prices", latin=False)
    assert all(kind != "verifier_only" for kind, _ in kinds(found))


def test_breakdown_of_verifier_is_not_marked():
    long = items("Мне нравится.", "Мне нравится Top of the Pops in every single way.")
    foreign = items("Это нормально.", "Это 바instant нормально.")
    assert all(kind != "verifier_only" for kind, _ in kinds(long) + kinds(foreign))


def test_replaced_word_is_not_marked():
    # Пусто должно быть у основной: замену ведёт правило канона.
    found = items("Открываю слак.", "Открываю Slack.")
    assert all(kind != "verifier_only" for kind, _ in kinds(found))


def test_report_puts_dropped_names_on_top_and_keeps_text():
    found = items(
        "И в том числе из-за этого понижает лимиты.",
        "И в том числе из-за этого OpenAI понижает лимиты.",
    )
    text = _review_markdown(found, [])
    assert "## Возможно, выпало из текста — 1" in text
    assert "## Слушать" not in text
    assert "«OpenAI» — в контексте: … том числе из-за этого ∅ понижает лимиты." in text


def test_long_sections_are_cut_with_pointer():
    many = [
        ReviewItem(
            float(i), float(i) + 1, f"контекст {i}", "", 0.5, (f"а{i} / б{i}",), "r",
            1, "substantive", f"а{i}", f"б{i}",
        )
        for i in range(REVIEW_SECTION_LIMIT + 7)
    ]
    text = _review_markdown(many, [])
    assert f"## Слушать — {REVIEW_SECTION_LIMIT + 7}, здесь первые {REVIEW_SECTION_LIMIT}" in text
    assert text.count("- [") == REVIEW_SECTION_LIMIT
    assert "Ещё 7 — в `segments.json` → `review_items`." in text


def test_typical_whisper_artifact_is_collapsed():
    found = find_review_items(
        Hypothesis("gigaam", "ru", 0.0, [Segment(0.0, 10.0, "Ну вот и всё на сегодня.")]),
        Hypothesis("whisper", "ru", 0.0, [Segment(0.0, 10.0, "Ну вот и всё на сегодня. Продолжение следует...")]),
    )
    assert [item.kind for item in found] == ["verifier_artifact"]
    text = _review_markdown(found, [])
    assert "## Слушать" not in text
    assert "## Мелкие расхождения — 1" in text


def test_artifact_phrase_inside_real_speech_is_not_collapsed():
    found = find_review_items(
        Hypothesis("gigaam", "ru", 0.0, [Segment(0.0, 10.0, "Ну и всё.")]),
        Hypothesis("whisper", "ru", 0.0, [Segment(0.0, 10.0, "Ну и всё. Продолжение следует, сказал он и ушёл домой")]),
    )
    assert [item.kind for item in found] == ["substantive"]
