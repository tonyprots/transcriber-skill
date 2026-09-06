import pytest

from audio_transcription.models import Hypothesis, Segment
from audio_transcription.reconcile import (
    align_segments,
    find_review_items,
    normalize_text,
    similarity,
)


def hypothesis(model: str, text: str) -> Hypothesis:
    return Hypothesis(model, "ru", 1.0, [Segment(0, 3, text)])


def test_normalization_ignores_case_punctuation_and_yo() -> None:
    assert normalize_text("Ёлка, ДА!") == "елка да"
    assert similarity("Ёлка, да!", "елка да") == 1.0


def test_large_difference_is_sent_to_review() -> None:
    items = find_review_items(
        hypothesis("readable", "Мы натянули трусы"),
        hypothesis("lexical", "Мы натянули тросы"),
        threshold=0.95,
    )
    assert len(items) == 1
    assert items[0].differing_tokens == ("трусы / тросы",)


def test_default_threshold_catches_known_trusy_trosy_case() -> None:
    items = find_review_items(
        hypothesis("readable", "Трусы."), hypothesis("lexical", "тросы")
    )
    assert len(items) == 1
    assert items[0].similarity == pytest.approx(0.8)


def test_close_hypotheses_do_not_create_review_item() -> None:
    assert not find_review_items(
        hypothesis("readable", "Добрый день!"),
        hypothesis("lexical", "добрый день"),
    )


def test_local_token_difference_is_not_hidden_by_high_global_similarity() -> None:
    items = find_review_items(
        hypothesis("readable", "Любимая Лись тут много одинаковых правильных слов"),
        hypothesis("verifier", "Любимая листья тут много одинаковых правильных слов"),
    )
    assert len(items) == 1
    assert items[0].differing_tokens == ("лись / листья",)


def test_word_level_verifier_is_grouped_by_readable_segment() -> None:
    readable = [Segment(1.0, 3.0, "два слова")]
    verifier = [Segment(1.1, 1.5, "два", 0.9), Segment(1.6, 2.2, "слова", 0.8)]
    pairs = align_segments(readable, verifier)
    assert pairs[0][1] is not None
    assert pairs[0][1].text == "два слова"
    assert pairs[0][1].confidence == pytest.approx(0.85)


def test_verifier_word_is_assigned_to_only_one_adjacent_segment() -> None:
    readable = [Segment(0, 1, "первый"), Segment(1, 2, "второй")]
    verifier = [Segment(0.95, 1.2, "второй", 0.9)]
    pairs = align_segments(readable, verifier)
    assert pairs[0][1] is None
    assert pairs[1][1] is not None
    assert pairs[1][1].text == "второй"

