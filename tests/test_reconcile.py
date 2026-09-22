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
    # Сравнение идёт по нормализованным формам, а в очередь попадает исходное
    # написание: из него словарь берёт канон, и его же читает человек.
    assert items[0].differing_tokens == ("Лись / листья",)


def test_review_item_points_at_the_span_not_at_the_whole_window() -> None:
    """Элемент очереди — разошедшееся место, а не двадцать секунд вокруг него.

    Пока единицей было окно, на часовом интервью в очередь уходило 254 окна
    из 272: слушать предлагалось всю запись.
    """
    long_text = " ".join(["слово"] * 20)
    items = find_review_items(
        Hypothesis("readable", "ru", 1.0, [Segment(0, 20, f"{long_text} трусы")]),
        Hypothesis("verifier", "ru", 1.0, [Segment(0, 20, f"{long_text} тросы")]),
    )
    assert len(items) == 1
    assert items[0].end - items[0].start < 3
    assert items[0].start > 15
    assert 0 <= items[0].start and items[0].end <= 20


def test_filler_and_repetition_do_not_become_work() -> None:
    items = find_review_items(
        hypothesis("readable", "ну вот мы сделали отчёт"),
        hypothesis("verifier", "мы как бы сделали отчёт"),
    )
    # Два разных места — «ну вот» в начале и «как бы» в середине, — и оба
    # остаются в данных, но не попадают в работу.
    assert items and all(item.kind == "filler" for item in items)
    stutter = find_review_items(
        hypothesis("readable", "и вот это это это отчёт"),
        hypothesis("verifier", "и вот это отчёт"),
    )
    assert all(item.kind == "disfluency" for item in stutter)


def test_same_number_written_differently_is_not_a_disagreement() -> None:
    for spoken, digits in (
        ("ninety five percent", "95%"),
        ("eight hundred thousand", "800,000"),
        ("sixty seventy percent", "60-70%"),
    ):
        items = find_review_items(
            hypothesis("readable", f"выручка выросла на {digits} за год"),
            hypothesis("verifier", f"выручка выросла на {spoken} за год"),
        )
        assert [item.kind for item in items] == ["equivalent"], (spoken, digits)


def test_queue_weight_counts_meaning_not_length() -> None:
    """Пять служебных слов весят меньше одного знаменательного."""
    verbose = find_review_items(
        hypothesis("readable", "мы поняли что это был он и потом ушли"),
        hypothesis("verifier", "мы поняли и потом ушли"),
    )
    single = find_review_items(
        hypothesis("readable", "мы поняли и потом уехали"),
        hypothesis("verifier", "мы поняли и потом ушли"),
    )
    assert max(item.weight for item in verbose) <= max(item.weight for item in single)


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

