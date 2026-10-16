"""`--review-threshold` выбирает формулировку причины, а не состав очереди.

Решение 2026-09-12: как фильтр порог ронял покрытие ошибок со 100% до 46%.
Код ему следует, а справка, quality.md и пустой отчёт обещали фильтр — и
пользователь, крутивший флаг ради короткой очереди, эффекта не получал.
"""
from __future__ import annotations

from pathlib import Path

from audio_transcription import cli
from audio_transcription.models import Hypothesis, Segment
from audio_transcription.reconcile import find_review_items
from audio_transcription.render import _review_markdown

SKILL = Path(__file__).resolve().parents[1] / "skills" / "transcriber"


def _help(flag: str) -> str:
    action = next(a for a in cli.build_parser()._actions if flag in a.option_strings)
    return action.help or ""


def test_threshold_changes_reason_not_queue() -> None:
    primary = Hypothesis("p", "ru", 1.0, [Segment(0, 3, "Мы натянули трусы и пошли домой")])
    verifier = Hypothesis("v", "ru", 1.0, [Segment(0, 3, "Мы натянули тросы и пошли домой")])
    found = {
        threshold: find_review_items(primary, verifier, threshold=threshold)
        for threshold in (0.0, 0.82, 1.0)
    }
    places = {threshold: [(i.start, i.end, i.kind) for i in items] for threshold, items in found.items()}
    assert places[0.0] == places[0.82] == places[1.0] and places[0.0]


def test_help_and_docs_do_not_promise_a_filter() -> None:
    text = _help("--review-threshold")
    assert "уходит в очередь" not in text
    assert "на состав очереди не влияет" in text.lower()
    quality = (SKILL / "references" / "quality.md").read_text(encoding="utf-8")
    assert "фильтром очереди" not in quality


def test_empty_report_does_not_mention_a_threshold() -> None:
    text = _review_markdown([], [])
    assert "порог" not in text
