"""Работа, результат которой никто не ждёт (аудит 2026-09-29).

Три находки: подсказки словаря перебирали весь текст против всего словаря и
давали шум, текст основной модели ждал проверяющую, а результат — уборку
временных WAV под антивирусом.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from audio_transcription import cli
from audio_transcription.glossary import GlossaryEntry, suggest_glossary_matches
from audio_transcription.models import ReviewItem, Segment


def test_suggestions_are_limited_to_disputed_words() -> None:
    """Слово, в котором модели сошлись, подсказкой не становится."""
    segments = [Segment(0.0, 4.0, "открыл Тэгэстат и трец вчера")]
    entries = [
        GlossaryEntry(canonical="TGStat", aliases=(), auto_apply=False),
        GlossaryEntry(canonical="Threads", aliases=(), auto_apply=False),
    ]
    everything = suggest_glossary_matches(segments, entries)
    assert {item["canonical"] for item in everything} == {"TGStat", "Threads"}

    limited = suggest_glossary_matches(segments, entries, only_words={"трец"})
    assert [item["canonical"] for item in limited] == ["Threads"]
    assert suggest_glossary_matches(segments, entries, only_words=set()) == []


def test_disputed_words_come_from_the_primary_side() -> None:
    item = ReviewItem(
        start=0.0,
        end=1.0,
        readable_text="",
        comparison_text="",
        similarity=0.5,
        differing_tokens=("Трец когда / threads",),
        reason="",
    )
    assert cli.disputed_primary_words([item]) == {"трец", "когда"}


def test_plain_text_joins_windows_until_sentence_end(tmp_path: Path) -> None:
    """Окно режет фразу посередине — склеиваем до точки, дальше абзац."""
    target = tmp_path / "sub" / "early.txt"
    cli.write_plain_text(
        target,
        [
            Segment(0.0, 1.0, "Поправь слайд"),
            Segment(1.0, 2.0, "про выручку."),
            Segment(3.0, 4.0, "Потом отчёт."),
        ],
    )
    assert target.read_text(encoding="utf-8") == "Поправь слайд про выручку.\n\nПотом отчёт.\n"
    assert not list(target.parent.glob(".*.tmp"))


def test_early_text_refuses_a_batch(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        [str(tmp_path / "a.wav"), str(tmp_path / "b.wav"), "--early-text", str(tmp_path / "t.txt")]
    )
    with pytest.raises(ValueError, match="--early-text"):
        cli._validate(args)


def test_scratch_dir_is_removed_without_blocking() -> None:
    with cli.scratch_dir("transcriber-test-") as path:
        (path / "window.wav").write_bytes(b"0" * 16)
        assert path.is_dir()
    deadline = time.monotonic() + 10
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not path.exists()
