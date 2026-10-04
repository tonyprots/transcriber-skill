"""Итог прогона сменяет быструю версию в буфере, но не чужое содержимое."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HANDY = Path(__file__).resolve().parents[1] / "integrations/handy"
sys.path.insert(0, str(HANDY))

import handy_watch  # noqa: E402

EARLY = "Из российских — это Bitrik4, VK Work Space"
FINAL = "Из российских — это Битрикс24, VK WorkSpace"


@pytest.fixture
def clipboard(monkeypatch) -> list[str]:
    board = [EARLY]
    monkeypatch.setattr(handy_watch, "from_clipboard", lambda: board[0] + "\n")
    monkeypatch.setattr(handy_watch, "to_clipboard", lambda text: board.__setitem__(0, text))
    return board


def test_final_text_replaces_untouched_early_text(clipboard) -> None:
    assert handy_watch.refresh_clipboard(EARLY, FINAL)
    assert clipboard == [FINAL]


def test_user_copy_during_run_is_kept(clipboard) -> None:
    clipboard[0] = "что-то своё"
    assert not handy_watch.refresh_clipboard(EARLY, FINAL)
    assert clipboard == ["что-то своё"]


def test_same_text_is_not_rewritten(clipboard) -> None:
    assert not handy_watch.refresh_clipboard(EARLY, EARLY)
