"""Пользователь узнаёт о новой версии, а сеть — не чаще раза в неделю."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_transcription import __version__, update_check

DAY = 24 * 3600


@pytest.fixture
def state(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "state.json"
    monkeypatch.setenv("TRANSCRIBER_UPDATE_STATE", str(path))
    monkeypatch.delenv("TRANSCRIBER_NO_UPDATE_CHECK")
    return path


def _bump(version: str) -> str:
    major, minor, *_ = update_check.parse_version(version)
    return f"v{major}.{minor + 1}.0"


def test_network_at_most_once_a_week(state: Path) -> None:
    assert update_check.due(state, now=0)
    update_check.refresh(state, fetch=lambda: "v0.1.0", now=0)
    assert not update_check.due(state, now=6 * DAY)
    assert update_check.due(state, now=7 * DAY)


def test_newer_release_is_reported_from_cache(state: Path) -> None:
    newer = _bump(__version__)
    update_check.refresh(state, fetch=lambda: newer, now=0)
    found = update_check.available(state)
    assert found["latest"] == newer.lstrip("v")
    assert found["installed"] == __version__
    assert found["how"]


@pytest.mark.parametrize("tag", [f"v{__version__}", "v0.1.0"])
def test_same_or_older_release_is_silent(state: Path, tag: str) -> None:
    update_check.refresh(state, fetch=lambda: tag, now=0)
    assert update_check.available(state) is None


def test_network_failure_keeps_last_answer_and_waits_a_week(state: Path) -> None:
    newer = _bump(__version__)
    update_check.refresh(state, fetch=lambda: newer, now=0)

    def offline():
        raise OSError("нет сети")

    update_check.refresh(state, fetch=offline, now=8 * DAY)
    assert json.loads(state.read_text())["latest"] == newer
    assert not update_check.due(state, now=9 * DAY)


def test_offline_and_opt_out_never_start(state: Path, monkeypatch) -> None:
    monkeypatch.setattr(update_check.threading, "Thread", pytest.fail)
    assert update_check.start(offline=True) is None
    monkeypatch.setenv("TRANSCRIBER_NO_UPDATE_CHECK", "1")
    assert update_check.start(offline=False) is None


def test_versions_compare_as_numbers() -> None:
    assert update_check.parse_version("v0.10.0") > update_check.parse_version("0.9.9")
    assert update_check.parse_version("smoke-test") is None


def test_install_hint_follows_install_method(tmp_path: Path) -> None:
    installed = Path.home() / ".local/share/transcriber-skill/skills/transcriber"
    assert "install.sh" in update_check.update_hint(installed)
    assert "/plugin" in update_check.update_hint(Path("/x/.claude/plugins/cache/transcriber/skills/transcriber"))


def test_transcription_result_carries_the_notice(state: Path, monkeypatch, capsys) -> None:
    from audio_transcription import cli

    update_check.refresh(state, fetch=lambda: _bump(__version__), now=0)
    monkeypatch.setattr(cli, "expand_inputs", lambda args: args.input)
    monkeypatch.setattr(cli, "run", lambda args: None)
    monkeypatch.setattr(cli, "summarize", lambda result: {"output": "x"})
    monkeypatch.setattr(update_check, "start", lambda **kwargs: None)
    assert cli.main(["x.wav"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["update_available"]["latest"] == _bump(__version__).lstrip("v")
    assert "Вышла версия скилла" in captured.err
