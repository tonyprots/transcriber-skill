from __future__ import annotations

import sys
from pathlib import Path

HANDY = Path(__file__).resolve().parents[1] / "integrations/handy"
sys.path.insert(0, str(HANDY))

import handy_watch  # noqa: E402


def run_once(tmp_path: Path, monkeypatch, outcomes: list[str], calls: list[tuple]) -> dict:
    recordings = tmp_path / "recordings"
    recordings.mkdir(exist_ok=True)
    (recordings / "handy-1791040137.wav").touch()
    monkeypatch.setattr(handy_watch, "log", lambda message: None)

    def fake_process(wav, args):
        calls.append((wav.name, args.clipboard))
        return outcomes.pop(0)

    monkeypatch.setattr(handy_watch, "process", fake_process)
    args = handy_watch.parse_args([
        "--python", "py", "--transcribe", "t.py", "--output-root", str(tmp_path / "out"),
        "--recordings", str(recordings), "--state", str(tmp_path / "state.json"),
    ])
    handy_watch.run(args)
    return handy_watch.load_state(tmp_path / "state.json")


def test_failed_recording_is_retried_on_next_run_without_clipboard(tmp_path, monkeypatch) -> None:
    calls: list[tuple] = []
    state = run_once(tmp_path, monkeypatch, ["failed rc=2"], calls)
    assert calls == [("handy-1791040137.wav", True)]  # в том же запуске не повторяем
    assert state["failures"] == {"handy-1791040137.wav": 1}

    state = run_once(tmp_path, monkeypatch, ["done x"], calls)
    assert calls[-1] == ("handy-1791040137.wav", False)
    assert state["seen"]["handy-1791040137.wav"] == "done x"
    assert state["failures"] == {}


def test_retries_stop_after_max_attempts(tmp_path, monkeypatch) -> None:
    calls: list[tuple] = []
    for _ in range(handy_watch.MAX_ATTEMPTS + 2):
        run_once(tmp_path, monkeypatch, ["failed rc=1"], calls)
    assert len(calls) == handy_watch.MAX_ATTEMPTS
