"""SIGTERM главному процессу гасит воркер и отпускает замок и временный каталог.

Python по умолчанию умирает от SIGTERM молча: не срабатывали ни `finally` у
scratch_dir и machine_slot, ни `except BaseException` в `_run_worker`. Воркер
досчитывал сиротой, копия записи оставалась в $TMPDIR. После SIGKILL родителя
воркер должен заметить это сам: перехватить SIGKILL нельзя.
"""
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "transcriber" / "scripts"

FAKE_WORKER = """
import json, os, sys, time
from pathlib import Path
from audio_transcription.exiting import run_entry_point

def main():
    request, result, progress = map(Path, sys.argv[1:4])
    Path(json.loads(request.read_text())["pid_file"]).write_text(str(os.getpid()))
    step = 0
    while True:
        step += 1
        progress.write_text(json.dumps({"completed": step}))
        time.sleep(0.2)

if __name__ == "__main__":
    run_entry_point(main, watch_parent_process=True)
"""

PARENT = """
import sys
from pathlib import Path
from audio_transcription.cli import scratch_dir
from audio_transcription.exiting import run_entry_point
from audio_transcription.isolated import _run_worker
from audio_transcription.machine_lock import machine_slot

def main():
    state = Path(sys.argv[1])
    with scratch_dir("transcriber-asr-") as work, machine_slot("тестовый прогон"):
        (state / "work").write_text(str(work))
        _run_worker(
            "fake_worker", "fake", {"pid_file": str(state / "worker.pid")}, work,
            stall_timeout_seconds=60,
        )
    return 0

if __name__ == "__main__":
    run_entry_point(main)
"""


def _wait(condition, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def running(tmp_path: Path):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "fake_worker.py").write_text(FAKE_WORKER, encoding="utf-8")
    (tmp_path / "parent.py").write_text(PARENT, encoding="utf-8")
    temp = tmp_path / "tmp"
    temp.mkdir()
    lock = tmp_path / "run.lock"
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(SCRIPTS), str(modules)]),
        TMPDIR=str(temp),
        TRANSCRIBER_LOCK=str(lock),
    )
    parent = subprocess.Popen(
        [sys.executable, str(tmp_path / "parent.py"), str(tmp_path)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid_file = tmp_path / "worker.pid"
    try:
        assert _wait(lambda: pid_file.is_file() and pid_file.read_text(), 30), parent.stderr
        worker = int(pid_file.read_text())
        yield parent, worker, Path((tmp_path / "work").read_text()), lock
    finally:
        for pid in (parent.pid, int(pid_file.read_text()) if pid_file.is_file() else None):
            if pid is not None and _alive(pid):
                os.kill(pid, signal.SIGKILL)
        parent.wait(timeout=10)


def test_sigterm_stops_worker_and_releases_lock_and_scratch(running) -> None:
    parent, worker, work, lock = running

    parent.send_signal(signal.SIGTERM)

    assert parent.wait(timeout=20) == 128 + signal.SIGTERM
    assert _wait(lambda: not _alive(worker), 10), "воркер пережил родителя"
    # Строку держателя machine_slot стирает только в finally: пустой файл —
    # знак, что уборка отработала, а не что ядро сняло flock за умершим.
    assert lock.read_text(encoding="utf-8") == ""
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert _wait(lambda: not work.exists(), 10), "временный каталог остался"


def test_worker_exits_when_parent_is_killed(running) -> None:
    parent, worker, _work, _lock = running

    parent.send_signal(signal.SIGKILL)
    parent.wait(timeout=10)

    assert _wait(lambda: not _alive(worker), 15), "воркер остался сиротой"


def test_ignored_hangup_stays_ignored() -> None:
    """Под `nohup` SIGHUP игнорируется, и закрытый терминал прогон не роняет."""
    from audio_transcription import exiting

    saved = {stop: signal.getsignal(stop) for stop in (signal.SIGTERM, signal.SIGHUP)}
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        exiting.stop_on_signals()
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, signal.SIG_IGN)
    finally:
        for stop, handler in saved.items():
            signal.signal(stop, handler)


def test_terminated_is_not_swallowed_by_except_exception() -> None:
    """Воркеры ловят `Exception` на границе процесса: сигнал должен пройти мимо."""
    from audio_transcription import exiting

    assert issubclass(exiting.Terminated, SystemExit)
    assert not issubclass(exiting.Terminated, Exception)
