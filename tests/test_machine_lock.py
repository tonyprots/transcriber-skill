"""Очередь прогонов на машину: второй ждёт первого и говорит, кого ждёт."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from audio_transcription.machine_lock import machine_slot


def test_second_run_waits_and_names_the_holder(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    seen: list[str] = []
    order: list[str] = []

    def second() -> None:
        with machine_slot("второй", path=lock, poll_seconds=0.01,
                          on_wait=lambda holder, waited: seen.append(holder)) as waited:
            order.append("второй")
            assert waited > 0

    with machine_slot("первый", path=lock) as waited:
        assert waited < 1
        worker = threading.Thread(target=second)
        worker.start()
        time.sleep(0.2)
        order.append("первый")
    worker.join(timeout=5)

    assert order == ["первый", "второй"]
    assert seen and "первый" in seen[0]
    # После прогона в замке пусто: следующему нечего показывать как «занято».
    assert lock.read_text(encoding="utf-8") == ""


def test_free_machine_is_taken_without_waiting(tmp_path: Path) -> None:
    calls: list[str] = []
    with machine_slot("один", path=tmp_path / "run.lock",
                      on_wait=lambda *_: calls.append("wait")) as waited:
        assert waited < 1
    assert calls == []
