from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "transcriber"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))


import pytest


@pytest.fixture(autouse=True)
def isolated_machine_lock(tmp_path_factory, monkeypatch):
    """Очередь прогонов общая на машину: тесты не должны ждать настоящую расшифровку."""
    monkeypatch.setenv("TRANSCRIBER_LOCK", str(tmp_path_factory.mktemp("lock") / "run.lock"))


# Нативные библиотеки (onnxruntime, MLX) падают в деструкторах при обычном
# выходе интерпретатора: `recursive_mutex lock failed`, rc 134 после зелёного
# прогона — примерно каждый второй запуск. Точки входа скилла выходят через
# `os._exit` по той же причине (`audio_transcription.exiting`); тесты делают
# то же самое, когда pytest уже всё напечатал.
_exit_status = 0


def pytest_sessionfinish(session, exitstatus) -> None:
    global _exit_status
    _exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config) -> None:
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status)
