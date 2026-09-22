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
