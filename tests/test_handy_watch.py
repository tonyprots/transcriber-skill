"""handy_watch.py: уведомления, свежесть раннего текста, метка идущего прогона."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HANDY = Path(__file__).resolve().parents[1] / "integrations/handy"
sys.path.insert(0, str(HANDY))

import handy_watch  # noqa: E402

# notify() зовёт subprocess.run, а тесты его подменяют: настоящий — отсюда.
REAL_RUN = subprocess.run
TITLE = "Расшифровка готова"
MESSAGE = "2.5 мин за 16 с — текст в буфере → «handy»"


def captured_notify(monkeypatch, returncode: int = 0, stderr: bytes = b"") -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, returncode, b"", stderr)

    monkeypatch.setattr(handy_watch.subprocess, "run", fake_run)
    return commands


def scripts_of(command: list[str]) -> list[str]:
    return [command[i + 1] for i, part in enumerate(command) if part == "-e"]


def test_notification_text_goes_as_arguments_not_into_script(monkeypatch) -> None:
    # json.dumps экранирует кириллицу в \uXXXX, а у AppleScript такого escape
    # нет: скрипт не компилировался, и уведомление молча не приходило.
    commands = captured_notify(monkeypatch)
    handy_watch.notify(TITLE, MESSAGE)
    [command] = commands
    assert command[-2:] == [TITLE, MESSAGE]
    for script in scripts_of(command):
        assert "\\u" not in script
        assert MESSAGE not in script and TITLE not in script


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("osacompile") is None,
    reason="AppleScript есть только на macOS",
)
def test_notification_script_compiles(monkeypatch, tmp_path: Path) -> None:
    commands = captured_notify(monkeypatch)
    handy_watch.notify(TITLE, MESSAGE)
    lines = [arg for script in scripts_of(commands[0]) for arg in ("-e", script)]
    # osacompile только собирает скрипт: уведомление тест не показывает.
    result = REAL_RUN(
        ["osacompile", "-o", str(tmp_path / "notify.scpt"), *lines],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_failed_notification_is_logged(monkeypatch) -> None:
    captured_notify(monkeypatch, returncode=1, stderr="ошибка -2741".encode())
    logged: list[str] = []
    monkeypatch.setattr(handy_watch, "log", logged.append)
    handy_watch.notify(TITLE, MESSAGE)
    assert any("ошибка -2741" in line for line in logged)
