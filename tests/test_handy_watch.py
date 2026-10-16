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


FAILS_WITHOUT_TEXT = """
import sys, time
time.sleep(0.5)
sys.exit(1)
"""

WRITES_NEW_TEXT = """
import sys, time, pathlib
args = sys.argv[1:]
out = pathlib.Path(args[args.index("--output") + 1])
early = pathlib.Path(args[args.index("--early-text") + 1])
time.sleep(0.3)
temporary = early.with_name(f".{early.name}.tmp")
temporary.write_text("новый текст\\n", encoding="utf-8")
temporary.replace(early)
time.sleep(0.3)
out.mkdir()
(out / "readable.md").write_text("полный", encoding="utf-8")
"""


def stale_job(tmp_path: Path, monkeypatch, script: str):
    """Повтор записи, после прошлой попытки которой остался ранний текст."""
    fake = tmp_path / "fake_transcribe.py"
    fake.write_text(script, encoding="utf-8")
    wav = tmp_path / "handy-1790899241.wav"
    out = handy_watch.output_dir_for(wav, tmp_path / "inbox")
    out.parent.mkdir()
    handy_watch.early_text_path(out).write_text("текст прошлой попытки\n", encoding="utf-8")
    titles: list[str] = []
    clipboard: list[str] = []
    monkeypatch.setattr(handy_watch, "notify", lambda title, message: titles.append(title))
    monkeypatch.setattr(handy_watch, "to_clipboard", clipboard.append)
    monkeypatch.setattr(handy_watch, "from_clipboard", lambda: clipboard[-1] if clipboard else "")
    monkeypatch.setattr(handy_watch, "log", lambda message: None)
    args = handy_watch.parse_args([
        "--python", sys.executable, "--transcribe", str(fake), "--output-root", str(out.parent),
    ])
    return wav, out, args, titles, clipboard


def test_failed_retry_with_old_early_text_reports_failure(tmp_path: Path, monkeypatch) -> None:
    wav, out, args, titles, clipboard = stale_job(tmp_path, monkeypatch, FAILS_WITHOUT_TEXT)
    assert handy_watch.run_job(wav, out, 60.0, args) == "failed rc=1"
    assert "Расшифровка готова" not in titles
    assert "Handy → transcriber" in titles
    assert clipboard == []


def test_old_early_text_is_not_taken_for_new_one(tmp_path: Path, monkeypatch) -> None:
    wav, out, args, titles, clipboard = stale_job(tmp_path, monkeypatch, WRITES_NEW_TEXT)
    assert handy_watch.run_job(wav, out, 60.0, args).startswith("done")
    assert clipboard == ["новый текст"]
    assert titles == ["Расшифровка готова"]
