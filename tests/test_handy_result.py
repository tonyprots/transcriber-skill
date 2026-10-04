"""handy_result.py: агент получает текст Handy, не запуская второй прогон."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

HANDY = Path(__file__).resolve().parents[1] / "integrations/handy"
sys.path.insert(0, str(HANDY))

import handy_watch  # noqa: E402

FAKE_TRANSCRIBE = """
import sys, time, pathlib
args = sys.argv[1:]
out = pathlib.Path(args[args.index("--output") + 1])
early = pathlib.Path(args[args.index("--early-text") + 1])
calls = out.parent / "calls.txt"
with calls.open("a") as f:
    f.write("x")
time.sleep(0.3)
early.write_text("текст основной модели", encoding="utf-8")
time.sleep(0.5)
out.mkdir()
(out / "readable.md").write_text("полный", encoding="utf-8")
"""


@pytest.fixture
def env(tmp_path: Path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    wav = recordings / "handy-1790899241.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000)
    old = time.time() - 600
    os.utime(wav, (old, old))
    fake = tmp_path / "fake_transcribe.py"
    fake.write_text(FAKE_TRANSCRIBE, encoding="utf-8")
    root = tmp_path / "inbox"
    root.mkdir()
    state = tmp_path / "state.json"
    out = handy_watch.output_dir_for(wav, root)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    clip = tmp_path / "clipboard.txt"
    pbcopy = bin_dir / "pbcopy"
    pbcopy.write_text(f"#!/bin/sh\ncat > '{clip}'\n", encoding="utf-8")
    pbcopy.chmod(0o755)
    return {"wav": wav, "fake": fake, "root": root, "state": state, "out": out, "bin": bin_dir, "clip": clip}


def run_result(env: dict, timeout: float = 20, *extra: str) -> subprocess.CompletedProcess:
    # Свой pbcopy: тест не трогает настоящий буфер и видит, что в него ушло.
    path = f"{env['bin']}{os.pathsep}{os.environ.get('PATH', '')}"
    return subprocess.run(
        [
            sys.executable, str(HANDY / "handy_result.py"), str(env["wav"]),
            "--output-root", str(env["root"]), "--state", str(env["state"]),
            "--python", sys.executable, "--transcribe", str(env["fake"]),
            "--timeout", str(timeout), *extra,
        ],
        capture_output=True, text=True, timeout=timeout + 10, env={**os.environ, "PATH": path},
    )


def calls(env: dict) -> int:
    path = env["root"] / "calls.txt"
    return len(path.read_text()) if path.exists() else 0


def set_status(env: dict, status: str) -> None:
    env["state"].write_text(json.dumps({"seen": {env["wav"].name: status}}), encoding="utf-8")


def test_ready_text_is_returned_without_a_run(env):
    handy_watch.early_text_path(env["out"]).write_text("готовый", encoding="utf-8")
    result = run_result(env)
    assert result.returncode == 0
    assert "готовый" in result.stdout
    assert calls(env) == 0
    # Готовый текст в буфер уже клал наблюдатель: не перетираем буфер повторно.
    assert not env["clip"].exists()


def test_waits_for_running_watcher_instead_of_second_run(env):
    marker = handy_watch.running_marker_path(env["out"])
    marker.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    early = handy_watch.early_text_path(env["out"])
    # Атомарно, как `write_plain_text`: иначе читатель застаёт файл созданным,
    # но пустым (CI на теге v0.22.2).
    temporary = early.with_name(f".{early.name}.tmp")

    def publish() -> None:
        temporary.write_text("от наблюдателя", encoding="utf-8")
        temporary.replace(early)

    threading.Timer(2.0, publish).start()
    result = run_result(env)
    assert result.returncode == 0
    assert "от наблюдателя" in result.stdout
    assert "дождался прогона наблюдателя" in result.stdout
    assert calls(env) == 0


def test_fresh_unseen_recording_waits_for_watcher_pickup(env):
    now = time.time()
    os.utime(env["wav"], (now, now))
    early = handy_watch.early_text_path(env["out"])
    threading.Timer(2.0, lambda: early.write_text("подхвачено", encoding="utf-8")).start()
    result = run_result(env)
    assert "подхвачено" in result.stdout
    assert calls(env) == 0


@pytest.mark.parametrize("status", ["short 36s", "failed rc=1", None])
def test_runs_itself_when_watcher_did_not_take_it(env, status):
    if status:
        set_status(env, status)
    result = run_result(env)
    assert result.returncode == 0, result.stderr
    assert "расшифровал сам" in result.stdout
    assert "текст основной модели" in result.stdout
    assert calls(env) == 1
    # Как у наблюдателя: свой прогон тоже кончается текстом в буфере.
    assert env["clip"].read_text(encoding="utf-8") == "текст основной модели"
    marker = handy_watch.running_marker_path(env["out"])
    deadline = time.time() + 10
    while marker.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert (env["out"] / "readable.md").exists()
    assert not marker.exists(), "метка своего прогона должна сниматься"


def test_stale_marker_does_not_block(env):
    marker = handy_watch.running_marker_path(env["out"])
    marker.write_text(json.dumps({"pid": 2**22 + 12345}), encoding="utf-8")
    set_status(env, "short 36s")
    result = run_result(env)
    assert result.returncode == 0
    assert calls(env) == 1


def test_watcher_skips_recording_with_live_marker(env, tmp_path):
    marker = handy_watch.running_marker_path(env["out"])
    marker.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    args = handy_watch.parse_args([
        "--python", sys.executable, "--transcribe", str(env["fake"]),
        "--output-root", str(env["root"]), "--min-seconds", "0",
    ])
    assert handy_watch.process(env["wav"], args).startswith("exists")
    assert calls(env) == 0


def test_full_waits_for_complete_set(env):
    set_status(env, "short 36s")
    result = run_result(env, 20, "--full")
    assert result.returncode == 0, result.stderr
    assert "# полный набор:" in result.stdout
    assert calls(env) == 1


def test_full_returns_text_when_run_died_after_primary(env):
    handy_watch.early_text_path(env["out"]).write_text("только текст", encoding="utf-8")
    set_status(env, "failed rc=1")
    result = run_result(env, 20, "--full")
    assert result.returncode == 0
    assert "только текст" in result.stdout
    assert "не досчитан" in result.stdout
    assert calls(env) == 0


def test_lists_names_heard_only_by_verifier(env):
    out = env["out"]
    out.mkdir()
    handy_watch.early_text_path(out).write_text("понижает лимиты", encoding="utf-8")
    (out / "readable.md").write_text("понижает лимиты", encoding="utf-8")
    (out / "segments.json").write_text(json.dumps({"readable": [], "review_items": [
        {"start": 157.8, "kind": "verifier_only", "verifier_span": "OpenAI"},
        {"start": 20.0, "kind": "substantive", "verifier_span": "не"},
    ]}), encoding="utf-8")
    result = run_result(env)
    assert "возможно, выпало из текста" in result.stdout
    assert "02:37 «OpenAI»" in result.stdout
    assert "«не»" not in result.stdout
