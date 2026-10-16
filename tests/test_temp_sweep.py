"""Прогон добирает свои временные каталоги, оставшиеся после убитого процесса.

После SIGKILL или выключения `finally` у scratch_dir не срабатывает, и в
$TMPDIR копились transcriber-asr-* и transcriber-download-* с prepared.wav на
сотни мегабайт — копии чужих записей. Порог и префиксы те же, что у уборки
Мостика (24 ч по mtime каталога и первого уровня), живого владельца не трогаем.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from audio_transcription import cli, scratch

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "transcriber" / "scripts"


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _make(root: Path, name: str, *, hours: float, owner: int | None = None, fresh_file: bool = False) -> Path:
    path = root / name
    path.mkdir()
    (path / "prepared.wav").write_bytes(b"0" * 16)
    if owner is not None:
        (path / scratch.OWNER_MARKER).write_text(str(owner), encoding="utf-8")
    old = time.time() - hours * 3600
    for item in path.iterdir():
        os.utime(item, (old, old))
    if fresh_file:
        os.utime(path / "prepared.wav", None)
    os.utime(path, (old, old))
    return path


def _gone(path: Path, seconds: float = 10) -> bool:
    deadline = time.monotonic() + seconds
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return not path.exists()


def test_sweep_removes_only_stale_orphans(tmp_path: Path) -> None:
    stale = _make(tmp_path, "transcriber-asr-old", hours=30)
    stale_download = _make(tmp_path, "transcriber-download-old", hours=30, owner=_dead_pid())
    kept = [
        _make(tmp_path, "transcriber-asr-fresh", hours=2),
        _make(tmp_path, "transcriber-asr-busy", hours=30, fresh_file=True),
        _make(tmp_path, "transcriber-download-live", hours=30, owner=os.getpid()),
        _make(tmp_path, "transcriber-mirror-old", hours=30),
        _make(tmp_path, "someone-else-old", hours=30),
    ]
    (tmp_path / "transcriber-asr-file").write_bytes(b"not a dir")

    removed = scratch.sweep_orphans(tmp_path)

    assert removed == sorted([stale, stale_download])
    assert _gone(stale) and _gone(stale_download)
    assert all(path.is_dir() for path in kept)
    assert (tmp_path / "transcriber-asr-file").is_file()


def test_missing_root_is_not_an_error(tmp_path: Path) -> None:
    assert scratch.sweep_orphans(tmp_path / "нет") == []


def test_scratch_dir_marks_its_owner(tmp_path: Path) -> None:
    with cli.scratch_dir("transcriber-asr-") as path:
        assert (path / scratch.OWNER_MARKER).read_text(encoding="utf-8") == str(os.getpid())
        old = time.time() - 48 * 3600
        for item in path.iterdir():
            os.utime(item, (old, old))
        os.utime(path, (old, old))
        assert path not in scratch.sweep_orphans(path.parent)
        assert path.is_dir()


class _Stop(Exception):
    pass


def test_run_sweeps_before_work(tmp_path: Path, monkeypatch, capsys) -> None:
    calls: list[None] = []

    def fake_sweep(*args, **kwargs):
        calls.append(None)
        return [tmp_path / "transcriber-asr-a", tmp_path / "transcriber-asr-b"]

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(cli, "sweep_orphans", fake_sweep)
    monkeypatch.setattr(cli, "prepare_audio", stop)
    args = cli.build_parser().parse_args([str(tmp_path / "a.wav"), "--no-cache"])
    with pytest.raises(_Stop):
        cli._transcribe(
            args,
            source=tmp_path / "a.wav",
            remote=None,
            section=None,
            output=tmp_path / "out",
            report=cli.ProgressReporter(enabled=True),
            download_seconds=None,
        )
    assert calls
    assert "убрано осиротевших: 2" in capsys.readouterr().err


def test_sweep_failure_does_not_stop_the_run(tmp_path: Path, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise OSError("нет доступа")

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(cli, "sweep_orphans", broken)
    monkeypatch.setattr(cli, "prepare_audio", stop)
    args = cli.build_parser().parse_args([str(tmp_path / "a.wav"), "--no-cache", "--quiet"])
    with pytest.raises(_Stop):
        cli._transcribe(
            args,
            source=tmp_path / "a.wav",
            remote=None,
            section=None,
            output=tmp_path / "out",
            report=cli.ProgressReporter(enabled=False),
            download_seconds=None,
        )


def _manage_cache():
    spec = importlib.util.spec_from_file_location("manage_cache_temp", SCRIPTS / "manage_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manage_cache_prune_temp(tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path / "tmp"
    root.mkdir()
    stale = _make(root, "transcriber-asr-old", hours=30)
    monkeypatch.setattr(tempfile, "tempdir", str(root))

    assert _manage_cache().main(["--cache-dir", str(tmp_path / "cache"), "--json", "prune", "--temp"]) == 0

    assert json.loads(capsys.readouterr().out)["removed_temp_dirs"] == 1
    assert _gone(stale)


def test_manage_cache_prune_needs_a_target(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _manage_cache().main(["--cache-dir", str(tmp_path / "cache"), "prune"])
