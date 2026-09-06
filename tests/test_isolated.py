from pathlib import Path

import pytest

from audio_transcription import isolated
from audio_transcription.backends import BackendUnavailable


class StuckProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout=None):
        return int(self.returncode or 0)

    def communicate(self):
        return "", ""


def test_watchdog_stops_process_without_progress(monkeypatch, tmp_path: Path) -> None:
    clock = [0.0]
    monkeypatch.setattr(isolated.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        isolated.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    process = StuckProcess()
    completed = isolated.wait_for_progress(
        process,
        tmp_path / "missing-progress.json",
        stall_timeout_seconds=0.3,
        poll_seconds=0.1,
    )
    assert completed is False
    assert process.terminated is True


def test_worker_reports_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailable, match="Неизвестный ASR-бэкенд"):
        isolated.run_isolated_backend(
            "unknown",
            [],
            "ru",
            tmp_path,
            config={},
            stall_timeout_seconds=5,
        )


def test_worker_activity_in_logs_counts_as_progress(monkeypatch, tmp_path: Path) -> None:
    """Загрузка модели пишет прогресс-бар в stderr, а не в progress.json.

    Пока лог растёт, watchdog не должен считать воркер зависшим.
    """
    clock = [0.0]
    monkeypatch.setattr(isolated.time, "monotonic", lambda: clock[0])
    log = tmp_path / "stderr.log"
    log.write_text("", encoding="utf-8")

    def sleep(seconds: float) -> None:
        clock[0] += seconds
        if clock[0] < 1.0:
            with log.open("a", encoding="utf-8") as handle:
                handle.write("downloading...\n")
        else:
            process.returncode = 0

    process = StuckProcess()
    monkeypatch.setattr(isolated.time, "sleep", sleep)
    completed = isolated.wait_for_progress(
        process,
        tmp_path / "missing-progress.json",
        stall_timeout_seconds=0.3,
        poll_seconds=0.1,
        activity_paths=(log,),
    )
    assert completed is True
    assert process.terminated is False


def test_worker_reports_missing_backend_with_dedicated_exit_code(tmp_path: Path) -> None:
    import json

    from audio_transcription import worker

    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": 1,
                "backend": "mlx-whisper",
                "language": "ru",
                "config": {"model_name": "no-such-model", "offline": True},
                "chunks": [
                    {"sequence": 0, "path": str(tmp_path / "chunk.wav"), "start": 0.0, "end": 1.0}
                ],
            }
        ),
        encoding="utf-8",
    )
    import sys

    monkeypatch_modules = {name: None for name in ("mlx_audio.stt.generate", "mlx_audio.stt.utils")}
    saved = {name: sys.modules.get(name) for name in monkeypatch_modules}
    sys.modules.update(monkeypatch_modules)
    try:
        code = worker.main([str(request), str(tmp_path / "result.json"), str(tmp_path / "progress.json")])
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert code == isolated.WORKER_MISSING_EXIT_CODE
