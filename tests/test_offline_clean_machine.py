"""Чистая машина без сети: `--diarize --offline` после prefetch не ходит в сеть.

Тест выпуска (`pytest -m slow` перед тегом, references/maintenance.md). Сквозная
проверка 2026-10-08 шла с сетью и в обжитом HOME, поэтому не заметила, что
модели FluidAudio лежали не там, где их читает бинарник, и он молча качал свой
набор: `--offline` сеть диаризации не запрещал.

Что делает: временный HOME (и CFFIXED_USER_HOME — бинарник берёт домашний
каталог у CoreFoundation, а не из $HOME), общий кэш Hugging Face, затем
`prefetch_models.py ru --diarize` — единственный шаг с сетью, около 35 МБ
моделей FluidAudio, — и прогон 20 секунд двух голосов `say` под
`sandbox-exec` без сети. Модели ASR берутся из общего кэша: без них тест
пропускается, а не качает гигабайты.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from audio_transcription.catalog import FLUIDAUDIO, hf_cache_dir, route_entries

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "transcriber" / "scripts"
SANDBOX = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)"]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        (platform.system(), platform.machine()) != ("Darwin", "arm64"), reason="диаризация только на macOS arm64"
    ),
    pytest.mark.skipif(shutil.which("say") is None or shutil.which("ffmpeg") is None, reason="нужны say и ffmpeg"),
]


def _missing_route_models(cache: Path) -> list[str]:
    missing = []
    for entry in route_entries("ru", apple=True):
        if not entry.repo or not entry.in_hf_cache:
            continue
        snapshot = cache / f"models--{entry.repo.replace('/', '--')}" / "snapshots" / str(entry.revision)
        if not snapshot.is_dir():
            missing.append(entry.label)
    return missing


def _sandbox_works() -> bool:
    # Из чужой песочницы (агент) вложенную macOS не даёт.
    try:
        return subprocess.run([*SANDBOX, "/usr/bin/true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _two_voices(target: Path, work: Path) -> Path:
    """20 секунд: Милена и она же ниже на треть — второй голос для диаризации."""
    first, second = work / "first.aiff", work / "second.aiff"
    subprocess.run(
        ["say", "-v", "Milena", "-o", str(first),
         "Добрый день. Сегодня мы обсуждаем план релиза и проверку без сети на чистой машине."],
        check=True,
    )
    subprocess.run(
        ["say", "-v", "Milena", "-o", str(second),
         "Согласен. Диаризация должна работать офлайн, если модели скачаны заранее."],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(first), "-i", str(second), "-filter_complex",
         "[0:a]aresample=16000[a];[1:a]asetrate=22050*0.7,aresample=16000[b];"
         "[a][b]concat=n=2:v=0:a=1,apad=whole_dur=20[out]",
         "-map", "[out]", "-ac", "1", "-ar", "16000", "-t", "20", str(target)],
        check=True,
    )
    return target


def test_offline_diarize_on_clean_machine(tmp_path: Path) -> None:
    cache = hf_cache_dir()
    missing = _missing_route_models(cache)
    if missing:
        pytest.skip(f"нет моделей маршрута ru в кэше Hugging Face ({', '.join(missing)}): prefetch_models.py ru")
    if not _sandbox_works():
        pytest.skip("sandbox-exec недоступен (запуск из чужой песочницы)")
    home = tmp_path / "home"
    home.mkdir()
    environment = {
        **os.environ,
        "HOME": str(home),
        "CFFIXED_USER_HOME": str(home),
        "HF_HUB_CACHE": str(cache),
        "TRANSCRIBER_NO_UPDATE_CHECK": "1",
    }
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TRANSCRIBER_FLUIDAUDIO_BIN"):
        environment.pop(name, None)

    prefetch = subprocess.run(
        [sys.executable, str(SCRIPTS / "prefetch_models.py"), "ru", "--diarize"],
        env=environment, capture_output=True, text=True, timeout=1800,
    )
    assert prefetch.returncode == 0, prefetch.stderr[-2000:]
    models = home / "Library" / "Application Support" / "FluidAudio" / "Models" / "speaker-diarization"
    assert (models / "Segmentation.mlmodelc" / "coremldata.bin").is_file()

    audio = _two_voices(tmp_path / "two-voices.wav", tmp_path)
    output = tmp_path / "out"
    run = subprocess.run(
        [*SANDBOX, sys.executable, str(SCRIPTS / "transcribe.py"), str(audio), "--mode", "fast",
         "--language", "ru", "--diarize", "--offline", "--no-cache", "--quiet", "--output", str(output)],
        env=environment, capture_output=True, text=True, timeout=1800,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_revisions"].get("FluidAudio Community-1 offline") == FLUIDAUDIO.revision
    assert "speaker_" in (output / "speakers.json").read_text(encoding="utf-8")
