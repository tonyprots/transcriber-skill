"""Сквозной прогон: настоящий ffmpeg, настоящий VAD, настоящая сборка каталога.

Остальные тесты проверяют детали по отдельности и проходят за полсекунды, но
между ними остаётся зазор: склейка окон, распаковка по таймкодам, сверка,
словарь и запись артефактов ни разу не встречаются в одном прогоне. Регрессия
в стыке видна только здесь.

Быстрый тест подменяет ASR фальшивыми бэкендами — весов не нужно, а весь путь
от файла до `manifest.json` проходится целиком. Помеченный `slow` гоняет тот же
путь на настоящих моделях, если они уже скачаны.

Речь берём у системного синтезатора macOS: он даёт живой сигнал, который проходит
VAD, и не тащит в публичный репозиторий чужой голос.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from audio_transcription import cli
from audio_transcription.catalog import GIGAAM_RUSSIAN, hf_cache_dir
from audio_transcription.models import Hypothesis, Segment

PRIMARY_TEXT = "сегодня смотрим отчёт в PostgreSQL и считаем бюджет"
VERIFIER_TEXT = "сегодня смотрим отчёт в постгрес и считаем бюджет"


def _say_available() -> bool:
    return bool(shutil.which("say") and shutil.which("ffmpeg") and shutil.which("ffprobe"))


requires_speech = pytest.mark.skipif(
    not _say_available(), reason="нужны say, ffmpeg и ffprobe (macOS)"
)


@pytest.fixture
def spoken_audio(tmp_path: Path) -> Path:
    source = tmp_path / "речь.aiff"
    subprocess.run(
        ["say", "-v", "Milena", "-o", str(source), "Сегодня смотрим отчёт и считаем бюджет"],
        check=True,
        capture_output=True,
    )
    return source


def fake_backend(backend: str, chunks, language: str, work_dir: Path, **kwargs) -> Hypothesis:
    """Гипотеза по тем же окнам, что пришли: расхождение ровно одно, на термине.

    `metadata["chunks"]` обязателен: по `sequence` результат проверяющей
    раскладывается обратно на исходные окна, и без него распаковка вернёт пусто.
    """
    text = PRIMARY_TEXT if backend == "gigaam" else VERIFIER_TEXT
    chunks = list(chunks)
    segments = [Segment(chunk.start, chunk.end, text) for chunk in chunks]
    metadata = {
        "chunks": [
            {"sequence": chunk.sequence, "start": chunk.start, "end": chunk.end, "text": text}
            for chunk in chunks
        ]
    }
    return Hypothesis(f"{backend}-fake", language, len(chunks), segments, metadata=metadata)


@requires_speech
def test_pipeline_writes_full_contract(tmp_path: Path, spoken_audio: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    output = tmp_path / "result"

    result = cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output",
                str(output),
                "--mode",
                "max",
                "--language",
                "ru",
                "--no-cache",
                "--quiet",
                # Склейку проверяет test_packing; здесь нужен прямой путь
                # «окно к окну», чтобы фальшивый бэкенд отвечал за то же окно.
                "--verifier-window-seconds",
                "0",
            ]
        )
    )

    assert result == output
    produced = {path.name for path in output.iterdir()}
    assert produced == {
        "manifest.json",
        "raw.json",
        "verbatim.md",
        "readable.md",
        "subtitles.srt",
        "subtitles.vtt",
        "segments.json",
        "glossary-audit.json",
        "review-needed.md",
    }

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 6
    assert manifest["mode"] == "max"
    assert manifest["language_route"]["verifier_used"] is not None
    assert manifest["timings_seconds"]["total"] > 0
    # VAD настоящий: окна пришли из речи, а не из пустого списка.
    assert manifest["source"]["duration_seconds"] > 0

    # Расхождение моделей дошло до очереди вместе с обеими формулировками.
    review = (output / "review-needed.md").read_text(encoding="utf-8")
    assert "PostgreSQL" in review and "постгрес" in review
    assert manifest["review_count"] >= 1

    # Текст основной модели не подменён вариантом проверяющей.
    assert "PostgreSQL" in (output / "verbatim.md").read_text(encoding="utf-8")
    assert "постгрес" not in (output / "readable.md").read_text(encoding="utf-8")

    srt = (output / "subtitles.srt").read_text(encoding="utf-8")
    assert "-->" in srt and "PostgreSQL" in srt


@requires_speech
def test_pipeline_applies_glossary_and_keeps_raw_intact(
    tmp_path: Path, spoken_audio: Path, monkeypatch
) -> None:
    """Словарь правит читаемый текст, но не аудит-след сырых гипотез."""
    monkeypatch.setattr(cli, "run_isolated_backend", fake_backend)
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text(
        "version: 1\nlanguage: ru\nentries:\n"
        "  - canonical: PostgreSQL\n    aliases: [постгрес]\n    auto_apply: true\n",
        encoding="utf-8",
    )
    output = tmp_path / "result"

    cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output",
                str(output),
                "--mode",
                "fast",
                "--language",
                "ru",
                "--glossary",
                str(glossary),
                "--no-cache",
                "--quiet",
            ]
        )
    )

    raw = json.loads((output / "raw.json").read_text(encoding="utf-8"))
    assert any(PRIMARY_TEXT in segment["text"] for h in raw["hypotheses"] for segment in h["segments"])
    audit = json.loads((output / "glossary-audit.json").read_text(encoding="utf-8"))
    assert isinstance(audit["corrections"], list)


@pytest.mark.slow
@requires_speech
def test_pipeline_on_real_models(tmp_path: Path, spoken_audio: Path) -> None:
    """То же самое, но настоящими моделями: запускать вручную `pytest -m slow`."""
    if not (hf_cache_dir() / (GIGAAM_RUSSIAN.cache_dir or "")).is_dir():
        pytest.skip("модели русского маршрута не скачаны")
    output = tmp_path / "result"

    cli.run(
        cli.build_parser().parse_args(
            [
                str(spoken_audio),
                "--output",
                str(output),
                "--mode",
                "fast",
                "--language",
                "ru",
                "--no-cache",
                "--quiet",
            ]
        )
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    # Ревизии весов проставлены настоящие — именно они делают прогон воспроизводимым.
    assert manifest["model_revisions"][GIGAAM_RUSSIAN.model]
    text = (output / "verbatim.md").read_text(encoding="utf-8")
    assert "бюджет" in text.lower()
