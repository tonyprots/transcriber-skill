"""Манифест называет модели, которые работали, а не умолчания флагов.

С 0.24 диаризацию по умолчанию делает FluidAudio, а в `model_revisions` и в
проверку устаревания шёл `--diarization-model` — Sortformer, который не
запускался. Ревизии FluidAudio в манифесте не было вовсе.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from audio_transcription import cli
from audio_transcription.backends import language_hypothesis
from audio_transcription.catalog import FLUIDAUDIO, SORTFORMER
from audio_transcription.models import Diarization, Hypothesis, Segment, SpeakerTurn
from audio_transcription.routing import language_route


def _args(*extra: str):
    return cli.build_parser().parse_args(["a.wav", *extra])


@pytest.mark.parametrize(
    ("kind", "present", "absent"),
    [("fluidaudio", FLUIDAUDIO.model, SORTFORMER.model), ("sortformer", SORTFORMER.model, FLUIDAUDIO.model)],
)
def test_used_models_name_the_diarization_that_ran(kind: str, present: str, absent: str) -> None:
    route = language_route("ru")
    models = cli.models_in_run(_args("--diarize"), route, route.verifier_for("max"), kind)
    assert models[0] == route.primary.model
    assert present in models and absent not in models


def test_used_models_without_diarization() -> None:
    route = language_route("ru")
    assert cli.models_in_run(_args(), route, None, None) == [route.primary.model]


requires_speech = pytest.mark.skipif(
    not (shutil.which("say") and shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="нужны say, ffmpeg и ffprobe (macOS)",
)


def _fake_backend(backend, chunks, language, work_dir, **kwargs) -> Hypothesis:
    if backend.endswith("-lid"):
        return language_hypothesis(backend, [{"ru": 0.97}] * len(chunks), 0.1)
    chunks = list(chunks)
    return Hypothesis(
        f"{backend}-fake",
        language,
        0.1,
        [Segment(chunk.start, chunk.end, "текст") for chunk in chunks],
        metadata={
            "chunks": [
                {"sequence": chunk.sequence, "start": chunk.start, "end": chunk.end, "text": "текст"}
                for chunk in chunks
            ]
        },
    )


@requires_speech
@pytest.mark.parametrize("backend", ["auto", "sortformer"])
def test_manifest_revisions_follow_diarization_backend(tmp_path: Path, monkeypatch, backend: str) -> None:
    source = tmp_path / "речь.aiff"
    subprocess.run(
        ["say", "-v", "Milena", "-o", str(source), "Сегодня смотрим отчёт и считаем бюджет"],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli, "run_isolated_backend", _fake_backend)
    monkeypatch.setattr(
        cli,
        "run_isolated_diarization",
        lambda kind, audio, work_dir, **kwargs: Diarization(
            "FluidAudio Community-1 offline" if kind == "fluidaudio" else SORTFORMER.model,
            0.1,
            [SpeakerTurn(0.0, 30.0, ("1",))],
            {"backend": kind},
        ),
    )
    checked: list[list[str]] = []
    monkeypatch.setattr(cli, "staleness_warnings", lambda models: checked.append(list(models)) or [])
    output = tmp_path / "out"

    cli.run(
        cli.build_parser().parse_args(
            [str(source), "--output", str(output), "--diarize", "--diarization-backend", backend,
             "--mode", "fast", "--language", "ru", "--no-cache", "--quiet"]
        )
    )

    revisions = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["model_revisions"]
    ran, idle = (FLUIDAUDIO, SORTFORMER) if backend == "auto" else (SORTFORMER, FLUIDAUDIO)
    assert revisions[ran.model] == ran.revision
    assert idle.model not in revisions
    assert ran.model in checked[0] and idle.model not in checked[0]
