"""Сторож и ошибки воркера называют модель, а не ключ воркера.

Parakeet и Vosk поднимаются тем же onnx-воркером, что GigaAM, и на английском
маршруте сторож писал «gigaam не сообщил о прогрессе» про Parakeet.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcription import cli, isolated
from audio_transcription.backends import BackendUnavailable
from audio_transcription.routing import language_route


@pytest.mark.parametrize("language", ["ru", "en"])
def test_onnx_family_gets_model_label(tmp_path: Path, monkeypatch, language: str) -> None:
    route = language_route(language)
    spec = next(
        candidate
        for candidate in (route.primary, route.verifier_for("max"), route.verifier_for("fast"))
        if candidate is not None and candidate.family in {"gigaam", "onnx"}
    )
    seen: dict = {}

    def fake(backend, chunks, language, work_dir, **kwargs):
        seen.update(backend=backend, **kwargs)
        raise BackendUnavailable("стоп")

    monkeypatch.setattr(cli, "run_isolated_backend", fake)
    args = cli.build_parser().parse_args(["a.wav", "--quiet"])
    with pytest.raises(BackendUnavailable):
        cli.run_route_backend(
            args, spec, [], [], tmp_path, role="primary", language=language, report=cli.ProgressReporter(enabled=False)
        )
    assert seen["backend"] == "gigaam"  # ключ воркера прежний
    assert seen["label"] == spec.model


def test_isolated_messages_use_label_and_safe_job_dir(tmp_path: Path, monkeypatch) -> None:
    seen: dict = {}

    def fake_worker(module, label, request, work_dir, **kwargs):
        seen["label"] = label
        # Имя каталога задания строится из подписи: слэш увёл бы его вглубь.
        Path(isolated.tempfile.mkdtemp(prefix=f"{label}-", dir=work_dir))
        return {"broken": True}

    monkeypatch.setattr(isolated, "_run_worker", fake_worker)
    with pytest.raises(BackendUnavailable, match="alphacep-vosk-model-ru"):
        isolated.run_isolated_backend(
            "gigaam", [], "ru", tmp_path, config={}, stall_timeout_seconds=5, label="alphacep/vosk-model-ru"
        )
    assert "gigaam" not in seen["label"]
