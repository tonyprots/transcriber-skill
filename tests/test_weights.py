"""Закреплённые веса: что качаем, что возвращаем, как падаем, если коммит пропал."""
from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub.errors import RevisionNotFoundError

from audio_transcription import catalog, weights


def _snapshot(tmp_path: Path, revision: str) -> str:
    path = tmp_path / "models--x" / "snapshots" / revision
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def test_fetch_asks_for_pinned_revision(tmp_path: Path) -> None:
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return _snapshot(tmp_path, kwargs["revision"])

    found = weights.fetch(catalog.GIGAAM_RUSSIAN, download=download)
    assert calls[0][0] == catalog.GIGAAM_RUSSIAN.repo
    assert calls[0][1]["revision"] == catalog.GIGAAM_RUSSIAN.revision
    # Только файлы E2E RNNT: репозиторий держит все варианты GigaAM v3 сразу.
    assert any("e2e_rnnt_encoder" in pattern for pattern in calls[0][1]["allow_patterns"])
    assert not any(pattern.startswith("v?_ctc") for pattern in calls[0][1]["allow_patterns"])
    assert found.revision == catalog.GIGAAM_RUSSIAN.revision
    assert found.pinned


def test_unpin_takes_main(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRANSCRIBER_UNPIN", "1")

    def download(repo, **kwargs):
        assert kwargs["revision"] is None
        return _snapshot(tmp_path, "f" * 40)

    found = weights.fetch(catalog.SILERO_VAD, download=download)
    assert found.revision == "f" * 40
    assert not found.pinned


class _Gone(RevisionNotFoundError):
    def __init__(self) -> None:  # без HTTP-ответа: нужен только тип
        Exception.__init__(self, "коммита больше нет")


def _gone() -> None:
    raise _Gone()


def test_vanished_commit_is_served_from_cache_first(tmp_path: Path) -> None:
    def download(repo, **kwargs):
        if not kwargs.get("local_files_only"):
            _gone()
        return _snapshot(tmp_path, kwargs["revision"])

    warnings: list[str] = []
    found = weights.fetch(catalog.SILERO_VAD, download=download, warn=warnings.append)
    assert found.pinned
    assert warnings == []


def test_vanished_commit_falls_back_to_main_loudly(tmp_path: Path) -> None:
    def download(repo, **kwargs):
        if kwargs.get("revision"):
            if kwargs.get("local_files_only"):
                raise FileNotFoundError("в кэше нет")
            _gone()
        return _snapshot(tmp_path, "e" * 40)

    warnings: list[str] = []
    found = weights.fetch(catalog.SILERO_VAD, download=download, warn=warnings.append)
    assert found.revision == "e" * 40
    assert not found.pinned
    assert "больше нет" in warnings[0]


def test_models_outside_catalog_load_by_name() -> None:
    assert weights.weights_for("large-v3") is None
    # FluidAudio ставится своим скриптом в свой каталог, не через кэш HF.
    assert weights.weights_for(catalog.FLUIDAUDIO.model) is None


@pytest.mark.parametrize("entry", [entry for entry in catalog.CATALOG if entry.family in {"vad", "gigaam", "onnx"}])
def test_onnx_patterns_follow_onnx_asr(entry) -> None:
    pytest.importorskip("onnx_asr")
    patterns = weights.allow_patterns(entry)
    assert "config.json" in patterns
    assert len(patterns) > 2, entry.key


def _args(**overrides):
    from types import SimpleNamespace

    return SimpleNamespace(**({"whisper_model": None} | overrides))


def test_cache_signature_follows_revision(monkeypatch, tmp_path: Path) -> None:
    """Перезалитые веса обязаны менять ключ кэша, иначе вернётся старая гипотеза."""
    from audio_transcription import cli

    spec = catalog.GIGAAM_RUSSIAN.spec
    pinned = cli.weights_signature(_args(), spec)
    assert pinned[catalog.GIGAAM_RUSSIAN.model] == catalog.GIGAAM_RUSSIAN.revision
    assert pinned[catalog.SILERO_VAD.model] == catalog.SILERO_VAD.revision

    monkeypatch.setenv("TRANSCRIBER_UNPIN", "1")
    monkeypatch.setattr(catalog, "hf_cache_dir", lambda: tmp_path)
    ref = tmp_path / catalog.GIGAAM_RUSSIAN.cache_dir / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("b" * 40, encoding="utf-8")
    unpinned = cli.weights_signature(_args(), spec)
    assert unpinned[catalog.GIGAAM_RUSSIAN.model] == "b" * 40
    assert cli.cache_key("primary", {"weights": pinned}) != cli.cache_key("primary", {"weights": unpinned})


def test_whisper_signature_covers_both_backends() -> None:
    from audio_transcription import cli

    signature = cli.weights_signature(_args(), catalog.WHISPER_TURBO.spec)
    assert signature[catalog.WHISPER_TURBO.model] == catalog.WHISPER_TURBO.revision
    assert signature[catalog.FASTER_WHISPER_FALLBACK.model] == catalog.FASTER_WHISPER_FALLBACK.revision


def test_foreign_weights_are_not_cached_under_pinned_key() -> None:
    from audio_transcription import cli

    signature = {"gigaam-v3-e2e-rnnt": "a" * 40}
    assert cli.weights_as_keyed({"weights": {"revision": "a" * 40, "pinned": True}}, signature)
    assert not cli.weights_as_keyed({"weights": {"revision": "e" * 40, "pinned": False}}, signature)
    # Старые гипотезы без поля и модели вне каталога кэшируются как раньше.
    assert cli.weights_as_keyed({}, signature)


def test_manifest_names_loaded_weights_and_warns_when_unpinned() -> None:
    from types import SimpleNamespace

    from audio_transcription import cli

    loaded = SimpleNamespace(model="gigaam-v3-e2e-rnnt", metadata={"weights": {"revision": "e" * 40, "pinned": False}})
    warnings: list[str] = []
    found = cli.loaded_revisions({"gigaam-v3-e2e-rnnt": "a" * 40}, [loaded], warnings)
    assert found == {"gigaam-v3-e2e-rnnt": "e" * 40}
    assert "не закреплённые" in warnings[0]
