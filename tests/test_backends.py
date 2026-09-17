import sys
from pathlib import Path
from types import SimpleNamespace

from audio_transcription import backends
from audio_transcription.models import AudioChunk


def test_empty_chunks_do_not_load_heavy_models() -> None:
    primary = backends.GigaAMBackend("gigaam-v3-e2e-rnnt").transcribe_chunks([])
    verifier = backends.MLXWhisperBackend().transcribe_chunks([])
    assert primary.segments == []
    assert verifier.segments == []


def test_gigaam_passes_quantization_to_onnx_asr(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeModel:
        def recognize(self, path: str) -> str:
            return "hello"

    def load_model(name: str, **kwargs):
        captured.update({"name": name, **kwargs})
        return FakeModel()

    monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(load_model=load_model))
    hypothesis = backends.GigaAMBackend(
        "gigaam-multilingual-ctc",
        quantization="int8",
    ).transcribe_chunks(
        [AudioChunk(0, tmp_path / "chunk.wav", 0.0, 1.0)],
        "en",
    )
    assert hypothesis.text == "hello"
    assert captured["quantization"] == "int8"
    assert hypothesis.metadata["quantization"] == "int8"


def test_hub_fallback_retries_offline_when_hub_is_down(monkeypatch) -> None:
    calls = []

    def load():
        calls.append(backends.os.environ.get("HF_HUB_OFFLINE"))
        if len(calls) == 1:
            raise ConnectionError("Server disconnected without sending a response")
        return "model"

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert backends.load_with_hub_fallback(load, offline=False, label="x") == "model"
    assert calls == [None, "1"]


def test_hub_fallback_reports_both_failures(monkeypatch) -> None:
    import pytest

    def load():
        raise ConnectionError("down")

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(backends.BackendUnavailable, match="ни с Hub, ни из локального кэша"):
        backends.load_with_hub_fallback(load, offline=False, label="x")


def test_hub_fallback_does_not_wait_for_silent_hub(monkeypatch) -> None:
    """Молчащая сеть исключения не даёт: срок ожидания и уводит нас в кэш."""
    import threading
    import time

    release = threading.Event()

    def load():
        if backends.os.environ.get("HF_HUB_OFFLINE") == "1":
            return "model-from-cache"
        release.wait(30)  # Hub не отвечает и не отказывает
        return "model-from-hub"

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    started = time.monotonic()
    result = backends.load_with_hub_fallback(
        load, offline=False, label="x", stall_seconds=0.2
    )
    release.set()
    assert result == "model-from-cache"
    assert time.monotonic() - started < 5


def test_hub_fallback_waits_out_a_real_download(monkeypatch) -> None:
    """Первая установка качает гигабайты: срок не должен её срывать."""
    import threading

    downloaded = threading.Event()

    def load():
        if backends.os.environ.get("HF_HUB_OFFLINE") == "1":
            raise FileNotFoundError("кэша ещё нет")
        downloaded.wait(30)
        return "model-from-hub"

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    threading.Timer(0.3, downloaded.set).start()
    result = backends.load_with_hub_fallback(
        load, offline=False, label="x", stall_seconds=0.1
    )
    assert result == "model-from-hub"
    assert backends.os.environ.get("HF_HUB_OFFLINE") == "0"
