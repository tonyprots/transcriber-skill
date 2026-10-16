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
        def recognize(self, waveform, sample_rate: int) -> str:
            # Окно приходит массивом из памяти, а не путём к файлу.
            captured["samples"] = len(waveform)
            captured["sample_rate"] = sample_rate
            return "hello"

    def load_model(name: str, path=None, **kwargs):
        captured.update({"name": name, "path": path, **kwargs})
        return FakeModel()

    from audio_transcription.weights import Weights

    pinned = Weights(tmp_path / "snapshots" / ("a" * 40), "a" * 40, True)
    monkeypatch.setattr(backends, "weights_for", lambda model, quantization=None: pinned)
    monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(load_model=load_model))
    import wave

    prepared = tmp_path / "prepared.wav"
    with wave.open(str(prepared), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\0\0" * 32_000)
    hypothesis = backends.GigaAMBackend(
        "gigaam-multilingual-ctc",
        quantization="int8",
    ).transcribe_chunks(
        [AudioChunk(0, prepared, 0.5, 1.5)],
        "en",
    )
    assert (captured["samples"], captured["sample_rate"]) == (16_000, 16_000)
    assert hypothesis.text == "hello"
    assert captured["quantization"] == "int8"
    assert hypothesis.metadata["quantization"] == "int8"
    # Модель грузится из закреплённого снапшота, и гипотеза помнит, из какого.
    assert captured["path"] == pinned.path
    assert hypothesis.metadata["weights"] == {"revision": "a" * 40, "pinned": True}


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


def _restore_hub_state(monkeypatch) -> None:
    """Переключение офлайна глобально для процесса: вернуть как было после теста."""
    import huggingface_hub.constants as constants

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", constants.HF_HUB_OFFLINE)
    monkeypatch.setattr(backends, "_ORIGINAL_HUB_ENV", None, raising=False)


def test_hub_fallback_does_not_build_model_twice(monkeypatch, tmp_path: Path) -> None:
    """Веса уже найдены, а сборка из кэша дольше срока: ждём её, а не строим вторую копию.

    Срок нужен сетевой части. Раньше под него попадала и сборка: после 45 с
    офлайн-попытка строила ещё одну модель параллельно первой, а флаг офлайна
    оставался в окружении и уходил во все воркеры прогона.
    """
    import threading
    import time

    from audio_transcription import catalog, weights

    _restore_hub_state(monkeypatch)
    snapshot = tmp_path / "models--x" / "snapshots" / catalog.GIGAAM_RUSSIAN.revision
    snapshot.mkdir(parents=True)
    lock = threading.Lock()
    state = {"calls": 0, "active": 0, "peak": 0}

    def load():
        weights.fetch(catalog.GIGAAM_RUSSIAN, download=lambda repo, **kwargs: str(snapshot))
        with lock:
            state["calls"] += 1
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.5)
        with lock:
            state["active"] -= 1
        return "model"

    assert backends.load_with_hub_fallback(load, offline=False, label="x", stall_seconds=0.1) == "model"
    assert (state["calls"], state["peak"]) == (1, 1)
    assert backends.os.environ.get("HF_HUB_OFFLINE") != "1"


def test_workers_do_not_inherit_forced_offline(monkeypatch, tmp_path: Path) -> None:
    """Офлайн, в который ушёл главный процесс, не наследуется воркерами.

    Иначе воркер с ещё не скачанной моделью падал сразу: и онлайн-, и
    офлайн-попытка видели HF_HUB_OFFLINE=1.
    """
    import pytest

    from audio_transcription import isolated

    _restore_hub_state(monkeypatch)
    captured: list[dict[str, str]] = []

    class Done:
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def popen(command, **kwargs):
        # Только флаги офлайна: всё окружение в отчёте pytest — это и токены.
        captured.append({name: kwargs["env"].get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")})
        return Done()

    monkeypatch.setattr(isolated.subprocess, "Popen", popen)
    backends.force_hub_offline()
    with pytest.raises(backends.BackendUnavailable):
        isolated._run_worker("x", "x", {}, tmp_path, stall_timeout_seconds=1.0)
    assert captured[0] == {"HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None}

    # Офлайн, который пользователь задал сам, воркеры получают как есть.
    _restore_hub_state(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    backends.force_hub_offline()
    backends._set_hub_offline(False)
    with pytest.raises(backends.BackendUnavailable):
        isolated._run_worker("x", "x", {}, tmp_path, stall_timeout_seconds=1.0)
    assert captured[1]["HF_HUB_OFFLINE"] == "1"
