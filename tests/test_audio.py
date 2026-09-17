import json
import hashlib
import wave
from pathlib import Path
from subprocess import CompletedProcess

from audio_transcription import audio
from audio_transcription.models import AudioChunk


def test_probe_media_parses_audio_stream(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "input.ogg"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(audio, "require_media_tools", lambda: ("ffmpeg", "ffprobe"))
    payload = {
        "streams": [{"codec_name": "opus", "sample_rate": "48000", "channels": 1}],
        "format": {"duration": "4.25"},
    }
    monkeypatch.setattr(
        audio.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, json.dumps(payload), ""),
    )
    info = audio.probe_media(source)
    assert info.codec == "opus"
    assert info.duration_seconds == 4.25
    assert info.sample_rate == 48000
    assert info.size_bytes == len(b"fixture")
    assert info.sha256 == hashlib.sha256(b"fixture").hexdigest()


def test_pack_speech_spans_keeps_context_and_respects_limit() -> None:
    spans = [(1_000, 20_000), (24_000, 80_000), (400_000, 430_000)]
    packed = audio.pack_speech_spans(
        spans,
        500_000,
        sample_rate=16_000,
        max_seconds=20,
        pad_seconds=0.2,
    )
    assert packed == [(0, 83_200), (396_800, 433_200)]


def test_split_audio_chunk_preserves_absolute_timeline(tmp_path: Path) -> None:
    source = tmp_path / "chunk.wav"
    with wave.open(str(source), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\0\0" * 64_000)
    left, right = audio.split_audio_chunk(
        AudioChunk(3, source, 12.0, 16.0),
        tmp_path / "split",
        label="test",
    )
    assert (left.start, left.end) == (12.0, 14.0)
    assert (right.start, right.end) == (14.0, 16.0)
    with wave.open(str(left.path), "rb") as part:
        assert part.getnframes() == 32_000


def test_vad_load_respects_offline(monkeypatch, tmp_path: Path) -> None:
    """VAD — первый поход в сеть за прогон, и он обязан слушаться `--offline`."""
    import sys
    from types import SimpleNamespace

    seen: dict[str, str | None] = {}

    def load_vad(name: str, **kwargs):
        seen["offline"] = audio.os.environ.get("HF_HUB_OFFLINE")
        raise RuntimeError("дальше загрузки тест не идёт")

    monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(load_vad=load_vad))
    monkeypatch.setitem(
        sys.modules,
        "onnx_asr.utils",
        SimpleNamespace(read_wav_files=lambda *a, **k: ([[0.0]], [1], audio.SAMPLE_RATE)),
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    import pytest

    with pytest.raises(Exception):
        audio.split_speech_windows(tmp_path / "prepared.wav", tmp_path, offline=True)
    assert seen["offline"] == "1"
