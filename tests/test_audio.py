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


def test_add_quiet_onsets_moves_start_back_but_not_into_neighbour() -> None:
    # Тихая фраза начинается через 0,6 с после первого окна, основной проход
    # услышал её на 1,9 с позже — начало сдвигается к ней.
    windows = [(16_000, 60_000), (100_000, 140_000)]
    sensitive = [(16_000, 60_000), (70_000, 145_000)]
    assert audio.add_quiet_onsets(windows, sensitive, 500_000, sample_rate=16_000) == [
        (16_000, 60_000),
        (66_800, 140_000),
    ]


def test_add_quiet_onsets_does_not_bridge_pause_inside_speech() -> None:
    # Чувствительный интервал тянется из предыдущего окна через паузу —
    # это сплошная речь, разрезанная по лимиту окна; окно не трогаем.
    windows = [(16_000, 60_000), (70_000, 140_000)]
    sensitive = [(16_000, 145_000)]
    assert audio.add_quiet_onsets(windows, sensitive, 500_000, sample_rate=16_000) == windows


def test_add_quiet_onsets_ignores_breath_and_adds_missed_speech() -> None:
    # Начало на 0,3 с раньше — вдох, окно не трогаем; сдвиг не больше 4 с;
    # речь вне окон — отдельным окном с запасом 0,2 с.
    windows = [(100_000, 140_000), (300_000, 340_000)]
    sensitive = [(98_400, 140_000), (16_000, 32_000), (200_000, 330_000)]
    assert audio.add_quiet_onsets(windows, sensitive, 500_000, sample_rate=16_000) == [
        (12_800, 35_200),
        (100_000, 140_000),
        (236_000, 340_000),
    ]


def test_add_quiet_onsets_keeps_loud_speech_as_is() -> None:
    windows = [(16_000, 48_000), (64_000, 96_000)]
    sensitive = [(19_200, 44_800), (67_200, 92_800)]
    assert audio.add_quiet_onsets(windows, sensitive, 500_000, sample_rate=16_000) == windows


def _write_ramp(path: Path, samples: int) -> None:
    """WAV, где значение сэмпла равно его номеру: по срезу видно, откуда он."""
    import numpy as np

    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes((np.arange(samples) % 32_000).astype("<i2").tobytes())


def test_split_audio_chunk_preserves_absolute_timeline(tmp_path: Path) -> None:
    source = tmp_path / "prepared.wav"
    _write_ramp(source, 16_000 * 20)
    left, right = audio.split_audio_chunk(AudioChunk(3, source, 12.0, 16.0))
    assert (left.start, left.end) == (12.0, 14.0)
    assert (right.start, right.end) == (14.0, 16.0)
    assert left.path == right.path == source
    assert len(audio.chunk_waveform(left)) == 32_000
    # Половинки стыкуются сэмпл в сэмпл: ни потерянного, ни задвоенного.
    whole = audio.chunk_waveform(AudioChunk(3, source, 12.0, 16.0))
    halves = [*audio.chunk_waveform(left), *audio.chunk_waveform(right)]
    assert list(whole) == halves
    assert sorted(path.name for path in tmp_path.iterdir()) == ["prepared.wav"]


def test_chunk_waveform_is_a_slice_of_prepared(tmp_path: Path) -> None:
    source = tmp_path / "prepared.wav"
    _write_ramp(source, 16_000 * 3)
    samples = audio.chunk_waveform(AudioChunk(0, source, 1.0, 1.5))
    assert samples.dtype.name == "float32"
    assert len(samples) == 8_000
    assert round(float(samples[0]) * 32768) == 16_000
    # Окно за концом записи обрезается, а не падает.
    tail = audio.chunk_waveform(AudioChunk(1, source, 2.5, 4.0))
    assert len(tail) == 8_000


def test_chunk_waveform_rejects_unprepared_audio(tmp_path: Path) -> None:
    import pytest

    source = tmp_path / "stereo.wav"
    with wave.open(str(source), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\0\0\0\0" * 100)
    with pytest.raises(audio.MediaToolError):
        audio.chunk_waveform(AudioChunk(0, source, 0.0, 0.001))


def test_vad_load_respects_offline(monkeypatch, tmp_path: Path) -> None:
    """VAD — первый поход в сеть за прогон, и он обязан слушаться `--offline`."""
    import sys
    from types import SimpleNamespace

    from audio_transcription import weights

    seen: dict[str, str | None] = {}

    def fetch(entry):
        # Сеть теперь трогает скачивание весов, а не сам onnx-asr.
        seen["offline"] = audio.os.environ.get("HF_HUB_OFFLINE")
        return weights.Weights(tmp_path, "0" * 40, True)

    def load_vad(name: str, path=None, **kwargs):
        seen["path"] = path
        raise RuntimeError("дальше загрузки тест не идёт")

    monkeypatch.setattr(weights, "fetch", fetch)

    monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(load_vad=load_vad))
    monkeypatch.setitem(
        sys.modules,
        "onnx_asr.utils",
        SimpleNamespace(read_wav_files=lambda *a, **k: ([[0.0]], [1], audio.SAMPLE_RATE)),
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    import pytest

    with pytest.raises(Exception):
        audio.split_speech_windows(tmp_path / "prepared.wav", offline=True)
    assert seen["offline"] == "1"
    assert seen["path"] == tmp_path


def test_gain_for_vad_lifts_quiet_speech_and_keeps_loud_and_silence() -> None:
    """Тихая фраза рядом с громкой поднимается до уровня речи сама по себе.

    Общий множитель на файл громкая фраза бы заблокировала; ноль и громкое
    при этом не меняются, а модели получают исходный звук.
    """
    import numpy as np

    rate = audio.SAMPLE_RATE
    t = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    silence = np.zeros(rate * 5, dtype=np.float32)
    quiet, loud = tone * 10 ** (-70 / 20), tone * 0.5
    waveform = np.concatenate([silence, quiet, silence, loud, silence])

    lifted = audio.gain_for_vad(waveform)

    def rms(part):
        return float(np.sqrt(np.mean(part**2)))

    quiet_part = lifted[rate * 5 : rate * 6]
    loud_part = lifted[rate * 11 : rate * 12]
    assert rms(quiet_part) > 0.05
    assert np.allclose(loud_part, loud)
    assert not lifted[: rate * 2].any()
    assert lifted.dtype == waveform.dtype
    assert rms(waveform[rate * 5 : rate * 6]) < 0.001


def test_gain_for_vad_short_input_is_untouched() -> None:
    import numpy as np

    short = np.full(10, 1e-4, dtype=np.float32)
    assert np.array_equal(audio.gain_for_vad(short), short)
