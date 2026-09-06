import wave
from pathlib import Path

from audio_transcription.models import AudioChunk
from audio_transcription.retry import recognize_with_retry


def write_wav(path: Path, seconds: int = 8) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\0\0" * 16_000 * seconds)


def test_empty_window_is_split_and_recovered(tmp_path: Path) -> None:
    source = tmp_path / "chunk.wav"
    write_wav(source)
    chunk = AudioChunk(7, source, 10.0, 18.0)

    def recognize(part: AudioChunk) -> dict:
        return {"text": "" if part.duration > 4 else f"часть-{part.start:g}"}

    result = recognize_with_retry(chunk, recognize, tmp_path / "retry")
    assert result["status"] == "recovered"
    assert result["retry_attempts"] == 3
    assert result["retry_depth"] == 1
    assert result["start"] == 10.0
    assert result["end"] == 18.0
    assert result["text"] == "часть-10 часть-14"


def test_unrecoverable_half_is_marked_partial(tmp_path: Path) -> None:
    source = tmp_path / "chunk.wav"
    write_wav(source)
    chunk = AudioChunk(0, source, 0.0, 8.0)

    def recognize(part: AudioChunk) -> dict:
        if part.duration > 4:
            return {"text": ""}
        if part.start < 4:
            return {"text": "левая часть"}
        raise RuntimeError("сбой декодера")

    result = recognize_with_retry(chunk, recognize, tmp_path / "retry")
    assert result["status"] == "partial"
    assert result["text"] == "левая часть"
    assert result["retry_depth"] == 2
    assert "сбой декодера" in str(result["subchunks"][1])


def test_short_failed_window_is_not_split(tmp_path: Path) -> None:
    source = tmp_path / "short.wav"
    write_wav(source, seconds=3)
    chunk = AudioChunk(0, source, 0.0, 3.0)
    result = recognize_with_retry(
        chunk,
        lambda _: {"text": ""},
        tmp_path / "retry",
    )
    assert result["status"] == "failed"
    assert result["retry_attempts"] == 1
    assert not (tmp_path / "retry").exists()
