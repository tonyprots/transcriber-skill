import pytest

from audio_transcription.models import AudioChunk, Hypothesis, Segment


def test_segment_validates_time() -> None:
    with pytest.raises(ValueError):
        Segment(2.0, 1.0, "текст")


def test_hypothesis_joins_segments() -> None:
    hypothesis = Hypothesis(
        model="test",
        language="ru",
        elapsed_seconds=1.0,
        segments=[Segment(0, 1, "Первый."), Segment(1, 2, "Второй.")],
    )
    assert hypothesis.text == "Первый. Второй."


def test_hypothesis_and_chunk_round_trip(tmp_path) -> None:
    hypothesis = Hypothesis("model", "ru", 1.2, [Segment(0, 1, "текст")], {"a": 1})
    assert Hypothesis.from_dict(hypothesis.to_dict()).to_dict() == hypothesis.to_dict()
    chunk = AudioChunk(2, tmp_path / "part.wav", 3.0, 4.0)
    payload = {**chunk.to_dict(), "path": str(chunk.path)}
    assert AudioChunk.from_dict(payload) == chunk
