from pathlib import Path

from audio_transcription.cache import (
    cache_key,
    load_diarization,
    load_hypothesis,
    save_diarization,
    save_hypothesis,
)
from audio_transcription.models import Diarization, Hypothesis, Segment, SpeakerTurn


def test_cache_round_trip_marks_hit_and_zeroes_current_elapsed(tmp_path: Path) -> None:
    hypothesis = Hypothesis("model", "ru", 1.5, [Segment(0, 1, "текст")])
    key = cache_key("primary", {"source": "abc"})
    save_hypothesis(tmp_path, key, hypothesis)
    loaded = load_hypothesis(tmp_path, key)
    assert loaded is not None
    assert loaded.text == "текст"
    assert loaded.elapsed_seconds == 0.0
    assert loaded.metadata["cache_hit"] is True
    assert loaded.metadata["cached_elapsed_seconds"] == 1.5


def test_cache_key_is_stable_and_sensitive_to_config() -> None:
    assert cache_key("primary", {"a": 1, "b": 2}) == cache_key(
        "primary", {"b": 2, "a": 1}
    )
    assert cache_key("primary", {"a": 1}) != cache_key("primary", {"a": 2})


def test_diarization_cache_round_trip(tmp_path: Path) -> None:
    diarization = Diarization(
        "model",
        2.5,
        [SpeakerTurn(0.0, 1.0, ("speaker_1",))],
    )
    save_diarization(tmp_path, "key", diarization)
    loaded = load_diarization(tmp_path, "key")
    assert loaded is not None
    assert loaded.elapsed_seconds == 0.0
    assert loaded.speakers == ("speaker_1",)
    assert loaded.metadata["cache_hit"] is True
