from pathlib import Path
import json

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


def _window(sequence: int, status: str, error: str | None = None) -> dict:
    window = {"sequence": sequence, "start": float(sequence), "end": sequence + 1.0,
              "text": "" if status == "failed" else "текст", "status": status}
    if error:
        window["error"] = error
    return window


def test_hypothesis_with_crashed_windows_is_not_cached(tmp_path: Path) -> None:
    """Сбой среды не должен стать ответом следующего прогона (2026-09-23)."""
    hypothesis = Hypothesis(
        "gigaam", "ru", 3.0, [Segment(0, 1, "текст")],
        {"chunks": [_window(0, "ok"),
                    _window(1, "failed", "dlopen: system is shutting down")]},
    )
    assert save_hypothesis(tmp_path, "key", hypothesis) is False
    assert not (tmp_path / "hypotheses" / "key.json").exists()
    assert load_hypothesis(tmp_path, "key") is None


def test_empty_answer_windows_are_still_cached(tmp_path: Path) -> None:
    """Пустой ответ на шуме повторяется — его кэшировать можно и нужно."""
    hypothesis = Hypothesis(
        "gigaam", "ru", 3.0, [Segment(0, 1, "текст")],
        {"chunks": [_window(0, "ok"),
                    _window(1, "failed", "модель вернула пустой текст")]},
    )
    assert save_hypothesis(tmp_path, "key", hypothesis) is True
    assert load_hypothesis(tmp_path, "key") is not None


def test_poisoned_entry_from_older_version_is_ignored(tmp_path: Path) -> None:
    """Запись, сохранённая до проверки, лечится при чтении, а не живёт вечно."""
    poisoned = Hypothesis(
        "gigaam", "ru", 0.2, [],
        {"chunks": [_window(n, "failed", "dlopen: system is shutting down")
                    for n in range(3)]},
    )
    directory = tmp_path / "hypotheses"
    directory.mkdir()
    (directory / "key.json").write_text(
        json.dumps(poisoned.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    assert load_hypothesis(tmp_path, "key") is None
