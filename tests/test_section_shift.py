"""С --section все таймкоды результата идут по исходнику, и вложенные тоже.

Раньше сдвигались только верхние `start`/`end`: пословные метки Whisper,
части окна после retry-split и сырые реплики диаризации оставались во времени
фрагмента, хотя контракт обещает время исходника во всех файлах.
"""
from __future__ import annotations

from audio_transcription import cli
from audio_transcription.models import Diarization, Hypothesis, Segment, SpeakerTurn


def _times(node, found=None) -> list[float]:
    found = [] if found is None else found
    if isinstance(node, dict):
        found.extend(float(node[key]) for key in ("start", "end") if key in node)
        for value in node.values():
            _times(value, found)
    elif isinstance(node, list):
        for value in node:
            _times(value, found)
    return found


def test_shift_hypothesis_moves_words_and_subchunks() -> None:
    chunk = {
        "sequence": 1,
        "start": 1.0,
        "end": 3.0,
        "text": "раз два",
        "words": [{"start": 1.2, "end": 1.5, "text": "раз"}],
        "segment_metrics": [{"avg_logprob": -0.2}],
        "subchunks": [
            {"start": 1.0, "end": 2.0, "words": [{"start": 1.1, "end": 1.3, "text": "раз"}]},
            {"start": 2.0, "end": 3.0, "words": [{"start": 2.1, "end": 2.4, "text": "два"}]},
        ],
    }
    hypothesis = Hypothesis("m", "ru", 0.1, [Segment(1.0, 3.0, "раз два")], {"chunks": [chunk]})

    shifted = cli._shift_hypothesis(hypothesis, 650.0)

    before = _times(chunk)
    after = _times(shifted.metadata["chunks"][0])
    assert after == [value + 650.0 for value in before]
    assert shifted.metadata["chunks"][0]["segment_metrics"] == chunk["segment_metrics"]
    assert chunk["words"][0]["start"] == 1.2  # исходник не тронут


def test_shift_diarization_moves_turns_and_raw_turns() -> None:
    diarization = Diarization(
        "community-1",
        1.0,
        [SpeakerTurn(0.0, 2.0, ("1",))],
        {"raw_turns": [{"start": 0.0, "end": 2.0, "speakers": ["1"]}], "threshold": 0.7},
    )

    shifted = cli._shift_diarization(diarization, 650.0)

    assert (shifted.turns[0].start, shifted.turns[0].end) == (650.0, 652.0)
    assert shifted.metadata["raw_turns"] == [{"start": 650.0, "end": 652.0, "speakers": ["1"]}]
    assert shifted.metadata["threshold"] == 0.7
    assert diarization.metadata["raw_turns"][0]["start"] == 0.0
