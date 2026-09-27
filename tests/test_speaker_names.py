"""Имена говорящих: при прогоне флагом и в готовом результате скриптом."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_transcription.models import Diarization, Hypothesis, MediaInfo, Section, Segment, SpeakerTurn
from audio_transcription.render import parse_speaker_names, rename_speakers, write_bundle


def test_parse_numbers_like_in_text() -> None:
    assert parse_speaker_names("1=Антон, 2 = Мария Петрова,unknown=Зал") == {
        "speaker_1": "Антон",
        "speaker_2": "Мария Петрова",
        "unknown": "Зал",
    }


@pytest.mark.parametrize("text", ["", "Антон", "1=", "x=Антон"])
def test_parse_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError):
        parse_speaker_names(text)


def _bundle(tmp_path: Path, **kwargs) -> Path:
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    media = MediaInfo(source, 4.0, "pcm_s16le", 16000, 1, section=kwargs.pop("section", None))
    segments = [
        Segment(0, 2, "Привет", speakers=("speaker_1",)),
        Segment(2, 4, "Здравствуйте", speakers=("speaker_2",)),
        Segment(4, 5, "Да", speakers=("speaker_1", "speaker_2")),
    ]
    return write_bundle(
        tmp_path / "result",
        media=media,
        hypotheses=[Hypothesis("model", "ru", 0.2, segments)],
        lexical_segments=segments,
        readable_segments=segments,
        review_items=[],
        corrections=[],
        suggestions=[],
        mode="fast",
        offline=True,
        diarization=Diarization("sortformer", 0.4, [SpeakerTurn(0, 4, ("speaker_1",))]),
        **kwargs,
    )


def test_names_at_transcription_time(tmp_path: Path) -> None:
    output = _bundle(tmp_path, speaker_names={"speaker_1": "Антон"})
    readable = (output / "readable.md").read_text()
    assert "Антон: Привет" in readable and "Спикер 2: Здравствуйте" in readable
    assert "Перекрытие (Антон + Спикер 2)" in readable
    assert "Антон: Привет" in (output / "subtitles.srt").read_text()
    assert json.loads((output / "manifest.json").read_text())["speaker_names"] == {"speaker_1": "Антон"}


def test_rename_finished_result_keeps_machine_labels(tmp_path: Path) -> None:
    output = _bundle(tmp_path, section=Section(10, 14))
    raw_before = (output / "raw.json").read_text()
    missing = rename_speakers(output, {"speaker_2": "Мария", "speaker_7": "Никто"})
    assert missing == ["speaker_7"]
    readable = (output / "readable.md").read_text()
    assert "Мария: Здравствуйте" in readable and "Спикер 1: Привет" in readable
    assert "фрагмент 00:00:10–00:00:14" in readable
    for name in ("verbatim.md", "subtitles.srt", "subtitles.vtt"):
        assert "Мария" in (output / name).read_text()
    assert (output / "raw.json").read_text() == raw_before
    # Второй вызов добавляет имя, прежнее остаётся.
    rename_speakers(output, {"speaker_1": "Антон"})
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["speaker_names"] == {"speaker_2": "Мария", "speaker_7": "Никто", "speaker_1": "Антон"}
    assert "Антон: Привет" in (output / "readable.md").read_text()


def test_rename_refuses_result_without_speakers(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixture")
    segments = [Segment(0, 1, "Текст")]
    output = write_bundle(
        tmp_path / "plain",
        media=MediaInfo(source, 1.0, "pcm_s16le", 16000, 1),
        hypotheses=[Hypothesis("model", "ru", 0.1, segments)],
        lexical_segments=segments,
        readable_segments=segments,
        review_items=[],
        corrections=[],
        suggestions=[],
        mode="fast",
        offline=True,
    )
    with pytest.raises(ValueError, match="--diarize"):
        rename_speakers(output, {"speaker_1": "Антон"})
