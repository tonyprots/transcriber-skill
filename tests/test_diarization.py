import wave
from pathlib import Path

from audio_transcription.diarization import partition_turns, split_chunks_by_diarization
from audio_transcription.models import AudioChunk, SpeakerTurn


def test_partition_turns_keeps_overlap_and_stable_labels() -> None:
    turns = partition_turns(
        [
            SpeakerTurn(0.0, 5.0, ("7",)),
            SpeakerTurn(3.0, 8.0, ("9",)),
            SpeakerTurn(9.0, 10.0, ("7",)),
        ]
    )
    assert [turn.to_dict() for turn in turns] == [
        {"start": 0.0, "end": 3.0, "speakers": ["speaker_1"], "overlap": False},
        {
            "start": 3.0,
            "end": 5.0,
            "speakers": ["speaker_1", "speaker_2"],
            "overlap": True,
        },
        {"start": 5.0, "end": 8.0, "speakers": ["speaker_2"], "overlap": False},
        {"start": 9.0, "end": 10.0, "speakers": ["speaker_1"], "overlap": False},
    ]


def test_split_chunks_uses_speaker_boundaries_without_duplication(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.wav"
    with wave.open(str(prepared), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\0\0" * 160_000)
    chunks = split_chunks_by_diarization(
        prepared,
        [AudioChunk(0, tmp_path / "unused.wav", 0.0, 10.0)],
        [
            SpeakerTurn(0.0, 4.0, ("speaker_1",)),
            SpeakerTurn(4.0, 6.0, ("speaker_1", "speaker_2")),
            SpeakerTurn(6.0, 10.0, ("speaker_2",)),
        ],
    )
    assert [(item.start, item.end, item.speakers) for item in chunks] == [
        (0.0, 4.0, ("speaker_1",)),
        (4.0, 6.0, ("speaker_1", "speaker_2")),
        (6.0, 10.0, ("speaker_2",)),
    ]
    assert sum(item.duration for item in chunks) == 10.0
    # Окна — отрезки prepared.wav, нарезка ничего не пишет на диск.
    assert {item.path for item in chunks} == {prepared}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["prepared.wav"]


def test_split_chunks_merges_short_silence_for_same_speaker(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.wav"
    with wave.open(str(prepared), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(b"\0\0" * 80_000)
    chunks = split_chunks_by_diarization(
        prepared,
        [
            AudioChunk(0, tmp_path / "a.wav", 0.0, 2.0),
            AudioChunk(1, tmp_path / "b.wav", 2.5, 5.0),
        ],
        [SpeakerTurn(0.0, 5.0, ("speaker_1",))],
    )
    assert [(item.start, item.end, item.speakers) for item in chunks] == [
        (0.0, 5.0, ("speaker_1",))
    ]


def test_numeric_speaker_ids_are_ordered_numerically() -> None:
    from audio_transcription.diarization import partition_turns
    from audio_transcription.models import SpeakerTurn

    turns = [SpeakerTurn(float(i), float(i) + 1.0, (str(i + 1),)) for i in range(11)]
    result = partition_turns(turns)
    assert result[1].speakers == ("speaker_2",)
    assert result[10].speakers == ("speaker_11",)


def test_window_inside_diarization_pause_takes_nearest_speaker(tmp_path: Path) -> None:
    # FluidAudio не размечает реплики короче секунды: такое окно посреди
    # монолога раньше становилось «unknown».
    chunks = split_chunks_by_diarization(
        tmp_path / "prepared.wav",
        [
            AudioChunk(0, tmp_path / "a.wav", 0.0, 10.0),
            AudioChunk(1, tmp_path / "b.wav", 11.0, 11.6),
            AudioChunk(2, tmp_path / "c.wav", 18.5, 19.0),
            AudioChunk(3, tmp_path / "d.wav", 20.0, 30.0),
        ],
        [SpeakerTurn(0.0, 10.0, ("speaker_1",)), SpeakerTurn(20.0, 30.0, ("speaker_2",))],
        merge_gap_seconds=0.0,
    )
    assert [item.speakers for item in chunks] == [
        ("speaker_1",),
        ("speaker_1",),
        ("speaker_2",),
        ("speaker_2",),
    ]
