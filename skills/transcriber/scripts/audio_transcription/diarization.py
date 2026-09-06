from __future__ import annotations

import wave
from pathlib import Path

from .models import AudioChunk, SpeakerTurn


def partition_turns(raw_turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Превращает пересекающиеся интервалы модели в непересекающуюся шкалу."""
    if not raw_turns:
        return []
    # Порядок по числу, если id числовой: у FluidAudio «10» не должен стоять раньше «2».
    speaker_ids = sorted(
        {speaker for turn in raw_turns for speaker in turn.speakers},
        key=lambda item: (0, int(item)) if item.isdigit() else (1, item),
    )
    labels = {speaker: f"speaker_{index}" for index, speaker in enumerate(speaker_ids, start=1)}
    boundaries = sorted({value for turn in raw_turns for value in (turn.start, turn.end)})
    result: list[SpeakerTurn] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < 0.01:
            continue
        midpoint = (start + end) / 2
        active = tuple(
            labels[speaker]
            for speaker in speaker_ids
            if any(
                speaker in turn.speakers and turn.start <= midpoint < turn.end
                for turn in raw_turns
            )
        )
        if not active:
            continue
        if result and result[-1].speakers == active and start - result[-1].end <= 0.05:
            previous = result[-1]
            result[-1] = SpeakerTurn(previous.start, end, active)
        else:
            result.append(SpeakerTurn(start, end, active))
    return result


def _speakers_for_interval(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> tuple[str, ...] | None:
    midpoint = (start + end) / 2
    for turn in turns:
        if turn.start <= midpoint < turn.end:
            return turn.speakers
    return None


def _fill_unassigned(
    intervals: list[tuple[float, float, tuple[str, ...] | None]],
) -> list[tuple[float, float, tuple[str, ...]]]:
    result: list[tuple[float, float, tuple[str, ...]]] = []
    for index, (start, end, speakers) in enumerate(intervals):
        if speakers is not None:
            result.append((start, end, speakers))
            continue
        previous = next(
            (item[2] for item in reversed(intervals[:index]) if item[2] is not None),
            None,
        )
        following = next(
            (item[2] for item in intervals[index + 1 :] if item[2] is not None),
            None,
        )
        if previous is None and following is None:
            result.append((start, end, ("unknown",)))
        elif previous is None:
            result.append((start, end, following))
        elif following is None or previous == following:
            result.append((start, end, previous))
        else:
            midpoint = (start + end) / 2
            result.append((start, midpoint, previous))
            result.append((midpoint, end, following))
    return result


def split_chunks_by_diarization(
    prepared: Path,
    base_chunks: list[AudioChunk],
    turns: list[SpeakerTurn],
    work_dir: Path,
    *,
    min_seconds: float = 0.1,
    merge_gap_seconds: float = 0.8,
) -> list[AudioChunk]:
    """Режет общие VAD-окна по сменам говорящих, не дублируя overlap-аудио."""
    if not base_chunks:
        return []
    with wave.open(str(prepared), "rb") as source:
        params = source.getparams()
        frame_count = source.getnframes()
        frames = source.readframes(frame_count)
    frame_size = params.nchannels * params.sampwidth
    sample_rate = params.framerate

    intervals: list[tuple[float, float, tuple[str, ...]]] = []
    for chunk in base_chunks:
        boundaries = {chunk.start, chunk.end}
        for turn in turns:
            if turn.end <= chunk.start or turn.start >= chunk.end:
                continue
            boundaries.add(max(chunk.start, turn.start))
            boundaries.add(min(chunk.end, turn.end))
        ordered = sorted(boundaries)
        raw_intervals = []
        for start, end in zip(ordered, ordered[1:]):
            if end - start < min_seconds:
                continue
            speakers = _speakers_for_interval(start, end, turns)
            raw_intervals.append((start, end, speakers))
        for start, end, speakers in _fill_unassigned(raw_intervals):
            if (
                intervals
                and intervals[-1][2] == speakers
                and start - intervals[-1][1] <= merge_gap_seconds
                and end - intervals[-1][0] <= 20.0
            ):
                previous = intervals[-1]
                intervals[-1] = (previous[0], end, speakers)
            else:
                intervals.append((start, end, speakers))

    chunk_dir = work_dir / "speaker-chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for sequence, (start, end, speakers) in enumerate(intervals):
        start_frame = max(0, min(frame_count, round(start * sample_rate)))
        end_frame = max(start_frame, min(frame_count, round(end * sample_rate)))
        path = chunk_dir / f"chunk-{sequence:04d}.wav"
        with wave.open(str(path), "wb") as target:
            target.setparams(params)
            target.writeframes(frames[start_frame * frame_size : end_frame * frame_size])
        result.append(AudioChunk(sequence, path, start, end, speakers))
    return result
