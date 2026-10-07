from __future__ import annotations

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


# Голос, у которого речи меньше 5 секунд и меньше 3% всей речи, — почти
# всегда призрак: кашель, смех, кусок того же голоса. На 33 монологах доля
# записей с одним голосом 73% → 94%, DER на VoxConverse не меняется
# (experiments/diarization-natural, 2026-10-07). При известном числе голосов
# фильтр не нужен и вреден: съедает собеседника, сказавшего пару фраз.
GHOST_MAX_SECONDS = 5.0
GHOST_MAX_SHARE = 0.03


def drop_ghost_speakers(
    turns: list[SpeakerTurn],
    *,
    max_seconds: float = GHOST_MAX_SECONDS,
    max_share: float = GHOST_MAX_SHARE,
) -> list[SpeakerTurn]:
    """Отдаёт речь призрачных голосов ближайшему настоящему соседу."""
    totals: dict[str, float] = {}
    for turn in turns:
        for speaker in turn.speakers:
            totals[speaker] = totals.get(speaker, 0.0) + turn.end - turn.start
    whole = sum(totals.values())
    ghosts = {
        speaker
        for speaker, seconds in totals.items()
        if seconds < max_seconds and seconds < max_share * whole
    }
    if not ghosts or len(ghosts) == len(totals):
        return turns

    def neighbour(index: int) -> tuple[str, ...]:
        best: tuple[float, tuple[str, ...]] | None = None
        for step in (-1, 1):
            other = index + step
            while 0 <= other < len(turns):
                real = tuple(s for s in turns[other].speakers if s not in ghosts)
                if real:
                    gap = (
                        turns[index].start - turns[other].end
                        if step < 0
                        else turns[other].start - turns[index].end
                    )
                    if best is None or gap < best[0]:
                        best = (gap, real)
                    break
                other += step
        assert best is not None  # хотя бы один настоящий голос есть
        return best[1]

    kept = [s for s in sorted(totals, key=_speaker_order) if s not in ghosts]
    renamed = {speaker: f"speaker_{index}" for index, speaker in enumerate(kept, start=1)}
    result: list[SpeakerTurn] = []
    for index, turn in enumerate(turns):
        real = tuple(s for s in turn.speakers if s not in ghosts) or neighbour(index)
        speakers = tuple(renamed[s] for s in real)
        if result and result[-1].speakers == speakers and turn.start - result[-1].end <= 0.05:
            result[-1] = SpeakerTurn(result[-1].start, turn.end, speakers)
        else:
            result.append(SpeakerTurn(turn.start, turn.end, speakers))
    return result


def _speaker_order(label: str) -> tuple[int, str]:
    tail = label.rsplit("_", 1)[-1]
    return (int(tail), label) if tail.isdigit() else (1 << 30, label)


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


def _nearest_speakers(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> tuple[str, ...]:
    """Голос ближайшей реплики для окна, целиком попавшего в паузу диаризации.

    FluidAudio не размечает реплики короче секунды; VAD их слышит. Без этого
    такое окно посреди монолога становилось «unknown».
    """
    before = max((turn for turn in turns if turn.end <= start), key=lambda turn: turn.end, default=None)
    after = min((turn for turn in turns if turn.start >= end), key=lambda turn: turn.start, default=None)
    if before is None and after is None:
        return ("unknown",)
    if after is None:
        return before.speakers
    if before is None:
        return after.speakers
    return before.speakers if start - before.end <= after.start - end else after.speakers


def _fill_unassigned(
    intervals: list[tuple[float, float, tuple[str, ...] | None]],
    turns: list[SpeakerTurn],
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
            # Окно целиком в паузе: голос по времени, не делим, чтобы не резать слово.
            span_start, span_end = intervals[0][0], intervals[-1][1]
            result.append((start, end, _nearest_speakers(span_start, span_end, turns)))
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
    *,
    min_seconds: float = 0.1,
    merge_gap_seconds: float = 0.8,
) -> list[AudioChunk]:
    """Режет общие VAD-окна по сменам говорящих, не дублируя overlap-аудио."""
    if not base_chunks:
        return []
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
        for start, end, speakers in _fill_unassigned(raw_intervals, turns):
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

    # Окно — отрезок prepared.wav; звук режет воркер срезом массива.
    return [
        AudioChunk(sequence, prepared, start, end, speakers)
        for sequence, (start, end, speakers) in enumerate(intervals)
    ]
