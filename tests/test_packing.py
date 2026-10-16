from pathlib import Path

from audio_transcription.models import AudioChunk, Hypothesis
from audio_transcription.packing import (
    PackedWindow,
    plan_packing,
    unpack_hypothesis,
)


def chunk(sequence: int, start: float, end: float, speakers: tuple[str, ...] = ()) -> AudioChunk:
    return AudioChunk(sequence, Path(f"chunk-{sequence}.wav"), start, end, speakers)


def test_adjacent_windows_are_packed_up_to_the_limit() -> None:
    chunks = [chunk(index, index * 2.0, index * 2.0 + 1.8) for index in range(20)]
    groups = plan_packing(chunks, max_seconds=10.0, max_gap_seconds=1.0)
    assert all(group[-1].end - group[0].start <= 10.0 for group in groups)
    assert sum(len(group) for group in groups) == len(chunks)
    assert len(groups) < len(chunks)


def test_long_pause_is_not_stitched_over() -> None:
    chunks = [chunk(0, 0.0, 2.0), chunk(1, 30.0, 32.0)]
    assert plan_packing(chunks, max_seconds=60.0, max_gap_seconds=3.0) == (
        [(chunks[0],), (chunks[1],)]
    )


def test_single_window_group_keeps_the_original_chunk() -> None:
    solo = chunk(7, 5.0, 6.0, ("speaker_1",))
    windows = [PackedWindow(chunk=solo, sources=(solo,))]
    packed = Hypothesis(
        "whisper",
        "ru",
        1.0,
        [],
        {"chunks": [{"sequence": 7, "start": 5.0, "end": 6.0, "text": "да"}]},
    )
    result = unpack_hypothesis(packed, windows)
    assert [segment.text for segment in result.segments] == ["да"]
    assert result.segments[0].speakers == ("speaker_1",)


def test_packed_text_is_split_back_by_word_timestamps() -> None:
    first = chunk(0, 0.0, 2.0, ("speaker_1",))
    second = chunk(1, 2.0, 4.0, ("speaker_2",))
    windows = [
        PackedWindow(
            chunk=chunk(0, 0.0, 4.0),
            sources=(first, second),
        )
    ]
    packed = Hypothesis(
        "whisper",
        "ru",
        1.0,
        [],
        {
            "chunks": [
                {
                    "sequence": 0,
                    "start": 0.0,
                    "end": 4.0,
                    "text": "привет как дела спасибо хорошо",
                    "words": [
                        {"text": "привет", "start": 0.1, "end": 0.6, "confidence": 0.9},
                        {"text": "как", "start": 0.7, "end": 1.0, "confidence": 0.9},
                        {"text": "дела", "start": 1.1, "end": 1.7, "confidence": 0.9},
                        {"text": "спасибо", "start": 2.2, "end": 2.8, "confidence": 0.8},
                        {"text": "хорошо", "start": 3.0, "end": 3.6, "confidence": 0.8},
                    ],
                }
            ]
        },
    )
    result = unpack_hypothesis(packed, windows)
    assert [(segment.start, segment.text) for segment in result.segments] == [
        (0.0, "привет как дела"),
        (2.0, "спасибо хорошо"),
    ]
    assert result.segments[0].speakers == ("speaker_1",)
    assert result.segments[1].speakers == ("speaker_2",)


def test_every_word_lands_in_exactly_one_window() -> None:
    sources = tuple(chunk(index, index * 2.0, index * 2.0 + 2.0) for index in range(5))
    words = [
        {"text": f"w{index}", "start": index * 0.5, "end": index * 0.5 + 0.4}
        for index in range(20)
    ]
    windows = [PackedWindow(chunk=chunk(0, 0.0, 10.0), sources=sources)]
    packed = Hypothesis(
        "whisper", "ru", 1.0, [], {"chunks": [{"sequence": 0, "start": 0.0, "end": 10.0,
                                               "text": " ".join(w["text"] for w in words),
                                               "words": words}]}
    )
    result = unpack_hypothesis(packed, windows)
    restored = " ".join(segment.text for segment in result.segments).split()
    assert restored == [word["text"] for word in words]


def test_fallback_does_not_copy_words_of_the_neighbours() -> None:
    """Пустое окно между соседями брало их крайние слова по допуску 0,35 с.

    «привет.» уже досталось A, «Классный» — C, а B получал «привет. Классный»,
    и в «Слушать» вставал ложный пункт «окно целиком под вопросом».
    """
    sources = (chunk(0, 0.0, 5.0), chunk(1, 5.2, 6.0), chunk(2, 6.1, 10.0))
    words = [
        {"text": "Я", "start": 3.6, "end": 3.8},
        {"text": "сказал", "start": 3.9, "end": 4.5},
        {"text": "привет.", "start": 4.6, "end": 5.1},
        {"text": "Классный", "start": 6.0, "end": 6.6},
        {"text": "день.", "start": 6.7, "end": 7.2},
    ]
    windows = [PackedWindow(chunk=chunk(0, 0.0, 10.0), sources=sources)]
    packed = Hypothesis(
        "whisper", "ru", 1.0, [], {"chunks": [{"sequence": 0, "start": 0.0, "end": 10.0,
                                               "text": " ".join(w["text"] for w in words),
                                               "words": words}]}
    )
    result = unpack_hypothesis(packed, windows)
    restored = " ".join(segment.text for segment in result.segments).split()
    assert restored == [word["text"] for word in words]
    assert [chunk_result["status"] for chunk_result in result.metadata["chunks"]] == [
        "ok", "empty", "ok",
    ]


def test_fallback_still_takes_a_word_from_the_gap() -> None:
    """Слово в зазоре между окнами никому не досталось — его берёт пустое окно."""
    sources = (chunk(0, 0.0, 2.0), chunk(1, 2.4, 3.0), chunk(2, 3.6, 6.0))
    words = [
        {"text": "раз", "start": 0.5, "end": 1.0},
        {"text": "два", "start": 2.95, "end": 3.1},
        {"text": "три", "start": 3.1, "end": 3.3},
        {"text": "четыре", "start": 4.0, "end": 4.5},
    ]
    windows = [PackedWindow(chunk=chunk(0, 0.0, 6.0), sources=sources)]
    packed = Hypothesis(
        "whisper", "ru", 1.0, [], {"chunks": [{"sequence": 0, "start": 0.0, "end": 6.0,
                                               "text": "раз два три четыре",
                                               "words": words}]}
    )
    result = unpack_hypothesis(packed, windows)
    assert [(segment.start, segment.text) for segment in result.segments] == [
        (0.0, "раз"),
        (2.4, "два три"),
        (3.6, "четыре"),
    ]


def test_missing_word_timestamps_do_not_lose_the_text() -> None:
    sources = (chunk(0, 0.0, 2.0), chunk(1, 2.0, 4.0))
    windows = [PackedWindow(chunk=chunk(0, 0.0, 4.0), sources=sources)]
    packed = Hypothesis(
        "whisper", "ru", 1.0, [],
        {"chunks": [{"sequence": 0, "start": 0.0, "end": 4.0, "text": "текст без слов"}]},
    )
    result = unpack_hypothesis(packed, windows)
    assert " ".join(segment.text for segment in result.segments) == "текст без слов"


def test_packed_window_sequences_are_unique(tmp_path: Path) -> None:
    """Одиночная группа получала номер исходного окна, а склейка — номер группы.

    Они совпадали, и unpack_hypothesis раскладывал текст не в то окно.
    """
    import wave

    from audio_transcription.packing import build_packed_windows

    prepared = tmp_path / "prepared.wav"
    with wave.open(str(prepared), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 16000 * 50)
    chunks = [
        chunk(0, 0.0, 2.0), chunk(1, 2.5, 4.0),   # группа 0 (склейка)
        chunk(2, 20.0, 22.0),                     # группа 1 (одиночная)
        chunk(3, 40.0, 42.0), chunk(4, 42.5, 44.0),  # группа 2 (склейка)
    ]
    windows = build_packed_windows(prepared, chunks, max_seconds=24.0)
    sequences = [window.chunk.sequence for window in windows]
    assert len(sequences) == len(set(sequences)) == 3
    assert windows[1].chunk.path == chunks[2].path
    # Склейка — отрезок того же prepared.wav, а не новый файл на диске.
    assert windows[0].chunk.path == prepared
    assert (windows[0].chunk.start, windows[0].chunk.end) == (0.0, 4.0)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["prepared.wav"]
