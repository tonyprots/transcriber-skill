from audio_transcription.fillers import strip_filler_segments, strip_fillers
from audio_transcription.models import Segment


def test_filler_at_sentence_start_is_removed_and_next_word_capitalized() -> None:
    assert strip_fillers("Э-э-э, я вообще зачем это делал?") == ("Я вообще зачем это делал?", 1)
    assert strip_fillers("А-а, у него все процессы стоят.") == ("У него все процессы стоят.", 1)
    assert strip_fillers("Хмм, твоя жизнь довольно разнообразная.") == ("Твоя жизнь довольно разнообразная.", 1)


def test_filler_as_own_sentence_and_at_end() -> None:
    assert strip_fillers("Все процессы стоят. Хмм. Да.") == ("Все процессы стоят. Да.", 1)
    assert strip_fillers("Нужно завести аккаунт. Э-э-э") == ("Нужно завести аккаунт.", 1)
    assert strip_fillers("подчинялись и-и-и ну и перешла в маркетинг") == ("подчинялись ну и перешла в маркетинг", 1)


def test_real_words_and_single_conjunctions_survive() -> None:
    text = "а эмоции её такие, что вы предательница, и амбиции, и мама, uh-huh, um"
    assert strip_fillers(text) == ("а эмоции её такие, что вы предательница, и амбиции, и мама, uh-huh,", 1)
    assert strip_fillers("Ну, это стандартно для любой работы.") == ("Ну, это стандартно для любой работы.", 0)


def test_segment_of_only_fillers_is_dropped_and_others_kept() -> None:
    segments = [
        Segment(0, 1, "Э-э-э."),
        Segment(1, 2, "Ам, привет", speakers=("speaker_1",)),
        Segment(2, 3, "Чистое окно"),
    ]
    result, removed = strip_filler_segments(segments)
    assert removed == 2
    assert [segment.text for segment in result] == ["Привет", "Чистое окно"]
    assert result[0].speakers == ("speaker_1",)
    assert result[1] is segments[2]
