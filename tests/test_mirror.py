"""Копия на YouTube: берём её субтитры только после сверки по звуку.

Пороги откалиброваны 2026-09-23 на трёх вебинарах (Рутуб ↔ YouTube):
своя пара даёт сходство 0,79–0,97, чужой выпуск той же серии — не выше 0,16.
Живой прогон — `fetch_media.py <ссылка Рутуба> --subtitles --mirror`.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

from audio_transcription import fetching, mirror
from audio_transcription.fetching import Cue, RemoteMedia, SubtitleTrack

SPEECH = (
    "сегодня мы покажем как доски и диск работают в одном окне сначала откроем "
    "файл потом добавим стикеры и позовём коллег обсудить макет прямо на доске "
    "а после встречи всё останется в диске вместе с историей изменений"
).split()
OTHER = (
    "автоматизация продаж начинается с воронки мы разберём роботов и триггеры "
    "покажем как настроить уведомления клиенту и как считать конверсию по этапам "
    "в конце ответим на вопросы из чата про интеграции и тарифы"
).split()
THIRD = (
    "в этом выпуске говорим про календарь и бронирование переговорок расскажем "
    "как пригласить внешних участников и как поставить встречу в повторяющийся "
    "слот а ещё покажем мобильное приложение и новые виджеты на рабочем столе"
).split()


def _media(duration: float = 1000.0) -> RemoteMedia:
    return RemoteMedia(
        url="https://rutube.ru/video/abc/",
        title="Доски и Диск",
        duration_seconds=duration,
        extractor="Rutube",
        tracks=(),
    )


def _cues_saying(text: list[str], at: dict[float, int], offset: float) -> list[Cue]:
    """Реплики по пять слов каждые 3 с; в точках `at` звучит `text`."""
    cues: list[Cue] = []
    filler = OTHER * 40
    moment = 0.0
    cursor = 0
    while moment < 1100:
        planted = next(
            (start for start in at if start + offset <= moment < start + offset + 45), None
        )
        if planted is not None:
            index = int((moment - planted - offset) / 3) * 5
            words = text[index : index + 5]
        else:
            words = filler[cursor : cursor + 5]
            cursor += 5
        if words:
            cues.append(Cue(moment, " ".join(words)))
        moment += 3.0
    return cues


def _hearing(text: list[str]):
    return lambda start, end: " ".join(text)


def test_same_recording_is_accepted_with_its_offset() -> None:
    cues = _cues_saying(SPEECH, {300.0: 0, 700.0: 0}, offset=-87.0)
    verdict = mirror.verify(
        _media(), mirror.Candidate("https://youtu.be/x", "копия", 913.0), cues, _hearing(SPEECH)
    )
    assert verdict.accepted, verdict.reason
    assert abs(verdict.offset + 87.0) <= 3.0


def test_another_episode_of_the_series_is_rejected() -> None:
    cues = _cues_saying(SPEECH, {300.0: 0, 700.0: 0}, offset=0.0)
    verdict = mirror.verify(
        _media(), mirror.Candidate("https://youtu.be/x", "копия", 1000.0), cues, _hearing(THIRD)
    )
    assert not verdict.accepted
    assert "не совпал" in verdict.reason


def test_reedited_copy_is_rejected() -> None:
    """Текст узнаётся, но сдвиг в двух точках разный — из копии вырезали кусок."""
    cues = _cues_saying(SPEECH, {300.0: 0}, offset=0.0)
    late = _cues_saying(SPEECH, {700.0: 0}, offset=-60.0)
    merged = [cue for cue in cues if cue.start < 500] + [cue for cue in late if cue.start >= 500]
    verdict = mirror.verify(
        _media(), mirror.Candidate("https://youtu.be/x", "копия", 1000.0), merged, _hearing(SPEECH)
    )
    assert not verdict.accepted
    assert "перемонтирована" in verdict.reason


def test_silent_anchor_proves_nothing() -> None:
    verdict = mirror.verify(
        _media(), mirror.Candidate("https://youtu.be/x", "копия", 1000.0), [], _hearing(["музыка"])
    )
    assert not verdict.accepted
    assert "нет речи" in verdict.reason


def test_captions_move_to_original_time() -> None:
    shifted = mirror.to_original_time([Cue(10.0, "заставка"), Cue(100.0, "речь")], offset=50.0)
    assert shifted == [Cue(50.0, "речь")]


def test_search_drops_clearly_different_lengths(monkeypatch) -> None:
    monkeypatch.setattr(fetching, "require_yt_dlp", lambda: "yt-dlp")
    stdout = "near\t3276\tBoards and Drive\nfar\t600\tTrailer\nlive\tNA\tStream\n"
    monkeypatch.setattr(
        fetching.subprocess, "run", lambda command, **_: CompletedProcess(command, 0, stdout, "")
    )
    found = mirror.search_youtube(_media(3240.0))
    assert [item.url.rsplit("=", 1)[-1] for item in found] == ["near", "live"]


def test_translated_track_of_a_copy_is_not_taken(monkeypatch, tmp_path: Path) -> None:
    """У английской копии дорожка `ru` — автоперевод, а не речь говорящих."""
    copy = RemoteMedia(
        url="https://www.youtube.com/watch?v=en",
        title="English talk",
        duration_seconds=1000.0,
        extractor="Youtube",
        tracks=(SubtitleTrack("ru", True), SubtitleTrack("en", True)),
        language="en",
    )
    monkeypatch.setattr(mirror, "probe_remote", lambda url: copy)
    monkeypatch.setattr(
        mirror, "fetch_subtitles", lambda *a, **k: (_ for _ in ()).throw(AssertionError("не качать"))
    )
    found, rejected = mirror.find_mirror(
        _media(),
        "ru",
        tmp_path,
        _hearing(SPEECH),
        search=lambda media: [mirror.Candidate(copy.url, copy.title, 1000.0)],
    )
    assert found is None
    assert "нет субтитров" in rejected[0]["reason"]


def test_captions_header_names_the_copy_and_the_proof() -> None:
    verdict = {
        "url": "https://www.youtube.com/watch?v=x",
        "title": "Копия",
        "offset_seconds": -86.7,
        "checks": [
            {"original_start": 1082.0, "mirror_start": 994.4, "similarity": 0.94},
            {"original_start": 2526.0, "mirror_start": 2440.2, "similarity": 0.944},
        ],
    }
    text = fetching.render_captions(
        _media(3608.0), SubtitleTrack("ru", True), [Cue(1.0, "речь")], mirror=verdict
    )
    assert "копии на YouTube" in text
    assert "watch?v=x" in text
    assert "сдвиг -86.7 с" in text
