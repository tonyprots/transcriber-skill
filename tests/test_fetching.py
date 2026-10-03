from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from audio_transcription import fetching
from audio_transcription.fetching import RemoteMedia, SubtitleTrack


@pytest.fixture
def yt_dlp(monkeypatch):
    """Подменяет бинарник и вызов, возвращая заранее заданный stdout."""

    def install(stdout: str = "", returncode: int = 0, stderr: str = "") -> list[list[str]]:
        calls: list[list[str]] = []
        monkeypatch.setattr(fetching, "require_yt_dlp", lambda: "yt-dlp")
        # Возраст yt-dlp проверяется только на отказе и отдельным вызовом;
        # здесь он свежий, чтобы счёт вызовов описывал саму операцию.
        monkeypatch.setattr(fetching, "yt_dlp_status", lambda today=None: {"stale": False})

        def fake_run(command, **kwargs):
            calls.append(command)
            return CompletedProcess(command, returncode, stdout, stderr)

        monkeypatch.setattr(fetching.subprocess, "run", fake_run)
        return calls

    return install


def test_url_is_told_apart_from_path() -> None:
    """Ссылка не должна уехать в Path: тот схлопнет `https://` в `https:/`."""
    assert fetching.looks_like_url("https://youtu.be/abc")
    assert fetching.looks_like_url("  HTTP://example.com/v  ")
    assert not fetching.looks_like_url("~/Downloads/интервью.m4a")
    assert not fetching.looks_like_url("/tmp/audio.wav")


def test_probe_splits_manual_and_automatic_tracks(yt_dlp) -> None:
    """Происхождение дорожки решает, каким флагом её потом качать."""
    payload = {
        "title": "Интервью",
        "duration": 4496.0,
        "extractor_key": "Youtube",
        "subtitles": {"en": [{"ext": "vtt"}]},
        "automatic_captions": {"en": [{"ext": "vtt"}], "ru": [{"ext": "vtt"}]},
    }
    yt_dlp(stdout=json.dumps(payload))
    media = fetching.probe_remote("https://youtu.be/abc")
    assert media.title == "Интервью"
    assert media.duration_seconds == 4496.0
    assert SubtitleTrack("en", automatic=False) in media.tracks
    assert SubtitleTrack("ru", automatic=True) in media.tracks


def test_probe_reports_failure_from_stderr(yt_dlp) -> None:
    """Причина отказа yt-dlp должна доезжать до пользователя, а не теряться."""
    yt_dlp(returncode=1, stderr="ERROR: Video unavailable")
    with pytest.raises(fetching.FetchError, match="Video unavailable"):
        fetching.probe_remote("https://youtu.be/abc")


@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        ((), SubtitleTrack("en", automatic=False)),
        (("ru",), SubtitleTrack("ru", automatic=True)),
        (("en",), SubtitleTrack("en", automatic=False)),
    ],
)
def test_requested_language_outranks_track_origin(languages, expected) -> None:
    """Авто на нужном языке лучше авторских на чужом; без запроса — авторские."""
    media = RemoteMedia(
        url="https://example.com/v",
        title="t",
        duration_seconds=None,
        extractor=None,
        tracks=(
            SubtitleTrack("en", automatic=False),
            SubtitleTrack("ru", automatic=True),
        ),
    )
    assert media.track_for(languages) == expected


def test_unnamed_language_falls_back_to_the_original() -> None:
    """Автодорожки отсортированы по алфавиту: без этого правила победит афарская."""
    media = RemoteMedia(
        url="https://example.com/v",
        title="t",
        duration_seconds=None,
        extractor=None,
        tracks=(
            SubtitleTrack("aa", automatic=True),
            SubtitleTrack("en", automatic=True),
        ),
        language="en-US",
    )
    assert media.track_for() == SubtitleTrack("en", automatic=True)


def test_unknown_original_language_refuses_to_guess() -> None:
    """Оригинал неизвестен и язык не назвали — лучше None, чем случайная дорожка."""
    media = RemoteMedia(
        url="https://example.com/v",
        title="t",
        duration_seconds=None,
        extractor=None,
        tracks=(SubtitleTrack("aa", automatic=True),),
    )
    assert media.track_for() is None


def test_regional_variants_count_as_one_language() -> None:
    """YouTube отдаёт `en-orig` и `en-US`; запрос `en` обязан их находить."""
    media = RemoteMedia(
        url="https://example.com/v",
        title="t",
        duration_seconds=None,
        extractor=None,
        tracks=(SubtitleTrack("en-orig", automatic=True),),
    )
    assert media.track_for(("en",)) == SubtitleTrack("en-orig", automatic=True)


def test_summary_counts_automatic_tracks_instead_of_listing(yt_dlp) -> None:
    """Сотня машинных переводов в выводе — это залитый контекст, а не польза."""
    media = RemoteMedia(
        url="https://example.com/v",
        title="t",
        duration_seconds=None,
        extractor=None,
        tracks=tuple(
            [SubtitleTrack("en", automatic=False)]
            + [SubtitleTrack(f"l{index}", automatic=True) for index in range(160)]
            + [SubtitleTrack("ru", automatic=True)]
        ),
    )
    summary = media.summary(("ru",))
    assert summary["subtitles"] == {
        "manual": ["en"],
        "automatic_count": 161,
        "automatic_matched": ["ru"],
    }


def test_auto_track_in_another_language_is_marked_as_translation() -> None:
    """Дорожка `ru` у английского ролика — перевод распознавания, а не речь."""
    media = RemoteMedia(
        url="https://example.com/v",
        title="t",
        duration_seconds=None,
        extractor=None,
        tracks=(),
        language="en",
    )
    assert media.is_translation(SubtitleTrack("ru", automatic=True))
    assert not media.is_translation(SubtitleTrack("en-orig", automatic=True))
    assert not media.is_translation(SubtitleTrack("ru", automatic=False))


def test_translation_warning_reaches_the_captions_file() -> None:
    """Предупреждение должно жить в самом файле: его переживёт любая пересылка."""
    media = RemoteMedia(
        url="https://example.com/v",
        title="Интервью",
        duration_seconds=600.0,
        extractor="Youtube",
        tracks=(),
        language="en",
    )
    track = SubtitleTrack("ru", automatic=True)
    rendered = fetching.render_captions(media, track, [fetching.Cue(1.0, "привет")])
    assert "машинный перевод" in rendered
    assert "[00:00:01] привет" in rendered


def test_rolling_captions_are_collapsed(tmp_path: Path) -> None:
    """Автосубтитры повторяют предыдущую реплику как контекст — иначе текст втрое длиннее."""
    source = tmp_path / "captions.vtt"
    source.write_text(
        "WEBVTT\n"
        "Kind: captions\n"
        "\n"
        "00:00:01.000 --> 00:00:03.000 align:start position:0%\n"
        "we have half a billion\n"
        "\n"
        "00:00:03.000 --> 00:00:05.000\n"
        "we have half a billion\n"
        "<c>monthly active users</c>\n"
        "\n"
        "00:00:05.000 --> 00:00:07.000\n"
        "&gt;&gt; that&#39;s a lot\n",
        encoding="utf-8",
    )
    cues = fetching.read_cues(source)
    assert [cue.text for cue in cues] == [
        "we have half a billion",
        "monthly active users",
        ">> that's a lot",
    ]
    assert cues[-1].timestamp == "00:00:05"


def test_distant_repetition_survives(tmp_path: Path) -> None:
    """Повтор за пределами окна — законная речь, вырезать его нельзя."""
    blocks = [f"00:00:{second:02d}.000 --> 00:00:{second + 1:02d}.000\nда\n" for second in (1,)]
    blocks += [
        f"00:00:{second:02d}.000 --> 00:00:{second + 1:02d}.000\nреплика {second}\n"
        for second in range(2, 8)
    ]
    blocks.append("00:00:09.000 --> 00:00:10.000\nда\n")
    source = tmp_path / "captions.vtt"
    source.write_text("WEBVTT\n\n" + "\n".join(blocks), encoding="utf-8")
    assert [cue.text for cue in fetching.read_cues(source)].count("да") == 2


def test_audio_prefers_light_track_in_parallel(yt_dlp, tmp_path: Path) -> None:
    """ВК отдавал 128 МБ по фрагменту: 100 минут на 68-минутный ролик.

    Лёгкая дорожка в восемь потоков — 30 секунд. Нижняя граница 64 кбит/с
    держит YouTube на прежней opus 103k вместо HE-AAC 49k.
    """
    calls = yt_dlp()
    (tmp_path / "audio.m4a").write_bytes(b"x")
    fetching.fetch_audio("https://vkvideo.ru/video-1_2", tmp_path)
    command = calls[0]
    assert command[command.index("-f") + 1].startswith("bestaudio[abr>=64][abr<=96]/")
    assert command[command.index("--concurrent-fragments") + 1] == "8"


def test_rate_limit_is_retried_then_reported(yt_dlp, monkeypatch) -> None:
    """429 от YouTube проходит через минуту — повторяем, но не бесконечно."""
    pauses: list[float] = []
    monkeypatch.setattr(fetching, "_sleep", pauses.append)
    calls = yt_dlp(returncode=1, stderr="ERROR: HTTP Error 429: Too Many Requests")
    with pytest.raises(fetching.FetchError, match="429"):
        fetching.probe_remote("https://www.youtube.com/watch?v=abc")
    assert len(calls) == 3 and pauses == list(fetching._RETRY_PAUSES_SECONDS)


def test_other_failures_are_not_retried(yt_dlp, monkeypatch) -> None:
    monkeypatch.setattr(fetching, "_sleep", lambda _: pytest.fail("незачем ждать"))
    calls = yt_dlp(returncode=1, stderr="ERROR: Video unavailable")
    with pytest.raises(fetching.FetchError):
        fetching.probe_remote("https://www.youtube.com/watch?v=abc")
    assert len(calls) == 1


def test_probe_reports_date_and_channel(yt_dlp) -> None:
    """По дате отсекают старые ролики до расшифровки, а не после."""
    yt_dlp(stdout=json.dumps({"title": "t", "upload_date": "20241010", "channel": "Канал"}))
    summary = fetching.probe_remote("https://vkvideo.ru/video-1_2").summary()
    assert summary["upload_date"] == "2024-10-10"
    assert summary["channel"] == "Канал"


def test_rutube_auto_track_is_not_passed_off_as_manual(yt_dlp) -> None:
    """Рутуб кладёт распознавание в `subtitles`; выдаёт его только имя дорожки."""
    payload = {
        "title": "Митап",
        "duration": 3600.0,
        "extractor_key": "Rutube",
        "subtitles": {"ru": [{"ext": "srt", "name": "Русский • Авто"}]},
    }
    yt_dlp(stdout=json.dumps(payload))
    media = fetching.probe_remote("https://rutube.ru/video/abc/")
    track = media.track_for(("ru",))
    assert track is not None and track.kind == "auto"
    assert not track.automatic, "качать всё равно через --write-subs"
    assert media.track_for(("ru",), allow_automatic=False) is None
    assert "автоматические субтитры" in fetching.render_captions(media, track, [])


def test_vk_track_origin_is_reported_as_unknown(yt_dlp) -> None:
    payload = {
        "title": "Подкаст",
        "duration": 3600.0,
        "extractor_key": "VKVideo",
        "subtitles": {"ru": [{"ext": "srt"}]},
    }
    yt_dlp(stdout=json.dumps(payload))
    media = fetching.probe_remote("https://vkvideo.ru/video-1_2")
    track = media.track_for(("ru",))
    assert track is not None and track.kind == "unknown"
    assert media.summary(("ru",))["subtitles"]["origin_unknown"] == ["ru"]
    assert "происхождение не сообщается" in fetching.render_captions(media, track, [])


def test_playlist_expands_to_videos_in_order(yt_dlp) -> None:
    payload = {
        "_type": "playlist",
        "entries": [
            {"url": "https://www.youtube.com/watch?v=a"},
            None,
            {"webpage_url": "https://www.youtube.com/watch?v=b", "url": "b"},
            {"url": "not-a-link"},
        ],
    }
    calls = yt_dlp(stdout=json.dumps(payload))
    assert fetching.expand_playlist("https://www.youtube.com/playlist?list=x", limit=5) == [
        "https://www.youtube.com/watch?v=a",
        "https://www.youtube.com/watch?v=b",
    ]
    assert "--no-playlist" not in calls[0]
    assert calls[0][calls[0].index("--playlist-end") + 1] == "5"


def test_single_video_is_its_own_playlist(yt_dlp) -> None:
    yt_dlp(stdout=json.dumps({"_type": "video", "id": "a"}))
    assert fetching.expand_playlist("https://youtu.be/a") == ["https://youtu.be/a"]


def test_empty_playlist_is_an_error(yt_dlp) -> None:
    yt_dlp(stdout=json.dumps({"_type": "playlist", "entries": []}))
    with pytest.raises(fetching.FetchError, match="нет доступных"):
        fetching.expand_playlist("https://www.youtube.com/playlist?list=x")


def test_stale_yt_dlp_is_named_in_the_error(yt_dlp, monkeypatch) -> None:
    yt_dlp(returncode=1, stderr="ERROR: Unable to extract")
    monkeypatch.setattr(
        fetching,
        "yt_dlp_status",
        lambda today=None: {"stale": True, "version": "2026.08.19", "age_days": 60, "path": "/opt/homebrew/bin/yt-dlp"},
    )
    with pytest.raises(fetching.FetchError, match="brew upgrade yt-dlp"):
        fetching.probe_remote("https://www.youtube.com/watch?v=abc")


def test_yt_dlp_age_from_version(monkeypatch) -> None:
    from datetime import date

    monkeypatch.setattr(fetching, "find_command", lambda name: "/opt/homebrew/bin/yt-dlp")
    monkeypatch.setattr(
        fetching.subprocess, "run", lambda *a, **k: CompletedProcess(a, 0, "2026.08.19\n", "")
    )
    status = fetching.yt_dlp_status(today=date(2026, 9, 27))
    assert status["version"] == "2026.08.19" and status["age_days"] == 39 and status["stale"]
    assert fetching.update_hint(status) == "brew upgrade yt-dlp"
    assert fetching.update_hint({"path": "/Users/x/.venv/bin/yt-dlp"}) == "pip install -U yt-dlp"


@pytest.mark.parametrize(
    ("raw", "public"),
    [
        ("https://www.youtube.com/watch?v=abc123&t=42&si=share-token", "https://www.youtube.com/watch?v=abc123"),
        ("https://youtu.be/abc123?si=share-token", "https://youtu.be/abc123"),
        ("https://vk.com/video?z=video-1_2&access_key=secret", "https://vk.com/video?z=video-1_2"),
        (
            "https://bucket.s3.amazonaws.com/a.mp3?X-Amz-Signature=deadbeef&X-Amz-Credential=AKIA",
            "https://bucket.s3.amazonaws.com/a.mp3",
        ),
        ("https://files.example.com/a.m4a?sig=abc&se=2026-10-04&sv=2024", "https://files.example.com/a.m4a"),
        ("https://user:password@intra.example.com/rec.mp4#t=10", "https://intra.example.com/rec.mp4"),
        ("https://music.yandex.ru/album/1/track/2", "https://music.yandex.ru/album/1/track/2"),
    ],
)
def test_public_url_strips_credentials(raw: str, public: str) -> None:
    assert fetching.public_url(raw) == public
