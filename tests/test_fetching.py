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

        def fake_execute(command, **kwargs):
            calls.append(command)
            return CompletedProcess(command, returncode, stdout, stderr)

        monkeypatch.setattr(fetching, "_execute", fake_execute)
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


def test_audio_prefers_light_track_in_parallel(monkeypatch, tmp_path: Path) -> None:
    """ВК отдавал 128 МБ по фрагменту: 100 минут на 68-минутный ролик.

    Лёгкая дорожка в восемь потоков — 30 секунд. Нижняя граница 64 кбит/с
    держит YouTube на прежней opus 103k вместо HE-AAC 49k.
    """
    calls = _writing_yt_dlp(monkeypatch, [{"m4a": "x"}])
    fetching.fetch_audio("https://vkvideo.ru/video-1_2", tmp_path)
    command = calls[0]
    selector = command[command.index("-f") + 1].split("/")
    assert "bestaudio[abr>=64][abr<=96]" in selector
    assert command[command.index("--concurrent-fragments") + 1] == "8"


def test_audio_prefers_original_over_auto_dub(monkeypatch, tmp_path: Path) -> None:
    """С автодубляжем оригинал шёл на 62k, фильтр 64–96 взял тамильский.

    Полтора часа Whisper распознавал дубляж как английский (2026-10-06).
    """
    calls = _writing_yt_dlp(monkeypatch, [{"webm": "x"}])
    fetching.fetch_audio("https://youtu.be/PkoX6R5sn8Y", tmp_path)
    selector = calls[0][calls[0].index("-f") + 1].split("/")
    light = selector.index("bestaudio[abr>=64][abr<=96]")
    assert all("original" in option for option in selector[:light])
    assert "bestaudio[format_note*=original][abr<=96]" in selector[:light]


def test_rate_limit_is_retried_then_reported(yt_dlp, monkeypatch) -> None:
    """429 от YouTube проходит через минуту — повторяем, но не бесконечно."""
    pauses: list[float] = []
    monkeypatch.setattr(fetching, "_sleep", pauses.append)
    calls = yt_dlp(returncode=1, stderr="ERROR: HTTP Error 429: Too Many Requests")
    with pytest.raises(fetching.FetchError, match="429"):
        fetching.probe_remote("https://www.youtube.com/watch?v=abc")
    assert len(calls) == 3 and pauses == list(fetching._RETRY_PAUSES_SECONDS)


def _scripted_yt_dlp(monkeypatch, results: list[tuple]) -> list[list[str]]:
    """Каждый вызов берёт следующий (returncode, stderr[, файлы]) из списка.

    Файлы — {то, что встанет вместо `%(ext)s`: содержимое} по шаблону `-o`.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(fetching, "require_yt_dlp", lambda: "yt-dlp")
    monkeypatch.setattr(fetching, "yt_dlp_status", lambda today=None: {"stale": False})

    def fake_execute(command, **kwargs):
        returncode, stderr, *written = results[len(calls)]
        calls.append(command)
        template = command[command.index("-o") + 1] if "-o" in command else ""
        for ext, body in (written[0] if written else {}).items():
            Path(template.replace("%(ext)s", ext)).write_text(body, encoding="utf-8")
        return CompletedProcess(command, returncode, "", stderr)

    monkeypatch.setattr(fetching, "_execute", fake_execute)
    return calls


_STREAM_403 = "ERROR: unable to download video data: HTTP Error 403: Forbidden"


def test_youtube_stream_403_falls_back_to_other_clients(monkeypatch, tmp_path: Path) -> None:
    """2026-10-04: метаданные отдались, поток visionos ответил 403.

    Повтор идёт свежим извлечением через android и mweb — они отдают ролик
    360p целиком. Огрызок первой попытки файлом результата не считается.
    """
    calls = _scripted_yt_dlp(
        monkeypatch, [(1, _STREAM_403, {"webm.part": "x"}), (0, "", {"mp4": "360p"})]
    )
    path = fetching.fetch_audio("https://youtu.be/MM-C3JqCXBk", tmp_path)
    assert path == tmp_path / "audio.mp4" and path.read_text(encoding="utf-8") == "360p"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["audio.mp4"]
    assert len(calls) == 2 and "--extractor-args" not in calls[0]
    retry = calls[1]
    assert retry[retry.index("--extractor-args") + 1] == fetching._YOUTUBE_FALLBACK_CLIENTS


def test_stream_403_twice_names_the_network(monkeypatch, tmp_path: Path) -> None:
    calls = _scripted_yt_dlp(monkeypatch, [(1, _STREAM_403), (1, _STREAM_403)])
    with pytest.raises(fetching.FetchError, match="VPN"):
        fetching.fetch_audio("https://www.youtube.com/watch?v=abc", tmp_path)
    assert len(calls) == 2


def test_stream_403_elsewhere_is_not_retried(monkeypatch, tmp_path: Path) -> None:
    calls = _scripted_yt_dlp(monkeypatch, [(1, _STREAM_403)])
    with pytest.raises(fetching.FetchError, match="403"):
        fetching.fetch_audio("https://vkvideo.ru/video-1_2", tmp_path)
    assert len(calls) == 1


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


def _script(tmp_path: Path, body: str) -> str:
    script = tmp_path / "fake-yt-dlp"
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def test_hung_metadata_call_is_stopped(tmp_path: Path, monkeypatch) -> None:
    """Зависший экстрактор не должен держать агента вечно."""
    monkeypatch.setattr(fetching, "require_yt_dlp", lambda: _script(tmp_path, "sleep 30\n"))
    monkeypatch.setattr(fetching, "METADATA_TIMEOUT_SECONDS", 0.5)
    with pytest.raises(fetching.FetchError, match="не ответил"):
        fetching._run_yt_dlp(["x"], failure="f", timeout=0.5)


def test_stalled_download_is_stopped_but_slow_one_is_not(tmp_path: Path, monkeypatch) -> None:
    # Запас вчетверо: при сроке 1 с тест падал на машине под нагрузкой (load
    # average 60, 2026-10-04) — запуск оболочки съедал разницу.
    monkeypatch.setattr(fetching, "STALL_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(fetching, "_WATCH_PERIOD_SECONDS", 0.2)
    target = tmp_path / "out"
    target.mkdir()
    # Медленно, но с данными: 3 с при сроке простоя в 2 с — живое скачивание.
    slow = _script(tmp_path, f"for i in 1 2 3 4 5 6; do echo x >> '{target}/audio.part'; sleep 0.5; done\necho done\n")
    monkeypatch.setattr(fetching, "require_yt_dlp", lambda: slow)
    assert fetching._run_yt_dlp(["x"], failure="f", watch=target).strip() == "done"
    stuck = _script(tmp_path, "sleep 30\n")
    monkeypatch.setattr(fetching, "require_yt_dlp", lambda: stuck)
    with pytest.raises(fetching.FetchError, match="без новых данных"):
        fetching._run_yt_dlp(["x"], failure="f", watch=target)


_VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n{text}\n"


def _writing_yt_dlp(monkeypatch, outputs: list[dict[str, str]]) -> list[list[str]]:
    """yt-dlp, который пишет файлы по шаблону `-o`: вызов N берёт outputs[N].

    Ключ — то, что yt-dlp подставит вместо `%(ext)s` (у субтитров `ru.vtt`).
    Готовый файл yt-dlp не перезаписывает («has already been downloaded»,
    «subtitle … is already present»), и подмена ведёт себя так же.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(fetching, "require_yt_dlp", lambda: "yt-dlp")
    monkeypatch.setattr(fetching, "yt_dlp_status", lambda today=None: {"stale": False})

    def fake_execute(command, **kwargs):
        written = outputs[len(calls)]
        calls.append(command)
        template = command[command.index("-o") + 1]
        for ext, body in written.items():
            target = Path(template.replace("%(ext)s", ext))
            if not target.exists():
                target.write_text(body, encoding="utf-8")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(fetching, "_execute", fake_execute)
    return calls


def _video(*languages: str) -> RemoteMedia:
    return RemoteMedia(
        url="https://www.youtube.com/watch?v=b",
        title="Ролик Б",
        duration_seconds=60.0,
        extractor="Youtube",
        tracks=tuple(SubtitleTrack(language, True) for language in languages),
        language=languages[0],
    )


def test_subtitles_ignore_other_files_in_the_folder(monkeypatch, tmp_path: Path) -> None:
    """Каталог по умолчанию общий (./media): там лежат свой `captions.md`
    и субтитры прошлых роликов. Брать надо только то, что скачано сейчас.

    До правки при `ru` первым по алфавиту оказывался `captions.md` (0 реплик,
    и он перезаписывался пустой шапкой), а `captions.en.vtt` чужого ролика
    уходил под шапкой нового.
    """
    (tmp_path / "captions.md").write_text("# Ролик Б\n\n[00:00:01] прошлый прогон\n", encoding="utf-8")
    (tmp_path / "captions.en.vtt").write_text(_VTT.format(text="video A"), encoding="utf-8")
    _writing_yt_dlp(monkeypatch, [{"ru.vtt": _VTT.format(text="ролик Б")}])
    raw, track = fetching.fetch_subtitles(_video("ru"), tmp_path, languages=("ru",))
    assert raw == tmp_path / "captions.ru.vtt"
    assert [cue.text for cue in fetching.read_cues(raw)] == ["ролик Б"]
    assert track.language == "ru"
    assert not [path.name for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_subtitles_replace_the_same_language_from_a_previous_video(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "captions.ru.vtt").write_text(_VTT.format(text="ролик А"), encoding="utf-8")
    _writing_yt_dlp(monkeypatch, [{"ru.vtt": _VTT.format(text="ролик Б")}])
    raw, _ = fetching.fetch_subtitles(_video("ru"), tmp_path, languages=("ru",))
    assert [cue.text for cue in fetching.read_cues(raw)] == ["ролик Б"]


def test_subtitles_missing_after_success_is_an_error(monkeypatch, tmp_path: Path) -> None:
    """Чужой файл в каталоге не выдаётся за скачанный."""
    (tmp_path / "captions.en.vtt").write_text(_VTT.format(text="video A"), encoding="utf-8")
    _writing_yt_dlp(monkeypatch, [{}])
    with pytest.raises(fetching.FetchError, match="файла субтитров"):
        fetching.fetch_subtitles(_video("ru"), tmp_path, languages=("ru",))


@pytest.mark.parametrize("previous", ["audio.m4a", "audio.webm"])
def test_audio_is_the_one_just_downloaded(monkeypatch, tmp_path: Path, previous: str) -> None:
    """В каталоге звук прошлой ссылки: вернуть надо новый.

    До правки другой формат проигрывал по алфавиту (`m4a` < `webm`), а тот же
    формат yt-dlp не перезаписывал — «has already been downloaded».
    """
    (tmp_path / previous).write_text("ролик А", encoding="utf-8")
    _writing_yt_dlp(monkeypatch, [{"webm": "ролик Б"}])
    path = fetching.fetch_audio("https://youtu.be/b", tmp_path)
    assert path == tmp_path / "audio.webm"
    assert path.read_text(encoding="utf-8") == "ролик Б"


def _channel_with_tabs() -> dict:
    """Ссылка на канал без вкладки: yt-dlp 2026.8.19 отдаёт плейлист из вкладок.

    `extractor/youtube/_tab.py`: при вкладках Shorts или Live каждая
    извлекается во вложенный плейлист с `webpage_url` вкладки — ролики даже
    при `--flat-playlist` лежат внутри.
    """
    return {
        "_type": "playlist",
        "title": "Канал",
        "entries": [
            {
                "_type": "playlist",
                "webpage_url": "https://www.youtube.com/@chan/videos",
                "entries": [
                    {"_type": "url", "url": "https://www.youtube.com/watch?v=V1"},
                    {"_type": "url", "url": "https://www.youtube.com/watch?v=V2"},
                ],
            },
            {
                "_type": "playlist",
                "webpage_url": "https://www.youtube.com/@chan/shorts",
                "entries": [{"_type": "url", "url": "https://www.youtube.com/shorts/S1"}],
            },
        ],
    }


def test_channel_tabs_expand_to_videos_not_tab_links(yt_dlp) -> None:
    """До правки наружу уходили адреса вкладок `/videos` и `/shorts`."""
    yt_dlp(stdout=json.dumps(_channel_with_tabs()))
    assert fetching.expand_playlist("https://www.youtube.com/@chan") == [
        "https://www.youtube.com/watch?v=V1",
        "https://www.youtube.com/watch?v=V2",
        "https://www.youtube.com/shorts/S1",
    ]


def test_playlist_limit_counts_videos_across_tabs(yt_dlp) -> None:
    yt_dlp(stdout=json.dumps(_channel_with_tabs()))
    assert fetching.expand_playlist("https://www.youtube.com/@chan", limit=2) == [
        "https://www.youtube.com/watch?v=V1",
        "https://www.youtube.com/watch?v=V2",
    ]


def test_nested_playlist_without_entries_is_not_taken_for_a_video(yt_dlp) -> None:
    payload = {
        "_type": "playlist",
        "entries": [
            {"_type": "playlist", "webpage_url": "https://www.youtube.com/@chan/streams"},
            {"_type": "url", "url": "https://www.youtube.com/watch?v=V1"},
        ],
    }
    yt_dlp(stdout=json.dumps(payload))
    assert fetching.expand_playlist("https://www.youtube.com/@chan") == [
        "https://www.youtube.com/watch?v=V1"
    ]


def test_playlist_link_without_flag_is_refused_with_a_hint(yt_dlp) -> None:
    """Ссылка на плейлист без `--playlist`: `--no-playlist` без id ролика не
    действует, и до правки yt-dlp извлекал каждый ролик целиком — у большого
    плейлиста упор в срок 300 с, у малого расшифровка одного файла под
    названием плейлиста.
    """
    payload = {
        "_type": "playlist",
        "title": "Курс",
        "entries": [{"_type": "url", "url": f"https://www.youtube.com/watch?v={n}"} for n in "abc"],
    }
    calls = yt_dlp(stdout=json.dumps(payload))
    with pytest.raises(fetching.FetchError, match=r"роликов: 3.*--playlist"):
        fetching.probe_remote("https://www.youtube.com/playlist?list=PL1")
    assert "--flat-playlist" in calls[0] and "--no-playlist" in calls[0]


def test_channel_link_without_flag_counts_videos_not_tabs(yt_dlp) -> None:
    calls = yt_dlp(stdout=json.dumps(_channel_with_tabs()))
    with pytest.raises(fetching.FetchError, match=r"роликов: 3.*--playlist"):
        fetching.probe_remote("https://www.youtube.com/@chan")
    assert len(calls) == 1


def test_single_video_probe_is_flat_safe(yt_dlp) -> None:
    """`--flat-playlist` на одиночный ролик не влияет: он извлекается целиком."""
    calls = yt_dlp(stdout=json.dumps({"_type": "video", "title": "Ролик", "duration": 60}))
    media = fetching.probe_remote("https://www.youtube.com/watch?v=a&list=PL1")
    assert media.title == "Ролик" and media.duration_seconds == 60
    assert "--flat-playlist" in calls[0]


def test_playlist_of_one_is_its_video(monkeypatch) -> None:
    """Плейлист из одного ролика (пост с одним видео) раньше расшифровывался —
    пусть и под названием плейлиста. Отказывать тут незачем: берём сам ролик.
    """
    answers = [
        {"_type": "playlist", "title": "Пост", "entries": [{"_type": "url", "url": "https://vk.com/video-1_2"}]},
        {"title": "Ролик", "duration": 30, "extractor_key": "VK"},
    ]
    calls: list[list[str]] = []
    monkeypatch.setattr(fetching, "require_yt_dlp", lambda: "yt-dlp")
    monkeypatch.setattr(fetching, "yt_dlp_status", lambda today=None: {"stale": False})

    def fake_execute(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 0, json.dumps(answers[len(calls) - 1]), "")

    monkeypatch.setattr(fetching, "_execute", fake_execute)
    media = fetching.probe_remote("https://vk.com/wall-1_2")
    assert media.title == "Ролик"
    assert media.url == "https://vk.com/video-1_2"
    assert calls[1][-1] == "https://vk.com/video-1_2"


def test_probe_keeps_chapters_and_a_short_description(yt_dlp) -> None:
    """yt-dlp и так отдаёт описание и главы — раньше они выбрасывались."""
    payload = {
        "title": "Доклад",
        "duration": 3600,
        "upload_date": "20261001",
        "description": "Слайды: https://example.com\n" + "х" * 5000,
        "chapters": [
            {"start_time": 0.0, "end_time": 95.0, "title": "Вступление"},
            {"start_time": 95.0, "end_time": 3600.0, "title": "Архитектура"},
        ],
    }
    yt_dlp(stdout=json.dumps(payload))
    media = fetching.probe_remote("https://www.youtube.com/watch?v=a")
    assert media.chapters == (
        {"start": 0.0, "title": "Вступление"},
        {"start": 95.0, "title": "Архитектура"},
    )
    assert media.description.startswith("Слайды: https://example.com\n")
    assert len(media.description) <= fetching.DESCRIPTION_LIMIT
    assert media.description.endswith("…")
    assert "chapters" not in media.summary(), "опрос остаётся компактным"


def test_probe_without_chapters_or_description(yt_dlp) -> None:
    yt_dlp(stdout=json.dumps({"title": "Ролик", "chapters": None, "description": ""}))
    media = fetching.probe_remote("https://www.youtube.com/watch?v=a")
    assert media.chapters == () and media.description is None


def _local(**fields):
    from audio_transcription.models import MediaInfo

    return MediaInfo(Path("/tmp/transcriber-download-x/audio.webm"), 60.0, "opus", 48000, 1, **fields)


def test_remote_source_carries_date_description_and_chapters() -> None:
    """В `manifest.source` по ссылке: адрес без хвостов, дата выпуска, описание, главы."""
    remote = RemoteMedia(
        url="https://www.youtube.com/watch?v=a&si=share",
        title="Доклад",
        duration_seconds=60.0,
        extractor="Youtube",
        tracks=(),
        upload_date="2026-10-01",
        description="Описание",
        chapters=({"start": 0.0, "title": "Вступление"},),
    )
    media = fetching.with_remote_source(
        _local(recorded_at="2026-10-16T12:00:00+03:00", recorded_at_source="file_mtime"), remote
    )
    source = media.to_dict()
    assert source["origin"] == "https://www.youtube.com/watch?v=a"
    assert (source["recorded_at"], source["recorded_at_source"]) == ("2026-10-01", "upload_date")
    assert source["description"] == "Описание"
    assert source["chapters"] == [{"start": 0.0, "title": "Вступление"}]


def test_remote_source_never_dates_by_the_downloaded_file() -> None:
    """Время изменения скачанного файла — момент загрузки (или заголовок
    сервера), а не записи. Без даты выпуска остаётся только метка контейнера."""
    remote = RemoteMedia("https://example.com/a.mp3", "Запись", 60.0, "generic", ())
    mtime = fetching.with_remote_source(
        _local(recorded_at="2026-10-16T12:00:00+03:00", recorded_at_source="file_mtime"), remote
    )
    assert (mtime.recorded_at, mtime.recorded_at_source) == (None, None)
    assert mtime.chapters is None and mtime.description is None
    container = fetching.with_remote_source(
        _local(recorded_at="2026-09-30T10:00:00Z", recorded_at_source="container"), remote
    )
    assert (container.recorded_at, container.recorded_at_source) == ("2026-09-30T10:00:00Z", "container")
