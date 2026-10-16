"""`fetch_media.py` пакетом: пауза между ссылками и отказ, который не рвёт пакет."""

from __future__ import annotations

import json

import fetch_media
from audio_transcription.fetching import FetchError, RemoteMedia, SubtitleTrack


def _media(url: str) -> RemoteMedia:
    return RemoteMedia(
        url=url,
        title=f"Ролик {url[-1]}",
        duration_seconds=60.0,
        extractor="Youtube",
        tracks=(SubtitleTrack("ru", True),),
        language="ru",
    )


def test_batch_probe_pauses_and_keeps_going(monkeypatch, capsys) -> None:
    pauses: list[float] = []
    monkeypatch.setattr(fetch_media, "_sleep", pauses.append)

    def probe(url: str) -> RemoteMedia:
        if url.endswith("2"):
            raise FetchError("HTTP Error 429")
        return _media(url)

    monkeypatch.setattr(fetch_media, "probe_remote", probe)
    code = fetch_media.main(["https://youtu.be/v1", "https://youtu.be/v2", "https://youtu.be/v3"])
    results = json.loads(capsys.readouterr().out)["results"]
    assert code == 2
    assert pauses == [fetch_media.BATCH_PAUSE_SECONDS] * 2
    assert [item["input"] for item in results] == [
        "https://youtu.be/v1",
        "https://youtu.be/v2",
        "https://youtu.be/v3",
    ]
    assert "429" in results[1]["error"]
    assert results[2]["picked"]["language"] == "ru"


def test_single_url_keeps_the_plain_payload(monkeypatch, capsys) -> None:
    monkeypatch.setattr(fetch_media, "probe_remote", _media)
    assert fetch_media.main(["https://youtu.be/v1"]) == 0
    assert "results" not in json.loads(capsys.readouterr().out)


def test_repeated_subtitles_into_the_same_folder_keep_the_text(monkeypatch, tmp_path, capsys) -> None:
    """Повтор `--subtitles --language ru` в тот же каталог (по умолчанию ./media).

    До правки второй прогон брал свой же `captions.md` как сырые субтитры:
    0 реплик, и файл перезаписывался одной шапкой.
    """
    from pathlib import Path
    from subprocess import CompletedProcess

    from audio_transcription import fetching

    monkeypatch.setattr(fetch_media, "probe_remote", _media)
    monkeypatch.setattr(fetching, "require_yt_dlp", lambda: "yt-dlp")
    monkeypatch.setattr(fetching, "yt_dlp_status", lambda today=None: {"stale": False})

    def fake_execute(command, **kwargs):
        template = command[command.index("-o") + 1]
        Path(template.replace("%(ext)s", "ru.vtt")).write_text(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nпривет\n", encoding="utf-8"
        )
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(fetching, "_execute", fake_execute)
    argv = ["https://youtu.be/v1", "--subtitles", "--language", "ru", "--output", str(tmp_path)]
    for _ in range(2):
        assert fetch_media.main(argv) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["raw"].endswith("captions.ru.vtt")
        assert payload["cues"] == 1
    assert "привет" in (tmp_path / "captions.md").read_text(encoding="utf-8")


def test_batch_videos_with_the_same_title_get_their_own_folders(monkeypatch, tmp_path, capsys) -> None:
    """В пакете каталог — по названию; у серии выпусков оно бывает общим.

    До правки второй ролик ложился в каталог первого, и его звук получал
    тот, кто читал первый результат.
    """
    from dataclasses import replace
    from pathlib import Path

    monkeypatch.setattr(fetch_media, "_sleep", lambda _: None)
    monkeypatch.setattr(fetch_media, "probe_remote", lambda url: replace(_media(url), title="Выпуск"))

    def fetch_audio(url: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / "audio.webm"
        target.write_text(url, encoding="utf-8")
        return target

    monkeypatch.setattr(fetch_media, "fetch_audio", fetch_audio)
    urls = ["https://youtu.be/v1", "https://youtu.be/v2", "https://youtu.be/v3"]
    assert fetch_media.main([*urls, "--audio", "--output", str(tmp_path)]) == 0
    results = json.loads(capsys.readouterr().out)["results"]
    folders = [Path(item["audio"]).parent.name for item in results]
    assert folders == ["выпуск", "выпуск-2", "выпуск-3"]
    assert [Path(item["audio"]).read_text(encoding="utf-8") for item in results] == urls
