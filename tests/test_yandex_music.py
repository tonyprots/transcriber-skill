"""Выпуск подкаста Яндекс Музыки: опрос и скачивание без yt-dlp.

Сеть подменена: проверяем разбор ссылок, отказы и сборку подписанной ссылки
на файл. Живой прогон — `fetch_media.py <ссылка на выпуск>`.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from audio_transcription import fetching, yandex_music
from audio_transcription.fetching import FetchError

TRACK = {
    "id": "140105021",
    "title": "Разговор о CRM",
    "durationMs": 347350,
    "pubDate": "2025-06-11",
    "availableFullWithoutPermission": True,
    "albums": [{"title": "Грачёв на максималках", "releaseDate": None}],
}
INFO_XML = (
    "<download-info><host>s1.storage</host><path>/abc/file.mp3</path>"
    "<ts>000ts</ts><region>-1</region><s>salt2</s></download-info>"
)


@pytest.fixture
def api(monkeypatch):
    opened: list[str] = []
    state = {"track": dict(TRACK)}

    def fake_open(url: str):
        opened.append(url)
        if url.endswith("/download-info"):
            body = json.dumps({"result": [
                {"codec": "mp3", "preview": False, "bitrateInKbps": 192, "downloadInfoUrl": "https://info/1"},
                {"codec": "mp3", "preview": False, "bitrateInKbps": 64, "downloadInfoUrl": "https://info/2"},
                {"codec": "aac", "preview": False, "bitrateInKbps": 32, "downloadInfoUrl": "https://info/3"},
            ]})
        elif url.startswith("https://api.music.yandex.net/tracks/"):
            body = json.dumps({"result": [state["track"]]})
        elif url.startswith("https://info/"):
            body = INFO_XML
        else:
            body = "ID3-audio"
        return io.BytesIO(body.encode())

    monkeypatch.setattr(yandex_music, "_open", fake_open)
    return opened, state


@pytest.mark.parametrize(
    "url",
    [
        "https://music.yandex.ru/album/36892662/track/140105021?utm_source=web",
        "https://music.yandex.kz/track/140105021",
    ],
)
def test_track_links_are_recognised(url: str) -> None:
    assert yandex_music.track_id(url) == "140105021"


def test_other_links_are_left_to_yt_dlp() -> None:
    assert yandex_music.track_id("https://www.youtube.com/watch?v=abc") is None


def test_whole_podcast_link_is_refused_with_a_hint() -> None:
    with pytest.raises(FetchError, match="/track/"):
        yandex_music.track_id("https://music.yandex.ru/album/36892662")


def test_probe_goes_through_api_not_yt_dlp(api, monkeypatch) -> None:
    """yt-dlp 2026.08.19 на Яндекс Музыке падает — до него дело не доходит."""
    monkeypatch.setattr(fetching, "require_yt_dlp", lambda: pytest.fail("yt-dlp не нужен"))
    media = fetching.probe_remote("https://music.yandex.ru/album/1/track/140105021")
    assert media.title == TRACK["title"]
    assert media.duration_seconds == pytest.approx(347.35)
    assert media.upload_date == "2025-06-11"
    assert media.channel == "Грачёв на максималках"
    assert media.tracks == ()


def test_download_takes_lightest_mp3_and_signs_the_link(api, tmp_path: Path) -> None:
    opened, _ = api
    path = fetching.fetch_audio("https://music.yandex.ru/album/1/track/140105021", tmp_path)
    assert path.read_bytes() == b"ID3-audio"
    assert "https://info/2" in opened  # 64 кбит/с, а не 192 и не aac
    sign = hashlib.md5((yandex_music._SIGN_SALT + "abc/file.mp3" + "salt2").encode()).hexdigest()
    assert opened[-1] == f"https://s1.storage/get-mp3/{sign}/000ts/abc/file.mp3?track-id=140105021"


def test_preview_only_track_is_refused(api, tmp_path: Path) -> None:
    """Музыка без входа — тридцать секунд превью; расшифровывать нечего."""
    _, state = api
    state["track"]["availableFullWithoutPermission"] = False
    with pytest.raises(FetchError, match="превью"):
        fetching.fetch_audio("https://music.yandex.ru/album/1/track/140105021", tmp_path)
    assert not any(tmp_path.iterdir())
