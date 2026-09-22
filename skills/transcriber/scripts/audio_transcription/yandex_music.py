"""Выпуск подкаста в Яндекс Музыке: что это и откуда взять звук.

Экстрактор yt-dlp для Яндекс Музыки сломан (2026.08.19 падает на
`'NoneType' object is not subscriptable` ещё до скачивания), а подкасты там
открыты без входа. Поэтому звук берётся тем же механизмом, что и у yt-dlp,
только метаданные — из публичного API, а не со страницы:

1. `api.music.yandex.net/tracks/<id>` — название, длительность, дата выпуска;
2. `/tracks/<id>/download-info` — ссылка на описание файла;
3. описание файла (XML) — хост и путь, подпись считается по общей
   константе из исходников yt-dlp (`extractor/yandexmusic.py`). Это не
   ключ пользователя: ни токена, ни входа в аккаунт здесь нет.

Музыку модуль не качает: без входа сервис отдаёт у неё только превью, и
расшифровывать тридцать секунд песни незачем. Такой трек — отказ с причиной.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

from .fetching import FetchError, RemoteMedia

API = "https://api.music.yandex.net"
# Та же константа, что в yt-dlp: подпись ссылки на файл, общая для всех.
_SIGN_SALT = "XGRlBW9FXlekgbPrRHuSiA"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT_SECONDS = 30
_TRACK_URL = re.compile(
    r"^https?://music\.yandex\.[a-z]+/(?:album/\d+/)?track/(\d+)", re.IGNORECASE
)
_ALBUM_URL = re.compile(r"^https?://music\.yandex\.[a-z]+/album/\d+/?(?:[?#]|$)", re.IGNORECASE)


def track_id(url: str) -> str | None:
    """Номер выпуска из ссылки или None, если ссылка не на Яндекс Музыку."""
    match = _TRACK_URL.match(url.strip())
    if match:
        return match.group(1)
    if _ALBUM_URL.match(url.strip()):
        raise FetchError(
            "Это ссылка на весь подкаст, а не на выпуск. Откройте выпуск и "
            "скопируйте его ссылку — в ней есть /track/<номер>."
        )
    return None


def probe(url: str) -> RemoteMedia:
    identifier = track_id(url)
    if identifier is None:
        raise FetchError(f"Не ссылка на выпуск Яндекс Музыки: {url}")
    track = _track(identifier)
    album = (track.get("albums") or [{}])[0]
    duration_ms = track.get("durationMs")
    return RemoteMedia(
        url=url,
        title=str(track.get("title") or "без названия"),
        duration_seconds=duration_ms / 1000 if duration_ms else None,
        extractor="YandexMusic",
        tracks=(),
        upload_date=track.get("pubDate") or album.get("releaseDate"),
        channel=album.get("title"),
    )


def download(url: str, dest_dir: Path) -> Path:
    identifier = track_id(url)
    if identifier is None:
        raise FetchError(f"Не ссылка на выпуск Яндекс Музыки: {url}")
    track = _track(identifier)
    if not track.get("availableFullWithoutPermission"):
        raise FetchError(
            "Без входа Яндекс Музыка отдаёт этот трек только превью. Скилл "
            "расшифровывает открытые выпуски подкастов; музыку и платный "
            "контент — нет."
        )
    variants = [
        item
        for item in _json(f"{API}/tracks/{identifier}/download-info").get("result") or []
        if item.get("codec") == "mp3" and not item.get("preview")
    ]
    if not variants:
        raise FetchError("Яндекс Музыка не отдала ни одного полного mp3 для выпуска")
    # Для распознавания речи битрейт выше 64 кбит/с ничего не даёт — берём
    # самый лёгкий: быстрее качается, конвейер всё равно сведёт в 16 кГц моно.
    variant = min(variants, key=lambda item: item.get("bitrateInKbps") or 0)
    info = ElementTree.fromstring(_get(variant["downloadInfoUrl"]))
    fields = {name: info.findtext(name) or "" for name in ("host", "path", "ts", "s")}
    if not all(fields.values()):
        raise FetchError("Описание файла Яндекс Музыки пришло без хоста или подписи")
    sign = hashlib.md5(
        (_SIGN_SALT + fields["path"][1:] + fields["s"]).encode()
    ).hexdigest()
    file_url = (
        f"https://{fields['host']}/get-mp3/{sign}/{fields['ts']}{fields['path']}"
        f"?track-id={identifier}"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "audio.mp3"
    try:
        with _open(file_url) as response, target.open("wb") as sink:
            shutil.copyfileobj(response, sink, length=1 << 20)
    except (urllib.error.URLError, OSError) as error:
        target.unlink(missing_ok=True)
        raise FetchError(f"Не удалось скачать выпуск Яндекс Музыки: {error}") from error
    if target.stat().st_size == 0:
        raise FetchError("Яндекс Музыка отдала пустой файл")
    return target


def _track(identifier: str) -> dict[str, Any]:
    result = _json(f"{API}/tracks/{identifier}").get("result") or []
    if not result:
        raise FetchError(f"Яндекс Музыка не знает выпуск {identifier}")
    return result[0]


def _json(url: str) -> dict[str, Any]:
    try:
        return json.loads(_get(url))
    except json.JSONDecodeError as error:
        raise FetchError(f"Яндекс Музыка вернула не JSON: {error}") from error


def _get(url: str) -> bytes:
    try:
        with _open(url) as response:
            return response.read()
    except urllib.error.URLError as error:
        raise FetchError(f"Яндекс Музыка не ответила: {error}") from error


def _open(url: str):
    request = urllib.request.Request(url, headers=_HEADERS)
    return urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS)
