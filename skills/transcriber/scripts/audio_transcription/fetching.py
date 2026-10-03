"""Источник-ссылка: что у видео есть и как это забрать.

Модуль знает только про yt-dlp и формат субтитров. Расшифровкой он не
занимается: либо отдаёт чужие субтитры как есть, либо кладёт рядом аудиофайл,
который дальше идёт обычным конвейером.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .audio import find_command


class FetchError(RuntimeError):
    """Не удалось получить медиа по ссылке."""


_URL_PREFIX = re.compile(r"^https?://", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_CUE_TIME = re.compile(r"^(\d\d):(\d\d):(\d\d)[.,](\d{3})\s+-->")
_SERVICE_PREFIXES = ("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE", "REGION")

# Автосубтитры YouTube идут «лентой»: каждая реплика повторяется в следующих
# репликах как контекст. Окно подобрано по их длине — двух мало, пяти много.
_DEDUP_WINDOW = 4
_AUTO_NAME = re.compile(r"\bавто|\bauto", re.IGNORECASE)

# Звук для распознавания сводится в 16 кГц моно, поэтому лучшая дорожка
# источнику не нужна. У ВК Видео `bestaudio` — это 128 МБ на час, которые
# yt-dlp тянул по фрагменту: 100 минут на 68-минутный ролик (2026-09-22).
# Лёгкая дорожка в восемь потоков скачалась за 30 секунд.
# Нижняя граница 64 кбит/с держит YouTube на прежней дорожке opus 103k:
# без неё выбиралась HE-AAC 49k, а её влияние на качество не мерили.
# Форматы с неизвестным битрейтом фильтр отсекает — они уходят в `bestaudio`.
_AUDIO_FORMAT = "bestaudio[abr>=64][abr<=96]/bestaudio/best[height<=480]/best"
_CONCURRENT_FRAGMENTS = "8"

# YouTube отвечает 429 на серию запросов субтитров подряд; через минуту
# тот же запрос проходит. Паузы короткие: дольше ждать дешевле руками.
_RETRY_PAUSES_SECONDS = (15.0, 45.0)
# Сокетный таймаут yt-dlp ловит молчащее соединение, но не зависший экстрактор
# (JS-задачу YouTube он решает через deno, и та может не вернуться). Без
# общего срока агент ждал бы такой вызов вечно. Метаданные приходят за секунды,
# плейлист канала — за минуту-другую.
METADATA_TIMEOUT_SECONDS = 300.0
# Скачивание общим сроком не ограничить: час с ВК идёт 30 секунд, с медленного
# сайта — десятки минут. Убиваем, когда в каталоге ничего не прибывает.
STALL_TIMEOUT_SECONDS = 300.0
_WATCH_PERIOD_SECONDS = 2.0
_sleep = time.sleep


def looks_like_url(value: str) -> bool:
    """Ссылку от пути отличаем только по схеме: файл с именем `https` — редкость."""
    return bool(_URL_PREFIX.match(value.strip()))


# Параметры, которые остаются в адресе для этих площадок: без них ссылка не
# ведёт на ролик. Всё прочее отрезается — см. `public_url`.
_IDENTITY_PARAMS = {
    "youtube.com": {"v", "list"},
    "youtu.be": set(),
    "vk.com": {"z"},
    "vkvideo.ru": {"z"},
    "rutube.ru": set(),
    "music.yandex.ru": set(),
    "music.yandex.com": set(),
}


def _identity_params(host: str) -> set[str]:
    host = host.lower().removeprefix("www.").removeprefix("m.")
    for domain, params in _IDENTITY_PARAMS.items():
        if host == domain or host.endswith("." + domain):
            return params
    return set()


def public_url(url: str) -> str:
    """Адрес, который можно записать в артефакт: без логина, токенов и подписей.

    Пользователь может дать presigned-ссылку на S3, адрес с `?token=` или
    временную корпоративную ссылку, и раньше она целиком попадала в manifest.
    Чёрный список имён тут не работает: у Azure подпись лежит в `sig`, а срок
    в `se`, у каждого облака свои имена. Поэтому белый список: у известных
    площадок остаётся то, без чего ссылка не ведёт на ролик, у остальных query
    отрезается целиком. Фрагмент и `user:pass@` не нужны никому.
    """
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        return url
    host = parts.hostname or ""
    netloc = host if parts.port is None else f"{host}:{parts.port}"
    keep = _identity_params(host)
    query = urlencode([(key, value) for key, value in parse_qsl(parts.query) if key in keep])
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def require_yt_dlp() -> str:
    found = find_command("yt-dlp")
    if not found:
        raise FetchError(
            "Ссылку без yt-dlp не открыть. Поставьте его (brew install yt-dlp) "
            "или передайте локальный файл."
        )
    return found


@dataclass(frozen=True)
class SubtitleTrack:
    """Дорожка субтитров: где она лежит у yt-dlp и кто её на самом деле сделал.

    `automatic` — раздел yt-dlp, он решает флаг скачивания. Происхождение —
    отдельный вопрос: Рутуб кладёт своё распознавание в обычные `subtitles`
    и выдаёт его только именем «Русский • Авто», а ВК происхождения не
    сообщает вовсе. Поэтому `origin` задаётся явно, когда раздел врёт.
    """

    language: str
    automatic: bool
    origin: str | None = None

    @property
    def kind(self) -> str:
        """`manual`, `auto` или `unknown` — то, что видит пользователь."""
        return self.origin or ("auto" if self.automatic else "manual")

    def to_dict(self) -> dict[str, object]:
        return {"language": self.language, "automatic": self.automatic, "origin": self.kind}


@dataclass(frozen=True)
class RemoteMedia:
    url: str
    title: str
    duration_seconds: float | None
    extractor: str | None
    tracks: tuple[SubtitleTrack, ...]
    language: str | None = None
    # Дата выпуска и канал решают, смотреть ли ролик вообще: 2026-09-22
    # за ними четыре раза ходили в yt-dlp отдельно, а три ролика до 2020 года
    # успели встать в очередь на расшифровку.
    upload_date: str | None = None
    channel: str | None = None

    def is_translation(self, track: SubtitleTrack) -> bool:
        """YouTube отдаёт автоперевод на любой язык, и по имени он неотличим.

        Дорожка `ru` у английского ролика — это машинный перевод машинного
        распознавания. Выдать такое за расшифровку нельзя, поэтому
        несовпадение с языком оригинала считаем переводом.
        """
        if track.kind != "auto" or not self.language:
            return False
        return not _same_language(track.language, self.language)

    def track_for(
        self,
        languages: Sequence[str] = (),
        allow_automatic: bool = True,
    ) -> SubtitleTrack | None:
        """Запрошенный язык важнее происхождения дорожки, при равенстве — авторская.

        Автоматические субтитры на нужном языке полезнее авторских на чужом:
        переводить чужие всё равно придётся, а ошибки распознавания видны и
        правятся по звуку. Пустой `languages` означает «любой» — так ведёт
        себя `--probe`, когда язык не назвали.
        """
        wanted = tuple(languages) or ((self.language,) if self.language else ())
        for kind in ("manual", "unknown", "auto"):
            automatic = kind == "auto"
            if automatic and not allow_automatic:
                return None
            candidates = [track for track in self.tracks if track.kind == kind]
            if not wanted:
                # Язык не назвали и оригинал неизвестен. У авторских дорожек
                # выбор очевиден, у автоматических — нет: их под две сотни и
                # отсортированы они по алфавиту, так что «первая» окажется
                # афарской. Пусть лучше язык назовут явно.
                if not automatic and candidates:
                    return candidates[0]
                continue
            for language in wanted:
                for track in candidates:
                    if _same_language(track.language, language):
                        return track
        return None

    def summary(self, languages: Sequence[str] = ()) -> dict[str, object]:
        """Сводка, а не перечень: автодорожек у YouTube больше сотни.

        Печатать их списком — залить контекст машинным переводом на языки,
        которых никто не просил. Поэтому наружу идут авторские дорожки
        целиком, автоматические — счётчиком и совпадениями с запросом.
        """
        manual = [track.language for track in self.tracks if track.kind == "manual"]
        unknown = [track.language for track in self.tracks if track.kind == "unknown"]
        automatic = [track.language for track in self.tracks if track.kind == "auto"]
        return {
            "url": public_url(self.url),
            "title": self.title,
            "duration_seconds": self.duration_seconds,
            "extractor": self.extractor,
            "language": self.language,
            "upload_date": self.upload_date,
            "channel": self.channel,
            "subtitles": {
                "manual": manual,
                **({"origin_unknown": unknown} if unknown else {}),
                "automatic_count": len(automatic),
                "automatic_matched": [
                    wanted
                    for wanted in languages
                    if any(_same_language(found, wanted) for found in automatic)
                ],
            },
        }


@dataclass(frozen=True)
class Cue:
    start: float
    text: str

    @property
    def timestamp(self) -> str:
        total = int(self.start)
        return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _same_language(track_language: str, wanted: str) -> bool:
    """`en` покрывает `en-US` и `en-orig`: региональные варианты — один язык."""
    return track_language.lower().split("-")[0] == wanted.lower().split("-")[0]


def probe_remote(url: str) -> RemoteMedia:
    """Что за видео и какие у него субтитры — без единого скачанного байта."""
    from . import yandex_music

    if yandex_music.track_id(url):
        return yandex_music.probe(url)
    payload = _run_yt_dlp(
        ["--skip-download", "--dump-single-json", url],
        failure=f"yt-dlp не открыл ссылку: {public_url(url)}",
    )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise FetchError(f"yt-dlp вернул не JSON: {error}") from error
    extractor = data.get("extractor_key") or data.get("extractor")
    tracks = tuple(
        SubtitleTrack(
            language=language,
            automatic=automatic,
            origin=None if automatic else _declared_origin(section[language], extractor),
        )
        for automatic, section in (
            (False, data.get("subtitles") or {}),
            (True, data.get("automatic_captions") or {}),
        )
        for language in sorted(section)
    )
    duration = data.get("duration")
    return RemoteMedia(
        url=url,
        title=str(data.get("title") or "без названия"),
        duration_seconds=float(duration) if duration is not None else None,
        extractor=extractor,
        tracks=tracks,
        language=data.get("language"),
        upload_date=_iso_date(data.get("upload_date")),
        channel=data.get("channel") or data.get("uploader"),
    )


def _declared_origin(formats: object, extractor: object) -> str | None:
    """Происхождение дорожки из раздела `subtitles`, если раздел его скрывает.

    Рутуб: «Русский • Авто» — распознавание площадки (проверено 2026-09-23).
    ВК: yt-dlp не передаёт ни имени, ни признака — честно «неизвестно».
    """
    names = " ".join(
        str(item.get("name") or "") for item in formats if isinstance(item, dict)
    ) if isinstance(formats, list) else ""
    if _AUTO_NAME.search(names):
        return "auto"
    if str(extractor or "").upper().startswith("VK"):
        return "unknown"
    return None


def _iso_date(value: object) -> str | None:
    """yt-dlp отдаёт `20241010`; в выводе нужна дата, которую читают глазами."""
    text = str(value or "")
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if re.fullmatch(r"\d{8}", text) else None


def fetch_subtitles(
    media: RemoteMedia,
    dest_dir: Path,
    languages: Sequence[str] = (),
    allow_automatic: bool = True,
) -> tuple[Path, SubtitleTrack]:
    """Кладёт дорожку субтитров как есть и возвращает путь и её происхождение."""
    track = media.track_for(languages, allow_automatic=allow_automatic)
    if track is None:
        available = ", ".join(sorted({item.language for item in media.tracks})) or "нет ни одной"
        raise FetchError(
            f"Подходящих субтитров нет (есть: {available}). "
            "Снимите ограничение по языку или расшифруйте звук."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    _run_yt_dlp(
        [
            "--skip-download",
            "--write-auto-subs" if track.automatic else "--write-subs",
            "--sub-langs",
            track.language,
            "--sub-format",
            "vtt/best",
            "-o",
            str(dest_dir / "captions.%(ext)s"),
            media.url,
        ],
        failure=f"yt-dlp не отдал субтитры {track.language}",
        watch=dest_dir,
    )
    files = sorted(path for path in dest_dir.glob("captions.*") if path.is_file())
    if not files:
        raise FetchError(
            f"yt-dlp отчитался об успехе, но файла субтитров в {dest_dir} нет"
        )
    return files[0], track


def expand_playlist(url: str, *, limit: int | None = None) -> list[str]:
    """Ссылки на ролики плейлиста или канала по порядку; у ролика — он сам.

    Без этого плейлист перечисляли руками: `--no-playlist` стоит во всех
    вызовах, и ссылка на плейлист давала один первый ролик.
    """
    from . import yandex_music

    if yandex_music.track_id(url):
        return [url]
    arguments = ["--flat-playlist", "--dump-single-json"]
    if limit:
        arguments += ["--playlist-end", str(limit)]
    payload = _run_yt_dlp(
        [*arguments, url], failure=f"yt-dlp не открыл плейлист: {public_url(url)}", playlist=True
    )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise FetchError(f"yt-dlp вернул не JSON: {error}") from error
    if data.get("_type") != "playlist":
        return [url]
    links = [
        str(link)
        for entry in data.get("entries") or []
        if isinstance(entry, dict)
        and (link := entry.get("webpage_url") or entry.get("url"))
        and looks_like_url(str(link))
    ]
    if not links:
        raise FetchError(f"В плейлисте нет доступных роликов: {public_url(url)}")
    return links[:limit] if limit else links


def fetch_audio(url: str, dest_dir: Path) -> Path:
    """Скачивает лёгкую аудиодорожку без перекодирования — WAV сделает конвейер."""
    from . import yandex_music

    if yandex_music.track_id(url):
        return yandex_music.download(url, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    _run_yt_dlp(
        [
            "-f",
            _AUDIO_FORMAT,
            "--concurrent-fragments",
            _CONCURRENT_FRAGMENTS,
            "-o",
            str(dest_dir / "audio.%(ext)s"),
            url,
        ],
        failure=f"yt-dlp не скачал аудио: {public_url(url)}",
        watch=dest_dir,
    )
    files = sorted(path for path in dest_dir.glob("audio.*") if path.is_file())
    if not files:
        raise FetchError(f"yt-dlp отчитался об успехе, но файла в {dest_dir} нет")
    return files[0]


def read_cues(path: Path) -> list[Cue]:
    """Разворачивает VTT или SRT в реплики, снимая «ленту» автосубтитров."""
    cues: list[Cue] = []
    start: float | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        moment = _CUE_TIME.match(stripped)
        if moment:
            hours, minutes, seconds, millis = (int(part) for part in moment.groups())
            start = hours * 3600 + minutes * 60 + seconds + millis / 1000
            continue
        text = _clean(stripped)
        if not text or start is None:
            continue
        if any(text == previous.text for previous in cues[-_DEDUP_WINDOW:]):
            continue
        cues.append(Cue(start=start, text=text))
    return cues


def render_captions(
    media: RemoteMedia,
    track: SubtitleTrack,
    cues: Sequence[Cue],
    mirror: dict[str, object] | None = None,
) -> str:
    """Субтитры источника с шапкой, которая не даёт спутать их с расшифровкой.

    `mirror` — сведения о копии на YouTube, если субтитры взяты с неё:
    читатель должен видеть, откуда текст и чем доказано, что запись та же.
    """
    origin = {
        "manual": "авторские",
        "auto": "автоматические",
        "unknown": "происхождение не сообщается площадкой — считайте автоматическими",
    }[track.kind]
    duration = (
        f"{media.duration_seconds / 60:.0f} мин"
        if media.duration_seconds
        else "неизвестна"
    )
    header = [
        f"# {media.title}",
        "",
        f"- **Источник:** {public_url(media.url)}",
        f"- **Дорожка:** {origin} субтитры, язык `{track.language}`",
        f"- **Длительность:** {duration}",
        "",
        "> Это субтитры источника, а не расшифровка скилла: одна чужая гипотеза "
        "без сверки, без очереди спорных мест и без словаря терминов. Имена, "
        "числа и термины в них не выверены. Нужна точность — расшифруйте звук.",
        "",
    ]
    if mirror is not None:
        checks = ", ".join(
            f"{check['original_start'] / 60:.0f}-я мин — сходство {check['similarity']:.2f}"
            for check in mirror["checks"]  # type: ignore[union-attr]
        )
        header += [
            f"> **Субтитры взяты с копии на YouTube:** {mirror['url']} "
            f"(«{mirror['title']}»). У источника своих не было. Что запись та же, "
            f"проверено по звуку: две точки оригинала расшифрованы и найдены в "
            f"субтитрах копии ({checks}). Таймкоды переведены во время источника "
            f"(сдвиг {mirror['offset_seconds']} с), точность — несколько секунд.",
            "",
        ]
    if media.is_translation(track):
        header += [
            f"> **Это машинный перевод с `{media.language}`, а не речь говорящих.** "
            "YouTube переводит собственное распознавание, ошибки складываются. "
            "Цитировать отсюда нельзя: берите оригинальную дорожку "
            f"`{media.language}` или расшифровку звука.",
            "",
        ]
    header += ["---", ""]
    body = [f"[{cue.timestamp}] {cue.text}" for cue in cues]
    return "\n".join(header + body) + "\n"


def _clean(line: str) -> str:
    if line.startswith(_SERVICE_PREFIXES) or line.isdigit():
        return ""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", line))).strip()


def _run_yt_dlp(
    arguments: Sequence[str],
    failure: str,
    *,
    playlist: bool = False,
    watch: Path | None = None,
    timeout: float = METADATA_TIMEOUT_SECONDS,
) -> str:
    """Вызов yt-dlp с повтором на 429 и сроком.

    `watch` — каталог скачивания: тогда срок не общий, а «ничего не прибыло за
    STALL_TIMEOUT_SECONDS». Без него — общий `timeout`.
    """
    # Ссылка на ролик внутри плейлиста (`watch?v=…&list=…`) без этого флага
    # тянет весь плейлист. Плейлист целиком разворачивает только `expand_playlist`.
    command = [require_yt_dlp(), *(() if playlist else ("--no-playlist",)), *arguments]
    for pause in (*_RETRY_PAUSES_SECONDS, None):
        completed = _execute(command, timeout=timeout, watch=watch)
        if completed.returncode == 0:
            return completed.stdout
        if pause is None or "HTTP Error 429" not in completed.stderr:
            break
        _sleep(pause)
    raise FetchError(_with_version_hint(_tail(completed.stderr) or failure))


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:  # фрагмент успели переименовать между обходом и stat
            continue
    return total


def _execute(
    command: list[str], *, timeout: float, watch: Path | None
) -> subprocess.CompletedProcess[str]:
    if watch is None:
        try:
            return subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise FetchError(
                f"yt-dlp не ответил за {timeout / 60:g} мин и остановлен: площадка или "
                "её экстрактор зависли. Повторите позже или обновите yt-dlp."
            ) from error
    # Вывод — во временные файлы: при PIPE длинный лог заполнил бы буфер, и
    # процесс встал бы, пока мы ждём файл.
    import tempfile

    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, text=True)
        size, changed = -1, time.monotonic()
        while process.poll() is None:
            time.sleep(_WATCH_PERIOD_SECONDS)
            current = _directory_size(watch)
            if current != size:
                size, changed = current, time.monotonic()
            elif time.monotonic() - changed > STALL_TIMEOUT_SECONDS:
                process.kill()
                process.wait()
                raise FetchError(
                    f"Скачивание стоит {STALL_TIMEOUT_SECONDS / 60:g} мин без новых данных и "
                    "остановлено. Повторите позже; если повторится — обновите yt-dlp."
                )
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(command, process.returncode, out.read(), err.read())


# yt-dlp обновляется раз в две-три недели, а сайты ломают его старые версии
# ещё чаще. Устаревший yt-dlp — самая частая причина отказа ссылки
# (references/remote-sources.md), поэтому о возрасте говорим прямо в ошибке.
YT_DLP_SHELF_LIFE_DAYS = 30
_YT_DLP_VERSION = re.compile(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})")


def yt_dlp_status(today: date | None = None) -> dict[str, object]:
    """Где yt-dlp, какой версии и сколько ей дней. Без сети."""
    path = find_command("yt-dlp")
    if not path:
        return {"path": None, "version": None, "age_days": None, "stale": False}
    try:
        version = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=20, check=False
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        version = ""
    age = None
    if match := _YT_DLP_VERSION.match(version):
        try:
            released = date(*map(int, match.groups()))
        except ValueError:
            released = None
        if released:
            age = ((today or date.today()) - released).days
    return {
        "path": path,
        "version": version or None,
        "age_days": age,
        "stale": age is not None and age > YT_DLP_SHELF_LIFE_DAYS,
    }


def update_hint(status: dict[str, object]) -> str:
    """Команда обновления под способ установки: brew, pip или сам бинарник."""
    path = str(status.get("path") or "")
    if "/homebrew/" in path or "/Cellar/" in path or path.startswith("/usr/local/bin"):
        return "brew upgrade yt-dlp"
    if "/.venv/" in path or "site-packages" in path or "/pipx/" in path:
        return "pip install -U yt-dlp"
    return "yt-dlp -U"


def _with_version_hint(message: str) -> str:
    status = yt_dlp_status()
    if not status["stale"]:
        return message
    return (
        f"{message}\n(yt-dlp {status['version']} — {status['age_days']} дн.; "
        f"сайты часто ломают старые версии — обновите, если есть новее: {update_hint(status)})"
    )


def _tail(stderr: str, limit: int = 400) -> str:
    """Полотно yt-dlp в исключении бесполезно — нужен только конец с причиной."""
    text = stderr.strip()
    return text[-limit:] if text else ""
