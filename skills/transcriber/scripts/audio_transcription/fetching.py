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
from pathlib import Path

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
_sleep = time.sleep


def looks_like_url(value: str) -> bool:
    """Ссылку от пути отличаем только по схеме: файл с именем `https` — редкость."""
    return bool(_URL_PREFIX.match(value.strip()))


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
            "url": self.url,
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
        failure=f"yt-dlp не открыл ссылку: {url}",
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
    )
    files = sorted(path for path in dest_dir.glob("captions.*") if path.is_file())
    if not files:
        raise FetchError(
            f"yt-dlp отчитался об успехе, но файла субтитров в {dest_dir} нет"
        )
    return files[0], track


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
        failure=f"yt-dlp не скачал аудио: {url}",
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
        f"- **Источник:** {media.url}",
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


def _run_yt_dlp(arguments: Sequence[str], failure: str) -> str:
    command = [require_yt_dlp(), "--no-playlist", *arguments]
    for pause in (*_RETRY_PAUSES_SECONDS, None):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            return completed.stdout
        if pause is None or "HTTP Error 429" not in completed.stderr:
            break
        _sleep(pause)
    raise FetchError(_tail(completed.stderr) or failure)


def _tail(stderr: str, limit: int = 400) -> str:
    """Полотно yt-dlp в исключении бесполезно — нужен только конец с причиной."""
    text = stderr.strip()
    return text[-limit:] if text else ""
