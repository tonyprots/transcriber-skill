"""Субтитры с копии на YouTube для ролика, у которого своих нет.

Рутуб и ВК Видео часто отдают ролик без субтитров, а тот же ролик лежит на
YouTube с автосубтитрами. Брать их можно только с доказательством, что это
та же запись: название совпадает у серий митапов с разными выпусками, а
длительность у копий одной трансляции расходится на 5–87 секунд — их
по-разному обрезают (замер 2026-09-23 на трёх вебинарах с Рутуба).

Поэтому доказательство — звук. Две точки оригинала (30% и 70%) расшифровываются
своим конвейером, по каждой в субтитрах копии ищется окно с тем же текстом.
Копия принимается, если обе точки узнаются и дают один и тот же сдвиг
времени; тогда таймкоды субтитров переводятся во время оригинала.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from pathlib import Path

from .fetching import (
    Cue,
    FetchError,
    RemoteMedia,
    SubtitleTrack,
    _run_yt_dlp,
    fetch_subtitles,
    probe_remote,
    read_cues,
)

ANCHORS = (0.3, 0.7)
ANCHOR_SECONDS = 45.0
# Порог сходства текста в точке и допуск на расхождение сдвигов между точками.
# Откалиброваны 2026-09-23: см. `references/remote-sources.md`.
MIN_SIMILARITY = 0.5
MAX_OFFSET_SPREAD_SECONDS = 4.0
_WORD = re.compile(r"[0-9a-zа-яё]+")


@dataclass(frozen=True)
class Candidate:
    url: str
    title: str
    duration_seconds: float | None


@dataclass(frozen=True)
class AnchorCheck:
    original_start: float
    mirror_start: float
    similarity: float

    @property
    def offset(self) -> float:
        return self.mirror_start - self.original_start

    def to_dict(self) -> dict[str, float]:
        return {
            "original_start": round(self.original_start, 1),
            "mirror_start": round(self.mirror_start, 1),
            "similarity": round(self.similarity, 3),
        }


@dataclass(frozen=True)
class MirrorVerdict:
    candidate: Candidate
    checks: tuple[AnchorCheck, ...]
    accepted: bool
    reason: str

    @property
    def offset(self) -> float:
        return sum(check.offset for check in self.checks) / len(self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "url": self.candidate.url,
            "title": self.candidate.title,
            "duration_seconds": self.candidate.duration_seconds,
            "accepted": self.accepted,
            "reason": self.reason,
            "offset_seconds": round(self.offset, 1) if self.checks else None,
            "checks": [check.to_dict() for check in self.checks],
        }


def search_youtube(media: RemoteMedia, limit: int = 5) -> list[Candidate]:
    """Кандидаты по названию: длительность — грубый фильтр, решает звук.

    Допуск широкий (10% или две минуты), потому что копии трансляций режут
    по-разному; явно чужие ролики он отсекает без единого скачанного байта.
    """
    try:
        payload = _run_yt_dlp(
            [
                "--flat-playlist",
                "--print",
                "%(id)s\t%(duration)s\t%(title)s",
                f"ytsearch{limit}:{media.title}",
            ],
            failure="поиск на YouTube не ответил",
        )
    except Exception:  # поиск — подсказка, а не условие работы
        return []
    found: list[Candidate] = []
    for line in payload.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        identifier, duration_text, title = parts
        try:
            duration = float(duration_text)
        except ValueError:
            duration = None
        if media.duration_seconds and duration:
            tolerance = max(120.0, media.duration_seconds * 0.1)
            if abs(duration - media.duration_seconds) > tolerance:
                continue
        found.append(
            Candidate(f"https://www.youtube.com/watch?v={identifier}", title, duration)
        )
    return found


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("ё", "е"))


def best_window(
    heard: Sequence[str], cues: Sequence[Cue], expected_start: float, search_seconds: float
) -> tuple[float, float]:
    """Где в субтитрах копии звучит услышанное: (начало окна, сходство)."""
    best = (expected_start, 0.0)
    for cue in cues:
        if abs(cue.start - expected_start) > search_seconds:
            continue
        window = [
            word
            for item in cues
            if cue.start <= item.start < cue.start + ANCHOR_SECONDS
            for word in words(item.text)
        ]
        if not window:
            continue
        similarity = SequenceMatcher(None, list(heard), window, autojunk=False).ratio()
        if similarity > best[1]:
            best = (cue.start, similarity)
    return best


def verify(
    media: RemoteMedia,
    candidate: Candidate,
    cues: Sequence[Cue],
    transcribe: Callable[[float, float], str],
) -> MirrorVerdict:
    """Сверяет копию со звуком оригинала в двух точках."""
    if not media.duration_seconds:
        return MirrorVerdict(candidate, (), False, "длительность оригинала неизвестна")
    delta = abs((candidate.duration_seconds or media.duration_seconds) - media.duration_seconds)
    search = delta + 60.0
    checks = []
    for share in ANCHORS:
        start = float(round(media.duration_seconds * share))
        heard = words(transcribe(start, start + ANCHOR_SECONDS))
        if len(heard) < 20:
            return MirrorVerdict(
                candidate, tuple(checks), False, f"в точке {start:.0f} с почти нет речи"
            )
        mirror_start, similarity = best_window(heard, cues, start, search)
        checks.append(AnchorCheck(start, mirror_start, similarity))
    weakest = min(check.similarity for check in checks)
    spread = max(check.offset for check in checks) - min(check.offset for check in checks)
    if weakest < MIN_SIMILARITY:
        return MirrorVerdict(
            candidate, tuple(checks), False, f"текст не совпал (сходство {weakest:.2f})"
        )
    if spread > MAX_OFFSET_SPREAD_SECONDS:
        return MirrorVerdict(
            candidate,
            tuple(checks),
            False,
            f"сдвиг времени плавает на {spread:.0f} с — копия перемонтирована",
        )
    return MirrorVerdict(candidate, tuple(checks), True, "та же запись")


def to_original_time(cues: Sequence[Cue], offset: float) -> list[Cue]:
    """Время копии → время оригинала; то, чего в оригинале нет, отбрасывается."""
    return [Cue(cue.start - offset, cue.text) for cue in cues if cue.start - offset >= 0]


@dataclass(frozen=True)
class MirrorFound:
    verdict: MirrorVerdict
    track: SubtitleTrack
    raw: Path
    cues: list[Cue]


def find_mirror(
    media: RemoteMedia,
    language: str,
    dest_dir: Path,
    transcribe: Callable[[float, float], str],
    *,
    search: Callable[[RemoteMedia], list[Candidate]] = search_youtube,
    limit: int = 3,
) -> tuple[MirrorFound | None, list[dict[str, object]]]:
    """Первая копия, прошедшая проверку, и причины отказа остальным.

    Дорожка копии должна быть на языке речи: автоперевод YouTube с чужого
    языка — не субтитры, а перевод распознавания, и сверка по звуку его
    всё равно не пропустит.
    """
    rejected: list[dict[str, object]] = []
    for candidate in search(media)[:limit]:
        try:
            copy = probe_remote(candidate.url)
            track = copy.track_for((language,))
            if track is None or copy.is_translation(track) or not _same(track, language):
                rejected.append({"url": candidate.url, "reason": f"нет субтитров `{language}`"})
                continue
            raw, track = fetch_subtitles(copy, dest_dir / "mirror" / candidate.url.rsplit("=", 1)[-1], languages=(language,))
        except FetchError as error:
            rejected.append({"url": candidate.url, "reason": str(error)})
            continue
        cues = read_cues(raw)
        verdict = verify(media, candidate, cues, transcribe)
        if verdict.accepted:
            return MirrorFound(verdict, track, raw, to_original_time(cues, verdict.offset)), rejected
        rejected.append(verdict.to_dict())
    return None, rejected


def _same(track: SubtitleTrack, language: str) -> bool:
    return track.language.lower().split("-")[0] == language.lower().split("-")[0]


__all__ = [
    "ANCHORS",
    "Candidate",
    "MirrorFound",
    "find_mirror",
    "MirrorVerdict",
    "search_youtube",
    "to_original_time",
    "verify",
    "words",
]
