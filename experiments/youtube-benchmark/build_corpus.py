#!/usr/bin/env python3
"""Корпус из ютьюба: эталон — ручные субтитры, соперник — автосубтитры.

Из каждого видео берётся один пятиминутный фрагмент. Пять минут — это около
тысячи слов эталона на видео: достаточно, чтобы разница в пару процентов WER
не тонула в шуме, и достаточно мало, чтобы прогон всех моделей по восьми
видео занимал минуты, а не часы.

Границы фрагмента подгоняются под реплики субтитров: обрезать эталон посреди
фразы — значит записать модели в ошибку слова, которых она не слышала.

Отдельная проверка: «ручная» дорожка на ютьюбе не обязана быть человеческой,
автор мог залить туда машинный текст. Если ручной и автоматический тексты
совпадают почти дословно, видео выбрасывается — эталон, списанный с
соперника, ничего не измеряет.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills" / "transcriber" / "scripts"))

from audio_transcription.fetching import fetch_audio, fetch_subtitles, probe_remote, read_cues  # noqa: E402
from textnorm import clean_cue, has_speaker_label, similarity  # noqa: E402

EXCERPT_SECONDS = 300.0
SKIP_INTRO_SECONDS = 90.0
# Выше этого сходства «ручная» дорожка — копия машинной, залитая автором.
# Порог близок к единице намеренно: высокое сходство само по себе значит лишь,
# что распознавание сработало хорошо (на студийной записи TED — 97%), и
# выбрасывать такие видео нельзя: это самые трудные и самые честные замеры.
MACHINE_LOOKALIKE = 0.99


def cue_text(cues, start: float, end: float) -> str:
    return " ".join(
        cleaned
        for cue in cues
        if start <= cue.start < end
        for cleaned in [clean_cue(cue.text)]
        if cleaned
    )


def pick_window(cues, duration: float) -> tuple[float, float]:
    """Окно от начала реплики до начала следующей за последней включённой.

    У реплики известно только начало (`Cue.start`), конца в нашем разборе нет.
    Поэтому правая граница звука — начало первой невключённой реплики: так
    внутри фрагмента звучат ровно те слова, что попали в эталон, и ни одного
    оборванного на полуслове.
    """
    starts = [cue.start for cue in cues]
    begin = next((value for value in starts if value >= SKIP_INTRO_SECONDS), starts[0] if starts else 0.0)
    limit = begin + EXCERPT_SECONDS
    after = [value for value in starts if value > limit]
    return begin, min(after[0] if after else duration, duration)


def auto_candidates(media, language: str) -> list:
    """Машинные дорожки нужного языка — сперва те, что названы им же.

    У видео с авторскими субтитрами список «автоматических» почти целиком
    состоит из автопереводов (`en-ar` — «English from Arabic»), и по имени они
    неотличимы от распознавания. Поэтому порядок — эвристика, а решает
    проверка содержимого.
    """
    matching = [
        track
        for track in media.tracks
        if track.automatic and track.language.lower().split("-")[0] == language
    ]
    preferred = (language, f"{language}-orig", f"{language}-{language}")
    return sorted(
        matching,
        key=lambda track: preferred.index(track.language.lower())
        if track.language.lower() in preferred
        else len(preferred),
    )


def is_recognition(path: Path) -> bool:
    """Распознавание ютьюб отдаёт лентой с пословными таймингами `<c>`.

    Автоперевод и залитый автором текст приходят обычными репликами. Это
    единственный надёжный признак: по имени дорожки не отличить.
    """
    return "<c>" in path.read_text(encoding="utf-8", errors="replace")[:200_000]


def cached(directory: Path, pattern: str = "captions.*") -> Path | None:
    """Уже скачанная дорожка. Перекачивать — зря дёргать ютьюб на каждой правке."""
    files = sorted(path for path in directory.glob(pattern) if path.is_file())
    return files[0] if files else None


def cut(source: Path, target: Path, start: float, end: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
        ],
        check=True,
    )


def speaker_labels(cues, start: float, end: float) -> int:
    """Сколько реплик подписаны именем — грубая мера «это диалог»."""
    return sum(1 for cue in cues if start <= cue.start < end and has_speaker_label(cue.text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    parser.add_argument("--candidates", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=here / "corpus")
    args = parser.parse_args()

    sources = args.candidates or [here / "candidates.json", here / "candidates-talks.json"]
    entries: list[dict] = []
    seen: set[str] = set()
    for path in sources:
        if not path.exists():
            continue
        for items in json.loads(path.read_text(encoding="utf-8")).values():
            for item in items:
                if item["id"] not in seen:
                    seen.add(item["id"])
                    entries.append(item)

    output = args.output.resolve()
    (output / "audio").mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(parents=True, exist_ok=True)
    manifest = []
    for entry in entries:
        identifier = f"{entry['language']}-{entry['id']}"
        print(f"→ {identifier}: {entry['title'][:60]}", flush=True)
        work = output / "raw" / identifier
        work.mkdir(parents=True, exist_ok=True)
        try:
            manual_path = cached(work / "manual")
            auto_path = cached(work / "auto", "*/captions.*")
            media = None
            if manual_path is None or auto_path is None:
                media = probe_remote(entry["url"])
            if manual_path is None:
                manual_path, _ = fetch_subtitles(
                    media, work / "manual", languages=(entry["manual_track"],), allow_automatic=False
                )
            if auto_path is None:
                for candidate in auto_candidates(media, entry["language"]):
                    # `track_for` при равенстве языков предпочитает авторскую
                    # дорожку, поэтому просить «автоматическую на en» бесполезно:
                    # вернётся та же ручная. Прячем от выбора всё остальное.
                    path, _ = fetch_subtitles(
                        replace(media, tracks=[candidate]),
                        work / "auto" / candidate.language,
                        languages=(candidate.language,),
                        allow_automatic=True,
                    )
                    if is_recognition(path):
                        auto_path = path
                        break
                    print(f"   {candidate.language}: не распознавание, а перевод", flush=True)
                if auto_path is None:
                    print("   пропуск: настоящей машинной дорожки нет", flush=True)
                    continue
            # Скачиваем сжатый звук как есть и режем фрагмент прямо из него:
            # перегонять в WAV всё видео ради пяти минут — лишняя работа.
            # Фрагмент уже вырезан — весь звук заново не нужен (после замера
            # скачанные дорожки удаляются, а фрагменты остаются).
            excerpt = output / "audio" / f"{identifier}.wav"
            downloads = list((work / "download").glob("audio.*"))
            if excerpt.exists():
                source_audio = None
            else:
                source_audio = downloads[0] if downloads else fetch_audio(entry["url"], work / "download")
        except Exception as error:  # noqa: BLE001 — видео могло стать недоступным
            print(f"   пропуск: {type(error).__name__}: {error}", flush=True)
            continue

        manual_cues = read_cues(manual_path)
        auto_cues = read_cues(auto_path)
        start, end = pick_window(manual_cues, entry["duration_seconds"])
        reference = cue_text(manual_cues, start, end)
        automatic = cue_text(auto_cues, start, end)
        if not reference.split() or not automatic.split():
            print("   пропуск: пустой фрагмент", flush=True)
            continue
        likeness = similarity(reference, automatic)
        if likeness >= MACHINE_LOOKALIKE:
            print(f"   пропуск: «ручная» дорожка совпадает с машинной на {likeness:.0%}", flush=True)
            continue

        if not excerpt.exists():
            cut(source_audio, excerpt, start, end)
        (output / f"{identifier}.reference.txt").write_text(reference + "\n", encoding="utf-8")
        (output / f"{identifier}.youtube-auto.txt").write_text(automatic + "\n", encoding="utf-8")
        manifest.append(
            {
                "id": identifier,
                "url": entry["url"],
                "title": entry["title"],
                "language": entry["language"],
                "audio": str(excerpt),
                "window": [round(start, 2), round(end, 2)],
                "reference_words": len(reference.split()),
                "auto_words": len(automatic.split()),
                "likeness_to_auto": round(likeness, 3),
                "speaker_labels": speaker_labels(manual_cues, start, end),
            }
        )
        print(
            f"   ✓ {end - start:.0f} с, эталон {manifest[-1]['reference_words']} слов, "
            f"сходство с автосубтитрами {likeness:.0%}",
            flush=True,
        )

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"videos": len(manifest), "manifest": str(output / "manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
