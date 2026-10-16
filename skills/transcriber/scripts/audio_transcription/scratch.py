"""Временные каталоги прогона: чьи они и как их убрать.

`transcriber-asr-*` и `transcriber-download-*` держат подготовленный WAV и
скачанный ролик — копию чужой записи. Обычно их убирает `scratch_dir` при
выходе, но после SIGKILL, вылета или выключения машины `finally` не
срабатывает, и в $TMPDIR копились сотни мегабайт (шесть каталогов, среди них
prepared.wav на 146 МБ). Прогон при старте добирает такие остатки по тем же
префиксам и тому же порогу 24 ч, что уборка Мостика (dictation-app 8ca5147):
две уборки друг другу не мешают.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

PREFIXES = ("transcriber-asr-", "transcriber-download-")
OWNER_MARKER = ".transcriber-owner"
ORPHAN_AGE_HOURS = 24.0


def mark_owner(path: Path) -> None:
    """Пишет в каталог pid прогона: живой владелец защищает его от уборки."""
    try:
        (path / OWNER_MARKER).write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass  # без метки каталог защищает свежесть, как у прежних версий


def remove_detached(path: Path) -> None:
    """Удаляет каталог отдельным процессом вне нашей сессии.

    Антивирус проверяет каждое удаление, и уборка сотен WAV растягивалась до
    десяти минут после готового результата (`.memory/defender-slow-temp-cleanup.md`).
    Отдельный процесс переживает выход интерпретатора и никого не задерживает.
    """
    try:
        subprocess.Popen(
            ["rm", "-rf", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        shutil.rmtree(path, ignore_errors=True)


def _owner_alive(path: Path) -> bool:
    try:
        pid = int((path / OWNER_MARKER).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False  # метки нет: решает возраст
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # процесс есть, но чужой — не наш случай решать
    return True


def _freshest_mtime(path: Path) -> float:
    latest = path.lstat().st_mtime
    with os.scandir(path) as entries:
        for entry in entries:
            try:
                latest = max(latest, entry.stat(follow_symlinks=False).st_mtime)
            except OSError:
                continue
    return latest


def sweep_orphans(
    root: Path | None = None,
    *,
    max_age_hours: float = ORPHAN_AGE_HOURS,
    now: float | None = None,
) -> list[Path]:
    """Отдаёт на удаление свои каталоги без хозяина; возвращает, какие.

    Каталог уходит, если самый свежий mtime у него и у первого уровня старше
    `max_age_hours`, а записанный в метке процесс не жив. Совпавший pid чужого
    процесса каталог только сбережёт: ошибка в безопасную сторону.
    """
    base = Path(root) if root is not None else Path(tempfile.gettempdir())
    cutoff = (time.time() if now is None else now) - max_age_hours * 3600
    removed: list[Path] = []
    try:
        entries = list(os.scandir(base))
    except OSError:
        return removed
    for entry in entries:
        if not entry.name.startswith(PREFIXES):
            continue
        path = Path(entry.path)
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
            if _freshest_mtime(path) > cutoff or _owner_alive(path):
                continue
        except OSError:
            continue
        remove_detached(path)
        removed.append(path)
    return sorted(removed)
