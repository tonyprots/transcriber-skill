"""Одна тяжёлая расшифровка на машину, остальные ждут в очереди.

2026-09-22 две сессии Claude одновременно запустили по два-три прогона:
VAD шёл 168–541 с вместо 8–30, прогон `fast` — с RTF 0,43 вместо 0,16–0,19.
Модели делят одни ядра и одну память, и параллельный запуск не ускоряет
пачку, а растягивает каждый прогон и прячет, какой из них завис.

Очередь держится на `flock`: ядро снимает блокировку, когда процесс умирает,
так что «вечного замка» после убитого прогона не бывает. Скачивание и
подготовка звука идут до очереди — это сеть и ffmpeg, им делить нечего.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: очереди нет, прогоны идут как раньше
    fcntl = None  # type: ignore[assignment]

DEFAULT_LOCK = Path.home() / ".transcriber" / "run.lock"


def lock_path() -> Path:
    override = os.environ.get("TRANSCRIBER_LOCK")
    return Path(override).expanduser() if override else DEFAULT_LOCK


@contextmanager
def machine_slot(
    label: str,
    *,
    on_wait: Callable[[str, float], None] | None = None,
    path: Path | None = None,
    poll_seconds: float = 2.0,
    report_every_seconds: float = 60.0,
) -> Iterator[float]:
    """Держит машину на время прогона; отдаёт, сколько секунд ждал очереди.

    `on_wait(кто держит, сколько ждём)` зовётся сразу и затем раз в
    `report_every_seconds`: ожидание длится десятки минут, и без признака
    жизни в логе оно неотличимо от зависания.
    """
    if fcntl is None:
        yield 0.0
        return
    target = path or lock_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with target.open("a+", encoding="utf-8") as handle:
        reported_at: float | None = None
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if on_wait is not None and (
                    reported_at is None or now - reported_at >= report_every_seconds
                ):
                    handle.seek(0)
                    on_wait(handle.read().strip() or "другой прогон", now - started)
                    reported_at = now
                time.sleep(poll_seconds)
        waited = time.monotonic() - started
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()} с {time.strftime('%H:%M:%S')}: {label}\n")
        handle.flush()
        try:
            yield waited
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)
