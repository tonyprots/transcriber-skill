"""Одна тяжёлая расшифровка на машину, остальные ждут в очереди.

2026-09-22 две сессии Claude одновременно запустили по два-три прогона:
VAD шёл 168–541 с вместо 8–30, прогон `fast` — с RTF 0,43 вместо 0,16–0,19.
Модели делят одни ядра и одну память, и параллельный запуск не ускоряет
пачку, а растягивает каждый прогон и прячет, какой из них завис.

Очередь держится на `flock`: ядро снимает блокировку, когда процесс умирает,
так что «вечного замка» после убитого прогона не бывает. Скачивание и
подготовка звука идут до очереди — это сеть и ffmpeg, им делить нечего.

Голосовое идёт вне очереди. 2026-10-10 диктовка Handy ждала 1517 с: пакет
подкастов считал 87-минутный выпуск, и одна основная модель шла 32 минуты.
Замок и тогда брался на одну запись пакета, а не на весь пакет, — уступать
только между записями было мало. Прогон с `priority`, пока ждёт, держит рядом
с замком метку `priority.wait`. Прогон без приоритета, видя свежую метку,
свободную машину не берёт, а занятую отдаёт между окнами модели
(`give_way` зовёт `isolated.wait_for_progress`): воркер встаёт с моделью в
памяти, голосовое считается, машина возвращается.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

try:
    import fcntl
except ImportError:  # Windows: очереди нет, прогоны идут как раньше
    fcntl = None  # type: ignore[assignment]

DEFAULT_LOCK = Path.home() / ".transcriber" / "run.lock"
PRIORITY_NAME = "priority.wait"
# Ждущий освежает метку на каждом опросе: старше получаса она бывает только у
# прогона, который умер, не успев её снять, — уступать некому.
PRIORITY_STALE_SECONDS = 30 * 60

OnWait = Callable[[str, float], None]


def lock_path() -> Path:
    override = os.environ.get("TRANSCRIBER_LOCK")
    return Path(override).expanduser() if override else DEFAULT_LOCK


class Slot:
    """Машина, занятая этим процессом.

    `waited` — сколько ждали очереди до старта, `yielded` — сколько отдали
    голосовым по ходу работы.
    """

    def __init__(
        self,
        label: str,
        handle: IO[str] | None,
        marker: Path | None,
        *,
        priority: bool,
        on_wait: OnWait | None,
        poll_seconds: float,
        report_every_seconds: float,
    ) -> None:
        self.label = label
        self.priority = priority
        self.waited = 0.0
        self.yielded = 0.0
        self._handle = handle
        self._marker = marker
        self._on_wait = on_wait
        self._poll_seconds = poll_seconds
        self._report_every = report_every_seconds
        self._held = False

    def wanted(self) -> bool:
        """Ждёт ли машину голосовое, которому этот прогон должен её отдать."""
        return self._held and not self.priority and _priority_waiter(self._marker) is not None

    def give_way(self) -> float:
        """Отдать машину ждущему голосовому и дождаться её обратно; секунды простоя."""
        if not self.wanted():
            return 0.0
        started = time.monotonic()
        self._release()
        self._take()
        spent = time.monotonic() - started
        self.yielded += spent
        return spent

    def _take(self) -> float:
        """Ждёт замок. Без приоритета пропускает вперёд ждущее голосовое,
        с приоритетом — держит метку, пока ждёт."""
        started = time.monotonic()
        reported_at: float | None = None
        try:
            while True:
                ahead = None if self.priority else _priority_waiter(self._marker)
                if ahead is None:
                    try:
                        fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if self.priority:
                            _post_marker(self._marker, self.label)
                now = time.monotonic()
                if self._on_wait is not None and (
                    reported_at is None or now - reported_at >= self._report_every
                ):
                    holder = f"голосовое вне очереди: {ahead}" if ahead is not None else self._holder()
                    self._on_wait(holder, now - started)
                    reported_at = now
                time.sleep(self._poll_seconds)
        finally:
            # Метка живёт, только пока ждём: взявший машину её снимает, и
            # упавший на ожидании тоже.
            if self.priority:
                _drop_marker(self._marker)
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid {os.getpid()} с {time.strftime('%H:%M:%S')}: {self.label}\n")
        self._handle.flush()
        self._held = True
        return time.monotonic() - started

    def _holder(self) -> str:
        self._handle.seek(0)
        return self._handle.read().strip() or "другой прогон"

    def _release(self) -> None:
        if not self._held:
            return
        self._held = False
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.flush()
        fcntl.flock(self._handle, fcntl.LOCK_UN)


# Машину процесс держит одну: второй `machine_slot` в том же процессе ждал бы
# сам себя. Поэтому точка уступки находит её здесь, а не через аргументы всех
# функций между `cli._transcribe` и ожиданием воркера.
_current: Slot | None = None


def priority_waiting() -> bool:
    """Ждёт ли голосовое машину, которую держит этот процесс без приоритета."""
    return _current is not None and _current.wanted()


def give_way() -> float:
    """Уступить голосовому машину этого процесса (см. `Slot.give_way`)."""
    return _current.give_way() if _current is not None else 0.0


@contextmanager
def machine_slot(
    label: str,
    *,
    priority: bool = False,
    on_wait: OnWait | None = None,
    path: Path | None = None,
    poll_seconds: float = 2.0,
    report_every_seconds: float = 60.0,
) -> Iterator[Slot]:
    """Держит машину на время прогона; `Slot.waited` — сколько ждали очереди.

    `on_wait(кто держит, сколько ждём)` зовётся сразу и затем раз в
    `report_every_seconds`: ожидание длится десятки минут, и без признака
    жизни в логе оно неотличимо от зависания. `priority` — голосовое: прогоны
    без приоритета пропускают его вперёд и отдают ему машину между окнами.
    """
    global _current
    options = {
        "priority": priority,
        "on_wait": on_wait,
        "poll_seconds": poll_seconds,
        "report_every_seconds": report_every_seconds,
    }
    if fcntl is None:
        yield Slot(label, None, None, **options)
        return
    target = path or lock_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+", encoding="utf-8") as handle:
        slot = Slot(label, handle, target.with_name(PRIORITY_NAME), **options)
        slot.waited = slot._take()
        _current = slot
        try:
            yield slot
        finally:
            _current = None
            slot._release()


def _priority_waiter(marker: Path | None) -> str | None:
    """Кто из голосовых ждёт машину: подпись свежей метки живого процесса."""
    owner = _marker_owner(marker)
    return owner[1] if owner is not None else None


def _marker_owner(marker: Path | None) -> tuple[int, str] | None:
    if marker is None:
        return None
    try:
        age = time.time() - marker.stat().st_mtime
        payload = json.loads(marker.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if age > PRIORITY_STALE_SECONDS or not _alive(pid):
        return None
    return pid, str(payload.get("label") or "")


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _post_marker(marker: Path, label: str) -> None:
    """Своя метка освежается, чужая или пропавшая пишется заново.

    Запись атомарная — временный файл и `os.replace`, — чтобы читающий не
    застал половину JSON. Двое ждущих голосовых перезаписывают метку друг
    другу, и это не мешает: метка есть, пока ждёт хоть один. Сбой записи
    приоритет теряет, но не роняет прогон: он просто ждёт как обычный.
    """
    try:
        owner = _marker_owner(marker)
        if owner is not None and owner[0] == os.getpid():
            os.utime(marker)
            return
        temporary = marker.with_name(f".{marker.name}.{os.getpid()}")
        temporary.write_text(
            json.dumps({"pid": os.getpid(), "label": label}, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, marker)
    except OSError:
        pass


def _drop_marker(marker: Path) -> None:
    """Снять свою метку; чужую не трогать — её хозяин ещё ждёт."""
    try:
        owner = _marker_owner(marker)
        if owner is not None and owner[0] == os.getpid():
            marker.unlink(missing_ok=True)
    except OSError:
        pass
