"""Завершение процесса без C++-teardown onnxruntime и с уборкой по сигналу.

При обычном выходе интерпретатора onnxruntime иногда падает в деструкторе
пула потоков («recursive_mutex lock failed»), и процесс, уже записавший
результат, возвращает 134. Поэтому точки входа сбрасывают буферы и выходят
через os._exit: результат к этому моменту уже на диске.

SIGTERM и SIGHUP Python по умолчанию не ловит и умирает молча: не срабатывают
ни `finally` у временного каталога и очереди машины, ни `except BaseException`,
который гасит дочерний воркер. Воркер досчитывал сиротой, копия записи
оставалась в $TMPDIR. `run_entry_point` превращает сигнал в исключение.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from collections.abc import Callable

_STOP_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)


class Terminated(SystemExit):
    """Процесс остановлен сигналом; код — 128 + номер сигнала, как у shell.

    Наследник SystemExit, а не Exception: границы процессов ловят `Exception`
    и превратили бы остановку в обычную ошибку бэкенда.
    """


def exit_after_flush(code: int) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(code)


def _terminate(signum: int, _frame: object) -> None:
    # Второй сигнал убивает сразу, как раньше: повисшая уборка не должна
    # держать процесс против повторного `kill`.
    for stop in _STOP_SIGNALS:
        signal.signal(stop, signal.SIG_DFL)
    raise Terminated(128 + signum)


def stop_on_signals() -> None:
    """SIGTERM и SIGHUP поднимают `Terminated` в основном потоке.

    Уже игнорируемый сигнал не трогаем: под `nohup` закрытый терминал не
    должен ронять прогон.
    """
    for stop in _STOP_SIGNALS:
        if signal.getsignal(stop) is not signal.SIG_IGN:
            signal.signal(stop, _terminate)


def watch_parent(*, poll_seconds: float = 1.0, grace_seconds: float = 10.0) -> None:
    """Воркер выходит сам, когда родителя не стало.

    SIGKILL родителю не перехватить, и его `except BaseException` воркер не
    погасит. Раз в секунду сверяем родителя; сменился — шлём себе SIGTERM, чтобы
    отработала уборка, а если основной поток застрял в нативном коде и сигнал
    не обработал, через `grace_seconds` выходим без неё.
    """
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(poll_seconds)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(grace_seconds)
        os._exit(1)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def run_entry_point(main: Callable[[], int], *, watch_parent_process: bool = False) -> None:
    """Запускает `main` точки входа и выходит через `exit_after_flush`."""
    stop_on_signals()
    if watch_parent_process:
        watch_parent()
    try:
        code = main()
    except Terminated as stop:
        code = int(stop.code)
    exit_after_flush(code)
