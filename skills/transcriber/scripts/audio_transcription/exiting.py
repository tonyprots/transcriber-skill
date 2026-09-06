"""Завершение процесса без C++-teardown onnxruntime.

При обычном выходе интерпретатора onnxruntime иногда падает в деструкторе
пула потоков («recursive_mutex lock failed»), и процесс, уже записавший
результат, возвращает 134. Поэтому точки входа сбрасывают буферы и выходят
через os._exit: результат к этому моменту уже на диске.
"""
from __future__ import annotations

import os
import sys


def exit_after_flush(code: int) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    os._exit(code)
