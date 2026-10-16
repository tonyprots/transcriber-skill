from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from . import machine_lock
from .backends import BackendMissing, BackendUnavailable, child_hub_environment
from .models import AudioChunk, Diarization, Hypothesis


class ProcessLike(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def communicate(self) -> tuple[str, str]: ...


def _progress_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


def _activity_signature(paths: tuple[Path, ...]) -> tuple[tuple[int, int] | None, ...]:
    """Признак жизни — любой из файлов: progress.json или лог воркера.

    Пока модель качается с Hub, окна не завершаются, зато прогресс-бар
    загрузки пишет в stderr. Без этого первая загрузка полутора гигабайт на
    медленной сети выглядела бы для watchdog как зависание.
    """
    return tuple(_progress_signature(path) for path in paths)


def _progress_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def wait_for_progress(
    process: ProcessLike,
    progress_path: Path,
    *,
    stall_timeout_seconds: float,
    poll_seconds: float = 0.2,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    activity_paths: tuple[Path, ...] = (),
) -> bool:
    """Ждёт процесс и возвращает False, если он остановлен из-за отсутствия прогресса."""
    last_progress = time.monotonic()
    signature = _progress_signature(progress_path)
    activity = _activity_signature(activity_paths)
    may_pause = True
    while process.poll() is None:
        if may_pause and machine_lock.priority_waiting():
            if _pause_and_give_way(progress_path):
                # Простой на голосовом — не зависание: отсчёт watchdog с нуля.
                last_progress = time.monotonic()
                activity = _activity_signature(activity_paths)
                continue
            # Метку паузы не записать: воркер не встанет, машина остаётся у
            # нас до конца этого шага. Watchdog считает как обычно — раньше
            # здесь был холостой цикл, обнулявший отсчёт на каждом обороте.
            may_pause = False
        current = _progress_signature(progress_path)
        current_activity = _activity_signature(activity_paths)
        if current_activity != activity:
            activity = current_activity
            last_progress = time.monotonic()
        if current != signature:
            signature = current
            last_progress = time.monotonic()
            if on_progress is not None:
                payload = _progress_payload(progress_path)
                if payload is not None:
                    on_progress(payload)
        if time.monotonic() - last_progress > stall_timeout_seconds:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            return False
        time.sleep(poll_seconds)
    return True


# Метка паузы лежит рядом с progress.json воркера; внутри — pid родителя.
PAUSE_NAME = "pause"


def _pause_and_give_way(progress_path: Path) -> bool:
    """Голосовое ждёт машину: воркер встаёт после текущего окна, машина уходит
    голосовому и возвращается, когда оно закончит.

    Конца окна не ждём: окно — секунды, а ждать подтверждения от воркера,
    который грузит или качает модель, пришлось бы минутами. Модель остаётся
    в памяти, ядра свободны. Воркер без точек паузы (бинарник FluidAudio)
    досчитывает рядом с голосовым — как до общей очереди.

    False — метку паузы записать не вышло: воркер не встанет, и машину не
    отдаём, голосовое ждёт, как до приоритета.
    """
    pause = progress_path.with_name(PAUSE_NAME)
    try:
        _post_pause(pause)
    except OSError as error:
        print(
            f"Внимание: метку паузы воркера не записать ({error}) — голосовому "
            "машину не отдаю, оно подождёт конца этого шага",
            file=sys.stderr, flush=True,
        )
        return False
    try:
        machine_lock.give_way()
    finally:
        pause.unlink(missing_ok=True)
    return True


def _post_pause(pause: Path) -> None:
    """Метка паузы атомарно — временный файл и `os.replace`, как `priority.wait`.

    Прямая запись на полном диске создавала файл, падала на записи и
    оставляла пустой `pause`, на котором воркер стоял, хотя машину никто не
    отдавал. Сбой — OSError, и ни метки, ни временного файла не остаётся.
    """
    temporary = pause.with_name(f".{pause.name}.{os.getpid()}")
    try:
        temporary.write_text(str(os.getpid()), encoding="utf-8")
        os.replace(temporary, pause)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def hold_while_paused(progress_path: Path, *, poll_seconds: float = 0.5) -> None:
    """Сторона воркера: между окнами стоять, пока родитель отдал машину.

    Родитель умер — ждать некого: выходим сразу, иначе сирота держал бы
    гигабайты весов вечно. Пустая или нечитаемая метка — не пауза: родитель
    пишет её атомарно, и такую оставил бы только сбой, после которого он
    машину не отдаёт.
    """
    pause = progress_path.with_name(PAUSE_NAME)
    while True:
        try:
            parent = int(pause.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if os.getppid() != parent:
            os._exit(2)
        time.sleep(poll_seconds)


WORKER_MISSING_EXIT_CODE = 3


def _tail(path: Path, limit: int = 4000) -> str:
    """Хвост лога воркера: в ошибку нужна причина, а не мегабайты прогресс-баров."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Последняя строка прогресс-бара tqdm перекрывает предыдущие через \r.
    text = text.replace("\r", "\n").strip()
    return text[-limit:]


def _chunk_payload(chunk: AudioChunk) -> dict[str, Any]:
    return {
        "sequence": chunk.sequence,
        "path": str(chunk.path.resolve()),
        "start": chunk.start,
        "end": chunk.end,
        "speakers": list(chunk.speakers),
    }


def _run_worker(
    module: str,
    label: str,
    request: dict[str, Any],
    work_dir: Path,
    *,
    stall_timeout_seconds: float,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    job_dir = Path(tempfile.mkdtemp(prefix=f"{label}-", dir=work_dir))
    request_path = job_dir / "request.json"
    result_path = job_dir / "result.json"
    progress_path = job_dir / "progress.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scripts_dir = Path(__file__).resolve().parents[1]
    environment = child_hub_environment(os.environ.copy())
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(scripts_dir), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # Вывод воркера — в файлы, а не в pipe: pipe никто не читает до конца
    # работы, и после 64 КБ (прогресс-бары загрузки, предупреждения
    # onnxruntime) воркер блокируется на записи, а watchdog его убивает.
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                module,
                str(request_path),
                str(result_path),
                str(progress_path),
            ],
            cwd=job_dir,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        try:
            completed = wait_for_progress(
                process,
                progress_path,
                stall_timeout_seconds=stall_timeout_seconds,
                on_progress=on_progress,
                activity_paths=(stdout_path, stderr_path),
            )
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise
        process.wait()
    stderr = _tail(stderr_path)
    stdout = _tail(stdout_path)
    if not completed:
        raise BackendUnavailable(
            f"{label} не сообщил о прогрессе за {stall_timeout_seconds:g} с и был остановлен"
        )
    if process.returncode == WORKER_MISSING_EXIT_CODE:
        raise BackendMissing(f"{label} недоступен на этой машине: {stderr or stdout}")
    if process.returncode != 0:
        detail = stderr or stdout or f"код {process.returncode}"
        raise BackendUnavailable(f"Изолированный {label} завершился с ошибкой: {detail}")
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BackendUnavailable(
            f"Изолированный {label} не вернул корректный результат: {error}"
        ) from error


def run_isolated_backend(
    backend: str,
    chunks: list[AudioChunk],
    language: str,
    work_dir: Path,
    *,
    config: dict[str, Any],
    stall_timeout_seconds: float,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    label: str | None = None,
) -> Hypothesis:
    """Запускает один ASR-бэкенд в дочернем процессе с watchdog по прогрессу.

    `label` — как называть модель в сообщениях сторожа и в имени каталога
    задания: один воркер `gigaam` поднимает и GigaAM, и Vosk, и Parakeet.
    """
    # Слэш из имени репозитория увёл бы каталог задания вглубь work_dir.
    label = (label or backend).replace("/", "-")
    try:
        payload = _run_worker(
            "audio_transcription.worker",
            label,
            {
                "schema": 1,
                "backend": backend,
                "language": language,
                "config": config,
                "chunks": [_chunk_payload(chunk) for chunk in chunks],
            },
            work_dir,
            stall_timeout_seconds=stall_timeout_seconds,
            on_progress=on_progress,
        )
        return Hypothesis.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise BackendUnavailable(
            f"Изолированный {label} не вернул корректный результат: {error}"
        ) from error


def run_isolated_diarization(
    backend: str,
    audio_path: Path,
    work_dir: Path,
    *,
    config: dict[str, Any],
    stall_timeout_seconds: float,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> Diarization:
    try:
        payload = _run_worker(
            "audio_transcription.diarization_worker",
            backend,
            {
                "schema": 1,
                "backend": backend,
                "audio_path": str(audio_path.resolve()),
                "config": config,
            },
            work_dir,
            stall_timeout_seconds=stall_timeout_seconds,
            on_progress=on_progress,
        )
        return Diarization.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise BackendUnavailable(
            f"Изолированная диаризация не вернула корректный результат: {error}"
        ) from error
