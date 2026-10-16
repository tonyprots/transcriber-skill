#!/usr/bin/env python3
"""Прогоняет длинные диктовки Handy через конвейер transcriber.

Запускается launchd-агентом по изменению папки записей Handy. Handy вставляет
свой текст сразу; здесь в фоне считается расшифровка. Текст основной модели
уходит в буфер обмена с уведомлением, как только готов, — проверяющая его не
меняет и досчитывается дальше ради словаря терминов. Только стандартная библиотека: интерпретатор агента — не `.venv`
скилла, а тот, у которого есть доступ к ~/Documents.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

HANDY_RECORDINGS = Path.home() / "Library/Application Support/com.pais.handy/recordings"
STATE_FILE = Path.home() / "Library/Application Support/handy-transcriber/state.json"
WAV_NAME = re.compile(r"^handy-(\d+)\.wav$")
SETTLE_SECONDS = 2.0


def log(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"seen": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def recordings(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("handy-*.wav") if WAV_NAME.match(p.name))


def wait_until_written(path: Path) -> bool:
    """Handy пишет wav параллельно с распознаванием: ждём, пока размер не замрёт."""
    previous = -1
    for _ in range(30):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == previous and time.time() - path.stat().st_mtime >= SETTLE_SECONDS:
            return True
        previous = size
        time.sleep(SETTLE_SECONDS)
    return False


def duration_seconds(path: Path) -> float:
    with wave.open(str(path)) as wav:
        return wav.getnframes() / wav.getframerate()


def output_dir_for(wav: Path, root: Path) -> Path:
    started = datetime.fromtimestamp(int(WAV_NAME.match(wav.name).group(1)))
    return root / f"{started:%Y-%m-%d-%H%M}-{wav.stem}"


def notify(title: str, message: str) -> None:
    # Текст — аргументами, а не внутри скрипта: json.dumps экранировал
    # кириллицу в \uXXXX, такого escape в AppleScript нет, и уведомление
    # молча не приходило ни разу.
    result = subprocess.run(
        [
            "osascript",
            "-e", "on run argv",
            "-e", "display notification (item 2 of argv) with title (item 1 of argv)",
            "-e", "end run",
            title, message,
        ],
        check=False, capture_output=True,
    )
    if result.returncode != 0:
        log(f"уведомление не показано: {result.stderr.decode('utf-8', 'replace').strip()}")


def to_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


def from_clipboard() -> str:
    result = subprocess.run(["pbpaste"], capture_output=True, check=False)
    return result.stdout.decode("utf-8", "replace")


def refresh_clipboard(pasted: str, final: str) -> bool:
    """Меняет быструю версию в буфере на итог, если её ещё никто не сменил.

    Быстрая версия — текст одной GigaAM с точными алиасами словаря; названия,
    которые верно написала только проверяющая («Bitrik4» → Битрикс24), в неё
    не попадают (2026-10-05). Скопированное поверх за время прогона не трогаем.
    """
    if not final or final == pasted or from_clipboard().strip() != pasted:
        return False
    to_clipboard(final)
    return True


def early_text_path(out: Path) -> Path:
    # Рядом с каталогом результата, а не внутри: каталог скилл создаёт сам и
    # отказывается писать в непустой.
    return out.with_name(f"{out.name}.txt")


def running_marker_path(out: Path) -> Path:
    # Каталог результата скилл создаёт только при сборке, поэтому без метки
    # идущий прогон снаружи не виден, и агент запускал второй (2 октября).
    return out.with_name(f"{out.name}.running")


def process_started(pid: int) -> str | None:
    """Время запуска процесса по `ps`, None — узнать не вышло.

    Метку пишет агент launchd без локали, а читает handy_result из оболочки
    со своей: формат даты и пояс заданы явно, иначе отпечатки не совпадут.
    """
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, check=False,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def write_marker(marker: Path, pid: int, wav_name: str) -> None:
    marker.write_text(
        json.dumps({"pid": pid, "started": process_started(pid), "wav": wav_name}), encoding="utf-8"
    )


def marker_alive(marker: Path) -> bool:
    """Метка живого прогона; метка упавшего процесса не считается.

    Убитый по SIGKILL или SIGTERM (выход из сессии, перезагрузка) наблюдатель
    метку не снимает, а его pid потом достаётся другому процессу. Чужой по
    пользователю процесс (EPERM) — точно не наш прогон; свой проверяем по
    времени запуска. Метка без него (до отпечатка) судится по одному pid.
    """
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        pid = int(data["pid"])
        os.kill(pid, 0)
    except (FileNotFoundError, ValueError, KeyError, TypeError, ProcessLookupError, PermissionError):
        return False
    recorded = data.get("started")
    if not recorded:
        return True
    current = process_started(pid)
    return current is None or current == recorded


def transcribe(
    wav: Path, out: Path, early: Path, args: argparse.Namespace, sink
) -> subprocess.Popen:
    # `--priority`: диктовку ждут сейчас, и пакет на машине уступает ей очередь
    # (2026-10-10 голосовое простояло 25 минут за выпуском подкаста).
    command = [
        args.python, args.transcribe, str(wav),
        "--mode", args.mode, "--language", args.language,
        "--output", str(out), "--early-text", str(early), "--priority", "--quiet",
    ]
    # Вывод в файл, а не в пайп: пока мы ждём текст, пайп никто не читает, и
    # переполненный буфер повесил бы прогон.
    return subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT)


def text_version(early: Path) -> tuple[int, int] | None:
    """Отпечаток файла текста: transcriber подменяет его целиком (новый inode)."""
    try:
        stat = early.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_mtime_ns


def wait_for_text(process: subprocess.Popen, early: Path, before: tuple[int, int] | None = None) -> bool:
    """Ждём текст основной модели или конец прогона, смотря что раньше.

    Файл с отпечатком `before` остался от прошлой попытки и текстом этой не
    считается: иначе повтор упавшей записи сразу сообщал «готово за 0 с»,
    а о новом сбое молчал.
    """
    while process.poll() is None:
        if text_version(early) not in (None, before):
            return True
        time.sleep(0.5)
    return text_version(early) not in (None, before)


def process(wav: Path, args: argparse.Namespace) -> str:
    if not wait_until_written(wav):
        return "unsettled"
    seconds = duration_seconds(wav)
    if seconds < args.min_seconds:
        return f"short {seconds:.0f}s"

    out = output_dir_for(wav, args.output_root)
    marker = running_marker_path(out)
    if out.exists() or marker_alive(marker):
        return f"exists {out.name}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    log(f"{wav.name}: {seconds:.0f} с → {out}")
    write_marker(marker, os.getpid(), wav.name)
    try:
        return run_job(wav, out, seconds, args)
    finally:
        marker.unlink(missing_ok=True)


def run_job(wav: Path, out: Path, seconds: float, args: argparse.Namespace) -> str:
    started = time.monotonic()
    early = early_text_path(out)
    pasted = None
    before = text_version(early)
    with tempfile.TemporaryFile() as sink:
        job = transcribe(wav, out, early, args, sink)
        fresh = wait_for_text(job, early, before)
        if fresh:
            text_ready = time.monotonic() - started
            if args.clipboard:
                pasted = early.read_text(encoding="utf-8").strip()
                to_clipboard(pasted)
            # Число спорных мест в уведомлении не нужно: свою диктовку никто не
            # переслушивает, а спорное агент уточнит вопросом, если от него
            # что-то зависит.
            where = "текст в буфере" if args.clipboard else early.name
            notify("Расшифровка готова", f"{seconds / 60:.1f} мин за {text_ready:.0f} с — {where}")
            log(f"{wav.name}: текст готов за {text_ready:.0f} с")
        job.wait()
        elapsed = time.monotonic() - started
        if job.returncode != 0 or not (out / "readable.md").exists():
            sink.seek(0)
            tail = sink.read().decode("utf-8", "replace").strip().splitlines()[-5:]
            log(f"{wav.name}: ошибка rc={job.returncode}: " + " | ".join(tail))
            if not fresh:
                notify("Handy → transcriber", f"Не вышло расшифровать {wav.name}, подробности в логе")
            return f"failed rc={job.returncode}"
    log(f"{wav.name}: проверка и словарь готовы за {elapsed:.0f} с")
    if pasted is not None and refresh_clipboard(pasted, early.read_text(encoding="utf-8").strip()):
        notify("Расшифровка уточнена", "Названия сверены с проверяющей — итог в буфере")
        log(f"{wav.name}: итог заменил быструю версию в буфере")
    return f"done {out.name}"


MAX_ATTEMPTS = 3


def attempt(wav: Path, args: argparse.Namespace, state: dict) -> None:
    seen: dict = state.setdefault("seen", {})
    failures: dict = state.setdefault("failures", {})
    try:
        seen[wav.name] = process(wav, args)
    except (OSError, wave.Error, EOFError) as error:
        seen[wav.name] = f"error {error}"
        log(f"{wav.name}: {error}")
    if str(seen[wav.name]).startswith("failed"):
        failures[wav.name] = failures.get(wav.name, 0) + 1
    else:
        failures.pop(wav.name, None)
    save_state(args.state, state)


def run(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    seen: dict = state.setdefault("seen", {})
    if args.init:
        for wav in recordings(args.recordings):
            seen.setdefault(wav.name, "baseline")
        save_state(args.state, state)
        log(f"baseline: {len(seen)} записей отмечено как уже виденные")
        return 0

    # Пока шёл прогон, Handy мог записать новую диктовку: пересканировать до тишины.
    tried: set[str] = set()
    while True:
        pending = [w for w in recordings(args.recordings) if w.name not in seen]
        if not pending:
            break
        for wav in pending:
            attempt(wav, args, state)
            tried.add(wav.name)

    # Упавшую запись повторяем при следующем запуске агента: 3 октября диктовка
    # упала на нехватке места и так и осталась без текста. Не в том же запуске,
    # где она упала, после новых записей и без буфера: старый текст не должен
    # затереть свежий.
    retry_args = argparse.Namespace(**{**vars(args), "clipboard": False})
    failures: dict = state.setdefault("failures", {})
    for wav in recordings(args.recordings):
        done_before = failures.get(wav.name, 0)
        if (wav.name not in tried and str(seen.get(wav.name, "")).startswith("failed")
                and done_before < MAX_ATTEMPTS):
            log(f"{wav.name}: повтор, попытка {done_before + 1} из {MAX_ATTEMPTS}")
            attempt(wav, retry_args, state)

    # Handy сам удаляет старые записи; забывать их, чтобы состояние не росло.
    present = {w.name for w in recordings(args.recordings)}
    for name in [n for n in seen if n not in present]:
        del seen[name]
    for name in [n for n in failures if n not in present]:
        del failures[name]
    save_state(args.state, state)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", required=True, help="питон из .venv скилла")
    parser.add_argument("--transcribe", required=True, help="путь к scripts/transcribe.py")
    parser.add_argument("--output-root", type=Path, required=True, help="куда класть расшифровки")
    parser.add_argument("--recordings", type=Path, default=HANDY_RECORDINGS)
    parser.add_argument("--state", type=Path, default=STATE_FILE)
    parser.add_argument("--min-seconds", type=float, default=20.0,
                        help="записи короче Handy распознаёт сам (по умолчанию 20)")
    parser.add_argument("--mode", default="max")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--no-clipboard", dest="clipboard", action="store_false",
                        help="не класть текст в буфер, только папка и уведомление")
    parser.add_argument("--init", action="store_true",
                        help="отметить уже лежащие записи виденными и выйти")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))
