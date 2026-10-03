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
    script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
    subprocess.run(["osascript", "-e", script], check=False, capture_output=True)


def to_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


def early_text_path(out: Path) -> Path:
    # Рядом с каталогом результата, а не внутри: каталог скилл создаёт сам и
    # отказывается писать в непустой.
    return out.with_name(f"{out.name}.txt")


def running_marker_path(out: Path) -> Path:
    # Каталог результата скилл создаёт только при сборке, поэтому без метки
    # идущий прогон снаружи не виден, и агент запускал второй (2 октября).
    return out.with_name(f"{out.name}.running")


def marker_alive(marker: Path) -> bool:
    """Метка с живым pid; метка упавшего процесса не считается."""
    try:
        pid = int(json.loads(marker.read_text(encoding="utf-8"))["pid"])
        os.kill(pid, 0)
    except (FileNotFoundError, ValueError, KeyError, TypeError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def transcribe(
    wav: Path, out: Path, early: Path, args: argparse.Namespace, sink
) -> subprocess.Popen:
    command = [
        args.python, args.transcribe, str(wav),
        "--mode", args.mode, "--language", args.language,
        "--output", str(out), "--early-text", str(early), "--quiet",
    ]
    # Вывод в файл, а не в пайп: пока мы ждём текст, пайп никто не читает, и
    # переполненный буфер повесил бы прогон.
    return subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT)


def wait_for_text(process: subprocess.Popen, early: Path) -> bool:
    """Ждём текст основной модели или конец прогона, смотря что раньше."""
    while process.poll() is None:
        if early.exists():
            return True
        time.sleep(0.5)
    return early.exists()


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
    marker.write_text(json.dumps({"pid": os.getpid(), "wav": wav.name}), encoding="utf-8")
    try:
        return run_job(wav, out, seconds, args)
    finally:
        marker.unlink(missing_ok=True)


def run_job(wav: Path, out: Path, seconds: float, args: argparse.Namespace) -> str:
    started = time.monotonic()
    early = early_text_path(out)
    with tempfile.TemporaryFile() as sink:
        job = transcribe(wav, out, early, args, sink)
        if wait_for_text(job, early):
            text_ready = time.monotonic() - started
            if args.clipboard:
                to_clipboard(early.read_text(encoding="utf-8").strip())
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
            if not early.exists():
                notify("Handy → transcriber", f"Не вышло расшифровать {wav.name}, подробности в логе")
            return f"failed rc={job.returncode}"
    log(f"{wav.name}: проверка и словарь готовы за {elapsed:.0f} с")
    return f"done {out.name}"


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
    while True:
        pending = [w for w in recordings(args.recordings) if w.name not in seen]
        if not pending:
            break
        for wav in pending:
            try:
                seen[wav.name] = process(wav, args)
            except (OSError, wave.Error, EOFError) as error:
                seen[wav.name] = f"error {error}"
                log(f"{wav.name}: {error}")
            save_state(args.state, state)

    # Handy сам удаляет старые записи; забывать их, чтобы состояние не росло.
    present = {w.name for w in recordings(args.recordings)}
    for name in [n for n in seen if n not in present]:
        del seen[name]
    save_state(args.state, state)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", required=True, help="питон из .venv скилла")
    parser.add_argument("--transcribe", required=True, help="путь к scripts/transcribe.py")
    parser.add_argument("--output-root", type=Path, required=True, help="куда класть расшифровки")
    parser.add_argument("--recordings", type=Path, default=HANDY_RECORDINGS)
    parser.add_argument("--state", type=Path, default=STATE_FILE)
    parser.add_argument("--min-seconds", type=float, default=60.0,
                        help="записи короче Handy распознаёт сам (по умолчанию 60)")
    parser.add_argument("--mode", default="max")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--no-clipboard", dest="clipboard", action="store_false",
                        help="не класть текст в буфер, только папка и уведомление")
    parser.add_argument("--init", action="store_true",
                        help="отметить уже лежащие записи виденными и выйти")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))
