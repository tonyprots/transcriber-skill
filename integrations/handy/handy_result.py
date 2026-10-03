#!/usr/bin/env python3
"""Отдаёт агенту расшифровку записи Handy: готовую, идущую или новую.

Один вход вместо «поищи в папке, не нашёл — запусти сам». 2 октября агент
не увидел идущий прогон наблюдателя, угадывал путь к скрипту и запустил
второй, когда текст уже 19 секунд как был готов.

    handy_result.py ~/Library/Application\\ Support/com.pais.handy/recordings/handy-<ts>.wav

Печатает пути и текст основной модели; с `--full` дожидается полного набора
и печатает правки и названия, которые слышит только проверяющая. Если
наблюдатель запись ещё считает, ждёт его; если он её не взял (короткая, упал, агент не установлен), сам
запускает transcriber в ту же раскладку и выходит, как только готов текст:
проверяющая и словарь досчитываются в фоне. Конфигурация берётся из
plist агента, флаги её перекрывают. Только стандартная библиотека.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from handy_watch import (  # noqa: E402
    STATE_FILE,
    WAV_NAME,
    early_text_path,
    marker_alive,
    output_dir_for,
    running_marker_path,
    to_clipboard,
)

PLIST = Path.home() / "Library/LaunchAgents/ru.tonyprots.handy-transcriber.plist"
REPO = Path(__file__).resolve().parents[2]
# Наблюдатель стартует по WatchPaths и ждёт, пока wav замрёт на 2 с.
PICKUP_GRACE_SECONDS = 20.0
POLL_SECONDS = 1.0


def agent_config(plist: Path) -> dict[str, str]:
    """--ключ значение из ProgramArguments агента."""
    try:
        argv = plistlib.loads(plist.read_bytes())["ProgramArguments"]
    except (FileNotFoundError, KeyError, plistlib.InvalidFileException):
        return {}
    return {
        key[2:]: value
        for key, value in zip(argv, argv[1:])
        if key.startswith("--") and not value.startswith("--")
    }


def seen_status(state_file: Path, wav: Path) -> str | None:
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))["seen"].get(wav.name)
    except (FileNotFoundError, KeyError, ValueError):
        return None


def watcher_busy(root: Path) -> bool:
    """Наблюдатель считает другую запись, наша стоит в очереди за ней."""
    return any(marker_alive(m) for m in root.glob("*.running"))


def report(wav: Path, out: Path, how: str, *, clipboard: bool = False) -> int:
    early = early_text_path(out)
    text = early.read_text(encoding="utf-8").strip()
    print(f"# {wav.name}: {how}")
    print(f"# текст: {early}")
    readable = out / "readable.md"
    if readable.exists():
        print(f"# полный набор: {out} (readable.md, review-needed.md)")
        # Текст ниже — от основной модели, до проверяющей. Термины, которые
        # она поправила по словарю, есть только в readable.md.
        try:
            audit = json.loads((out / "glossary-audit.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            audit = {}
        taken = [
            f"«{item['before']}» → {item['after']}"
            for item in audit.get("corrections", [])
            if item.get("rule") == "verifier_canonical"
        ]
        if taken:
            print(f"# в readable.md поправлено по проверяющей: {', '.join(taken)}")
        # Названия, которые основная модель уронила, а проверяющая услышала:
        # в текст они не вставлены, но без них фраза может потерять адресата.
        try:
            items = json.loads((out / "segments.json").read_text(encoding="utf-8"))["review_items"]
        except (FileNotFoundError, KeyError, ValueError):
            items = []
        dropped = [
            f"{_clock(item['start'])} «{item['verifier_span'].strip()}»"
            for item in items
            if item.get("kind") == "verifier_only"
        ]
        if dropped:
            print(f"# возможно, выпало из текста (слышит только проверяющая): {', '.join(dropped)}")
    else:
        print(f"# полный набор досчитывается: {out}")
    if clipboard:
        # Запись, которую наблюдатель не взял (короче порога), кончается так
        # же, как взятая: текст в буфере, вставить его может сам человек.
        try:
            to_clipboard(text)
            print("# текст скопирован в буфер обмена")
        except (OSError, subprocess.CalledProcessError):
            pass
    print()
    print(text)
    return 0


def _clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"


def run_log(out: Path) -> Path:
    # Не во входящие: там лежат только результаты.
    return Path(tempfile.gettempdir()) / f"{out.name}.log"


def start_own_run(wav: Path, out: Path, args: argparse.Namespace) -> subprocess.Popen:
    out.parent.mkdir(parents=True, exist_ok=True)
    marker = running_marker_path(out)
    log = run_log(out).open("w", encoding="utf-8")
    # Оболочка снимает метку, когда прогон кончится: мы выходим раньше, на тексте.
    job = subprocess.Popen(
        [
            "/bin/sh", "-c", '"$@"; rc=$?; rm -f "$0"; exit $rc', str(marker),
            args.python, args.transcribe, str(wav),
            "--mode", args.mode, "--language", args.language,
            "--output", str(out), "--early-text", str(early_text_path(out)), "--quiet",
        ],
        stdout=log, stderr=subprocess.STDOUT,
        # Своя сессия: прогон досчитывает проверку после нашего выхода.
        start_new_session=True,
    )
    marker.write_text(json.dumps({"pid": job.pid, "wav": wav.name}), encoding="utf-8")
    return job


def run(args: argparse.Namespace) -> int:
    wav = args.wav.expanduser().resolve()
    if not WAV_NAME.match(wav.name) or not wav.exists():
        print(f"Не запись Handy или файла нет: {wav}", file=sys.stderr)
        return 2
    out = output_dir_for(wav, args.output_root)
    early = early_text_path(out)
    marker = running_marker_path(out)
    deadline = time.monotonic() + args.timeout
    if marker.exists() and not marker_alive(marker):
        marker.unlink()  # след упавшего прогона
    job: subprocess.Popen | None = None
    waited_for = "готова"

    full = out / "readable.md"
    while time.monotonic() < deadline:
        # Каталог результата скилл подменяет целиком, поэтому readable.md
        # появляется только вместе с полным набором.
        if (full if args.full else early).exists():
            return report(wav, out, waited_for, clipboard=job is not None)
        if job is not None:
            if job.poll() is not None:
                if (full if args.full else early).exists():
                    return report(wav, out, waited_for, clipboard=True)
                break
        elif marker_alive(marker):
            waited_for = "дождался прогона наблюдателя"
        elif args.full and early.exists():
            # Текст есть, прогона нет, набора нет: прогон умер после основной
            # модели. Перезапуск упрётся в занятый каталог — отдаём что есть.
            return report(wav, out, "полный набор не досчитан, только текст")
        else:
            status = seen_status(args.state, wav)
            young = time.time() - wav.stat().st_mtime < PICKUP_GRACE_SECONDS
            if status is None and (young or watcher_busy(args.output_root)):
                waited_for = "дождался наблюдателя"
            else:
                # Короткую запись наблюдатель пропускает, упавший прогон не
                # повторяет: считаем сами. Минута аудио — около 5 с до текста.
                job = start_own_run(wav, out, args)
                waited_for = f"расшифровал сам (наблюдатель: {status or 'не взял'})"
        time.sleep(POLL_SECONDS)

    log = run_log(out)
    tail = log.read_text(encoding="utf-8").strip().splitlines()[-5:] if log.exists() else []
    print(f"Текста нет за {args.timeout:.0f} с: {out}", file=sys.stderr)
    for line in tail:
        print(f"  {line}", file=sys.stderr)
    return 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    config = agent_config(PLIST)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path(config.get("output-root", REPO / "out/handy")))
    parser.add_argument("--python", default=config.get("python", str(REPO / ".venv/bin/python")))
    parser.add_argument("--transcribe", default=config.get(
        "transcribe", str(REPO / "skills/transcriber/scripts/transcribe.py")))
    parser.add_argument("--state", type=Path, default=STATE_FILE)
    parser.add_argument("--mode", default=config.get("mode", "max"))
    parser.add_argument("--language", default=config.get("language", "ru"))
    parser.add_argument("--full", action="store_true",
                        help="ждать полный набор (проверяющая, словарь), а не только текст")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="сколько ждать текст, секунд (по умолчанию 900)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))
