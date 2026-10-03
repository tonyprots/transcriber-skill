#!/usr/bin/env python3
"""Замер на ютьюбовском корпусе: мы, автосубтитры и «просто Whisper».

Три соперника на одном и том же звуке и одном и том же эталоне:

* `youtube-auto` — машинные субтитры ютьюба, взятые из того же видео;
* `whisper-plain` — как работает обычный скилл расшифровки: одна модель на весь
  файл, без нарезки по речи, без второй модели и без сверки;
* `transcriber-fast` и `transcriber-max` — наш конвейер в двух режимах.

Что считаем. Филлеры («э-э», «um») снимаются у всех, включая эталон: человек,
писавший субтитры, их почти никогда не записывает, и оставить их только у машин
значило бы записать в ошибку то, что никто не считает ошибкой. Ударения и
различие е/ё снимаются там же — их не слышно.

Словари выключены намеренно (`--no-glossary`): скилл ведёт свой словарь сам,
его содержимое зависит от истории прогонов, и повторить такой замер на другой
машине было бы нельзя.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "transcriber" / "scripts"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "multilingual-bakeoff"))

from audio_transcription.fillers import strip_fillers  # noqa: E402
from score import error_rate  # noqa: E402
from textnorm import fold  # noqa: E402

TRANSCRIBE = ROOT / "skills" / "transcriber" / "scripts" / "transcribe.py"
PLAIN_MODEL = "mlx-community/whisper-large-v3-turbo-asr-fp16"


def prepared(text: str) -> str:
    return fold(strip_fillers(text)[0])


def measure(reference: str, hypothesis: str) -> dict[str, float | int]:
    errors, words = error_rate(prepared(reference), prepared(hypothesis))
    char_errors, chars = error_rate(prepared(reference), prepared(hypothesis), characters=True)
    return {
        "wer": round(errors / words, 4) if words else 0.0,
        "cer": round(char_errors / chars, 4) if chars else 0.0,
        "errors": errors,
        "words": words,
    }


def whisper_plain(audio: Path, language: str) -> str:
    """Одна модель на весь файл — так выглядит расшифровка без конвейера.

    Настройки оставлены по умолчанию, включая `condition_on_previous_text`:
    именно так вызывает Whisper скилл, который просто передаёт ему файл. Это и
    есть предмет сравнения — не другая модель, а её вызов без обвязки.
    """
    generate = import_module("mlx_audio.stt.generate")
    utils = import_module("mlx_audio.stt.utils")
    model = utils.load_model(PLAIN_MODEL)
    answer = generate.generate_transcription(
        model=model, audio=str(audio), language=language, verbose=None
    )
    text = answer["text"] if isinstance(answer, dict) else getattr(answer, "text", "")
    return str(text)


def parse_system(system: str) -> tuple[str, list[str]]:
    """`transcriber-fast@w28` — режим fast с речевым окном 28 секунд.

    Вариант параметра — отдельная система в результатах, иначе прогон с другой
    настройкой молча затрёт предыдущий и сравнивать станет не с чем.
    """
    name, _, variant = system.partition("@")
    mode = name.split("-", 1)[1]
    flags = ["--speech-window-seconds", variant[1:]] if variant.startswith("w") else []
    return mode, flags


def run_pipeline(audio: Path, language: str, mode: str, output: Path, flags: list[str]) -> str:
    subprocess.run(
        [
            sys.executable, str(TRANSCRIBE), str(audio),
            "--language", language,
            "--mode", mode,
            *flags,
            "--no-glossary",
            "--no-cache",
            "--quiet",
            "--overwrite",
            "--output", str(output),
        ],
        check=True,
    )
    segments = json.loads((output / "segments.json").read_text(encoding="utf-8"))
    return " ".join(segment["text"] for segment in segments["readable"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus")
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--system", action="append", help="Ограничить набор систем")
    parser.add_argument("--only", action="append", help="Ограничить список фрагментов")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    args = parser.parse_args()

    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    if args.only:
        manifest = [item for item in manifest if item["id"] in set(args.only)]
    systems = args.system or ["youtube-auto", "whisper-plain", "transcriber-fast", "transcriber-max"]
    args.runs.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    if args.output.exists():
        results = json.loads(args.output.read_text(encoding="utf-8"))

    for item in manifest:
        reference = (args.corpus / f"{item['id']}.reference.txt").read_text(encoding="utf-8")
        duration = item["window"][1] - item["window"][0]
        for system in systems:
            key = f"{system}/{item['id']}"
            if key in results:
                continue
            print(f"→ {key}", flush=True)
            started = time.monotonic()
            try:
                if system == "youtube-auto":
                    text = (args.corpus / f"{item['id']}.youtube-auto.txt").read_text(encoding="utf-8")
                    elapsed = None
                elif system == "whisper-plain":
                    text = whisper_plain(Path(item["audio"]), item["language"])
                    elapsed = time.monotonic() - started
                else:
                    mode, flags = parse_system(system)
                    text = run_pipeline(
                        Path(item["audio"]),
                        item["language"],
                        mode,
                        args.runs / system / item["id"],
                        flags,
                    )
                    elapsed = time.monotonic() - started
            except Exception as error:  # noqa: BLE001 — падение одной системы не должно валить замер
                print(f"   ошибка: {type(error).__name__}: {error}", flush=True)
                continue
            scores = measure(reference, text)
            results[key] = {
                "system": system,
                "item": item["id"],
                "language": item["language"],
                "duration_seconds": round(duration, 1),
                "seconds": round(elapsed, 1) if elapsed else None,
                "rtfx": round(duration / elapsed, 2) if elapsed else None,
                "text": text.strip(),
                **scores,
            }
            print(
                f"   WER {scores['wer']:.1%}  CER {scores['cer']:.1%}"
                + (f"  RTFx {results[key]['rtfx']}" if elapsed else ""),
                flush=True,
            )
            args.output.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"measurements": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
