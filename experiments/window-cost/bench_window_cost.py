#!/usr/bin/env python3
"""Сколько стоит одно окно Whisper Turbo MLX и от чего эта цена зависит.

Проверяем две вещи:
  1. Растёт ли время с длиной окна. Если нет — кодировщик всегда считает
     30 секунд, и дробить речь на короткие окна бессмысленно дорого.
  2. Сколько сверху берут word_timestamps.

Запуск: .venv/bin/python experiments/window-cost/bench_window_cost.py <media>
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import tempfile
import time
from importlib import import_module
from pathlib import Path

DURATIONS = [1.0, 2.5, 5.0, 10.0, 20.0, 29.0]
REPEATS = 3
OFFSET = 300.0  # с начала записи, чтобы попасть в живую речь


def cut(media: Path, out: Path, start: float, duration: float) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{start}", "-t", f"{duration}", "-i", str(media),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out),
        ],
        check=True,
    )
    return out


def main() -> None:
    media = Path(sys.argv[1]).expanduser().resolve()
    generate = import_module("mlx_audio.stt.generate")
    utils = import_module("mlx_audio.stt.utils")

    load_started = time.monotonic()
    model = utils.load_model("mlx-community/whisper-large-v3-turbo-asr-fp16")
    print(f"загрузка модели: {time.monotonic() - load_started:.1f} с\n")

    with tempfile.TemporaryDirectory(prefix="bench-window-") as tmp:
        clips = {
            d: cut(media, Path(tmp) / f"clip-{d}.wav", OFFSET, d) for d in DURATIONS
        }
        # Прогрев: первый вызов всегда дороже из-за компиляции графа.
        generate.generate_transcription(
            model=model, audio=str(clips[5.0]), language="ru",
            word_timestamps=False, condition_on_previous_text=False, verbose=None,
        )

        print(f"{'окно, с':>8} {'со словами':>12} {'без слов':>10} {'× реалтайма':>12}")
        rows = []
        for d in DURATIONS:
            times = {}
            for words in (True, False):
                runs = []
                for _ in range(REPEATS):
                    started = time.monotonic()
                    generate.generate_transcription(
                        model=model, audio=str(clips[d]), language="ru",
                        word_timestamps=words, condition_on_previous_text=False,
                        verbose=None,
                    )
                    runs.append(time.monotonic() - started)
                times[words] = statistics.median(runs)
            rows.append((d, times[True], times[False]))
            print(f"{d:>8.1f} {times[True]:>12.2f} {times[False]:>10.2f} {times[True]/d:>12.1f}")

    print("\nЕсли столбец «со словами» почти не растёт с длиной окна — цена")
    print("фиксирована на окно, и упаковка коротких окон в длинные экономит время.")


if __name__ == "__main__":
    main()
