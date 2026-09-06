#!/usr/bin/env python3
"""Цена окна у основной модели: длина окна, батч, провайдер.

Проверяем три рычага:
  1. Растёт ли время с длиной окна (то же, что у Whisper).
  2. Даёт ли что-то батч вместо recognize по одному файлу.
  3. Быстрее ли CoreML, чем CPU: сейчас в коде жёстко CPUExecutionProvider,
     хотя CoreML на этой машине доступен.

Запуск: .venv/bin/python experiments/window-cost/bench_gigaam.py <media>
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODEL = "gigaam-v3-e2e-rnnt"
DURATIONS = [2.5, 5.0, 10.0, 20.0]
REPEATS = 3
OFFSET = 300.0
BATCH_CLIPS = 16  # столько окон гоняем пачкой против того же числа по одному


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


def load(providers: list[str]):
    import onnx_asr

    started = time.monotonic()
    model = onnx_asr.load_model(MODEL, providers=providers)
    return model, time.monotonic() - started


def main() -> None:
    media = Path(sys.argv[1]).expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="bench-gigaam-") as tmp:
        work = Path(tmp)
        clips = {d: cut(media, work / f"clip-{d}.wav", OFFSET, d) for d in DURATIONS}
        batch = [
            str(cut(media, work / f"batch-{i}.wav", OFFSET + i * 6, 5.0))
            for i in range(BATCH_CLIPS)
        ]

        for providers in (["CPUExecutionProvider"], ["CoreMLExecutionProvider", "CPUExecutionProvider"]):
            label = providers[0].replace("ExecutionProvider", "")
            try:
                model, load_seconds = load(providers)
            except Exception as exc:  # noqa: BLE001
                print(f"\n{label}: недоступен — {exc}")
                continue
            print(f"\n=== {label} (загрузка {load_seconds:.1f} с) ===")
            model.recognize(str(clips[5.0]))  # прогрев

            print(f"{'окно, с':>8} {'время, с':>10} {'× реалтайма':>12}")
            for d in DURATIONS:
                runs = []
                for _ in range(REPEATS):
                    started = time.monotonic()
                    model.recognize(str(clips[d]))
                    runs.append(time.monotonic() - started)
                t = statistics.median(runs)
                print(f"{d:>8.1f} {t:>10.2f} {t / d:>12.2f}")

            started = time.monotonic()
            for path in batch:
                model.recognize(path)
            one_by_one = time.monotonic() - started

            batched = None
            for size in (8,):
                try:
                    started = time.monotonic()
                    model.recognize(batch, batch_size=size)
                    batched = time.monotonic() - started
                except Exception as exc:  # noqa: BLE001
                    print(f"  батч {size}: не поддержан — {exc}")
            print(f"  {BATCH_CLIPS} окон по 5 с поодиночке: {one_by_one:.1f} с")
            if batched is not None:
                print(f"  то же пачкой по 8:              {batched:.1f} с  "
                      f"(быстрее в {one_by_one / batched:.1f} раза)")


if __name__ == "__main__":
    main()
