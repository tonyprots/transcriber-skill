#!/usr/bin/env python3
"""Где на самом деле уходит время: замер на настоящих окнах пайплайна.

Синтетический бенчмарк говорит, что GigaAM считает 0.06× реалтайма, а в
прогоне вышло 2.25 с на окно. Разница где-то между нарезкой окон, чтением
файлов и обёрткой retry — здесь мы её локализуем.

Запуск: .venv/bin/python experiments/window-cost/profile_pipeline.py <media> [N]
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "skills" / "transcriber" / "scripts"),
)

from audio_transcription.audio import prepare_audio, split_speech_windows  # noqa: E402
from audio_transcription.backends import GigaAMBackend  # noqa: E402
from audio_transcription.retry import recognize_with_retry  # noqa: E402


class Timer:
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self):
        self.started = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.monotonic() - self.started
        print(f"{self.label}: {self.elapsed:.1f} с")
        return False


def main() -> None:
    media = Path(sys.argv[1]).expanduser().resolve()
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    with tempfile.TemporaryDirectory(prefix="profile-") as tmp:
        work = Path(tmp)
        with Timer("подготовка аудио"):
            prepared, media_info = prepare_audio(media, work)
        print(f"  длительность {media_info.duration_seconds:.0f} с")

        with Timer("VAD + нарезка окон, pack=False (как при --diarize)") as t_split:
            unpacked = split_speech_windows(prepared, work / "unpacked", pack=False)
        print(f"  окон: {len(unpacked)}, "
              f"на окно {t_split.elapsed / len(unpacked) * 1000:.0f} мс")

        with Timer("VAD + нарезка окон, pack=True") as t_packed:
            packed = split_speech_windows(prepared, work / "packed", pack=True)
        print(f"  окон: {len(packed)}, "
              f"на окно {t_packed.elapsed / len(packed) * 1000:.0f} мс")

        sample = unpacked[:limit]
        audio_seconds = sum(c.duration for c in sample)
        print(f"\nвыборка: {len(sample)} окон, {audio_seconds:.0f} с аудио")

        import onnx_asr

        model = onnx_asr.load_model("gigaam-v3-e2e-rnnt", providers=["CPUExecutionProvider"])
        model.recognize(str(sample[0].path))  # прогрев

        with Timer("  голый model.recognize по одному") as t_raw:
            for chunk in sample:
                model.recognize(str(chunk.path))
        print(f"    {t_raw.elapsed / len(sample):.2f} с на окно, "
              f"{t_raw.elapsed / audio_seconds:.3f}× реалтайма")

        retry_dir = work / "retry"
        with Timer("  через recognize_with_retry") as t_retry:
            for chunk in sample:
                recognize_with_retry(
                    chunk,
                    lambda part: {"text": str(model.recognize(str(part.path))).strip()},
                    retry_dir,
                )
        print(f"    {t_retry.elapsed / len(sample):.2f} с на окно")

        with Timer("  полный backend.transcribe_chunks") as t_backend:
            GigaAMBackend("gigaam-v3-e2e-rnnt").transcribe_chunks(sample)
        print(f"    {t_backend.elapsed / len(sample):.2f} с на окно")

        packed_sample = [c for c in packed if c.start < sample[-1].end]
        packed_audio = sum(c.duration for c in packed_sample)
        with Timer(f"\nтот же кусок записи упакованными окнами ({len(packed_sample)} шт.)") as t_pack:
            for chunk in packed_sample:
                model.recognize(str(chunk.path))
        print(f"    {packed_audio:.0f} с аудио, "
              f"{t_pack.elapsed / packed_audio:.3f}× реалтайма")


if __name__ == "__main__":
    main()
