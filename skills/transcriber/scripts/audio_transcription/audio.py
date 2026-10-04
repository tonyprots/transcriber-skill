from __future__ import annotations

import json
import os
import hashlib
import shutil
import subprocess
import wave
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .models import AudioChunk, MediaInfo, Section


SAMPLE_RATE = 16_000


class MediaToolError(RuntimeError):
    """Ошибка подготовки медиафайла."""


def require_media_tools() -> tuple[str, str]:
    ffmpeg = find_command("ffmpeg")
    ffprobe = find_command("ffprobe")
    if not ffmpeg or not ffprobe:
        raise MediaToolError(
            "Нужны ffmpeg и ffprobe. Установите ffmpeg и повторите запуск."
        )
    return ffmpeg, ffprobe


def find_command(name: str) -> str | None:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    for directory in (Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def probe_media(source: Path) -> MediaInfo:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Медиафайл не найден: {source}")
    _, ffprobe = require_media_tools()
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:format=duration",
        "-of",
        "json",
        str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise MediaToolError(completed.stderr.strip() or f"ffprobe не прочитал {source}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise MediaToolError(f"В файле нет аудиодорожки: {source}")
    stream = streams[0]
    duration_raw = payload.get("format", {}).get("duration")
    digest = hashlib.sha256()
    with source.open("rb") as stream_handle:
        for block in iter(lambda: stream_handle.read(1024 * 1024), b""):
            digest.update(block)
    return MediaInfo(
        source=source,
        duration_seconds=float(duration_raw) if duration_raw is not None else None,
        codec=stream.get("codec_name"),
        sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        channels=int(stream["channels"]) if stream.get("channels") else None,
        size_bytes=source.stat().st_size,
        sha256=digest.hexdigest(),
    )


def prepare_audio(
    source: Path, work_dir: Path, section: Section | None = None
) -> tuple[Path, MediaInfo]:
    """Создаёт нормализованный WAV PCM16/16 kHz/mono, не меняя исходник.

    С `section` в WAV попадает только кусок, а исходник по-прежнему
    опознаётся по SHA-256 целиком. Резать здесь, а не при скачивании: один
    файл обслуживает сколько угодно фрагментов, а `yt-dlp --download-sections`
    качал бы каждый заново и 2026-09-22 падал на ffmpeg с кодом 8.
    """
    info = probe_media(source)
    trim: list[str] = []
    if section is not None:
        total = info.duration_seconds
        if total is not None and section.start >= total:
            raise ValueError(
                f"Фрагмент {section.label} начинается после конца записи "
                f"({total:.0f} с)"
            )
        if total is not None and section.end > total:
            section = Section(section.start, total)
        trim = ["-ss", f"{section.start:.3f}", "-to", f"{section.end:.3f}"]
        info = replace(info, duration_seconds=section.end - section.start, section=section)
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared = work_dir / "prepared.wav"
    ffmpeg, _ = require_media_tools()
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *trim,
        "-i",
        str(info.source),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(prepared),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not prepared.is_file():
        raise MediaToolError(completed.stderr.strip() or "ffmpeg не подготовил WAV")
    return prepared, info


def pack_speech_spans(
    spans: list[tuple[int, int]],
    waveform_samples: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_seconds: float = 20.0,
    pad_seconds: float = 0.2,
) -> list[tuple[int, int]]:
    """Пакует соседние VAD-интервалы в общие для всех ASR контекстные окна."""
    if not spans:
        return []
    limit = round(max_seconds * sample_rate)
    pad = round(pad_seconds * sample_rate)
    packed: list[tuple[int, int]] = []
    group_start, group_end = spans[0]
    for start, end in spans[1:]:
        if end - group_start <= limit:
            group_end = end
            continue
        packed.append((max(0, group_start - pad), min(waveform_samples, group_end + pad)))
        group_start, group_end = start, end
    packed.append((max(0, group_start - pad), min(waveform_samples, group_end + pad)))
    return packed


@lru_cache(maxsize=2)
def _pcm16(path: Path) -> object:
    """Весь подготовленный WAV в памяти, один раз на процесс.

    Окна раньше нарезались на диск отдельными WAV, по одному на окно, склейку
    и ретрай. Антивирус проверяет каждое создание и удаление файла, и VAD
    часовой записи растягивался с 5 секунд до пяти минут
    (`.memory/defender-slow-temp-cleanup.md`). Теперь окно — это отрезок
    `prepared.wav`, а звук режется срезом массива. Час записи — 115 МБ int16.
    """
    import numpy as np

    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
            1,
            2,
            SAMPLE_RATE,
        ):
            raise MediaToolError(f"Ожидался PCM16 mono {SAMPLE_RATE} Hz: {path}")
        frames = source.readframes(source.getnframes())
    return np.frombuffer(frames, dtype="<i2")


def _frame_bounds(chunk: AudioChunk, total: int) -> tuple[int, int]:
    first = max(0, min(total, round(chunk.start * SAMPLE_RATE)))
    return first, max(first, min(total, round(chunk.end * SAMPLE_RATE)))


def chunk_waveform(chunk: AudioChunk) -> object:
    """Звук окна как float32 16 kHz — то, что принимают все три рантайма ASR."""
    import numpy as np

    samples = _pcm16(chunk.path.resolve())
    first, last = _frame_bounds(chunk, len(samples))
    return samples[first:last].astype(np.float32) / 32768.0


def split_audio_chunk(chunk: AudioChunk) -> tuple[AudioChunk, AudioChunk]:
    """Делит окно пополам по сэмплу, сохраняя абсолютные таймкоды родителя."""
    first = round(chunk.start * SAMPLE_RATE)
    last = round(chunk.end * SAMPLE_RATE)
    if last - first < 2:
        raise MediaToolError(f"Окно слишком короткое для retry-split: {chunk.start:.3f} с")
    boundary = (first + last) // 2 / SAMPLE_RATE
    return (
        AudioChunk(chunk.sequence, chunk.path, chunk.start, boundary, chunk.speakers),
        AudioChunk(chunk.sequence, chunk.path, boundary, chunk.end, chunk.speakers),
    )


VAD_TARGET_RMS = 0.1  # −20 dBFS: обычный уровень речи в записи
VAD_MAX_GAIN = 1000.0  # +60 dB: дальний микрофон даёт −60…−76 dBFS
VAD_ENVELOPE_SECONDS = 2.0


def gain_for_vad(waveform):
    """Поднимает тихие участки до уровня речи — только для Silero VAD.

    Silero принимает речь на −60…−76 dBFS за тишину, и фраза не доходит до
    моделей, хотя GigaAM её узнаёт: на Golos farfield 2026-10-04 пустыми
    вышли 82 фразы из 500. Усиление скользящее, по огибающей ±2 с: общий
    множитель на файл не поднимает тихого собеседника рядом с громким
    (480 фраз из 500 против 492). Огибающая — максимум, а не среднее, чтобы
    фраза поднималась целиком, а не по слогам. Цифровой ноль остаётся нулём,
    громкое не ослабляется. Модели получают звук без изменений.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    hop = SAMPLE_RATE * 30 // 1000
    frames = len(waveform) // hop
    if frames == 0:
        return waveform
    rms = np.sqrt((waveform[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    reach = max(1, round(VAD_ENVELOPE_SECONDS * SAMPLE_RATE / hop))
    envelope = sliding_window_view(np.pad(rms, reach, mode="edge"), 2 * reach + 1).max(axis=1)
    gain = np.clip(VAD_TARGET_RMS / np.maximum(envelope, 1e-6), 1.0, VAD_MAX_GAIN)
    gain = np.repeat(gain, hop)
    gain = np.pad(gain, (0, len(waveform) - len(gain)), mode="edge")
    return np.clip(waveform * gain, -1.0, 1.0).astype(waveform.dtype)


def split_speech_windows(
    prepared: Path,
    *,
    max_seconds: float = 20.0,
    pad_seconds: float = 0.2,
    pack: bool = True,
    offline: bool = False,
) -> list[AudioChunk]:
    """Один раз применяет Silero VAD и создаёт одинаковые окна для всех моделей.

    VAD — первый поход в сеть за весь прогон, и раньше единственный, который
    шёл мимо `load_with_hub_fallback`: 2026-09-17 прогон встал здесь на
    неотвечающем Hub при полном кэше, а `--offline` до этого места не доходил
    вовсе. Теперь загрузка живёт по общим правилам.
    """
    from .backends import load_with_hub_fallback  # backends зависит от audio через retry

    try:
        import onnx_asr
        from onnx_asr.utils import read_wav_files
    except ImportError as error:
        raise MediaToolError("Не установлен onnx-asr с Silero VAD") from error

    waveforms, lengths, sample_rate = read_wav_files(str(prepared), SAMPLE_RATE, "mean")
    if sample_rate != SAMPLE_RATE:
        raise MediaToolError(f"Ожидалось {SAMPLE_RATE} Hz, получено {sample_rate} Hz")
    from .catalog import SILERO_VAD
    from .weights import fetch

    vad = load_with_hub_fallback(
        lambda: onnx_asr.load_vad(
            "silero", fetch(SILERO_VAD).path, providers=["CPUExecutionProvider"]
        ),
        offline=offline,
        label="Silero VAD",
    )
    raw_spans = list(
        next(
            vad.segment_batch(
                gain_for_vad(waveforms[0])[None, :],
                lengths,
                SAMPLE_RATE,
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_s=max_seconds,
                min_silence_duration_ms=300,
                speech_pad_ms=0,
            )
        )
    )
    spans = (
        pack_speech_spans(
            raw_spans,
            int(lengths[0]),
            max_seconds=max_seconds,
            pad_seconds=pad_seconds,
        )
        if pack
        else raw_spans
    )
    return [
        AudioChunk(
            sequence=sequence,
            path=prepared,
            start=start / SAMPLE_RATE,
            end=end / SAMPLE_RATE,
        )
        for sequence, (start, end) in enumerate(spans)
    ]
