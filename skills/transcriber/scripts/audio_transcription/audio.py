from __future__ import annotations

import json
import os
import hashlib
import shutil
import subprocess
import wave
from pathlib import Path

from .models import AudioChunk, MediaInfo


SAMPLE_RATE = 16_000


class MediaToolError(RuntimeError):
    """Ошибка подготовки медиафайла."""


def require_media_tools() -> tuple[str, str]:
    ffmpeg = _find_command("ffmpeg")
    ffprobe = _find_command("ffprobe")
    if not ffmpeg or not ffprobe:
        raise MediaToolError(
            "Нужны ffmpeg и ffprobe. Установите ffmpeg и повторите запуск."
        )
    return ffmpeg, ffprobe


def _find_command(name: str) -> str | None:
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


def prepare_audio(source: Path, work_dir: Path) -> tuple[Path, MediaInfo]:
    """Создаёт нормализованный WAV PCM16/16 kHz/mono, не меняя исходник."""
    info = probe_media(source)
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


def _write_pcm16(path: Path, samples: object) -> None:
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def split_audio_chunk(
    chunk: AudioChunk,
    output_dir: Path,
    *,
    label: str,
) -> tuple[AudioChunk, AudioChunk]:
    """Делит PCM WAV пополам, сохраняя абсолютные таймкоды родительского окна."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(chunk.path), "rb") as source:
        params = source.getparams()
        frame_count = source.getnframes()
        frames = source.readframes(frame_count)
    if frame_count < 2:
        raise MediaToolError(f"Окно слишком короткое для retry-split: {chunk.path}")

    frame_size = params.nchannels * params.sampwidth
    midpoint = frame_count // 2
    boundary = chunk.start + chunk.duration * midpoint / frame_count
    parts = (
        (0, midpoint, chunk.start, boundary),
        (midpoint, frame_count, boundary, chunk.end),
    )
    result: list[AudioChunk] = []
    for index, (start_frame, end_frame, start, end) in enumerate(parts):
        path = output_dir / f"chunk-{chunk.sequence:04d}-{label}-{index}.wav"
        with wave.open(str(path), "wb") as target:
            target.setparams(params)
            target.writeframes(frames[start_frame * frame_size : end_frame * frame_size])
        result.append(AudioChunk(chunk.sequence, path, start, end, chunk.speakers))
    return result[0], result[1]


def split_speech_windows(
    prepared: Path,
    work_dir: Path,
    *,
    max_seconds: float = 20.0,
    pad_seconds: float = 0.2,
    pack: bool = True,
) -> list[AudioChunk]:
    """Один раз применяет Silero VAD и создаёт одинаковые окна для всех моделей."""
    try:
        import onnx_asr
        from onnx_asr.utils import read_wav_files
    except ImportError as error:
        raise MediaToolError("Не установлен onnx-asr с Silero VAD") from error

    waveforms, lengths, sample_rate = read_wav_files(str(prepared), SAMPLE_RATE, "mean")
    if sample_rate != SAMPLE_RATE:
        raise MediaToolError(f"Ожидалось {SAMPLE_RATE} Hz, получено {sample_rate} Hz")
    vad = onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
    raw_spans = list(
        next(
            vad.segment_batch(
                waveforms,
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
    chunk_dir = work_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for sequence, (start, end) in enumerate(spans):
        path = chunk_dir / f"chunk-{sequence:04d}.wav"
        _write_pcm16(path, waveforms[0, start:end])
        chunks.append(
            AudioChunk(
                sequence=sequence,
                path=path,
                start=start / SAMPLE_RATE,
                end=end / SAMPLE_RATE,
            )
        )
    return chunks
