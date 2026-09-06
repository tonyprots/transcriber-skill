from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from .backends import BackendMissing, load_with_hub_fallback
from .diarization import partition_turns
from .models import Diarization, SpeakerTurn
from .exiting import exit_after_flush


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fluid_binary(config: dict[str, Any]) -> Path:
    candidates = [
        config.get("binary"),
        os.environ.get("TRANSCRIBER_FLUIDAUDIO_BIN"),
        shutil.which("fluidaudiocli"),
        Path(__file__).resolve().parents[2] / "bin" / "macos-arm64" / "fluidaudiocli",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "Не найден fluidaudiocli. Укажите --fluidaudio-bin или установите "
        "бинарник в bin/macos-arm64/fluidaudiocli внутри скилла"
    )


def _run_sortformer(
    audio_path: Path,
    config: dict[str, Any],
    progress_path: Path,
) -> Diarization:
    try:
        from mlx_audio.vad import load
    except ImportError as error:
        raise BackendMissing(
            "Диаризация Sortformer требует mlx-audio на Apple Silicon"
        ) from error

    started = time.monotonic()
    _write_json(progress_path, {"stage": "loading", "completed": 0})
    model = load_with_hub_fallback(
        lambda: load(str(config["model_name"])),
        offline=bool(config.get("offline", False)),
        label="Sortformer",
    )
    _write_json(progress_path, {"stage": "loaded", "completed": 0})
    with wave.open(str(audio_path), "rb") as source:
        duration = source.getnframes() / source.getframerate()
    chunk_seconds = float(config.get("chunk_seconds", 30.0))
    total = max(1, int((duration + chunk_seconds - 1e-9) // chunk_seconds))
    raw_turns: list[SpeakerTurn] = []
    for index, output in enumerate(
        model.generate_stream(
            str(audio_path),
            chunk_duration=chunk_seconds,
            threshold=float(config.get("threshold", 0.4)),
            min_duration=float(config.get("min_duration", 0.25)),
            merge_gap=float(config.get("merge_gap", 0.2)),
        ),
        start=1,
    ):
        raw_turns.extend(
            SpeakerTurn(float(item.start), float(item.end), (str(item.speaker),))
            for item in output.segments
        )
        _write_json(
            progress_path,
            {"stage": "inference", "completed": index, "total": total},
        )
    turns = partition_turns(raw_turns)
    return Diarization(
        model=str(config["model_name"]),
        elapsed_seconds=time.monotonic() - started,
        turns=turns,
        metadata={
            "backend": "sortformer",
            "process_isolated": True,
            "threshold": float(config.get("threshold", 0.4)),
            "chunk_seconds": chunk_seconds,
            "max_speakers": 4,
            "raw_turns": [turn.to_dict() for turn in raw_turns],
        },
    )


def _run_fluidaudio(
    audio_path: Path,
    config: dict[str, Any],
    progress_path: Path,
    result_path: Path,
) -> Diarization:
    binary = _fluid_binary(config)
    output_path = result_path.with_name("fluidaudio-output.json")
    threshold = float(config.get("threshold", 0.8))
    started = time.monotonic()
    _write_json(progress_path, {"stage": "loading", "completed": 0})
    process = subprocess.Popen(
        [
            str(binary),
            "process",
            str(audio_path),
            "--mode",
            "offline",
            "--threshold",
            str(threshold),
            "--output",
            str(output_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    heartbeat = 0
    try:
        while process.poll() is None:
            heartbeat += 1
            _write_json(
                progress_path,
                {"stage": "inference", "completed": heartbeat, "total": None},
            )
            time.sleep(1.0)
        stdout, stderr = process.communicate()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or f"код {process.returncode}"
        raise RuntimeError(f"FluidAudio завершился с ошибкой: {detail}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    raw_turns = [
        SpeakerTurn(
            float(item["startTimeSeconds"]),
            float(item["endTimeSeconds"]),
            (str(item["speakerId"]),),
        )
        for item in payload.get("segments", [])
        if float(item["endTimeSeconds"]) > float(item["startTimeSeconds"])
    ]
    turns = partition_turns(raw_turns)
    quality_scores = [
        float(item["qualityScore"])
        for item in payload.get("segments", [])
        if item.get("qualityScore") is not None
    ]
    return Diarization(
        model="FluidAudio Community-1 offline",
        elapsed_seconds=time.monotonic() - started,
        turns=turns,
        metadata={
            "backend": "fluidaudio",
            "process_isolated": True,
            "threshold": threshold,
            "max_speakers": None,
            "reported_speaker_count": int(payload.get("speakerCount", len({speaker for turn in raw_turns for speaker in turn.speakers}))),
            "processing_seconds": payload.get("processingTimeSeconds"),
            "average_quality_score": (
                sum(quality_scores) / len(quality_scores) if quality_scores else None
            ),
            "raw_turns": [turn.to_dict() for turn in raw_turns],
        },
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3:
        print("Ожидались request.json, result.json и progress.json", file=sys.stderr)
        return 2
    request_path, result_path, progress_path = map(Path, arguments)
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        config = dict(payload["config"])
        audio_path = Path(payload["audio_path"])
        backend = str(payload.get("backend", "sortformer"))
        if bool(config.get("offline", False)):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if backend == "sortformer":
            diarization = _run_sortformer(audio_path, config, progress_path)
        elif backend == "fluidaudio":
            diarization = _run_fluidaudio(
                audio_path,
                config,
                progress_path,
                result_path,
            )
        else:
            raise ValueError(f"Неизвестный backend диаризации: {backend}")
        _write_json(result_path, diarization.to_dict())
    except BackendMissing as error:
        print(str(error), file=sys.stderr)
        return 3
    except Exception as error:
        print(f"Диаризация завершилась с ошибкой: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    exit_after_flush(main())
