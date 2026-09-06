from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .backends import BackendMissing, BackendUnavailable, GigaAMBackend, MLXWhisperBackend, WhisperBackend
from .models import AudioChunk, Hypothesis
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


def _chunks(payload: dict[str, Any]) -> list[AudioChunk]:
    return [AudioChunk.from_dict(item) for item in payload.get("chunks", [])]


def _transcribe(payload: dict[str, Any], progress_path: Path) -> Hypothesis:
    backend = str(payload["backend"])
    language = str(payload.get("language", "ru"))
    config = dict(payload.get("config", {}))
    chunks = _chunks(payload)

    def progress(completed: int, total: int) -> None:
        _write_json(progress_path, {"completed": completed, "total": total})

    progress(0, len(chunks))
    if backend == "gigaam":
        model = GigaAMBackend(
            str(config["model_name"]),
            quantization=(
                str(config["quantization"])
                if config.get("quantization") is not None
                else None
            ),
            providers=tuple(config.get("providers", ["CPUExecutionProvider"])),
            offline=bool(config.get("offline", False)),
        )
        hypothesis = model.transcribe_chunks(chunks, language, progress=progress)
    elif backend == "mlx-whisper":
        model = MLXWhisperBackend(
            str(config["model_name"]),
            offline=bool(config.get("offline", False)),
        )
        hypothesis = model.transcribe_chunks(
            chunks,
            language,
            hotwords=list(config.get("hotwords", [])),
            progress=progress,
        )
    elif backend == "faster-whisper":
        model = WhisperBackend(
            str(config["model_name"]),
            offline=bool(config.get("offline", False)),
            beam_size=int(config.get("beam_size", 5)),
        )
        hypothesis = model.transcribe_chunks(
            chunks,
            language,
            hotwords=list(config.get("hotwords", [])),
            progress=progress,
        )
    else:
        raise BackendUnavailable(f"Неизвестный ASR-бэкенд: {backend}")
    hypothesis.metadata["process_isolated"] = True
    return hypothesis


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3:
        print("Ожидались request.json, result.json и progress.json", file=sys.stderr)
        return 2
    request_path, result_path, progress_path = map(Path, arguments)
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        if int(payload.get("schema", 0)) != 1:
            raise ValueError("Неподдерживаемая схема ASR-задания")
        hypothesis = _transcribe(payload, progress_path)
        _write_json(result_path, hypothesis.to_dict())
    except BackendMissing as error:
        # Код 3: бэкенда нет на машине. Только на него родитель уходит в fallback.
        print(str(error), file=sys.stderr)
        return 3
    except (BackendUnavailable, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception as error:  # граница отдельного процесса должна вернуть понятную причину
        print(f"ASR-процесс завершился с ошибкой: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    exit_after_flush(main())
