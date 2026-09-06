from __future__ import annotations

import os
import time
from importlib import import_module
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .models import AudioChunk, Hypothesis, Segment
from .retry import recognize_with_retry


ProgressCallback = Callable[[int, int], None]


class BackendUnavailable(RuntimeError):
    """Запрошенный ASR-бэкенд недоступен."""


class BackendMissing(BackendUnavailable):
    """Бэкенда нет на этой машине (нет библиотеки или платформы).

    Отличается от сбоя в работе: только на это допустим переход на CPU-fallback.
    Сбой на трёхсотом окне или срабатывание watchdog — не повод молча начинать
    те же часы работы на медленной модели.
    """


def force_hub_offline() -> None:
    """Переводит huggingface_hub в офлайн для текущего процесса.

    Переменной окружения мало: константу huggingface_hub читает при импорте,
    а библиотеки моделей импортируют его раньше нас. Поэтому правим и её.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        constants = import_module("huggingface_hub.constants")
        constants.HF_HUB_OFFLINE = True
    except ImportError:
        pass


def load_with_hub_fallback(load: Callable[[], Any], *, offline: bool, label: str) -> Any:
    """Грузит модель; если Hub не отвечает, а веса уже в кэше — повторяет офлайн.

    Бенчмарк 2026-09-06: Sortformer упал с «Server disconnected without
    sending a response» при полностью скачанной модели — библиотека сходила
    на Hub проверить обновления и не дождалась ответа. Без сети должна
    работать любая модель, которая уже была загружена.
    """
    if offline:
        force_hub_offline()
        return load()
    try:
        return load()
    except Exception as online_error:  # ошибки сети у библиотек разные
        force_hub_offline()
        try:
            return load()
        except Exception as offline_error:
            raise BackendUnavailable(
                f"{label}: не удалось загрузить модель ни с Hub, ни из локального кэша. "
                f"Hub: {online_error}. Кэш: {offline_error}"
            ) from offline_error


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _confidence(item: Any) -> float | None:
    for name in ("confidence", "probability", "score"):
        value = _value(item, name)
        if value is not None:
            try:
                number = float(value)
                return max(0.0, min(1.0, number))
            except (TypeError, ValueError):
                pass
    return None


def _segments(items: Iterable[Any]) -> list[Segment]:
    result: list[Segment] = []
    for item in items:
        text = str(_value(item, "text", "")).strip()
        if not text:
            continue
        start = float(_value(item, "start", 0.0) or 0.0)
        end = float(_value(item, "end", start) or start)
        result.append(
            Segment(
                start=max(0.0, start),
                end=max(max(0.0, start), end),
                text=text,
                confidence=_confidence(item),
            )
        )
    return result


class GigaAMBackend:
    def __init__(
        self,
        model_name: str,
        *,
        quantization: str | None = None,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        offline: bool = False,
    ) -> None:
        self.model_name = model_name
        self.quantization = quantization
        self.providers = list(providers)
        self.offline = offline

    def transcribe_chunks(
        self,
        chunks: list[AudioChunk],
        language: str = "ru",
        *,
        progress: ProgressCallback | None = None,
    ) -> Hypothesis:
        if self.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if not chunks:
            return Hypothesis(
                self.model_name,
                language,
                0.0,
                [],
                {"vad": "shared-silero", "chunks": []},
            )
        try:
            import onnx_asr
        except ImportError as exc:
            raise BackendMissing("Не установлен onnx-asr. Установите зависимости скилла.") from exc

        started = time.monotonic()
        model = load_with_hub_fallback(
            lambda: onnx_asr.load_model(
                self.model_name,
                quantization=self.quantization,
                providers=self.providers,
            ),
            offline=self.offline,
            label=self.model_name,
        )
        try:
            if progress:
                progress(0, len(chunks))
            segments: list[Segment] = []
            chunk_results: list[dict[str, Any]] = []
            retry_dir = chunks[0].path.parent / "retry-gigaam"
            for index, chunk in enumerate(chunks, start=1):
                result = recognize_with_retry(
                    chunk,
                    lambda part: {"text": str(model.recognize(str(part.path))).strip()},
                    retry_dir,
                )
                text = str(result["text"])
                chunk_results.append(result)
                if text:
                    segments.append(
                        Segment(chunk.start, chunk.end, text, speakers=chunk.speakers)
                    )
                if progress:
                    progress(index, len(chunks))
        except Exception as exc:
            raise BackendUnavailable(f"Не удалось запустить {self.model_name}: {exc}") from exc
        return Hypothesis(
            model=self.model_name,
            language=language,
            elapsed_seconds=time.monotonic() - started,
            segments=segments,
            metadata={
                "vad": "shared-silero",
                "quantization": self.quantization,
                "chunks": chunk_results,
            },
        )


class WhisperBackend:
    """Независимая проверка через faster-whisper с временными метками слов."""

    def __init__(
        self,
        model_name: str = "medium",
        *,
        offline: bool = False,
        beam_size: int = 5,
    ) -> None:
        self.model_name = model_name
        self.offline = offline
        self.beam_size = beam_size

    def transcribe_chunks(
        self,
        chunks: list[AudioChunk],
        language: str = "ru",
        *,
        hotwords: list[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> Hypothesis:
        if self.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if not chunks:
            return Hypothesis(
                f"faster-whisper-{self.model_name}",
                language,
                0.0,
                [],
                {
                    "vad": "shared-silero",
                    "role": "independent_verifier_cpu_fallback",
                    "chunks": [],
                },
            )
        try:
            module = import_module("faster_whisper")
            model_class = module.WhisperModel
        except (ImportError, AttributeError) as exc:
            raise BackendMissing("Не установлен faster-whisper для CPU-fallback.") from exc
        started = time.monotonic()
        model = load_with_hub_fallback(
            lambda: model_class(
                self.model_name,
                device="cpu",
                compute_type="int8",
                local_files_only=self.offline or os.environ.get("HF_HUB_OFFLINE") == "1",
            ),
            offline=self.offline,
            label=f"faster-whisper {self.model_name}",
        )
        try:
            if progress:
                progress(0, len(chunks))
            segments: list[Segment] = []
            chunk_results: list[dict[str, Any]] = []
            detected = language
            retry_dir = chunks[0].path.parent / "retry-faster-whisper"

            def recognize(part: AudioChunk) -> dict[str, Any]:
                raw_segments, info = model.transcribe(
                    str(part.path),
                    language=language,
                    beam_size=self.beam_size,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    hotwords=", ".join(hotwords) if hotwords else None,
                )
                text_parts: list[str] = []
                words_meta: list[dict[str, Any]] = []
                confidences: list[float] = []
                for phrase in raw_segments:
                    phrase_text = str(getattr(phrase, "text", "")).strip()
                    if phrase_text:
                        text_parts.append(phrase_text)
                    for word in getattr(phrase, "words", None) or []:
                        word_text = str(getattr(word, "word", "")).strip()
                        if not word_text:
                            continue
                        confidence = _confidence(word)
                        if confidence is not None:
                            confidences.append(confidence)
                        words_meta.append(
                            {
                                "start": part.start + max(0.0, float(word.start)),
                                "end": part.start + max(0.0, float(word.end)),
                                "text": word_text,
                                "confidence": confidence,
                            }
                        )
                return {
                    "text": " ".join(text_parts).strip(),
                    "words": words_meta,
                    "confidence": (
                        sum(confidences) / len(confidences) if confidences else None
                    ),
                    "language": getattr(info, "language", None) or language,
                }

            for index, chunk in enumerate(chunks, start=1):
                result = recognize_with_retry(chunk, recognize, retry_dir)
                text = str(result["text"])
                confidence = result.get("confidence")
                if confidence is None:
                    values = [
                        float(word["confidence"])
                        for word in result.get("words", [])
                        if word.get("confidence") is not None
                    ]
                    confidence = sum(values) / len(values) if values else None
                if text:
                    segments.append(
                        Segment(
                            chunk.start,
                            chunk.end,
                            text,
                            confidence,
                            chunk.speakers,
                        )
                    )
                chunk_results.append(result)
                detected = result.get("language") or detected
                if progress:
                    progress(index, len(chunks))
        except Exception as exc:
            raise BackendUnavailable(f"Не удалось запустить Whisper {self.model_name}: {exc}") from exc
        return Hypothesis(
            model=f"faster-whisper-{self.model_name}",
            language=str(detected),
            elapsed_seconds=time.monotonic() - started,
            segments=segments,
            metadata={
                "vad": "shared-silero",
                "beam_size": self.beam_size,
                "word_timestamps": True,
                "role": "independent_verifier_cpu_fallback",
                "hotwords": hotwords or [],
                "chunks": chunk_results,
            },
        )


class MLXWhisperBackend:
    """Независимая проверка Whisper Turbo, оптимизированная для Apple Silicon."""

    def __init__(
        self,
        model_name: str = "mlx-community/whisper-large-v3-turbo-asr-fp16",
        *,
        offline: bool = False,
    ) -> None:
        self.model_name = model_name
        self.offline = offline

    def transcribe_chunks(
        self,
        chunks: list[AudioChunk],
        language: str = "ru",
        *,
        hotwords: list[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> Hypothesis:
        if self.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if not chunks:
            return Hypothesis(
                self.model_name,
                language,
                0.0,
                [],
                {"vad": "shared-silero", "role": "independent_verifier", "chunks": []},
            )
        try:
            generate = import_module("mlx_audio.stt.generate")
            utils = import_module("mlx_audio.stt.utils")
        except (ImportError, AttributeError, RuntimeError) as exc:
            raise BackendMissing(
                f"Whisper Turbo MLX недоступен на этой машине: {exc}"
            ) from exc
        model = load_with_hub_fallback(
            lambda: utils.load_model(self.model_name),
            offline=self.offline,
            label="Whisper Turbo MLX",
        )

        started = time.monotonic()
        segments: list[Segment] = []
        chunk_results: list[dict[str, Any]] = []
        detected = language
        try:
            if progress:
                progress(0, len(chunks))
            retry_dir = chunks[0].path.parent / "retry-mlx-whisper"

            def recognize(part: AudioChunk) -> dict[str, Any]:
                answer = generate.generate_transcription(
                    model=model,
                    audio=str(part.path),
                    language=language,
                    hotwords=hotwords or None,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    verbose=None,
                )
                text = str(_value(answer, "text", "")).strip()
                answer_language = _value(answer, "language", language) or language
                raw_segments = _value(answer, "segments", []) or []
                words_meta = []
                confidences = []
                segment_metrics = []
                for phrase in raw_segments:
                    words = _value(phrase, "words", []) or []
                    for word in words:
                        word_text = str(_value(word, "word", "")).strip()
                        if not word_text:
                            continue
                        confidence = _confidence(word)
                        if confidence is not None:
                            confidences.append(confidence)
                        words_meta.append(
                            {
                                "start": part.start + max(0.0, float(_value(word, "start", 0.0))),
                                "end": part.start + max(0.0, float(_value(word, "end", 0.0))),
                                "text": word_text,
                                "confidence": confidence,
                            }
                        )
                    segment_metrics.append(
                        {
                            "avg_logprob": _value(phrase, "avg_logprob"),
                            "compression_ratio": _value(phrase, "compression_ratio"),
                            "no_speech_prob": _value(phrase, "no_speech_prob"),
                        }
                    )
                return {
                    "text": text,
                    "words": words_meta,
                    "segment_metrics": segment_metrics,
                    "confidence": (
                        sum(confidences) / len(confidences) if confidences else None
                    ),
                    "language": answer_language,
                }

            for index, chunk in enumerate(chunks, start=1):
                result = recognize_with_retry(chunk, recognize, retry_dir)
                text = str(result["text"])
                confidence = result.get("confidence")
                if confidence is None:
                    values = [
                        float(word["confidence"])
                        for word in result.get("words", [])
                        if word.get("confidence") is not None
                    ]
                    confidence = sum(values) / len(values) if values else None
                if text:
                    segments.append(
                        Segment(
                            chunk.start,
                            chunk.end,
                            text,
                            confidence,
                            chunk.speakers,
                        )
                    )
                chunk_results.append(result)
                detected = result.get("language") or detected
                if progress:
                    progress(index, len(chunks))
        except Exception as exc:
            raise BackendUnavailable(f"Не удалось запустить Whisper Turbo MLX: {exc}") from exc
        return Hypothesis(
            model=self.model_name,
            language=str(detected),
            elapsed_seconds=time.monotonic() - started,
            segments=segments,
            metadata={
                "vad": "shared-silero",
                "word_timestamps": True,
                "role": "independent_verifier",
                "hotwords": hotwords or [],
                "chunks": chunk_results,
            },
        )
