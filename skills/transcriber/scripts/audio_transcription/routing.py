from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSpec:
    family: str
    model: str
    quantization: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "family": self.family,
            "model": self.model,
            "quantization": self.quantization,
        }


@dataclass(frozen=True)
class LanguageRoute:
    identifier: str
    language: str
    primary: BackendSpec
    verifier: BackendSpec | None
    locally_calibrated: bool
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "language": self.language,
            "primary": self.primary.to_dict(),
            "verifier": self.verifier.to_dict() if self.verifier else None,
            "locally_calibrated": self.locally_calibrated,
            "warning": self.warning,
        }


WHISPER_TURBO = BackendSpec(
    "whisper",
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
)
GIGAAM_RUSSIAN = BackendSpec("gigaam", "gigaam-v3-e2e-rnnt")
GIGAAM_MULTILINGUAL_FAST = BackendSpec(
    "gigaam",
    "gigaam-multilingual-ctc",
    "int8",
)


def normalize_language(language: str) -> str:
    normalized = language.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("--language не может быть пустым")
    return normalized.split("-", 1)[0]


def language_route(language: str) -> LanguageRoute:
    normalized = normalize_language(language)
    if normalized == "ru":
        return LanguageRoute(
            "ru-gigaam-v3e2e-whisper-turbo",
            "ru",
            GIGAAM_RUSSIAN,
            WHISPER_TURBO,
            True,
        )
    if normalized == "en":
        return LanguageRoute(
            "en-whisper-turbo-gigaam-multilingual-fast",
            "en",
            WHISPER_TURBO,
            GIGAAM_MULTILINGUAL_FAST,
            True,
        )
    warning = (
        f"Маршрут для языка '{normalized}' локально не откалиброван; "
        "используется только Whisper Turbo без независимой акустической проверки"
    )
    return LanguageRoute(
        f"{normalized}-whisper-turbo-unverified",
        normalized,
        WHISPER_TURBO,
        None,
        False,
        warning,
    )


def diarization_backend(requested: str, expected_speakers: int | None) -> str:
    if requested not in {"auto", "sortformer", "fluidaudio"}:
        raise ValueError(f"Неизвестный backend диаризации: {requested}")
    if expected_speakers is not None and expected_speakers < 1:
        raise ValueError("--expected-speakers должен быть положительным")
    if requested == "auto":
        return "fluidaudio" if expected_speakers and expected_speakers > 4 else "sortformer"
    if requested == "sortformer" and expected_speakers and expected_speakers > 4:
        raise ValueError(
            "Sortformer поддерживает не более четырёх голосов; "
            "используйте --diarization-backend fluidaudio"
        )
    return requested
