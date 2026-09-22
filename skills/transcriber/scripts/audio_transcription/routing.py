from __future__ import annotations

from dataclasses import dataclass

from .catalog import BackendSpec
from .catalog import PARAKEET_ENGLISH as _PARAKEET_ENGLISH
from .catalog import GIGAAM_RUSSIAN as _GIGAAM_RUSSIAN
from .catalog import VOSK_RUSSIAN as _VOSK_RUSSIAN
from .catalog import WHISPER_TURBO as _WHISPER_TURBO

__all__ = [
    "BackendSpec",
    "LanguageRoute",
    "diarization_backend",
    "language_route",
]


@dataclass(frozen=True)
class LanguageRoute:
    identifier: str
    language: str
    primary: BackendSpec
    verifier: BackendSpec | None
    locally_calibrated: bool
    warning: str | None = None
    # Проверяющая для режима `fast`: дешёвая модель, которой хватает, чтобы
    # собрать очередь, но не хватает, чтобы править текст. Есть не у всех
    # маршрутов — только там, где её измерили.
    light_verifier: BackendSpec | None = None

    def verifier_for(self, mode: str) -> BackendSpec | None:
        return self.verifier if mode == "max" else self.light_verifier

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "language": self.language,
            "primary": self.primary.to_dict(),
            "verifier": self.verifier.to_dict() if self.verifier else None,
            "light_verifier": self.light_verifier.to_dict() if self.light_verifier else None,
            "locally_calibrated": self.locally_calibrated,
            "warning": self.warning,
        }


# Идентификаторы моделей живут в catalog.py вместе с датой калибровки —
# здесь только сборка маршрутов из них.
WHISPER_TURBO = _WHISPER_TURBO.spec
GIGAAM_RUSSIAN = _GIGAAM_RUSSIAN.spec
PARAKEET_ENGLISH = _PARAKEET_ENGLISH.spec
VOSK_RUSSIAN = _VOSK_RUSSIAN.spec


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
            light_verifier=VOSK_RUSSIAN,
        )
    if normalized == "en":
        return LanguageRoute(
            "en-whisper-turbo-parakeet-tdt-v3",
            "en",
            WHISPER_TURBO,
            PARAKEET_ENGLISH,
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
