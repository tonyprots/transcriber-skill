import pytest

from audio_transcription.routing import diarization_backend, language_route


def test_russian_route_preserves_existing_primary() -> None:
    route = language_route("ru-RU")
    assert route.primary.family == "gigaam"
    assert route.primary.model == "gigaam-v3-e2e-rnnt"
    assert route.verifier is not None
    assert route.verifier.family == "whisper"
    assert route.locally_calibrated is True


def test_english_route_uses_independent_fast_verifier() -> None:
    route = language_route("en_US")
    assert route.primary.family == "whisper"
    assert route.verifier is not None
    assert route.verifier.model == "gigaam-multilingual-ctc"
    assert route.verifier.quantization == "int8"


def test_uncalibrated_language_is_whisper_only_with_warning() -> None:
    route = language_route("kk")
    assert route.primary.family == "whisper"
    assert route.verifier is None
    assert route.locally_calibrated is False
    assert "не откалиброван" in str(route.warning)


def test_large_meeting_selects_clustering_backend() -> None:
    assert diarization_backend("auto", None) == "sortformer"
    assert diarization_backend("auto", 4) == "sortformer"
    assert diarization_backend("auto", 6) == "fluidaudio"


def test_sortformer_rejects_known_large_meeting() -> None:
    with pytest.raises(ValueError, match="не более четырёх"):
        diarization_backend("sortformer", 6)
