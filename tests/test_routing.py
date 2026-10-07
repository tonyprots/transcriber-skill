import pytest

from audio_transcription.routing import diarization_backend, language_route


def test_russian_route_preserves_existing_primary() -> None:
    route = language_route("ru-RU")
    assert route.primary.family == "gigaam"
    assert route.primary.model == "gigaam-v3-e2e-rnnt"
    assert route.verifier is not None
    assert route.verifier.family == "whisper"
    assert route.locally_calibrated is True


def test_english_route_verifies_with_a_model_of_its_own_weight() -> None:
    """Проверяющая должна быть сопоставима с основной, а не слабее её.

    GigaAM Multilingual int8 ошибался в полтора раза чаще Whisper и отправлял
    в очередь 98% окон: расхождение означало не сомнение, а разницу в классе.
    """
    route = language_route("en_US")
    assert route.primary.family == "whisper"
    assert route.verifier is not None
    assert route.verifier.model == "nemo-parakeet-tdt-0.6b-v3"
    assert route.verifier.family != route.primary.family


def test_fast_mode_verifies_russian_with_the_light_model() -> None:
    route = language_route("ru")
    assert route.verifier_for("max") is route.verifier
    light = route.verifier_for("fast")
    assert light is not None
    assert light.model == "alphacep/vosk-model-ru"
    # Лёгкая проверяющая обязана быть другого семейства: две родственные
    # модели независимой проверкой не являются.
    assert light.family != route.primary.family


def test_fast_mode_leaves_unmeasured_routes_without_verification() -> None:
    # Лёгкую проверяющую ставим только там, где её измерили; молча подставлять
    # русскую модель другому языку нельзя.
    assert language_route("en").verifier_for("fast") is None
    assert language_route("kk").verifier_for("fast") is None


def test_route_reports_both_verifiers() -> None:
    route = language_route("ru").to_dict()
    assert route["verifier"]["model"].startswith("mlx-community/")
    assert route["light_verifier"]["model"] == "alphacep/vosk-model-ru"


def test_uncalibrated_language_is_whisper_only_with_warning() -> None:
    route = language_route("kk")
    assert route.primary.family == "whisper"
    assert route.verifier is None
    assert route.locally_calibrated is False
    assert "не откалиброван" in str(route.warning)


def test_large_meeting_selects_clustering_backend() -> None:
    assert diarization_backend("auto", None) == "fluidaudio"
    assert diarization_backend("auto", 2) == "fluidaudio"
    assert diarization_backend("auto", 6) == "fluidaudio"
    assert diarization_backend("sortformer", 4) == "sortformer"


def test_sortformer_rejects_known_large_meeting() -> None:
    with pytest.raises(ValueError, match="не более четырёх"):
        diarization_backend("sortformer", 6)


def test_primary_whisper_on_cpu_is_named_in_warnings() -> None:
    """На Linux английский маршрут получал medium вместо Turbo молча."""
    from audio_transcription.catalog import BackendSpec
    from audio_transcription.cli import cpu_fallback_warning
    from audio_transcription.models import Hypothesis

    whisper = BackendSpec("whisper", "mlx-community/whisper-large-v3-turbo-asr-fp16")
    gigaam = BackendSpec("gigaam", "gigaam-v3-e2e-rnnt")
    cpu = Hypothesis("faster-whisper-medium", "en", 0.0, [], {})
    mlx = Hypothesis(whisper.model, "en", 0.0, [], {})
    warning = cpu_fallback_warning(whisper, cpu)
    assert warning and "faster-whisper-medium" in warning
    assert cpu_fallback_warning(whisper, mlx) is None
    assert cpu_fallback_warning(gigaam, Hypothesis("gigaam-v3-e2e-rnnt", "ru", 0.0, [], {})) is None
