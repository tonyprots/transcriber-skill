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
    assert diarization_backend("auto", None) == "sortformer"
    assert diarization_backend("auto", 4) == "sortformer"
    assert diarization_backend("auto", 6) == "fluidaudio"


def test_sortformer_rejects_known_large_meeting() -> None:
    with pytest.raises(ValueError, match="не более четырёх"):
        diarization_backend("sortformer", 6)
