"""Каталог моделей: сравнение версий, устаревание замера, размеры, ревизии.

Логика здесь тихая: она ничего не переключает, только предупреждает. Поэтому
ошибка в ней не роняет прогон, а молча лишает пользователя сигнала — и увидеть
её можно только тестом.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from audio_transcription import catalog
from audio_transcription.catalog import (
    CATALOG,
    GIGAAM_RUSSIAN,
    SILERO_VAD,
    VOSK_RUSSIAN,
    WHISPER_TURBO,
    entry_for_model,
    format_gb,
    local_revision,
    revisions_for_models,
    route_download_gb,
    route_entries,
    search_term,
    staleness_warnings,
)


def test_newer_variants_finds_next_major() -> None:
    assert GIGAAM_RUSSIAN.newer_variants(["gigaam-v4-e2e-rnnt"]) == ["gigaam-v4-e2e-rnnt"]


def test_newer_variants_ignores_other_families_and_older() -> None:
    """Сосед по семейству — не более новая версия той же модели."""
    candidates = [
        "gigaam-v3-e2e-ctc",  # другая архитектура
        "gigaam-v2-rnnt",  # другое имя и версия ниже
        "gigaam-v3-e2e-rnnt",  # она сама
        "whisper-v9",  # другое семейство
    ]
    assert GIGAAM_RUSSIAN.newer_variants(candidates) == []


def test_newer_variants_sorts_by_number_not_text() -> None:
    """«v10» больше «v4», хотя лексикографически меньше — иначе совет устареет."""
    found = GIGAAM_RUSSIAN.newer_variants(
        ["gigaam-v4-e2e-rnnt", "gigaam-v10-e2e-rnnt", "gigaam-v7-e2e-rnnt"]
    )
    assert found == ["gigaam-v10-e2e-rnnt", "gigaam-v7-e2e-rnnt", "gigaam-v4-e2e-rnnt"]


def test_newer_variants_without_version_in_name() -> None:
    """У Vosk в имени версии нет: сравнивать нечего, и выдумывать не надо."""
    assert VOSK_RUSSIAN.newer_variants(["alphacep/vosk-model-ru-v2"]) == []


def test_newer_variants_survives_garbage() -> None:
    assert GIGAAM_RUSSIAN.newer_variants(None) == []
    assert GIGAAM_RUSSIAN.newer_variants([None, 42, object()]) == []


def test_newer_repos_compares_full_repository_name() -> None:
    assert GIGAAM_RUSSIAN.newer_repos(["istupakov/gigaam-v4-onnx"]) == ["istupakov/gigaam-v4-onnx"]
    assert GIGAAM_RUSSIAN.newer_repos(["someone-else/gigaam-v4-onnx"]) == []


def test_search_term_covers_neighbouring_versions() -> None:
    assert search_term("istupakov/gigaam-v3-onnx") == "gigaam"
    assert search_term("alphacep/vosk-model-ru") == "vosk-model-ru"


def test_age_and_staleness_use_shelf_life() -> None:
    fresh = date(2026, 9, 6)
    assert GIGAAM_RUSSIAN.age_days(fresh) == 0
    assert not GIGAAM_RUSSIAN.is_stale(date(2027, 3, 4))  # 179 дней
    assert GIGAAM_RUSSIAN.is_stale(date(2027, 3, 6))  # 181 день
    # У VAD замера нет, и «устаревшим» он не становится никогда.
    assert SILERO_VAD.age_days(date(2030, 1, 1)) is None
    assert not SILERO_VAD.is_stale(date(2030, 1, 1))


def test_staleness_warnings_silent_while_calibration_is_fresh(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "known_asr_names", lambda: ())
    assert staleness_warnings([GIGAAM_RUSSIAN.model], today=date(2026, 9, 16)) == []


def test_staleness_warnings_report_oldest_measurement(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "known_asr_names", lambda: ())
    warnings = staleness_warnings(
        [GIGAAM_RUSSIAN.model, VOSK_RUSSIAN.model], today=date(2027, 6, 1)
    )
    assert len(warnings) == 1
    # Оба устарели, названо самое старое число дней — у GigaAM.
    assert "268 дн." in warnings[0]
    assert GIGAAM_RUSSIAN.label in warnings[0] and VOSK_RUSSIAN.label in warnings[0]


def test_staleness_warnings_notice_newer_model_in_library(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "known_asr_names", lambda: ("gigaam-v4-e2e-rnnt",))
    warnings = staleness_warnings([GIGAAM_RUSSIAN.model], today=date(2026, 9, 16))
    assert len(warnings) == 1
    assert "gigaam-v4-e2e-rnnt" in warnings[0]
    assert "замера WER" in warnings[0]


def test_staleness_warnings_ignore_unknown_models(monkeypatch) -> None:
    monkeypatch.setattr(catalog, "known_asr_names", lambda: ())
    assert staleness_warnings(["модель-которой-нет"], today=date(2030, 1, 1)) == []


def test_entry_for_model_respects_quantization() -> None:
    assert entry_for_model("gigaam-multilingual-ctc", "int8") is not None
    assert entry_for_model("gigaam-multilingual-ctc") is None


def test_cache_dir_matches_hugging_face_layout() -> None:
    assert GIGAAM_RUSSIAN.cache_dir == "models--istupakov--gigaam-v3-onnx"
    # FluidAudio лежит не в кэше HF, а в Application Support.
    assert catalog.FLUIDAUDIO.cache_dir is None


def test_route_entries_match_what_prefetch_downloads() -> None:
    russian = route_entries("ru")
    assert GIGAAM_RUSSIAN in russian and VOSK_RUSSIAN in russian and WHISPER_TURBO in russian
    assert catalog.GIGAAM_MULTILINGUAL_FAST not in russian
    # Вне Apple Silicon Whisper MLX заменяется CPU-сборкой.
    assert catalog.FASTER_WHISPER_FALLBACK in route_entries("ru", apple=False)
    assert catalog.SORTFORMER in route_entries("ru", diarize=True)
    # Диаризации вне Apple Silicon нет, и качать её незачем.
    assert catalog.SORTFORMER not in route_entries("ru", apple=False, diarize=True)


def test_route_download_size_is_sum_of_catalog() -> None:
    assert route_download_gb("ru") == pytest.approx(
        sum(entry.download_gb for entry in route_entries("ru"))
    )
    assert route_download_gb("en") == pytest.approx(
        sum(entry.download_gb for entry in route_entries("en"))
    )
    # Какой из маршрутов тяжелее — следствие выбора моделей, а не контракт:
    # с приходом Canary английский обогнал русский. Держим только то, что
    # обязано быть верным всегда.
    assert route_download_gb("all") > max(route_download_gb("ru"), route_download_gb("en"))


def test_unknown_route_is_an_error() -> None:
    with pytest.raises(ValueError):
        route_entries("de")


def test_format_gb_switches_to_megabytes_for_small_models() -> None:
    assert format_gb(2.66) == "2,7 ГБ"
    assert format_gb(0.034) == "35 МБ"
    assert format_gb(0.2) == "0,2 ГБ"


def test_local_revision_reads_refs_main(tmp_path: Path) -> None:
    ref = tmp_path / GIGAAM_RUSSIAN.cache_dir / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("322c3b294926abc\n", encoding="utf-8")
    assert local_revision(GIGAAM_RUSSIAN, tmp_path) == "322c3b294926abc"


def test_local_revision_absent_without_cache(tmp_path: Path) -> None:
    assert local_revision(GIGAAM_RUSSIAN, tmp_path) is None
    # У модели вне кэша HF ревизии нет по устройству, а не по отсутствию файла.
    assert local_revision(catalog.FLUIDAUDIO, tmp_path) is None


def test_revisions_for_models_skips_unknown(tmp_path: Path) -> None:
    ref = tmp_path / GIGAAM_RUSSIAN.cache_dir / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("deadbeef", encoding="utf-8")
    found = revisions_for_models([GIGAAM_RUSSIAN.model, "модель-которой-нет"], tmp_path)
    assert found == {GIGAAM_RUSSIAN.model: "deadbeef"}


def test_every_entry_documents_its_origin() -> None:
    """Каталог — ещё и ответ на «откуда эти веса»: пустых полей в нём быть не должно."""
    for entry in CATALOG:
        assert entry.upstream, entry.key
        assert entry.role, entry.key
        assert entry.calibrated or entry.calibration_note, entry.key


def test_fluidaudio_binary_found_outside_skill_folder(tmp_path, monkeypatch) -> None:
    """Бинарник ищется в env и PATH, а не только в папке скилла.

    Раньше поиск был написан дважды, и doctor смотрел лишь в папку скилла:
    у собравшего бинарник самостоятельно он печатал «✗» там, где диаризация
    работала.
    """
    skill_dir = tmp_path / "skill"           # «скилл» без bin/
    skill_dir.mkdir()
    monkeypatch.delenv("TRANSCRIBER_FLUIDAUDIO_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    assert catalog.fluidaudio_binary_path(skill_dir=skill_dir) is None

    elsewhere = tmp_path / "bin"
    elsewhere.mkdir()
    binary = elsewhere / "fluidaudiocli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)

    monkeypatch.setenv("PATH", str(elsewhere))
    assert catalog.fluidaudio_binary_path(skill_dir=skill_dir) == binary.resolve()


def test_fluidaudio_binary_priority(tmp_path, monkeypatch) -> None:
    """Явный путь бьёт переменную, переменная — PATH, PATH — папку скилла."""
    def executable(name: str) -> Path:
        path = tmp_path / name
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
        return path.resolve()

    explicit, from_env = executable("explicit"), executable("from-env")
    monkeypatch.setenv("TRANSCRIBER_FLUIDAUDIO_BIN", str(from_env))
    monkeypatch.setenv("PATH", "")

    assert catalog.fluidaudio_binary_path(explicit) == explicit
    assert catalog.fluidaudio_binary_path() == from_env


def test_fluidaudio_binary_ignores_non_executable(tmp_path, monkeypatch) -> None:
    """Файл без флага исполнения — не бинарник: молча запускать его нельзя."""
    monkeypatch.setenv("PATH", "")
    plain = tmp_path / "fluidaudiocli"
    plain.write_text("текст, а не программа", encoding="utf-8")
    plain.chmod(0o644)
    monkeypatch.setenv("TRANSCRIBER_FLUIDAUDIO_BIN", str(plain))
    assert catalog.fluidaudio_binary_path(skill_dir=tmp_path / "skill") is None
