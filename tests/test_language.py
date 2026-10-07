"""Язык по умолчанию определяется, а не подставляется.

До 0.16 умолчанием был русский, и английская запись без `--language en`
уходила в GigaAM, которая молча отдавала мусор.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcription import cli
from audio_transcription.backends import BackendUnavailable, language_hypothesis
from audio_transcription.fetching import RemoteMedia
from audio_transcription.models import AudioChunk
from audio_transcription.routing import language_route


def _chunks(count: int) -> list[AudioChunk]:
    return [
        AudioChunk(sequence=index, start=index * 10.0, end=index * 10.0 + 8.0, path=Path(f"/tmp/{index}.wav"))
        for index in range(count)
    ]


def _args(tmp_path: Path, *extra: str):
    return cli.build_parser().parse_args(["x.wav", "--cache-dir", str(tmp_path), *extra])


def _resolve(args, *, remote=None, chunks=None):
    return cli.resolve_language(
        args,
        remote=remote,
        chunks=_chunks(5) if chunks is None else chunks,
        work_dir=Path("/tmp"),
        source_identity="sha",
        report=cli.ProgressReporter(enabled=False),
    )


def _lid(shares: dict[str, float], calls: list | None = None):
    def run(backend, chunks, language, work_dir, **kwargs):
        if calls is not None:
            calls.append((backend, [chunk.sequence for chunk in chunks]))
        return language_hypothesis(backend, [shares] * len(chunks), 0.1)

    return run


def test_default_is_auto(tmp_path: Path) -> None:
    assert _args(tmp_path).language == "auto"


def _remote(language: str) -> RemoteMedia:
    return RemoteMedia(url="https://x", title="t", duration_seconds=1.0, extractor="Youtube", tracks=(), language=language)


def test_explicit_language_of_link_is_checked_against_audio(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_isolated_backend", _lid({"en": 0.97, "de": 0.03}))
    language, detection = _resolve(_args(tmp_path, "--language", "en-US"), remote=_remote(""))
    assert language == "en"
    assert detection["method"] == "argument"
    assert "warning" not in detection
    assert "heard" in detection


def test_explicit_language_of_local_file_costs_nothing(tmp_path: Path, monkeypatch) -> None:
    """Проба стоит 10–22 с, а чужая дорожка бывает только у ссылок."""
    monkeypatch.setattr(cli, "run_isolated_backend", pytest.fail)
    assert _resolve(_args(tmp_path, "--language", "ru")) == (
        "ru",
        {"method": "argument", "language": "ru"},
    )


def test_link_metadata_wins_over_detection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_isolated_backend", _lid({"en": 0.9, "ru": 0.1}))
    language, detection = _resolve(_args(tmp_path), remote=_remote("en"))
    assert language == "en" and detection["method"] == "source-metadata"


def test_dubbed_track_stops_before_recognition(tmp_path: Path, monkeypatch) -> None:
    """2026-10-06: тамильский автодубляж полтора часа распознавался как английский."""
    monkeypatch.setattr(cli, "run_isolated_backend", _lid({"ta": 0.9, "en": 0.05, "ml": 0.05}))
    with pytest.raises(cli.LanguageMismatch, match="ta.*--language ta"):
        _resolve(_args(tmp_path, "--language", "en"), remote=_remote(""))
    with pytest.raises(cli.LanguageMismatch, match="описании ссылки"):
        _resolve(_args(tmp_path), remote=_remote("en-US"))


def test_foreign_intro_is_not_a_mismatch(tmp_path: Path, monkeypatch) -> None:
    """Заставка на другом языке в одном окне из трёх не останавливает прогон."""
    windows = iter([{"en": 0.95}, {"ru": 0.9, "en": 0.1}, {"ru": 0.95}])

    def run(backend, chunks, language, work_dir, **kwargs):
        return language_hypothesis(backend, [next(windows) for _ in chunks], 0.1)

    monkeypatch.setattr(cli, "run_isolated_backend", run)
    assert _resolve(_args(tmp_path, "--language", "ru"), remote=_remote(""))[0] == "ru"


def test_language_check_can_be_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_isolated_backend", pytest.fail)
    assert _resolve(_args(tmp_path, "--language", "en", "--skip-language-check"), remote=_remote("")) == (
        "en",
        {"method": "argument", "language": "en"},
    )


def test_unavailable_check_does_not_block(tmp_path: Path, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise BackendUnavailable("нет весов")

    monkeypatch.setattr(cli, "run_isolated_backend", broken)
    language, detection = _resolve(_args(tmp_path, "--language", "en"), remote=_remote(""))
    assert language == "en" and "не сверен" in detection["warning"]


def test_detection_samples_start_middle_end(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(cli, "run_isolated_backend", _lid({"en": 0.9, "ru": 0.1}, calls))
    language, detection = _resolve(_args(tmp_path))
    assert language == "en"
    assert detection["method"] == "whisper-language-id"
    assert detection["probability"] == pytest.approx(0.9)
    assert "warning" not in detection
    assert calls == [("mlx-whisper-lid", [0, 2, 4])]
    assert language_route(language).primary.family == "whisper"


def test_detection_is_cached(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(cli, "run_isolated_backend", _lid({"ru": 0.99}, calls))
    _resolve(_args(tmp_path))
    _resolve(_args(tmp_path))
    assert len(calls) == 1


def test_unsure_detection_warns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_isolated_backend", _lid({"ru": 0.5, "uk": 0.45, "en": 0.05}))
    language, detection = _resolve(_args(tmp_path))
    assert language == "ru"
    assert "неуверенно" in detection["warning"] and "uk 45%" in detection["warning"]


def test_missing_whisper_falls_back_to_russian_loudly(tmp_path: Path, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise BackendUnavailable("нет весов")

    monkeypatch.setattr(cli, "run_isolated_backend", broken)
    language, detection = _resolve(_args(tmp_path))
    assert language == "ru"
    assert detection["method"] == "default"
    assert "--language" in detection["warning"]


def test_no_speech_needs_no_detection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_isolated_backend", pytest.fail)
    assert _resolve(_args(tmp_path), chunks=[])[0] == "ru"


def test_auto_has_no_route() -> None:
    with pytest.raises(ValueError):
        language_route("auto")


def test_language_votes_are_averaged() -> None:
    result = language_hypothesis("m", [{"ru": 1.0}, {"ru": 0.2, "en": 0.8}, {"ru": 0.9, "en": 0.1}], 0.0)
    assert result.language == "ru"
    assert result.metadata["language_probability"] == pytest.approx(0.7)
