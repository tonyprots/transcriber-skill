"""Ранние проверки и признаки жизни CLI.

Инцидент 2026-09-03: отказ «каталог уже существует» приходил только на записи
результата, после 17 минут расшифровки, а до этого прогон молчал.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from audio_transcription import cli


def parse(argv: list[str]):
    return cli.build_parser().parse_args(argv)


def test_run_rejects_non_empty_output_before_touching_audio(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "run.log").write_text("", encoding="utf-8")
    missing_source = tmp_path / "нет-такого-файла.wav"

    with pytest.raises(FileExistsError, match="Каталог уже существует"):
        cli.run(parse([str(missing_source), "--output", str(output)]))

    # Источника нет: до подготовки аудио дело не дошло — отказ пришёл раньше.
    assert list(output.iterdir()) == [output / "run.log"]


def test_run_allows_existing_empty_output(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    missing_source = tmp_path / "нет-такого-файла.wav"

    with pytest.raises(FileNotFoundError):
        cli.run(parse([str(missing_source), "--output", str(output)]))


def test_run_accepts_non_empty_output_with_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "manifest.json").write_text("{}", encoding="utf-8")
    missing_source = tmp_path / "нет-такого-файла.wav"

    with pytest.raises(FileNotFoundError):
        cli.run(parse([str(missing_source), "--output", str(output), "--overwrite"]))


def test_main_reports_existing_output_directory(tmp_path: Path, capsys) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "run.log").write_text("", encoding="utf-8")

    code = cli.main([str(tmp_path / "audio.wav"), "--output", str(output)])

    assert code == 2
    assert "Каталог уже существует" in capsys.readouterr().err


def test_reporter_writes_stages_to_its_stream() -> None:
    stream = io.StringIO()
    reporter = cli.ProgressReporter(stream=stream)
    reporter.stage("Подготовка аудио")
    assert "Подготовка аудио" in stream.getvalue()


def test_reporter_throttles_windows_but_keeps_first_and_last() -> None:
    stream = io.StringIO()
    reporter = cli.ProgressReporter(stream=stream, interval_seconds=10_000.0)
    report = reporter.windows("GigaAM")
    assert report is not None
    report({"completed": 0, "total": 3})
    report({"completed": 1, "total": 3})  # придавлено интервалом
    report({"completed": 3, "total": 3})  # последнее окно проходит всегда
    lines = stream.getvalue().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("GigaAM: окно 0/3")
    assert lines[1].endswith("GigaAM: окно 3/3")


def test_quiet_reporter_stays_silent() -> None:
    stream = io.StringIO()
    reporter = cli.ProgressReporter(enabled=False, stream=stream)
    reporter.stage("Подготовка аудио")
    assert reporter.windows("GigaAM") is None
    assert stream.getvalue() == ""
