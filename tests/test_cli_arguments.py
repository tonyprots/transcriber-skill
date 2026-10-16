"""Аргументы, которые раньше проходили проверку и ломали прогон позже.

Каждый отказ здесь должен прийти до замка машины и до звука: прогон, который
упал после часа расшифровки, стоит час, а отказ на входе — секунду.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcription import cli


def _forbidden(*args, **kwargs):
    raise AssertionError("до тяжёлой работы дойти не должно")


@pytest.mark.parametrize("value", ["0", "-5", "0.00001", "nan"])
def test_speech_window_must_be_at_least_a_second(tmp_path: Path, monkeypatch, value: str) -> None:
    # 0 уводил Silero VAD в бесконечный цикл (`_merge_segments` не двигает
    # начало), пока прогон держал замок машины и стоял Handy.
    monkeypatch.setattr(cli, "machine_slot", _forbidden)
    monkeypatch.setattr(cli, "split_speech_windows", _forbidden)
    monkeypatch.setattr(cli, "prepare_audio", _forbidden)
    source = tmp_path / "a.wav"
    source.write_bytes(b"x")
    args = cli.build_parser().parse_args(
        [str(source), "--output", str(tmp_path / "out"), "--speech-window-seconds", value, "--quiet"]
    )
    with pytest.raises(ValueError, match="--speech-window-seconds"):
        cli.run(args)
    assert not (tmp_path / "out").exists()


def test_speech_window_default_and_one_second_pass() -> None:
    for argv in (["a.wav"], ["a.wav", "--speech-window-seconds", "1"]):
        cli._validate(cli.build_parser().parse_args(argv))


@pytest.mark.parametrize("inside", ["text.txt", "nested/text.txt", ""])
def test_early_text_inside_output_is_rejected_before_work(
    tmp_path: Path, monkeypatch, inside: str
) -> None:
    # Раньше файл раннего текста создавал каталог результата, и write_bundle в
    # самом конце отказывал «Каталог уже существует», а с --overwrite увозил
    # ранний текст в архив.
    monkeypatch.setattr(cli, "_transcribe", _forbidden)
    source = tmp_path / "a.wav"
    source.write_bytes(b"x")
    output = tmp_path / "out"
    early = output / inside if inside else output
    args = cli.build_parser().parse_args(
        [str(source), "--output", str(output), "--early-text", str(early), "--quiet"]
    )
    with pytest.raises(ValueError, match="OUTPUT_DIR.txt"):
        cli.run(args)
    assert not output.exists()


def test_early_text_next_to_output_passes(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "_transcribe", lambda args, **kw: calls.append(kw["output"]) or kw["output"])
    source = tmp_path / "a.wav"
    source.write_bytes(b"x")
    output = tmp_path / "out"
    args = cli.build_parser().parse_args(
        [str(source), "--output", str(output), "--early-text", str(tmp_path / "out.txt"), "--quiet"]
    )
    assert cli.run(args) == output.resolve()
    assert calls == [output.resolve()]


def test_early_text_inside_batch_record_directory_fails_that_record(
    tmp_path: Path, monkeypatch
) -> None:
    # Плейлист из одного ролика идёт пакетом: каталог записи — внутри --output.
    monkeypatch.setattr(cli, "_transcribe", _forbidden)
    source = tmp_path / "a.wav"
    source.write_bytes(b"x")
    base = tmp_path / "out"
    args = cli.build_parser().parse_args(
        [str(source), "--output", str(base), "--early-text", str(base / "a" / "t.txt"), "--quiet"]
    )
    results = cli.run_batch(args)
    assert len(results) == 1 and "OUTPUT_DIR.txt" in results[0]["error"]
