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
