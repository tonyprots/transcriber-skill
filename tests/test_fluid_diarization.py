"""FluidAudio по умолчанию: число голосов в модель, фильтр призраков, свои модели."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_transcription import diarization_worker, fluid_models
from audio_transcription.backends import BackendMissing
from audio_transcription.diarization import drop_ghost_speakers
from audio_transcription.models import SpeakerTurn


def _turn(start: float, end: float, *speakers: str) -> SpeakerTurn:
    return SpeakerTurn(start, end, tuple(speakers))


def test_ghost_speech_goes_to_nearest_real_neighbour() -> None:
    turns = [
        _turn(0.0, 60.0, "speaker_1"),
        _turn(60.0, 61.0, "speaker_2"),  # призрак: 1 с из 116
        _turn(65.0, 120.0, "speaker_3"),
    ]
    assert drop_ghost_speakers(turns) == [
        _turn(0.0, 61.0, "speaker_1"),
        _turn(65.0, 120.0, "speaker_2"),
    ]


def test_ghost_filter_keeps_rare_but_real_speaker() -> None:
    # 4 с из 60 — больше 3% речи: это собеседник, а не призрак.
    turns = [_turn(0.0, 30.0, "speaker_1"), _turn(30.0, 34.0, "speaker_2"), _turn(34.0, 60.0, "speaker_1")]
    assert drop_ghost_speakers(turns) == turns


def test_ghost_filter_never_removes_every_speaker() -> None:
    turns = [_turn(0.0, 1.0, "speaker_1"), _turn(1.5, 2.0, "speaker_2")]
    assert drop_ghost_speakers(turns, max_seconds=5.0, max_share=1.0) == turns


def _fake_fluid(monkeypatch, tmp_path: Path) -> list[list[str]]:
    commands: list[list[str]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **_kwargs) -> None:
            commands.append(list(command))
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {
                        "speakerCount": 2,
                        "segments": [
                            {"startTimeSeconds": 0.0, "endTimeSeconds": 2.0, "speakerId": "1"},
                            {"startTimeSeconds": 2.0, "endTimeSeconds": 4.0, "speakerId": "2"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

        def poll(self) -> int:
            return 0

        def communicate(self):
            return "", ""

    monkeypatch.setattr(diarization_worker, "_fluid_binary", lambda _config: tmp_path / "fluidaudiocli")
    monkeypatch.setattr(diarization_worker, "ensure_fluid_models", lambda **_kwargs: "ready")
    monkeypatch.setattr(diarization_worker.subprocess, "Popen", FakeProcess)
    return commands


@pytest.mark.parametrize(("num_speakers", "expected_tail"), [(2, ["--num-speakers", "2"]), (None, [])])
def test_expected_speakers_reach_fluidaudio(monkeypatch, tmp_path: Path, num_speakers, expected_tail) -> None:
    commands = _fake_fluid(monkeypatch, tmp_path)
    result = diarization_worker._run_fluidaudio(
        tmp_path / "audio.wav",
        {"num_speakers": num_speakers},
        tmp_path / "progress.json",
        tmp_path / "result.json",
    )
    command = commands[0]
    tail = command[command.index("--output") + 2:]
    assert tail == expected_tail
    assert result.metadata["num_speakers"] == num_speakers
    assert [turn.speakers for turn in result.turns] == [("speaker_1",), ("speaker_2",)]


def test_missing_models_offline_is_backend_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(diarization_worker, "_fluid_binary", lambda _config: tmp_path / "fluidaudiocli")

    def unavailable(**_kwargs):
        raise fluid_models.ModelsUnavailable("нет моделей")

    monkeypatch.setattr(diarization_worker, "ensure_fluid_models", unavailable)
    with pytest.raises(BackendMissing):
        diarization_worker._run_fluidaudio(
            tmp_path / "audio.wav", {"offline": True}, tmp_path / "p.json", tmp_path / "r.json"
        )


def _complete(target: Path) -> None:
    for relative in fluid_models.REQUIRED:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


def test_models_ready_skip_download(tmp_path: Path) -> None:
    _complete(tmp_path)
    assert fluid_models.ensure(tmp_path, offline=True) == "ready"


def test_models_offline_without_cache_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(fluid_models.ModelsUnavailable, match="офлайн"):
        fluid_models.ensure(tmp_path / "missing", offline=True)


def test_partial_models_are_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "plda-parameters.json").write_text("{}", encoding="utf-8")
    with pytest.raises(fluid_models.ModelsUnavailable, match="неполон"):
        fluid_models.ensure(tmp_path)


def test_download_uses_pinned_revision(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs)
        _complete(Path(kwargs["local_dir"]))

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    assert fluid_models.ensure(tmp_path / "models") == "downloaded"
    assert calls[0]["revision"] == fluid_models.FLUIDAUDIO.revision
    assert calls[0]["repo_id"] == fluid_models.FLUIDAUDIO.repo


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = [payload, b""]

    def read(self, _size: int) -> bytes:
        return self._payload.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _skill(tmp_path: Path, payload: bytes) -> Path:
    import hashlib

    folder = tmp_path / "skill" / "bin" / "macos-arm64"
    folder.mkdir(parents=True)
    (folder / "fluidaudiocli.url").write_text("https://example.invalid/fluidaudiocli\n", encoding="utf-8")
    (folder / "fluidaudiocli.sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  fluidaudiocli\n", encoding="utf-8"
    )
    return tmp_path / "skill"


def test_binary_is_downloaded_and_checked(tmp_path: Path) -> None:
    # Установка через /plugin, npx или корп-стор не запускает setup.sh.
    skill = _skill(tmp_path, b"binary")
    path = fluid_models.ensure_binary(skill, opener=lambda url, timeout: _Response(b"binary"))
    assert path.read_bytes() == b"binary"
    assert path.stat().st_mode & 0o111
    assert fluid_models.ensure_binary(skill, opener=pytest.fail) == path


def test_tampered_binary_is_rejected(tmp_path: Path) -> None:
    skill = _skill(tmp_path, b"binary")
    with pytest.raises(fluid_models.ModelsUnavailable, match="не совпал"):
        fluid_models.ensure_binary(skill, opener=lambda url, timeout: _Response(b"evil"))
    assert not fluid_models.bundled_binary(skill).exists()
    assert not fluid_models.bundled_binary(skill).with_name("fluidaudiocli.part").exists()


def test_binary_offline_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(fluid_models.ModelsUnavailable, match="офлайн"):
        fluid_models.ensure_binary(_skill(tmp_path, b"x"), offline=True, opener=pytest.fail)
