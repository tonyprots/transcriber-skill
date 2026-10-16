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


class _Calls(list):
    """Команды поддельного fluidaudiocli; окружение каждой — в `environments`."""

    def __init__(self) -> None:
        super().__init__()
        self.environments: list[dict[str, str]] = []


def _fake_fluid(monkeypatch, tmp_path: Path, returncodes: tuple[int, ...] = ()) -> _Calls:
    commands = _Calls()
    codes = list(returncodes)

    class FakeProcess:
        def __init__(self, command, **kwargs) -> None:
            commands.append(list(command))
            commands.environments.append(dict(kwargs.get("env") or {}))
            self.returncode = codes.pop(0) if codes else 0
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
            return self.returncode

        def wait(self, timeout=None) -> int:
            return self.returncode

    _silence(tmp_path / "audio.wav")
    monkeypatch.setattr(diarization_worker, "_fluid_binary", lambda _config, *_rest: tmp_path / "fluidaudiocli")
    monkeypatch.setattr(diarization_worker, "ensure_fluid_models", lambda **_kwargs: "ready")
    monkeypatch.setattr(diarization_worker.subprocess, "Popen", FakeProcess)
    # Ни песочницы, ни настоящего каталога моделей в тестах.
    monkeypatch.setattr(diarization_worker, "no_network_command", lambda command: list(command))
    monkeypatch.setattr(fluid_models, "default_target", lambda: tmp_path / "models")
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


def test_fluidaudio_never_reaches_model_registry(monkeypatch, tmp_path: Path) -> None:
    # Без набора в своём каталоге бинарник сам качает `main` репозитория:
    # адрес реестра — закрытый порт, команда — под песочницей без сети.
    commands = _fake_fluid(monkeypatch, tmp_path)
    monkeypatch.setattr(diarization_worker, "no_network_command", lambda command: ["sandbox", *command])
    result = diarization_worker._run_fluidaudio(
        tmp_path / "audio.wav", {}, tmp_path / "progress.json", tmp_path / "result.json"
    )
    assert commands[0][:2] == ["sandbox", str(tmp_path / "fluidaudiocli")]
    for name in ("MODEL_REGISTRY_URL", "REGISTRY_URL"):
        assert commands.environments[0][name] == fluid_models.CLOSED_REGISTRY
    assert result.metadata["network_sandbox"] is True


def test_fluidaudio_retries_without_sandbox_once(monkeypatch, tmp_path: Path) -> None:
    # Песочница сама может помешать (политика машины, новая macOS): повтор
    # без неё, реестр по-прежнему закрыт.
    commands = _fake_fluid(monkeypatch, tmp_path, returncodes=(1, 0))
    monkeypatch.setattr(diarization_worker, "no_network_command", lambda command: ["sandbox", *command])
    result = diarization_worker._run_fluidaudio(
        tmp_path / "audio.wav", {}, tmp_path / "progress.json", tmp_path / "result.json"
    )
    assert [command[0] for command in commands] == ["sandbox", str(tmp_path / "fluidaudiocli")]
    assert commands.environments[1]["MODEL_REGISTRY_URL"] == fluid_models.CLOSED_REGISTRY
    assert result.metadata["network_sandbox"] is False


def test_fluidaudio_failure_without_sandbox_is_not_retried(monkeypatch, tmp_path: Path) -> None:
    commands = _fake_fluid(monkeypatch, tmp_path, returncodes=(1, 0))
    with pytest.raises(RuntimeError, match="FluidAudio"):
        diarization_worker._run_fluidaudio(
            tmp_path / "audio.wav", {}, tmp_path / "progress.json", tmp_path / "result.json"
        )
    assert len(commands) == 1


def _hub_metadata(target: Path, commit: str, names=fluid_models.FILES) -> None:
    folder = target / ".cache" / "huggingface" / "download"
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / f"{name}.metadata").write_text(f"{commit}\netag\n1760000000.0\n", encoding="utf-8")


def test_fluidaudio_reports_installed_revision(monkeypatch, tmp_path: Path) -> None:
    _fake_fluid(monkeypatch, tmp_path)
    _hub_metadata(tmp_path / "models", fluid_models.FLUIDAUDIO.revision)
    result = diarization_worker._run_fluidaudio(
        tmp_path / "audio.wav", {}, tmp_path / "progress.json", tmp_path / "result.json"
    )
    assert result.metadata["weights"] == {"revision": fluid_models.FLUIDAUDIO.revision, "pinned": True}


def test_installed_revision_sources(tmp_path: Path) -> None:
    ours = tmp_path / "ours"
    _hub_metadata(ours, "a" * 40)
    assert fluid_models.installed_revision(ours) == "a" * 40
    assert fluid_models.weights_metadata(ours) == {"revision": "a" * 40, "pinned": False}
    # Набор, который бинарник скачал себе сам по 0.24.1.
    own = tmp_path / "own"
    own.mkdir()
    (own / "provenance.json").write_text(json.dumps({"artifact_snapshot": "b" * 40}), encoding="utf-8")
    assert fluid_models.installed_revision(own) == "b" * 40
    mixed = tmp_path / "mixed"
    _hub_metadata(mixed, "a" * 40, names=fluid_models.FILES[:1])
    _hub_metadata(mixed, "c" * 40, names=fluid_models.FILES[1:])
    assert fluid_models.installed_revision(mixed) is None
    assert fluid_models.weights_metadata(mixed) == {"revision": "unknown", "pinned": False}
    assert fluid_models.installed_revision(tmp_path / "missing") is None


def test_models_live_where_binary_reads_them(monkeypatch, tmp_path: Path) -> None:
    # `fluidaudiocli process --mode offline` читает только этот каталог;
    # в соседний `speaker-diarization-coreml` он не смотрит.
    monkeypatch.setenv("HOME", str(tmp_path))
    assert fluid_models.default_target() == (
        tmp_path / "Library" / "Application Support" / "FluidAudio" / "Models" / "speaker-diarization"
    )


def test_ready_needs_offline_set_with_weights(tmp_path: Path) -> None:
    _complete(tmp_path)
    assert fluid_models.ready(tmp_path)
    (tmp_path / "Embedding.mlmodelc" / "weights" / "weight.bin").unlink()
    assert not fluid_models.ready(tmp_path)
    assert all("pyannote" not in path and "wespeaker" not in path for path in fluid_models.REQUIRED)


def test_no_network_command_probes_sandbox(monkeypatch) -> None:
    calls: list[list[str]] = []

    class Probe:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    monkeypatch.setattr(fluid_models.os, "access", lambda *_args: True)
    monkeypatch.setattr(fluid_models.subprocess, "run", lambda command, **_kw: calls.append(command) or Probe(0))
    wrapped = fluid_models.no_network_command(["fluidaudiocli", "process"])
    assert wrapped == [fluid_models.SANDBOX_EXEC, "-p", fluid_models.NO_NETWORK_PROFILE, "fluidaudiocli", "process"]
    assert calls[0][-1] == "/usr/bin/true"
    # Вложенная песочница (агент в своей) отказывает: запуск без префикса.
    monkeypatch.setattr(fluid_models.subprocess, "run", lambda command, **_kw: Probe(71))
    assert fluid_models.no_network_command(["fluidaudiocli"]) == ["fluidaudiocli"]


def test_missing_models_offline_is_backend_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(diarization_worker, "_fluid_binary", lambda _config, *_rest: tmp_path / "fluidaudiocli")

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


def test_foreign_content_is_not_overwritten(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    (tmp_path / "plda-parameters.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("чужое", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_kwargs: pytest.fail("перезапись чужого"))
    with pytest.raises(fluid_models.ModelsUnavailable, match="notes.txt"):
        fluid_models.ensure(tmp_path)
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "чужое"


def test_interrupted_download_resumes(monkeypatch, tmp_path: Path) -> None:
    # Обрыв первой загрузки (Ctrl-C, сторож, сеть) оставляет каталог непустым:
    # раньше каждый следующий --diarize отказывал до ручной уборки.
    import huggingface_hub

    download = tmp_path / ".cache" / "huggingface" / "download"
    download.mkdir(parents=True)
    (download / "plda-parameters.json.incomplete").write_bytes(b"half")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "Segmentation.mlmodelc").mkdir()
    (tmp_path / "Segmentation.mlmodelc" / "model.mil").write_text("x", encoding="utf-8")
    (tmp_path / ".DS_Store").write_bytes(b"")
    calls: list[dict] = []

    def resume(**kwargs):
        calls.append(kwargs)
        _complete(Path(kwargs["local_dir"]))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", resume)
    assert fluid_models.ensure(tmp_path) == "downloaded"
    assert len(calls) == 1 and calls[0]["revision"] == fluid_models.FLUIDAUDIO.revision


@pytest.mark.parametrize("error", [OSError("сеть"), RuntimeError("Hub ответил 500")])
def test_download_failure_is_models_unavailable(monkeypatch, tmp_path: Path, error) -> None:
    import huggingface_hub

    def broken(**_kwargs):
        raise error

    monkeypatch.setattr(huggingface_hub, "snapshot_download", broken)
    with pytest.raises(fluid_models.ModelsUnavailable, match="не скачались"):
        fluid_models.ensure(tmp_path / "models")


def test_incomplete_after_download_is_reported(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_kwargs: None)
    with pytest.raises(fluid_models.ModelsUnavailable, match="неполон"):
        fluid_models.ensure(tmp_path / "models")


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
    assert sorted(calls[0]["allow_patterns"]) == sorted(
        [*fluid_models.FILES, *(f"{name}.mlmodelc/**" for name in ("Segmentation", "Embedding", "FBank", "PldaRho"))]
    )


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


def test_diarization_cache_key_follows_installed_models(monkeypatch) -> None:
    from audio_transcription import cli

    monkeypatch.setattr(cli, "installed_fluid_revision", lambda: None)
    name, revision = cli.diarization_weights("fluidaudio", cli.SORTFORMER.model)
    assert (name, revision) == ("FluidAudio Community-1 offline", fluid_models.FLUIDAUDIO.revision)
    pinned = {"weights": {"revision": fluid_models.FLUIDAUDIO.revision, "pinned": True}}
    assert cli.weights_as_keyed(pinned, {name: revision})
    # Набор, который бинарник скачал себе сам с другого коммита, — свой ключ.
    monkeypatch.setattr(cli, "installed_fluid_revision", lambda: "b" * 40)
    name, revision = cli.diarization_weights("fluidaudio", cli.SORTFORMER.model)
    assert revision == "b" * 40
    assert cli.weights_as_keyed({"weights": {"revision": "b" * 40, "pinned": False}}, {name: revision})
    assert not cli.weights_as_keyed({"weights": {"revision": "unknown", "pinned": False}}, {name: revision})
    assert cli.diarization_weights("sortformer", cli.SORTFORMER.model) == (
        cli.SORTFORMER.model,
        cli.revision_for_model(cli.SORTFORMER.model),
    )


def _silence(path: Path, seconds: float = 1.0) -> Path:
    import wave

    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\0\0" * int(16000 * seconds))
    return path


def _script_fluid(monkeypatch, tmp_path: Path, body: str) -> Path:
    """Настоящий процесс вместо fluidaudiocli: проверяем пайпы и сроки, а не модель."""
    script = tmp_path / "fluidaudiocli"
    script.write_text(
        "#!/bin/sh\n"
        f"echo $$ > '{tmp_path / 'pid'}'\n"
        'while [ $# -gt 0 ]; do [ "$1" = "--output" ] && out="$2"; shift; done\n'
        f"{body}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setattr(diarization_worker, "_fluid_binary", lambda _config, *_rest: script)
    monkeypatch.setattr(diarization_worker, "ensure_fluid_models", lambda **_kwargs: "ready")
    monkeypatch.setattr(diarization_worker, "no_network_command", lambda command: list(command))
    monkeypatch.setattr(fluid_models, "default_target", lambda: tmp_path / "models")
    return script


def _alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Зомби: процесс завершён, но ещё не прибран — для нас он уже мёртв.
    import subprocess

    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(state.strip()) and not state.strip().startswith("Z")


def _run_bounded(function, seconds: float):
    """Вызов с потолком: на старом коде бинарник висит на полном пайпе вечно."""
    import threading

    outcome: dict = {}

    def target() -> None:
        try:
            outcome["value"] = function()
        except BaseException as error:  # noqa: BLE001 — отдаём в тест как есть
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    return thread.is_alive(), outcome


def test_chatty_fluidaudio_does_not_fill_pipe(monkeypatch, tmp_path: Path) -> None:
    _script_fluid(
        monkeypatch,
        tmp_path,
        "head -c 204800 /dev/zero | tr '\\000' x >&2\n"
        'printf \'{"speakerCount":1,"segments":[{"startTimeSeconds":0,"endTimeSeconds":1,"speakerId":"1"}]}\' > "$out"',
    )
    audio = _silence(tmp_path / "audio.wav")
    hung, outcome = _run_bounded(
        lambda: diarization_worker._run_fluidaudio(audio, {}, tmp_path / "progress.json", tmp_path / "result.json"),
        15.0,
    )
    if hung:
        import os
        import signal

        os.kill(int((tmp_path / "pid").read_text()), signal.SIGKILL)
    assert not hung, "FluidAudio повис на полном пайпе stderr"
    assert "error" not in outcome, outcome.get("error")
    assert [turn.speakers for turn in outcome["value"].turns] == [("speaker_1",)]
    assert (tmp_path / "fluidaudio.stderr.log").stat().st_size == 204800


def test_fluidaudio_error_carries_stderr_tail(monkeypatch, tmp_path: Path) -> None:
    _script_fluid(monkeypatch, tmp_path, "echo 'CoreML: модель не открылась' >&2\nexit 7")
    with pytest.raises(RuntimeError, match="модель не открылась"):
        diarization_worker._run_fluidaudio(
            _silence(tmp_path / "audio.wav"), {}, tmp_path / "progress.json", tmp_path / "result.json"
        )


def test_stuck_fluidaudio_is_killed_after_deadline(monkeypatch, tmp_path: Path) -> None:
    # Сердцебиение раз в секунду глушит сторож зависаний родителя; без своего
    # срока зависший бинарник держал бы прогон вечно.
    _script_fluid(monkeypatch, tmp_path, "exec sleep 1000")
    monkeypatch.setattr(diarization_worker, "FLUIDAUDIO_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(diarization_worker, "FLUIDAUDIO_DEADLINE_PER_AUDIO_SECOND", 0.0)
    hung, outcome = _run_bounded(
        lambda: diarization_worker._run_fluidaudio(
            _silence(tmp_path / "audio.wav"), {}, tmp_path / "progress.json", tmp_path / "result.json"
        ),
        15.0,
    )
    pid = int((tmp_path / "pid").read_text())
    if hung:
        import os
        import signal

        os.kill(pid, signal.SIGKILL)
    assert not hung, "зависший FluidAudio не остановлен"
    assert isinstance(outcome.get("error"), RuntimeError)
    assert "не уложился" in str(outcome["error"])
    assert not _alive(pid)


def _diarize_args(tmp_path: Path, *extra: str):
    from audio_transcription import cli

    return cli.build_parser().parse_args([str(tmp_path / "in.wav"), "--diarize", *extra])


@pytest.fixture
def apple_silicon(monkeypatch):
    from audio_transcription import cli

    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    return cli


def test_wrong_fluidaudio_bin_fails_before_audio(apple_silicon, monkeypatch, tmp_path: Path) -> None:
    # Раньше бинарник искали в воркере — после подготовки звука, SHA-256, VAD
    # и выбора языка, — а неверный путь молча подменяли поставочным.
    cli = apple_silicon
    monkeypatch.setattr(cli, "prepare_audio", lambda *_a, **_k: pytest.fail("звук готовить рано"))
    args = _diarize_args(tmp_path, "--fluidaudio-bin", str(tmp_path / "my-build"))
    with pytest.raises(ValueError, match="my-build"):
        cli.run(args)


def test_offline_diarize_without_models_fails_before_audio(apple_silicon, monkeypatch, tmp_path: Path) -> None:
    cli = apple_silicon
    own = tmp_path / "fluidaudiocli"
    own.write_text("#!/bin/sh\n", encoding="utf-8")
    own.chmod(0o755)
    monkeypatch.setattr(fluid_models, "default_target", lambda: tmp_path / "models")
    args = _diarize_args(tmp_path, "--offline", "--fluidaudio-bin", str(own))
    with pytest.raises(ValueError, match="prefetch_models.py --diarize"):
        cli._validate(args)
    _complete(tmp_path / "models")
    cli._validate(args)


def test_sortformer_does_not_need_fluidaudio(apple_silicon, tmp_path: Path) -> None:
    args = _diarize_args(tmp_path, "--diarization-backend", "sortformer", "--fluidaudio-bin", str(tmp_path / "nope"))
    apple_silicon._validate(args)


def test_diarization_cache_key_tracks_binary(tmp_path: Path) -> None:
    from audio_transcription import cli
    from audio_transcription.cache import cache_key

    args = _diarize_args(tmp_path)

    def key(fingerprint: str | None) -> str:
        return cache_key(
            "diarization", cli.diarization_cache_payload(args, "fluidaudio", 0.7, "sha", fingerprint)
        )

    assert key("aaaaaaaaaaaa") != key("bbbbbbbbbbbb")
    assert key("aaaaaaaaaaaa") == key("aaaaaaaaaaaa")


def test_worker_takes_chosen_binary_as_is(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        diarization_worker, "ensure_fluid_binary", lambda **_kwargs: pytest.fail("свой путь не докачивают")
    )
    with pytest.raises(FileNotFoundError, match="my-build"):
        diarization_worker._fluid_binary({"binary": str(tmp_path / "my-build")})
    bundled = tmp_path / "bundled"
    bundled.write_text("#!/bin/sh\n", encoding="utf-8")
    bundled.chmod(0o755)
    monkeypatch.setattr(diarization_worker, "ensure_fluid_binary", lambda **_kwargs: bundled)
    monkeypatch.setattr(diarization_worker, "bundled_binary_problem", lambda _path: None)
    assert diarization_worker._fluid_binary({"binary": None}) == bundled


def test_fluidaudio_metadata_names_binary(monkeypatch, tmp_path: Path) -> None:
    _fake_fluid(monkeypatch, tmp_path)
    result = diarization_worker._run_fluidaudio(
        tmp_path / "audio.wav",
        {"binary_source": "explicit", "binary_fingerprint": "abcdef012345"},
        tmp_path / "progress.json",
        tmp_path / "result.json",
    )
    assert result.metadata["binary"] == {"source": "explicit", "sha256": "abcdef012345"}


def test_diarization_pipeline_moves_with_worker_command(monkeypatch, tmp_path: Path) -> None:
    """Сменилась команда бинарника — меняется и `pipeline` ключа кэша.

    0.24 начал передавать `--num-speakers`, а ключ остался прежним: разметка 0.23
    без подсказки возвращалась из кэша ещё 90 дней. Снимок связывает одно с
    другим: правка команды без смены `pipeline` ломает этот тест.
    """
    from audio_transcription import cli

    commands = _fake_fluid(monkeypatch, tmp_path)
    diarization_worker._run_fluidaudio(
        tmp_path / "audio.wav", {"num_speakers": 2}, tmp_path / "progress.json", tmp_path / "result.json"
    )
    flags = [part for part in commands[0] if part.startswith("--")]
    payload = cli.diarization_cache_payload(_diarize_args(tmp_path), "fluidaudio", 0.7, "sha", None)
    assert (payload["pipeline"], flags) == ("0.6", ["--mode", "--threshold", "--output", "--num-speakers"])


def test_diarization_worker_reports_model_download(monkeypatch, tmp_path: Path) -> None:
    """Загрузка моделей диаризации видна сторожу так же, как у ASR-воркера."""
    from audio_transcription import weights
    from audio_transcription.models import Diarization

    def downloading(audio_path, config, progress_path, result_path):
        bar = weights._silent_tqdm()(total=10_000_000, unit="B")
        bar.update(3_000_000)
        bar.close()
        snapshot.append(json.loads(progress_path.read_text(encoding="utf-8")))
        return Diarization(model="fake", elapsed_seconds=0.0, turns=[], metadata={})

    snapshot: list[dict] = []
    monkeypatch.setattr(diarization_worker, "_run_fluidaudio", downloading)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"backend": "fluidaudio", "audio_path": str(tmp_path / "a.wav"), "config": {}}), encoding="utf-8"
    )
    code = diarization_worker.main([str(request), str(tmp_path / "result.json"), str(tmp_path / "progress.json")])
    assert code == 0
    assert snapshot[0]["stage"] == "загрузка моделей: 3 МБ"


def test_models_download_quietly_in_worker(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    calls: list[dict] = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs)
        _complete(Path(kwargs["local_dir"]))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    fluid_models.ensure(tmp_path / "quiet", quiet=True)
    fluid_models.ensure(tmp_path / "loud")
    assert calls[0]["tqdm_class"].__name__ == "Silent"
    assert "tqdm_class" not in calls[1]  # prefetch и setup показывают полосу


class _Blocks(_Response):
    def __init__(self, blocks: list[bytes]) -> None:
        self._payload = [*blocks, b""]


def test_binary_download_reports_each_block(tmp_path: Path) -> None:
    # 20 МБ на медленной сети шли молча, и сторож родителя убивал воркер.
    blocks = [b"a" * 10, b"b" * 10, b"c" * 5]
    skill = _skill(tmp_path, b"".join(blocks))
    heard: list[int] = []
    path = fluid_models.ensure_binary(skill, opener=lambda url, timeout: _Blocks(list(blocks)), progress=heard.append)
    assert heard == [10, 10, 5]
    assert path.read_bytes() == b"".join(blocks)


def test_fluidaudio_reports_loading_before_binary(monkeypatch, tmp_path: Path) -> None:
    _fake_fluid(monkeypatch, tmp_path)
    seen: list[dict] = []

    def binary(_config, progress=None):
        seen.append(json.loads((tmp_path / "progress.json").read_text(encoding="utf-8")))
        progress(2_000_000)
        seen.append(json.loads((tmp_path / "progress.json").read_text(encoding="utf-8")))
        return tmp_path / "fluidaudiocli"

    monkeypatch.setattr(diarization_worker, "_fluid_binary", binary)
    diarization_worker._run_fluidaudio(tmp_path / "audio.wav", {}, tmp_path / "progress.json", tmp_path / "result.json")
    assert seen[0]["stage"] == "loading"
    assert seen[1]["stage"] == "загрузка fluidaudiocli: 2 МБ"


def test_worker_passes_progress_to_binary_download(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    bundled = tmp_path / "bundled"
    bundled.write_text("#!/bin/sh\n", encoding="utf-8")
    bundled.chmod(0o755)
    monkeypatch.setattr(diarization_worker, "ensure_fluid_binary", lambda **kwargs: calls.append(kwargs) or bundled)
    monkeypatch.setattr(diarization_worker, "bundled_binary_problem", lambda _path: None)
    marker = object()
    diarization_worker._fluid_binary({}, marker)
    assert calls[0]["progress"] is marker
