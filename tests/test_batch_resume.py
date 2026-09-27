"""Прерванную пачку можно продолжить, а занятый каталог не роняет её целиком."""
from __future__ import annotations

import json
from pathlib import Path

from audio_transcription import cli


def _fake_transcribe(args, *, source, remote, section, output, report, download_seconds):
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({"source": {}, "review_count": 0}))
    return output


def _args(tmp_path: Path, *extra: str):
    inputs = [str(tmp_path / f"{name}.wav") for name in ("done", "busy", "fresh")]
    for path in inputs:
        Path(path).write_bytes(b"fixture")
    return cli.build_parser().parse_args([*inputs, "--output", str(tmp_path / "out"), "--quiet", *extra])


def _prepare(tmp_path: Path) -> None:
    done = tmp_path / "out" / "done"
    done.mkdir(parents=True)
    (done / "manifest.json").write_text(json.dumps({"source": {}, "review_count": 3}))
    busy = tmp_path / "out" / "busy"
    busy.mkdir()
    (busy / "leftover.txt").write_text("x")


def test_skip_done_continues_interrupted_batch(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        cli, "_transcribe", lambda args, **kw: calls.append(kw["output"].name) or _fake_transcribe(args, **kw)
    )
    results = cli.run_batch(_args(tmp_path, "--skip-done"))
    by_input = {Path(item["input"]).stem: item for item in results}
    assert by_input["done"]["skipped"] is True and by_input["done"]["review_count"] == 3
    assert "уже существует" in by_input["busy"]["error"]
    assert by_input["fresh"]["output"].endswith("fresh")
    assert calls == ["fresh"]


def test_busy_directory_fails_one_record_not_batch(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path)
    monkeypatch.setattr(cli, "_transcribe", _fake_transcribe)
    results = cli.run_batch(_args(tmp_path))
    errors = {Path(item["input"]).stem for item in results if "error" in item}
    assert errors == {"done", "busy"}
    assert any(item.get("output", "").endswith("fresh") for item in results)


def test_skip_done_single_record(tmp_path: Path, monkeypatch) -> None:
    _prepare(tmp_path)
    monkeypatch.setattr(cli, "_transcribe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("не должен считать")))
    source = tmp_path / "done.wav"
    source.write_bytes(b"fixture")
    args = cli.build_parser().parse_args(
        [str(source), "--output", str(tmp_path / "out" / "done"), "--skip-done", "--quiet"]
    )
    assert cli.run(args) == (tmp_path / "out" / "done").resolve()
