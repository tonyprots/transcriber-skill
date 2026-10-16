"""Каталоги пачки: какой записи какой каталог достаётся и когда запись готова.

`--skip-done` раньше узнавал готовую запись только по имени каталога, а имя
зависит от заголовка и порядка входов. У канала с одинаковыми названиями
новый ролик сдвигал порядок и получал чужую расшифровку.
"""
from __future__ import annotations

import json
from pathlib import Path

from audio_transcription import cli
from audio_transcription.fetching import RemoteMedia, public_url


def _remote(url: str, title: str = "Стрим") -> RemoteMedia:
    return RemoteMedia(url=url, title=title, duration_seconds=60.0, extractor="youtube", tracks=())


def _fake_transcribe(calls: list):
    """Пишет манифест так же, как настоящий прогон: с источником записи."""

    def transcribe(args, *, source, remote, section, output, report, download_seconds):
        calls.append((remote.url if remote else str(source), output))
        output.mkdir(parents=True, exist_ok=True)
        origin = public_url(remote.url) if remote else None
        (output / "manifest.json").write_text(
            json.dumps({"source": {"source": str(source), "origin": origin, "sha256": "x"}}),
            encoding="utf-8",
        )
        return output

    return transcribe


def _patch_remote(monkeypatch, titles: dict[str, str]) -> list:
    calls: list = []
    monkeypatch.setattr(cli, "probe_remote", lambda url: _remote(url, titles[url]))
    monkeypatch.setattr(cli, "_download", lambda remote, dest, report: (dest / "a.mp3", 0.0))
    monkeypatch.setattr(cli, "_transcribe", _fake_transcribe(calls))
    return calls


def test_playlist_of_one_video_is_still_a_batch(tmp_path: Path, monkeypatch, capsys) -> None:
    # Плейлист, развернувшийся в один ролик, шёл одиночной записью: --output
    # становился каталогом результата, а не родителем пачки.
    base = tmp_path / "channel"
    earlier = base / "old-video"
    earlier.mkdir(parents=True)
    (earlier / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "expand_playlist", lambda url, limit=None: ["https://youtu.be/v1"])
    calls = _patch_remote(monkeypatch, {"https://youtu.be/v1": "Новый ролик"})

    code = cli.main(
        ["https://youtube.com/@chan", "--playlist", "--playlist-limit", "1", "--output", str(base), "--quiet"]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["output"] for item in payload["results"]] == [str(base.resolve() / "новый-ролик")]
    assert calls == [("https://youtu.be/v1", base.resolve() / "новый-ролик")]
    assert (earlier / "manifest.json").read_text(encoding="utf-8") == "{}"
