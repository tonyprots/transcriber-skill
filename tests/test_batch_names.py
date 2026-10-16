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


def test_skip_done_matches_record_not_directory_name(tmp_path: Path, monkeypatch) -> None:
    # Канал отдаёт новые ролики первыми: новый «Стрим» сдвигал порядок, садился
    # на каталог прошлого и числился готовым с его расшифровкой.
    a, b, c = "https://youtu.be/a", "https://youtu.be/b", "https://youtu.be/c"
    calls = _patch_remote(monkeypatch, {a: "Стрим", b: "Стрим", c: "Стрим"})
    base = tmp_path / "out"

    def run(*urls: str) -> list[dict]:
        args = cli.build_parser().parse_args([*urls, "--output", str(base), "--skip-done", "--quiet"])
        return cli.run_batch(args)

    run(b, a)
    calls.clear()
    results = {item["input"]: item for item in run(c, b, a)}

    assert [url for url, _ in calls] == [c]
    assert not results[c].get("skipped")
    assert results[b]["skipped"] and results[a]["skipped"]
    for url in (a, b, c):
        manifest = json.loads(Path(results[url]["manifest"]).read_text(encoding="utf-8"))
        assert manifest["source"]["origin"] == public_url(url)


def test_skip_done_single_run_does_not_take_foreign_manifest(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"source": {"source": str(tmp_path / "other.wav"), "origin": None, "sha256": "f" * 64}}),
        encoding="utf-8",
    )
    source = tmp_path / "mine.wav"
    source.write_bytes(b"mine")
    monkeypatch.setattr(cli, "_transcribe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("занято")))
    args = cli.build_parser().parse_args([str(source), "--output", str(output), "--skip-done", "--quiet"])
    try:
        cli.run(args)
    except FileExistsError as error:
        assert "уже существует" in str(error)
    else:
        raise AssertionError("чужой манифест не делает запись готовой")


def test_skip_done_recognizes_moved_file_by_sha256(tmp_path: Path, monkeypatch) -> None:
    import hashlib

    source = tmp_path / "moved" / "talk.wav"
    source.parent.mkdir()
    source.write_bytes(b"same bytes")
    output = tmp_path / "out" / "talk"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "source": str(tmp_path / "old-place" / "talk.wav"),
                    "origin": None,
                    "sha256": hashlib.sha256(b"same bytes").hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "other.wav").write_bytes(b"other")
    calls: list = []
    monkeypatch.setattr(cli, "_transcribe", _fake_transcribe(calls))
    args = cli.build_parser().parse_args(
        [str(source), str(tmp_path / "other.wav"), "--output", str(tmp_path / "out"), "--skip-done", "--quiet"]
    )
    results = {Path(item["input"]).name: item for item in cli.run_batch(args)}
    assert results["talk.wav"]["skipped"] is True
    assert results["talk.wav"]["output"] == str(output.resolve())
    assert [Path(name).name for name, _ in calls] == ["other.wav"]


def test_batch_directory_names_differ_beyond_letter_case(tmp_path: Path, monkeypatch) -> None:
    # APFS не различает регистр: Meeting и meeting — один каталог, и вторая
    # запись падала только при записи результата, после всей расшифровки.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, second = tmp_path / "a" / "Meeting.wav", tmp_path / "b" / "meeting.wav"
    first.write_bytes(b"1")
    second.write_bytes(b"2")
    calls: list = []
    monkeypatch.setattr(cli, "_transcribe", _fake_transcribe(calls))
    args = cli.build_parser().parse_args([str(first), str(second), "--output", str(tmp_path / "out"), "--quiet"])
    cli.run_batch(args)
    names = [output.name for _, output in calls]
    assert len({name.casefold() for name in names}) == 2, names
