"""doctor.py говорит о yt-dlp: без него ссылки не открываются."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "transcriber" / "scripts" / "doctor.py"
spec = importlib.util.spec_from_file_location("doctor", SCRIPT)
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)  # type: ignore[union-attr]


def _status(**overrides):
    base = {"path": "/opt/homebrew/bin/yt-dlp", "version": "2026.08.19", "age_days": 39, "stale": True}
    return lambda today=None: {**base, **overrides}


def test_old_but_latest_yt_dlp_is_not_stale(monkeypatch) -> None:
    import io
    import urllib.request

    monkeypatch.setattr(doctor, "yt_dlp_status", _status())
    monkeypatch.setattr(doctor, "find_command", lambda name: "/opt/homebrew/bin/deno")
    assert doctor.check_yt_dlp()["stale"] is True
    # Сравнение с релизом снимает догадку по возрасту.
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b'{"tag_name": "2026.08.19"}')
    )
    status = doctor.check_yt_dlp(check_updates=True)
    assert status["latest"] == "2026.08.19" and status["stale"] is False


def test_missing_yt_dlp_is_a_warning_not_a_problem(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "check_yt_dlp",
        lambda check_updates=False: {"path": None, "version": None, "age_days": None, "stale": False, "deno": None, "latest": None},
    )
    report = doctor.collect()
    assert any("нет yt-dlp" in text for text in report["warnings"])
    assert not any("yt-dlp" in text for text in report["problems"])
    assert "yt-dlp (ссылки) ✗ не установлен" in doctor.render(report)


def test_newer_release_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor,
        "check_yt_dlp",
        lambda check_updates=False: {"path": "/opt/homebrew/bin/yt-dlp", "version": "2026.08.19", "age_days": 5, "stale": False, "deno": "/d", "latest": "2026.09.20"},
    )
    assert any("вышел yt-dlp 2026.09.20" in text for text in doctor.collect()["warnings"])
