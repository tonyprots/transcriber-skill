"""Установщики: корневой install.sh, integrations/handy/install.sh и умолчания handy_result."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HANDY = ROOT / "integrations/handy"
sys.path.insert(0, str(HANDY))

import handy_result  # noqa: E402

SKILL_VENV = Path("skills/transcriber/.venv/bin/python")  # кладёт setup.sh
ROOT_VENV = Path(".venv/bin/python")  # разработка по README, так у Антона

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="нужен bash")


def executable(path: Path, body: str = "exit 0") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def make_repo(tmp_path: Path, *venvs: Path) -> Path:
    repo = tmp_path / "repo"
    script = repo / "integrations/handy/install.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(HANDY / "install.sh", script)
    for venv in venvs:
        executable(repo / venv)
    return repo


def install(tmp_path: Path, repo: Path) -> subprocess.CompletedProcess:
    home = tmp_path / "home"
    (home / "Library/Application Support/com.pais.handy/recordings").mkdir(parents=True)
    (home / "Library/LaunchAgents").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    # launchd не трогаем: `print` отвечает «агента нет», остальное — успех.
    executable(bin_dir / "launchctl", '[ "$1" = print ] && exit 1\nexit 0')
    agent_python = executable(tmp_path / "agent-python")
    return subprocess.run(
        [
            "bash", str(repo / "integrations/handy/install.sh"),
            "--output-root", str(tmp_path / "out"), "--agent-python", str(agent_python),
        ],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
    )


def agent_python_arg(tmp_path: Path) -> str:
    plist = tmp_path / "home/Library/LaunchAgents/ru.tonyprots.handy-transcriber.plist"
    argv = plistlib.loads(plist.read_bytes())["ProgramArguments"]
    return argv[argv.index("--python") + 1]


@pytest.mark.parametrize("venv", [SKILL_VENV, ROOT_VENV])
def test_installer_finds_skill_environment_where_it_lives(tmp_path: Path, venv: Path) -> None:
    repo = make_repo(tmp_path, venv)
    result = install(tmp_path, repo)
    assert result.returncode == 0, result.stderr
    assert agent_python_arg(tmp_path) == str(repo / venv)


def test_installer_without_environment_names_both_places(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    result = install(tmp_path, repo)
    assert result.returncode == 1
    assert str(repo / "skills/transcriber/.venv") in result.stderr
    assert str(repo / ".venv") in result.stderr
    assert "setup.sh" in result.stderr


@pytest.mark.parametrize("venv", [SKILL_VENV, ROOT_VENV])
def test_result_default_python_without_agent(tmp_path: Path, monkeypatch, venv: Path) -> None:
    repo = tmp_path / "repo"
    executable(repo / venv)
    monkeypatch.setattr(handy_result, "REPO", repo)
    monkeypatch.setattr(handy_result, "PLIST", tmp_path / "нет.plist")
    args = handy_result.parse_args([str(tmp_path / "handy-1790899241.wav")])
    assert args.python == str(repo / venv)


def test_root_installer_explains_unreachable_repository(tmp_path: Path) -> None:
    # set -e с pipefail обрывали `REF="$(latest_tag)"` молча: rc 128, ни строки.
    missing = tmp_path / "нет-такого-репозитория"
    result = subprocess.run(
        ["bash", str(ROOT / "install.sh")],
        capture_output=True, text=True, timeout=60,
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "TRANSCRIBER_HOME": str(tmp_path / "target"),
            "TRANSCRIBER_REPO": str(missing),
            "TRANSCRIBER_REF": "",
        },
    )
    assert result.returncode != 0
    assert str(missing) in result.stderr
    assert not (tmp_path / "target").exists()
