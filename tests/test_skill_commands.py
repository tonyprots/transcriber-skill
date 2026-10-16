"""Команды запуска из SKILL.md работают так, как агент их скопирует."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SKILL_MD = Path(__file__).resolve().parents[1] / "skills/transcriber/SKILL.md"
PLACEHOLDERS = ("OUTPUT_DIR", "PARENT_DIR")


def background_runs() -> list[str]:
    blocks = re.findall(r"```bash\n(.*?)```", SKILL_MD.read_text(encoding="utf-8"), re.S)
    return [b for b in blocks if any(f"> {name}.log" in b for name in PLACEHOLDERS)]


def test_skill_has_background_runs() -> None:
    assert len(background_runs()) >= 2


@pytest.mark.skipif(shutil.which("bash") is None, reason="нужен bash")
@pytest.mark.parametrize("block", background_runs())
def test_background_run_starts_in_new_folder(tmp_path: Path, block: str) -> None:
    # Редирект лога оболочка открывает до запуска python: без родителя
    # OUTPUT_DIR команда падала с «No such file or directory», а SKILL.md
    # при этом запрещал mkdir.
    target = tmp_path / "новая" / "папка" / "запись"
    command = block
    for name in PLACEHOLDERS:
        command = command.replace(name, str(target))
    stub = tmp_path / "python"
    stub.write_text("#!/bin/sh\necho запущен\n", encoding="utf-8")
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", command + "\nwait $!\n"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "ASR_PYTHON": str(stub), "SKILL_DIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    log = target.with_name(f"{target.name}.log")
    assert log.read_text(encoding="utf-8").strip() == "запущен"
    assert not target.exists(), "сам каталог результата создаёт скрипт"
