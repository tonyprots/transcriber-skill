"""Вышла ли версия скилла новее установленной.

Сам скилл не обновляется и никого не уведомлял: до 0.24 пользователь узнавал о
новой версии, только повторив установку. Теперь прогон раз в неделю спрашивает
у GitHub номер последнего релиза — в фоне, с коротким таймаутом, без данных о
записях. Ответ хранится в ~/.transcriber/update-check.json, поэтому напоминание
о найденной версии печатается в каждом прогоне, а сеть — не чаще раза в неделю.
`--offline` и TRANSCRIBER_NO_UPDATE_CHECK=1 проверку выключают.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from . import __version__

RELEASES_URL = "https://api.github.com/repos/tonyprots/transcriber-skill/releases/latest"
INTERVAL_SECONDS = 7 * 24 * 3600
TIMEOUT_SECONDS = 3.0
DEFAULT_STATE = Path.home() / ".transcriber" / "update-check.json"
SKILL_DIR = Path(__file__).resolve().parents[2]


def state_path() -> Path:
    override = os.environ.get("TRANSCRIBER_UPDATE_STATE")
    return Path(override).expanduser() if override else DEFAULT_STATE


def disabled() -> bool:
    return os.environ.get("TRANSCRIBER_NO_UPDATE_CHECK", "") not in ("", "0")


def parse_version(text: str) -> tuple[int, ...] | None:
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", text.strip())
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def _read_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def fetch_latest(timeout: float = TIMEOUT_SECONDS) -> str | None:
    request = urllib.request.Request(
        RELEASES_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": f"transcriber/{__version__}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 — адрес постоянный
        tag = json.load(response).get("tag_name")
    return str(tag) if tag and parse_version(str(tag)) else None


def refresh(path: Path | None = None, *, fetch=fetch_latest, now: float | None = None) -> None:
    """Спрашивает GitHub и записывает ответ; любая ошибка — молча, прогон важнее."""
    path = path or state_path()
    state = {"checked_at": time.time() if now is None else now}
    try:
        latest = fetch()
    except Exception:  # noqa: BLE001 — нет сети или GitHub не ответил
        latest = None
    previous = _read_state(path).get("latest")
    state["latest"] = latest or previous
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def due(path: Path | None = None, *, now: float | None = None) -> bool:
    checked = _read_state(path or state_path()).get("checked_at")
    current = time.time() if now is None else now
    return not isinstance(checked, (int, float)) or current - checked >= INTERVAL_SECONDS


def start(*, offline: bool) -> threading.Thread | None:
    """Запускает проверку в фоне, если с прошлой прошла неделя."""
    if offline or disabled() or not due():
        return None
    thread = threading.Thread(target=refresh, name="transcriber-update-check", daemon=True)
    thread.start()
    return thread


def update_hint(skill_dir: Path = SKILL_DIR) -> str:
    text = str(skill_dir)
    home = os.environ.get("TRANSCRIBER_HOME")
    if "/.local/share/transcriber-skill/" in text or (home and str(Path(home).expanduser()) in text):
        return (
            "повторить установку: curl -fsSL https://raw.githubusercontent.com/"
            "tonyprots/transcriber-skill/main/install.sh | bash"
        )
    if "/plugins/" in text:
        return "обновить плагин transcriber в Claude Code (/plugin)"
    if (skill_dir.parents[1] / ".git").exists():
        return f"git -C {skill_dir.parents[1]} pull, затем scripts/setup.sh"
    return "повторить установку тем же способом, что и в первый раз"


def available(path: Path | None = None) -> dict[str, str] | None:
    """Новая версия по последнему ответу GitHub, без сети."""
    if disabled():
        return None
    latest = _read_state(path or state_path()).get("latest")
    found = parse_version(str(latest)) if latest else None
    installed = parse_version(__version__)
    if not found or not installed or found <= installed:
        return None
    return {
        "installed": __version__,
        "latest": str(latest).lstrip("v"),
        "how": update_hint(),
        "notes": "https://github.com/tonyprots/transcriber-skill/releases",
    }
