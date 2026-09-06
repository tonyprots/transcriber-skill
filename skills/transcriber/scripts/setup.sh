#!/usr/bin/env bash
# Создаёт .venv рядом со скиллом и ставит зависимости. Повторный запуск безопасен.
# Использование: bash scripts/setup.sh [путь к python3]
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${1:-}"

if [ -z "$PYTHON" ]; then
  for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "Не найден python3 (нужен 3.10+)" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Не найден ffmpeg. macOS: brew install ffmpeg; Debian/Ubuntu: sudo apt install ffmpeg" >&2
  exit 1
fi

VENV="$SKILL_DIR/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Создаю $VENV на $("$PYTHON" --version)"
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r "$SKILL_DIR/requirements.txt"

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  echo "Apple Silicon: ставлю mlx-audio (Whisper Turbo MLX и диаризация)"
  "$VENV/bin/python" -m pip install --quiet -r "$SKILL_DIR/requirements-apple.txt"
  chmod +x "$SKILL_DIR/bin/macos-arm64/fluidaudiocli" 2>/dev/null || true
  # Архив, скачанный браузером, получает карантин: снимаем его с бинарника.
  xattr -d com.apple.quarantine "$SKILL_DIR/bin/macos-arm64/fluidaudiocli" 2>/dev/null || true
fi

echo
"$VENV/bin/python" "$SKILL_DIR/scripts/doctor.py"
echo
echo "Модели загружаются при первом запуске transcribe.py (русский маршрут — около 2,5 ГБ)."
