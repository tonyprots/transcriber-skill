#!/usr/bin/env bash
# Установка transcriber одной командой:
#   curl -fsSL https://raw.githubusercontent.com/tonyprots/transcriber-skill/main/install.sh | bash
# Что делает: клонирует или обновляет репозиторий в ~/.local/share/transcriber-skill,
# подключает скилл симлинком в ~/.claude/skills и, если есть Codex, в ~/.codex/skills,
# создаёт окружение (около 700 МБ) и скачивает модели русского маршрута (около 2,5 ГБ).
# Переменные: TRANSCRIBER_HOME — куда клонировать; TRANSCRIBER_MODELS=ru|en|all|none —
# какие модели скачать сразу (по умолчанию ru); TRANSCRIBER_DIARIZE=1 — плюс Sortformer.
set -euo pipefail

REPO="${TRANSCRIBER_REPO:-https://github.com/tonyprots/transcriber-skill}"
TARGET="${TRANSCRIBER_HOME:-$HOME/.local/share/transcriber-skill}"
MODELS="${TRANSCRIBER_MODELS:-ru}"
DIARIZE="${TRANSCRIBER_DIARIZE:-}"

for tool in git python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Нужен $tool. macOS: xcode-select --install; Debian/Ubuntu: sudo apt install git python3-venv" >&2
    exit 1
  fi
done

if [ -d "$TARGET/.git" ]; then
  echo "Обновляю $TARGET"
  git -C "$TARGET" pull --ff-only --quiet
else
  echo "Клонирую $REPO в $TARGET"
  mkdir -p "$(dirname "$TARGET")"
  git clone --quiet --depth 1 "$REPO" "$TARGET"
fi

SKILL="$TARGET/skills/transcriber"
link() {
  local dir="$1"
  mkdir -p "$dir"
  if [ -e "$dir/transcriber" ] && [ ! -L "$dir/transcriber" ]; then
    echo "В $dir уже есть каталог transcriber, не трогаю его" >&2
    return
  fi
  ln -sfn "$SKILL" "$dir/transcriber"
  echo "Подключено: $dir/transcriber → $SKILL"
}
link "$HOME/.claude/skills"
if [ -d "$HOME/.codex" ]; then link "$HOME/.codex/skills"; fi

SETUP_ARGS=()
if [ "$MODELS" != "none" ]; then SETUP_ARGS+=(--models "$MODELS"); fi
if [ -n "$DIARIZE" ]; then SETUP_ARGS+=(--diarize); fi
bash "$SKILL/scripts/setup.sh" "${SETUP_ARGS[@]}"

echo
echo "Готово. В Claude Code или Codex достаточно попросить: «расшифруй эту запись»."
