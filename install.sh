#!/usr/bin/env bash
# Установка transcriber одной командой:
#   curl -fsSL https://raw.githubusercontent.com/tonyprots/transcriber-skill/main/install.sh | bash
# Что делает: клонирует или обновляет репозиторий в ~/.local/share/transcriber-skill,
# подключает скилл симлинком в ~/.claude/skills и, если есть Codex, в ~/.codex/skills,
# создаёт окружение (около 1 ГБ) и скачивает модели русского маршрута (2,7 ГБ).
# Точный размер маршрута печатает scripts/prefetch_models.py --print-size.
# Переменные: TRANSCRIBER_HOME — куда клонировать; TRANSCRIBER_MODELS=ru|en|all|none —
# какие модели скачать сразу (по умолчанию ru); TRANSCRIBER_DIARIZE=1 — плюс Sortformer;
# TRANSCRIBER_REF — тег или ветка (по умолчанию последний тег релиза; main — свежий код
# без гарантий: релиз получает тег только после прогона тестов на настоящих моделях).
set -euo pipefail

REPO="${TRANSCRIBER_REPO:-https://github.com/tonyprots/transcriber-skill}"
TARGET="${TRANSCRIBER_HOME:-$HOME/.local/share/transcriber-skill}"
MODELS="${TRANSCRIBER_MODELS:-ru}"
DIARIZE="${TRANSCRIBER_DIARIZE:-}"
REF="${TRANSCRIBER_REF:-}"

for tool in git python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Нужен $tool. macOS: xcode-select --install; Debian/Ubuntu: sudo apt install git python3-venv" >&2
    exit 1
  fi
done

# Последний релиз — по номеру версии, а не по дате тега.
latest_tag() {
  git ls-remote --tags --refs "$REPO" 'v*' 2>/dev/null \
    | sed 's|.*refs/tags/||' | sort -t. -k1,1V -k2,2n -k3,3n | tail -n 1
}
if [ -z "$REF" ]; then
  REF="$(latest_tag)"
  if [ -z "$REF" ]; then
    echo "Не удалось узнать последний релиз в $REPO: ставлю main" >&2
    REF="main"
  fi
fi

if [ -d "$TARGET/.git" ]; then
  echo "Обновляю $TARGET до $REF"
  # Клон с тега однобранчевый: ветки запрашиваем явно, иначе main не найдётся.
  # Теги — без --force: тег релиза не двигается (релизы на GitHub неизменяемы),
  # и если на сервере он вдруг указывает на другой коммит, это повод
  # остановиться, а не молча поставить другой код под тем же номером.
  if ! git -C "$TARGET" fetch --quiet --tags origin "+refs/heads/*:refs/remotes/origin/*"; then
    echo "Не обновляю: тег релиза на сервере указывает не на тот коммит, что уже скачан," >&2
    echo "или сервер недоступен. Подмена релиза — повод разобраться, а не ставить поверх." >&2
    exit 1
  fi
  # Ветку обновляем до её состояния на сервере, тег берём как есть.
  if git -C "$TARGET" rev-parse --verify --quiet "refs/remotes/origin/$REF" >/dev/null; then
    git -C "$TARGET" checkout --quiet -B "$REF" "origin/$REF"
  else
    git -C "$TARGET" -c advice.detachedHead=false checkout --quiet "$REF"
  fi
else
  echo "Клонирую $REPO ($REF) в $TARGET"
  mkdir -p "$(dirname "$TARGET")"
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$REF" "$REPO" "$TARGET"
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

# macOS ships bash 3.2, where an empty array under set -u is an error: собираем строку.
SETUP_ARGS=""
if [ "$MODELS" != "none" ]; then SETUP_ARGS="--models $MODELS"; fi
if [ -n "$DIARIZE" ]; then SETUP_ARGS="$SETUP_ARGS --diarize"; fi
# shellcheck disable=SC2086
bash "$SKILL/scripts/setup.sh" $SETUP_ARGS

echo
echo "Готово. В Claude Code или Codex достаточно попросить: «расшифруй эту запись»."
