#!/usr/bin/env bash
# Создаёт .venv рядом со скиллом и ставит зависимости. Повторный запуск безопасен.
# Использование: bash scripts/setup.sh [--models ru|en|all] [--diarize] [--python /путь/к/python3]
#   --models  сразу скачать модели маршрута, чтобы первая расшифровка не ждала загрузки
#   --diarize вместе с моделями скачать Sortformer для диаризации
#   --allow-unlocked  если проверенные версии не встают, ставить по диапазонам
#                     (то же — TRANSCRIBER_ALLOW_UNLOCKED=1)
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=""
MODELS=""
DIARIZE=""
ALLOW_UNLOCKED="${TRANSCRIBER_ALLOW_UNLOCKED:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS="${2:-ru}"; shift 2 ;;
    --models=*) MODELS="${1#--models=}"; shift ;;
    --diarize) DIARIZE="--diarize"; shift ;;
    --allow-unlocked) ALLOW_UNLOCKED=1; shift ;;
    --python) PYTHON="${2:-}"; shift 2 ;;
    -h|--help) sed -n 2,7p "$0"; exit 0 ;;
    *) PYTHON="$1"; shift ;;   # старый вызов: bash setup.sh /путь/к/python3
  esac
done

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
  if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    echo "Не найден ffmpeg, ставлю через Homebrew"
    brew install ffmpeg
  else
    echo "Не найден ffmpeg. macOS: brew install ffmpeg; Debian/Ubuntu: sudo apt install ffmpeg" >&2
    exit 1
  fi
fi

# yt-dlp нужен только ссылкам, поэтому его отсутствие не повод падать.
# Ставим из brew: там он обновляется вместе с остальным, а в .venv его
# версию никто бы не поднимал.
if ! command -v yt-dlp >/dev/null 2>&1; then
  if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    echo "Не найден yt-dlp (нужен для ссылок), ставлю через Homebrew вместе с deno"
    brew install yt-dlp deno || echo "yt-dlp не поставился: ссылки не откроются, файлы работают" >&2
  else
    echo "Не найден yt-dlp: ссылки не откроются. Установка: pipx install yt-dlp или пакет дистрибутива" >&2
  fi
fi

VENV="$SKILL_DIR/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Создаю $VENV на $("$PYTHON" --version)"
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --quiet --upgrade pip
# Точные версии с хешами — те, на которых проходят тесты релиза. mlx-audio в
# lock-файле помечен маркером Apple Silicon и на других машинах не ставится.
# Не встали — останавливаемся: обычная установка должна давать именно
# проверенное окружение, а не то, что pip выберет сегодня.
if ! "$VENV/bin/python" -m pip install --quiet --require-hashes -r "$SKILL_DIR/requirements.lock"; then
  if [ -z "$ALLOW_UNLOCKED" ]; then
    echo >&2
    echo "Проверенные версии из requirements.lock не встали на $("$VENV/bin/python" --version)." >&2
    echo "Чаще всего помогает другой Python: bash scripts/setup.sh --python python3.12" >&2
    echo "Поставить по диапазонам из requirements.txt (окружение не совпадёт с проверенным):" >&2
    echo "  bash scripts/setup.sh --allow-unlocked" >&2
    exit 1
  fi
  echo "Ставлю по диапазонам из requirements.txt: окружение не совпадёт с проверенным" >&2
  "$VENV/bin/python" -m pip install --quiet -r "$SKILL_DIR/requirements.txt"
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    "$VENV/bin/python" -m pip install --quiet -r "$SKILL_DIR/requirements-apple.txt"
  fi
fi

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  echo "Apple Silicon: Whisper Turbo MLX и диаризация доступны"
  FLUID_DIR="$SKILL_DIR/bin/macos-arm64"
  # fluidaudiocli (диаризация 5+ голосов) собирается в CI репозитория из
  # закреплённого коммита FluidAudio и лежит в его Release, а не в git. Ставим
  # только после сверки с суммой, опубликованной в скилле; curl не вешает на
  # файл карантин, так что снимать его не нужно.
  if [ -f "$FLUID_DIR/fluidaudiocli.sha256" ] && [ -f "$FLUID_DIR/fluidaudiocli.url" ] \
    && ! (cd "$FLUID_DIR" && shasum -a 256 -c fluidaudiocli.sha256 >/dev/null 2>&1); then
    echo "Скачиваю fluidaudiocli для встреч с 5+ участниками (около 20 МБ)"
    PART="$FLUID_DIR/fluidaudiocli.part"
    if curl -fsSL --retry 2 -o "$PART" "$(cat "$FLUID_DIR/fluidaudiocli.url")" \
      && [ "$(shasum -a 256 "$PART" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$FLUID_DIR/fluidaudiocli.sha256")" ]; then
      chmod +x "$PART" && mv "$PART" "$FLUID_DIR/fluidaudiocli"
    else
      rm -f "$PART"
      echo "fluidaudiocli не скачался или не совпал с опубликованной суммой: диаризация 5+ голосов" >&2
      echo "недоступна. Соберите свой по third-party/fluidaudio/NOTICE.md и передайте --fluidaudio-bin" >&2
    fi
  fi
fi

if [ -n "$MODELS" ]; then
  echo
  "$VENV/bin/python" "$SKILL_DIR/scripts/prefetch_models.py" "$MODELS" $DIARIZE
fi

echo
"$VENV/bin/python" "$SKILL_DIR/scripts/doctor.py"
if [ -z "$MODELS" ]; then
  # Размер спрашиваем у каталога моделей: вписанное руками число разъезжается
  # с ним при первой же замене модели.
  SIZE="$("$VENV/bin/python" "$SKILL_DIR/scripts/prefetch_models.py" ru --print-size 2>/dev/null || echo "около 3 ГБ")"
  echo
  echo "Модели загрузятся при первом запуске transcribe.py (русский маршрут — $SIZE)."
  echo "Чтобы скачать заранее: bash scripts/setup.sh --models ru"
fi
