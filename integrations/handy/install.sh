#!/bin/bash
# Ставит launchd-агента, который прогоняет длинные диктовки Handy через transcriber.
#
#   integrations/handy/install.sh --output-root <папка> [--agent-python <путь>] [--min-seconds 60]
#   integrations/handy/install.sh --uninstall
#
# --agent-python — интерпретатор, которому macOS уже дала доступ к папке
# с репозиторием (TCC выдаёт грант бинарю, а не агенту). Если репозиторий не в
# ~/Documents, ~/Desktop или ~/Downloads, подойдёт /usr/bin/python3.
set -euo pipefail

LABEL="ru.tonyprots.handy-transcriber"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/$LABEL.log"
RECORDINGS="$HOME/Library/Application Support/com.pais.handy/recordings"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

AGENT_PYTHON="/usr/bin/python3"
OUTPUT_ROOT=""
MIN_SECONDS="60"
UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --agent-python) AGENT_PYTHON="$2"; shift 2 ;;
    --min-seconds) MIN_SECONDS="$2"; shift 2 ;;
    --uninstall) UNINSTALL=1; shift ;;
    *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
  esac
done

unload() {
  launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
  # bootout возвращается раньше, чем launchd отпустит метку: bootstrap сразу за ним падает.
  for _ in $(seq 1 50); do
    launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1 || return 0
    sleep 0.1
  done
}

if [ "$UNINSTALL" = 1 ]; then
  unload
  rm -f "$PLIST"
  echo "агент снят"
  exit 0
fi

[ -n "$OUTPUT_ROOT" ] || { echo "нужен --output-root" >&2; exit 2; }
[ -d "$RECORDINGS" ] || { echo "нет папки записей Handy: $RECORDINGS" >&2; exit 1; }
ASR_PYTHON="$REPO/.venv/bin/python"
[ -x "$ASR_PYTHON" ] || { echo "нет окружения скилла: $ASR_PYTHON (запусти setup.sh)" >&2; exit 1; }
WATCH="$REPO/integrations/handy/handy_watch.py"
TRANSCRIBE="$REPO/skills/transcriber/scripts/transcribe.py"

xml() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' <<<"$1"; }

"$AGENT_PYTHON" "$WATCH" --python "$ASR_PYTHON" --transcribe "$TRANSCRIBE" \
  --output-root "$OUTPUT_ROOT" --init

unload
cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(xml "$AGENT_PYTHON")</string>
        <string>-u</string>
        <string>$(xml "$WATCH")</string>
        <string>--python</string>
        <string>$(xml "$ASR_PYTHON")</string>
        <string>--transcribe</string>
        <string>$(xml "$TRANSCRIBE")</string>
        <string>--output-root</string>
        <string>$(xml "$OUTPUT_ROOT")</string>
        <string>--min-seconds</string>
        <string>$MIN_SECONDS</string>
    </array>
    <key>WatchPaths</key>
    <array>
        <string>$(xml "$RECORDINGS")</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardOutPath</key>
    <string>$(xml "$LOG")</string>
    <key>StandardErrorPath</key>
    <string>$(xml "$LOG")</string>
</dict>
</plist>
EOF
launchctl bootstrap "gui/$UID" "$PLIST"
echo "агент $LABEL стоит; лог: $LOG"
