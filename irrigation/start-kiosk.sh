#!/bin/bash

set -u

LOG_FILE="$HOME/irrigation-kiosk.log"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_URL="file://$APP_DIR/index.html"

{
  echo "----- $(date) -----"
  echo "Starting irrigation kiosk"
  echo "User: $(whoami)"
  echo "Display: ${DISPLAY:-not set}"
  echo "App URL: $APP_URL"
} >> "$LOG_FILE" 2>&1

sleep 8

xset s off >> "$LOG_FILE" 2>&1 || true
xset -dpms >> "$LOG_FILE" 2>&1 || true
xset s noblank >> "$LOG_FILE" 2>&1 || true

pkill -f "chromium.*irrigation" >> "$LOG_FILE" 2>&1 || true

if command -v chromium-browser >/dev/null 2>&1; then
  BROWSER="chromium-browser"
elif command -v chromium >/dev/null 2>&1; then
  BROWSER="chromium"
else
  echo "Could not find chromium-browser or chromium" >> "$LOG_FILE" 2>&1
  exit 1
fi

echo "Using browser: $BROWSER" >> "$LOG_FILE" 2>&1

exec "$BROWSER" \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --check-for-update-interval=31536000 \
  "$APP_URL" >> "$LOG_FILE" 2>&1
