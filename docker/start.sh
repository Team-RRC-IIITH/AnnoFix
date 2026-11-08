#!/usr/bin/env bash
# Inside the container: a virtual screen, a window manager that maximises AnnoFix, VNC on it,
# noVNC on port 6080, then AnnoFix on /data/samples (read-only) writing to /data/fixed.
# Extra AnnoFix options come from ANNOFIX_ARGS or the command line.
set -euo pipefail

export DISPLAY=:1
Xvfb :1 -screen 0 "${SCREEN_SIZE:-1600x900}x24" -nolisten tcp >/dev/null 2>&1 &
for _ in $(seq 100); do [[ -e /tmp/.X11-unix/X1 ]] && break; sleep 0.1; done
openbox --config-file /app/docker/openbox.xml >/dev/null 2>&1 &
x11vnc -display :1 -localhost -nopw -forever -shared -quiet -rfbport 5900 >/dev/null 2>&1 &
websockify --web /usr/share/novnc 6080 localhost:5900 >/dev/null 2>&1 &

if [[ ! -d /data/samples ]]; then
    echo "annofix: no samples mounted at /data/samples (set SAMPLES, see README)" >&2
    exit 2
fi
echo "AnnoFix is starting: open http://localhost:6080 in your browser"
# shellcheck disable=SC2086  # ANNOFIX_ARGS is a list of options
exec python -m annofix.cvat /data/samples --out /data/fixed ${ANNOFIX_ARGS:-} "$@"
