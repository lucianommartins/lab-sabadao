#!/usr/bin/env bash
# Entrypoint for the gbench-ui-control-osworld image. Starts a virtual display (OSWorld's agent deps
# pyautogui/pynput import an X display at module load), then runs the launcher, which drives the
# OSWorld harness docker-out-of-docker (it spawns sibling QEMU-VM containers on the host daemon).
set -uo pipefail

ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
export OSWORLD_DIR="${OSWORLD_DIR:-/app/OSWorld}"
export OSWORLD_WORKDIR="${OSWORLD_WORKDIR:-/out}"
mkdir -p "$OSWORLD_WORKDIR"

# The docker SDK must reach the mounted host daemon (docker-out-of-docker).
if ! python3 -c "import docker; docker.from_env().ping()" >/dev/null 2>&1; then
  echo "ui_control_osworld: cannot reach the Docker daemon via the SDK - mount /var/run/docker.sock." >&2
  exit 3
fi

# Virtual display for pyautogui/pynput imports.
Xvfb :99 -screen 0 1920x1080x24 >/dev/null 2>&1 &
XVFB_PID=$!
export DISPLAY=:99
sleep 1
trap 'kill "$XVFB_PID" 2>/dev/null || true' EXIT

if [ ! -e /dev/kvm ]; then
  echo "ui_control_osworld: WARNING /dev/kvm not present in this container - the sibling QEMU VM will" >&2
  echo "  fall back to TCG software emulation and almost certainly miss the 300s VM-ready timeout." >&2
fi

exec python3 "$ADAPTER_DIR/ui_control_osworld_run.py"
