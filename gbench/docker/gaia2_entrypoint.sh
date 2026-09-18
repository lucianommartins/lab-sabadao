#!/usr/bin/env bash
# Entrypoint for the gbench-gaia2 image. Optionally starts the Gemini cascade-judge proxy (ARE's
# judge calls hit it instead of the canonical Llama-3.3-70B), then runs the launcher which drives
# `are-benchmark`. No docker-out-of-docker here (ARE is a pure in-process simulator).
set -uo pipefail

ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
export GAIA2_WORKDIR="${GAIA2_WORKDIR:-/out}"
mkdir -p "$GAIA2_WORKDIR"

JUDGE_PORT="${GAIA2_JUDGE_PORT:-18790}"
# Start the cascade proxy only when GEMINI_API_KEY is present AND no explicit judge endpoint override
# was given (an override, e.g. self-judge against the served model, is used for smoke tests).
if [ -n "${GEMINI_API_KEY:-}" ] && [ -z "${GAIA2_JUDGE_ENDPOINT:-}" ]; then
  WILDCLAW_JUDGE_PORT="$JUDGE_PORT" nohup python3 "$ADAPTER_DIR/gaia2_cascade_judge.py" \
      > "$GAIA2_WORKDIR/cascade_judge.log" 2>&1 &
  JUDGE_PID=$!
  sleep 1
  if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
    echo "gaia2: cascade judge proxy failed to start; see $GAIA2_WORKDIR/cascade_judge.log" >&2
  fi
  trap 'kill "$JUDGE_PID" 2>/dev/null || true' EXIT
elif [ -z "${GAIA2_JUDGE_ENDPOINT:-}" ]; then
  echo "gaia2: WARNING neither GEMINI_API_KEY nor GAIA2_JUDGE_ENDPOINT is set - ARE's load-bearing" >&2
  echo "       LLM judge has no model; most scenarios will score no_validation." >&2
fi

exec python3 "$ADAPTER_DIR/gaia2_run.py"
