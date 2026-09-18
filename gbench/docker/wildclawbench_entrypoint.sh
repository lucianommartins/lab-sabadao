#!/usr/bin/env bash
# Entrypoint for the gbench-wildclawbench orchestrator image. Starts the Gemini cascade-judge proxy
# (the tasks' verbatim OpenAI-SDK judge calls hit it instead of OpenRouter/gpt-5.4), then runs the
# launcher, which drives the pinned InternLM harness docker-out-of-docker.
#
# DooD identity-mount contract (skillsbench pitfall): run_batch bind-mounts each task's
# <workspace>/exec into the sibling task container, and `-v <path>` from inside a container resolves
# on the HOST. So WILDCLAW_HARNESS_DIR (with its workspace/ data) must be an IDENTITY mount (same
# path on host + orchestrator); the gbench runner arranges that. TMPDIR is pointed at the workdir.
set -uo pipefail

HARNESS_DIR="${WILDCLAW_HARNESS_DIR:-/app/WildClawBench}"
ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
export WILDCLAW_WORKDIR="${WILDCLAW_WORKDIR:-/out}"
export TMPDIR="${WILDCLAW_WORKDIR}"
mkdir -p "$WILDCLAW_WORKDIR"

# The Docker CLI must reach the mounted host daemon (docker-out-of-docker).
if ! docker version >/dev/null 2>&1; then
  echo "wildclawbench: cannot reach the Docker daemon - mount /var/run/docker.sock into the container." >&2
  exit 3
fi

# The harness root must be present (identity-mounted checkout with its workspace/ data).
if [ ! -f "$HARNESS_DIR/eval/run_batch.py" ]; then
  echo "wildclawbench: harness not found at $HARNESS_DIR (identity-mount the provisioned WildClawBench checkout)." >&2
  exit 4
fi

# Start the cascade-judge proxy (OpenAI-compatible) in the background.
JUDGE_PORT="${WILDCLAW_JUDGE_PORT:-18790}"
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "wildclawbench: WARNING GEMINI_API_KEY is unset - the judge proxy will return 503 and the" >&2
  echo "               43 judged tasks will silently regex-fall-back (recorded as judge_fallback)." >&2
fi
WILDCLAW_JUDGE_PORT="$JUDGE_PORT" nohup python3 "$ADAPTER_DIR/wildclawbench_cascade_judge.py" \
    > "$WILDCLAW_WORKDIR/cascade_judge.log" 2>&1 &
JUDGE_PID=$!
sleep 1
if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
  echo "wildclawbench: cascade judge proxy failed to start; see $WILDCLAW_WORKDIR/cascade_judge.log" >&2
fi
trap 'kill "$JUDGE_PID" 2>/dev/null || true' EXIT

exec python3 "$ADAPTER_DIR/wildclawbench_run.py"
