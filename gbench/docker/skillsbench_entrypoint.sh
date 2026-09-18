#!/usr/bin/env bash
# Entrypoint for the gbench-skillsbench image. Runs the launcher, which drives BenchFlow 0.6.3
# docker-out-of-docker. CRITICAL: BenchFlow bind-mounts its working dirs into sibling task
# containers, so SKILLSBENCH_WORKDIR must be an IDENTITY mount (same path on host + orchestrator)
# and TMPDIR must point there; the harness sets this up. The model-under-test is the gbench /v1
# endpoint (must be reachable FROM the sibling task containers); the deterministic verifier scores.
set -uo pipefail

SKILLSBENCH_DIR="${SKILLSBENCH_DIR:-/app/skillsbench}"
ADAPTER_DIR="${ADAPTER_DIR:-/app/adapter}"
export SKILLSBENCH_WORKDIR="${SKILLSBENCH_WORKDIR:-/out}"
export TMPDIR="${SKILLSBENCH_WORKDIR}"
mkdir -p "$SKILLSBENCH_WORKDIR"
cd "$SKILLSBENCH_DIR"

AGENT="${SKILLSBENCH_AGENT:-deepagents}"
if [ "$AGENT" != "oracle" ]; then
  : "${GBENCH_MODEL_BASE_URL:?GBENCH_MODEL_BASE_URL (a /v1 reachable from task containers) is required}"
  : "${GBENCH_MODEL_NAME:?GBENCH_MODEL_NAME (served model name) is required}"
fi
export GBENCH_MODEL_API_KEY="${GBENCH_MODEL_API_KEY:-dummy}"

# The Docker CLI must reach the mounted host daemon (docker-out-of-docker).
if ! docker version >/dev/null 2>&1; then
  echo "skillsbench: cannot reach the Docker daemon - mount /var/run/docker.sock into the container." >&2
  exit 3
fi

exec python3 "$ADAPTER_DIR/skillsbench_run.py"
