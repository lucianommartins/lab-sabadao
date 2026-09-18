# Canonical SkillsBench orchestrator image for gbench's `skillsbench` suite, built LOCALLY.
#
# SkillsBench (benchflow-ai/skillsbench) is a DETERMINISTIC, execution-based agentic benchmark:
# 87 tasks, each a self-contained per-task Docker environment scored by a pytest verifier that
# writes /logs/verifier/reward.txt (reward in [0,1]) - NO LLM judge anywhere. This orchestrator
# bakes the BenchFlow runner (`bench` CLI, via `uv sync`) + the pinned task set, and drives it
# docker-OUT-of-docker: `bench eval run --backend docker` builds/runs each TASK container as a
# sibling on the host daemon (mount /var/run/docker.sock at run time). The OpenCode agent runs
# INSIDE each task container, pointed at the gbench /v1 endpoint via an injected openai-compatible
# provider; the deterministic verifier produces the reward. No judge, no GEMINI key.
#
# Build (context = gbench/docker):
#   docker build -t gbench-skillsbench -f gbench/docker/skillsbench.Dockerfile gbench/docker
#
# A graded run needs: a reachable Docker daemon (socket mounted), network egress (the ~28 tasks
# that fetch public toolchains/data at build + the OpenCode agent install), and a served model at
# /v1 reachable FROM the sibling task containers. The model-free `oracle` agent needs only Docker
# + egress. See docs/evals/skillsbench.md.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Base: git, curl, python, and the Docker CLI (client only - it talks to the mounted host daemon).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gnupg python3 python3-venv \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# uv (astral) for `uv sync` / `uv run bench`.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
ENV PATH="/usr/local/bin:${PATH}"

# Clone SkillsBench at a pinned commit (bakes tasks/ + the open-model runner) and uv-sync it so
# `uv run bench` resolves the BenchFlow harness.
ARG SKILLSBENCH_REF=9a1f4dd5f7659f75707435da3ce854b6e48321d1
RUN git clone https://github.com/benchflow-ai/skillsbench /app/skillsbench \
    && cd /app/skillsbench && git checkout "${SKILLSBENCH_REF}"
WORKDIR /app/skillsbench
RUN uv sync --locked || uv sync

# BenchFlow's docker sandbox drives `docker compose` (v2) against the host daemon, so the
# orchestrator needs the compose plugin (the CLI alone lacks the `compose` subcommand).
RUN apt-get update && apt-get install -y --no-install-recommends docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# gbench adapter: the launcher (delegates to the upstream runner with a gbench model target +
# parses rewards) and the entrypoint.
RUN mkdir -p /app/adapter
COPY skillsbench_run.py /app/adapter/skillsbench_run.py
COPY skillsbench_entrypoint.sh /app/adapter/skillsbench_entrypoint.sh
RUN chmod +x /app/adapter/skillsbench_entrypoint.sh

ENV SKILLSBENCH_DIR=/app/skillsbench
ENV ADAPTER_DIR=/app/adapter
ENTRYPOINT ["/app/adapter/skillsbench_entrypoint.sh"]
