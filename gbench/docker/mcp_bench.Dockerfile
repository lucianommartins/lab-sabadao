# Canonical MCP-Bench image for gbench's `mcp_bench` suite, built LOCALLY (never pulled).
#
# Bundles Accenture's MCP-Bench runner + all 28 vendored MCP servers, built from source by the
# upstream install.sh (which tolerates per-server failures: one flaky server does not abort the
# build). The gbench launcher (mcp_bench_run.py) injects the gbench Gemini cascade judge, filters
# tasks to the servers actually provisioned, and delegates to upstream benchmark.runner.main().
#
# Build (context = gbench/docker, so the adapter files are in scope):
#   docker build -t gbench-mcp-bench -f gbench/docker/mcp_bench.Dockerfile gbench/docker
#
# HEAVY image: builds ~9 TypeScript servers (tsc/tsup) + ~19 Python servers (uv), including the
# fragile BioMCP/alphagenome, metmuseum (hand-built tsc), and numpy/scipy servers. Tens of
# minutes. A graded run additionally needs a served model at /v1, GEMINI_API_KEY (judge), and
# network egress (17 servers hit public APIs); see docs/evals/mcp_bench.md.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Make uv pip installs land in the system interpreter so `python <server>.py` servers resolve
# their deps (the `uv run` servers still use their own uv-managed venvs).
ENV UV_SYSTEM_PYTHON=1

# Base toolchain: python 3.11 (deadsnakes, matches upstream's tested version), build tools, git,
# sudo (install.sh calls sudo), and Node 22 LTS (NodeSource). Pre-installing these makes
# install.sh's version checks skip re-installing them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl wget git gnupg sudo \
        build-essential gcc g++ make \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/python3 + get pip for it.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py && rm -f /tmp/get-pip.py

# uv (astral) on PATH for all users.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
ENV PATH="/usr/local/bin:${PATH}"

# Clone MCP-Bench at a pinned commit (self-contained; gbench never pulls a prebuilt image).
ARG MCPBENCH_REF=7a8eaeae83a842a2949080acc5473f65e1569daf
RUN git clone https://github.com/Accenture/mcp-bench /app/mcp-bench \
    && cd /app/mcp-bench && git checkout "${MCPBENCH_REF}"

WORKDIR /app/mcp-bench

# Build all 28 servers + install runner deps via the upstream installer (tolerant of per-server
# failures). Run from mcp_servers/ (where install.sh lives and its server loops are rooted).
RUN cd /app/mcp-bench/mcp_servers && bash install.sh || true

# Post-install fixes for the servers install.sh leaves incomplete on a fresh Linux image (each
# tolerant so one failure never aborts the build): OKX needs typescript before tsc; Google Maps
# needs its tsup build run; and the 4 `uv run` servers need a real .venv (UV_SYSTEM_PYTHON, which
# we set so `python <server>.py` servers resolve system-wide, otherwise suppresses `uv sync`'s venv).
RUN set -x; cd /app/mcp-bench/mcp_servers; \
    ( cd okx-mcp && npm install typescript --no-save && npx tsc ) || echo "WARN okx build failed"; \
    ( cd mcp-google-map && npm install && npm run build ) || echo "WARN google-map build failed"; \
    ( cd openapi-mcp-server && npm install ) || echo "WARN openapi npm failed"; \
    for d in wikipedia-mcp biomcp mcp-reddit nasa-mcp; do \
      ( cd "/app/mcp-bench/mcp_servers/$d" && env -u UV_SYSTEM_PYTHON uv sync ) || echo "WARN uv sync failed: $d"; \
    done; \
    true

# The `fastmcp`-based servers need mcp>=2 + full fastmcp, which conflicts with the mcp<2 the v1
# servers require. Give each its OWN venv (isolation, like the uv-run servers); the launcher
# repoints their commands.json entry at that venv's python at start-up.
RUN set -x; for d in unit-converter-mcp game-trends-mcp mcp-osint-server paper-search-mcp; do \
      ( cd "/app/mcp-bench/mcp_servers/$d" \
        && env -u UV_SYSTEM_PYTHON uv venv \
        && ( [ -f uv.lock ] && env -u UV_SYSTEM_PYTHON uv sync || true ) \
        && ( [ -f pyproject.toml ] && env -u UV_SYSTEM_PYTHON uv pip install -e . || true ) \
        && ( [ -f requirements.txt ] && env -u UV_SYSTEM_PYTHON uv pip install -r requirements.txt || true ) \
        && ( [ -f mcp_osint_server/requirements.txt ] && env -u UV_SYSTEM_PYTHON uv pip install -r mcp_osint_server/requirements.txt || true ) \
        && env -u UV_SYSTEM_PYTHON uv pip install fastmcp ) || echo "WARN fastmcp venv failed: $d"; \
    done; \
    true

# Runner + judge deps in the system interpreter (belt-and-suspenders over install.sh's uv step):
# the launcher/judge import openai + json_repair, the runner imports mcp/jsonschema/etc, and the
# `python <file>.py` v1 servers need mcp<2 (pinned LAST so it wins over anything that pulled v2).
RUN python3.11 -m pip install --no-cache-dir openai json_repair \
    && ( python3.11 -m pip install --no-cache-dir -r /app/mcp-bench/mcp_servers/requirements.txt || true ) \
    && ( python3.11 -m pip install --no-cache-dir mcp jsonschema pyyaml aiohttp tenacity || true ) \
    && ( python3.11 -m pip install --no-cache-dir -e /app/mcp-bench/mcp_servers/time-mcp || true ) \
    && python3.11 -m pip install --no-cache-dir "mcp<2"

# gbench adapter (cascade judge + launcher + entrypoint), from the build context (gbench/docker).
RUN mkdir -p /app/adapter
COPY mcp_bench_cascade_judge.py /app/adapter/mcp_bench_cascade_judge.py
COPY mcp_bench_run.py /app/adapter/mcp_bench_run.py
COPY mcp_bench_entrypoint.sh /app/adapter/mcp_bench_entrypoint.sh
RUN chmod +x /app/adapter/mcp_bench_entrypoint.sh

ENV MCPBENCH_DIR=/app/mcp-bench
ENV ADAPTER_DIR=/app/adapter
ENTRYPOINT ["/app/adapter/mcp_bench_entrypoint.sh"]
