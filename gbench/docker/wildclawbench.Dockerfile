# Canonical WildClawBench orchestrator image for gbench's `wildclawbench` suite, built LOCALLY.
#
# WildClawBench (github.com/internlm/WildClawBench) is a live, long-horizon agentic benchmark: 60
# tasks, each run in its own Docker container from the prebaked image `wildclawbench-ubuntu:v1.3`
# (a 13.5GB `docker load` from HF), with the OpenClaw agent inside and per-task grading (programmatic
# checks + an LLM judge) run afterwards by `docker exec`. This orchestrator bakes the pinned InternLM
# harness's Python runtime + the gbench adapter, and drives `eval/run_batch.py` docker-OUT-of-docker
# (mount /var/run/docker.sock at run time). The OpenClaw agent is pointed at the gbench /v1 endpoint
# via an injected `my_api.json` custom provider; the tasks' verbatim OpenAI-SDK judge calls are
# pointed at the gbench Gemini cascade proxy (wildclawbench_cascade_judge.py) instead of
# OpenRouter/gpt-5.4.
#
# Build (context = gbench/docker):
#   docker build -t gbench-wildclawbench -f gbench/docker/wildclawbench.Dockerfile gbench/docker
#
# PROVISIONED HOST CHECKOUT (swe_lancer-style): the large, partly un-redistributable workspace (3
# YouTube videos) and the 13.5GB task image are NOT baked. A graded run needs, on the host:
#   * a WildClawBench checkout with its HF `workspace/` downloaded + `bash script/prepare.sh` run
#     (videos + SAM3 weights); this dir is IDENTITY-mounted into the orchestrator (DooD requires the
#     workspace bind-mount source to resolve on the host);
#   * `docker load` of wildclawbench-ubuntu:v1.3 on the host daemon;
#   * GEMINI_API_KEY (judge cascade), a served model at /v1 reachable FROM the task containers, and
#     BRAVE_API_KEY for the Search & Retrieval tasks.
# See docs/evals/wildclawbench.md.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Base: git, curl, python, and the Docker CLI (client only - it talks to the mounted host daemon).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gnupg python3 python3-pip \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# The only third-party deps run_batch.py's openclaw path needs at import + run time are python-dotenv
# and pyyaml (weasyprint/pymupdf/yt-dlp/modelscope in the harness requirements are HOST-side
# provisioning / task-image-side grading tools, not orchestrator deps). The judge proxy is stdlib.
RUN pip3 install --no-cache-dir --break-system-packages python-dotenv pyyaml

# Bake the pinned harness as a provenance/version reference (the graded run uses the IDENTITY-mounted
# host checkout, not this copy - a container path can't be a host bind-mount source). WILDCLAW_REF
# is the SHA gbench pins and docs against.
ARG WILDCLAW_REF=316334ccc4a87b9b5635ad73da99b4dfc0b3887e
RUN git clone https://github.com/internlm/WildClawBench /opt/wildclawbench-ref \
    && cd /opt/wildclawbench-ref && git checkout "${WILDCLAW_REF}" \
    && echo "${WILDCLAW_REF}" > /opt/wildclawbench-ref/.gbench_pinned_ref
ENV WILDCLAW_REF=316334ccc4a87b9b5635ad73da99b4dfc0b3887e

# gbench adapter: the launcher (drives run_batch + parses the score tree), the cascade-judge proxy,
# and the entrypoint.
RUN mkdir -p /app/adapter
COPY wildclawbench_run.py /app/adapter/wildclawbench_run.py
COPY wildclawbench_cascade_judge.py /app/adapter/wildclawbench_cascade_judge.py
COPY wildclawbench_entrypoint.sh /app/adapter/wildclawbench_entrypoint.sh
RUN chmod +x /app/adapter/wildclawbench_entrypoint.sh

ENV ADAPTER_DIR=/app/adapter
ENTRYPOINT ["/app/adapter/wildclawbench_entrypoint.sh"]
