# Canonical GAIA2 / Meta-ARE orchestrator image for gbench's `gaia2` suite, built LOCALLY.
#
# GAIA2 (Meta Agents Research Environments, github.com/facebookresearch/meta-agents-research-environments)
# is a STATEFUL multi-turn agentic benchmark: the agent acts inside an IN-PROCESS Python simulator
# (contacts/calendar/email/files with time-sensitive + async events), issuing tool calls over many
# turns; scoring is HYBRID - deterministic "hard validation" + a load-bearing LLM judge for
# soft/semantic checks. Unlike skillsbench/wildclawbench there is NO docker-out-of-docker and NO
# sibling task containers here: `are-benchmark` runs the whole simulation in this one container. It
# MUST be a container (not the host environment): ARE 1.2.0 hard-pins ~22 deps with `==` (numpy==2.2.6,
# litellm==1.71.1, mcp==1.11.0, pydantic==2.10.6, ...) that collide with the vLLM/torch stack.
#
# Build (context = gbench/docker):
#   docker build -t gbench-gaia2 -f gbench/docker/gaia2.Dockerfile gbench/docker
#
# The model-under-test is served at gbench /v1 (OpenAI-compatible) and reached with --network host.
# The judge is swapped to gbench's Gemini cascade via an in-container OpenAI-compat proxy
# (gaia2_cascade_judge.py, reused verbatim from wildclawbench) with ARE's judge PROMPTS untouched.
# A graded run needs: a reachable served model + GEMINI_API_KEY (judge). The public cc-by-4.0
# dataset is BAKED at build (no HF token). See docs/evals/gaia2.md.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*

# Pin the ARE harness. Its 22 hard `==` pins resolve inside this isolated image only.
ARG ARE_VERSION=1.2.0
RUN pip install "meta-agents-research-environments==${ARE_VERSION}"
ENV ARE_VERSION=1.2.0

# Bake the public GAIA2 datasets into the HF hub cache so a graded run is self-contained + offline
# (cc-by-4.0, anonymous download - no token): the scenarios (`gaia2`, all 7 configs) AND the
# filesystem-app assets (`gaia2_filesystem`, lazy-loaded per scenario) - baking the latter avoids the
# anonymous paths-info 429 rate-limiting seen at runtime.
RUN python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("meta-agents-research-environments/gaia2",
             "meta-agents-research-environments/gaia2_filesystem"):
    p = snapshot_download(repo, repo_type="dataset")
    print("cached", repo, "at", p)
PY
# NOTE: do NOT force HF_HUB_OFFLINE/HF_DATASETS_OFFLINE here - datasets' load_dataset needs online
# hub resolution and returns 0 scenarios offline. The baked cache makes file CONTENT local; ARE's
# filesystem app still lazy-loads file STATS via the HF paths-info API and may hit anonymous 429
# rate-limits, which it handles via fallback_file_system (non-fatal). Provide HF_TOKEN to remove the
# rate-limit warnings on a full run (see docs).

# gbench adapter: the launcher (drives are-benchmark + parses benchmark_stats.json), the reused
# Gemini cascade-judge proxy, and the entrypoint.
RUN mkdir -p /app/adapter
COPY gaia2_run.py /app/adapter/gaia2_run.py
COPY wildclawbench_cascade_judge.py /app/adapter/gaia2_cascade_judge.py
COPY gaia2_entrypoint.sh /app/adapter/gaia2_entrypoint.sh
RUN chmod +x /app/adapter/gaia2_entrypoint.sh

ENV ADAPTER_DIR=/app/adapter
ENTRYPOINT ["/app/adapter/gaia2_entrypoint.sh"]
