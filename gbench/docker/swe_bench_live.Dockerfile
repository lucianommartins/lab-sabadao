# Canonical SWE-bench-Live scoring image for gbench's `swe_bench_live` suite, built LOCALLY.
#
# SWE-bench-Live ships the `swebench` harness as a FORK: same package name, a version incompatible
# with the upstream `swebench` the main gbench env uses (swe_bench_multilingual / copilot_bench_swe).
# Only one `swebench` can be installed per environment, so gbench isolates the fork HERE instead of a
# separate host virtualenv. gbench runs `swebench.harness.run_evaluation` inside this image; the
# harness spawns the per-instance task containers (DockerHub namespace `starryzhang`) on the HOST
# daemon via the mounted docker socket (docker-out-of-docker). The model rollout (patch generation)
# stays in the gbench process against the served /v1 endpoint and needs only `datasets`, so nothing
# fork-versioned leaks into the main env.
#
# Build (context = gbench/docker):
#   docker build -t gbench-swe-bench-live -f gbench/docker/swe_bench_live.Dockerfile gbench/docker
#
# A graded run needs: a reachable Docker daemon (socket mounted), network egress (the per-instance
# `starryzhang/*` task images are pulled from DockerHub), and a served model at /v1. See
# docs/evals/swe_bench_live.md.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Base: python + git, and the Docker CLI (client only - it talks to the mounted host daemon).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gnupg python3 python3-pip \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# The SWE-bench-Live fork of `swebench` at a pinned commit, plus `datasets` (the harness loads the
# dataset inside the container). --break-system-packages: ubuntu 24.04's python is externally managed
# and this image is single-purpose.
ARG SWE_BENCH_LIVE_REF=a145aa87c62361532dacaa243398978164b234b7
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "git+https://github.com/SWE-bench-Live/SWE-bench-Live@${SWE_BENCH_LIVE_REF}" datasets \
    && ln -sf "$(command -v python3)" /usr/local/bin/python

# No ENTRYPOINT: gbench's scorer invokes `docker run <image> python -m swebench.harness.run_evaluation ...`
# with the predictions + workdir identity-mounted and the docker socket mounted for the task containers.
