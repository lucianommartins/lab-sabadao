# Canonical OSWorld orchestrator image for gbench's `ui_control_osworld` suite, built LOCALLY.
#
# OSWorld (github.com/xlang-ai/OSWorld) is an EXECUTION-BASED desktop computer-use benchmark: 369
# real tasks over an Ubuntu-desktop VM. A computer-use agent observes screenshots and emits
# pyautogui mouse/keyboard actions over many steps; each task is scored by a DETERMINISTIC per-task
# evaluator (getters read the final VM state + compare to a reference) -> NO LLM judge. This
# orchestrator bakes the pinned OSWorld harness + its deps + the docker SDK, and drives the run
# docker-OUT-of-docker: OSWorld's `docker` provider (DesktopEnv(provider_name="docker")) spawns a
# SIBLING QEMU-VM container (happysixd/osworld-docker) on the host daemon per task, bind-mounting the
# Ubuntu qcow2 disk. The agent (mm_agents PromptAgent) runs host-side in this orchestrator and calls
# the served model over /v1.
#
# Build (context = gbench/docker):
#   docker build -t gbench-ui-control-osworld -f gbench/docker/ui_control_osworld.Dockerfile gbench/docker
#
# A GRADED run REQUIRES a host with /dev/kvm (nested virtualization): the OSWorld VM only becomes
# ready (300s /screenshot poll) with hardware acceleration; without KVM QEMU falls back to TCG
# software emulation and the desktop will not boot in time. It also needs the ~12GB Ubuntu.qcow2
# (HF xlangai/ubuntu_osworld) at an identity-mounted path and the served model. NO judge / NO
# GEMINI. See docs/evals/ui_control_osworld.md.
FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System libs: xvfb (pyautogui/pynput import a display at module load), opencv runtime libs, git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates xvfb libgl1 libglib2.0-0 x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Pin the OSWorld harness (no upstream tags -> pin a commit).
ARG OSWORLD_REF=fc31a9049664292fcb35d6e501ee1dc839f2cf6d
RUN git clone https://github.com/xlang-ai/OSWorld /app/OSWorld \
    && cd /app/OSWorld && git checkout "${OSWORLD_REF}"
ENV OSWORLD_REF=fc31a9049664292fcb35d6e501ee1dc839f2cf6d
WORKDIR /app/OSWorld
# Harness deps + the docker SDK (the `docker` provider uses docker.from_env(), not the CLI).
RUN pip install -r requirements.txt && pip install "docker>=7,<8"

# gbench adapter.
RUN mkdir -p /app/adapter
COPY ui_control_osworld_run.py /app/adapter/ui_control_osworld_run.py
COPY ui_control_osworld_entrypoint.sh /app/adapter/ui_control_osworld_entrypoint.sh
RUN chmod +x /app/adapter/ui_control_osworld_entrypoint.sh

ENV OSWORLD_DIR=/app/OSWorld
ENV ADAPTER_DIR=/app/adapter
ENTRYPOINT ["/app/adapter/ui_control_osworld_entrypoint.sh"]
