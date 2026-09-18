# TerminalBench 2.1 (TB2) Evaluation Setup Guide

> **Docker setup:** this suite runs one container/network per task and can exhaust the
> Docker address pool at high `--sandboxes`. See [docker-sandboxes.md](docker-sandboxes.md)
> for the recommended `default-address-pools` config and pre-run prune, strongly advised
> before running at concurrency.


## Overview
`terminal_bench` evaluates autonomous agent execution across interactive Linux terminal troubleshooting tasks, Bash scripts, and CLI workflows.

## Prerequisites
Terminal-Bench executes candidate commands inside isolated Docker sandbox containers managed by the **Harbor** agent evaluation framework.

---

### 1. System Requirements (Docker Engine)
`pip install docker` only installs the Python SDK. The actual **Docker Engine daemon** must be installed on your Linux host:

```bash
# 1. Check if Docker is already installed
docker --version && docker info

# 2. If Docker is NOT installed, install Docker Engine:
# Option A: Standard Ubuntu/Debian package (zero config)
sudo apt-get update && sudo apt-get install -y docker.io
# Option B: Upstream Docker CE (if preferred)
# Follow https://docs.docker.com/engine/install/

# 3. Ensure Docker service is running
sudo systemctl enable --now docker

# 4. Grant non-root user permissions to access /var/run/docker.sock
sudo groupadd -f docker
sudo usermod -aG docker $USER
newgrp docker
```

---

### 2. Python & Framework Dependencies
Install the official Docker Python SDK and Harbor framework:

```bash
# Via pip
pip install docker harbor

# Or via uv (recommended)
uv tool install harbor
```

---

### 3. Install Terminal-Bench 2.1 (TB2.1) Dataset
Download TB2.1 tasks directly from the [Harbor Registry](https://hub.harborframework.com/datasets):

```bash
harbor dataset download terminal-bench/terminal-bench-2-1
```

---

## Running the Evaluation in gbench

### Full Benchmark Run (All Tasks)
```bash
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals terminal_bench \
       --eval-thinking \
       --sandboxes 32
```

### Fast Smoke Test (`--eval-limit`)
Run a quick single-task or small-sample smoke test to verify setup:

```bash
# Smoke test on 1 task
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals terminal_bench \
       --eval-thinking \
       --eval-limit 1

# Quick test on 5 tasks with 16 parallel sandbox containers
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals terminal_bench \
       --eval-thinking \
       --eval-limit 5 \
       --sandboxes 16
```

### Concurrency Separation (`--sandboxes` vs `--batch-sizes`)
When running mixed evaluation campaigns (e.g. `--evals all` or combining academic QA evals with agent benchmarks):
- `--batch-sizes 128` controls concurrent HTTP API requests for high-throughput academic benchmarks (`mmlu_pro`, `aime`, `gsm8k`).
- `--sandboxes 32` or `--sandboxes 64` explicitly caps the number of simultaneous Docker containers spawned on the host system to prevent container daemon exhaustion while maintaining maximum throughput.

## Behavior when prerequisites are missing

This suite **hard-errors** (never skips, never reports a fabricated 0%): if Docker or the `harbor`
CLI is unavailable, or if the Harbor harness crashes or completes with **zero parseable trials**,
the run raises `infra_required` and is recorded as a `status:"error"` row (the sweep continues).
Provision the prerequisites and re-run. A run that lost some trials to parse/harness errors is
marked `partial: true` (with `parse_failures` / `trial_errors` counts) and is not comparable.

## Harbor version

The result-parsing here follows Harbor **0.20.x**'s `result.json` schema (`n_completed_trials`,
`evals[<name>].metrics[].mean`, `verifier_result.rewards`). Pin the tested version:

```bash
pip install 'harbor==0.20.*'   # or: uv tool install harbor
```

Older Harbor releases emit a different schema and will parse to zero trials (→ hard-error).

## Configuration knobs

| Env var | Default | Effect |
| --- | --- | --- |
| `GBENCH_TERMINAL_BENCH_TEMPERATURE` | *(run default)* | per-suite sampling override; else 0.0 greedy (no-think) / 1.0 (`--thinking`). Precedes `--temperature`. |
| `GBENCH_TB_PARSER` | `xml` | Terminus-2 action parser. The `xml` parser can salvage a turn that hit the token cap; the canonical `json` default cannot (a truncated turn is re-asked → livelock). `xml` is non-canonical (see leaderboard note). |
| `GBENCH_TB_TURN_MAX_TOKENS` | `8192` (plain) / `32768` (`--thinking`) | per-turn `max_tokens` backstop for the agent's LLM calls. `0` restores uncapped (the canonical setting, but risks a single turn consuming the whole agent budget on a slow endpoint). |
| `GBENCH_TB_TIMEOUT_MULTIPLIER` | `2.0` (plain) / `4.0` (`--thinking`) | multiplies Harbor's per-task agent timeout so a slow local model is not cut off. Canonical is `1.0`; anything else is non-comparable. |
| `GBENCH_TB_CONTAINER_THREADS` | `1` | thread cap inside each task container (tasks pin `cpus=1`, but `nproc` still reports the host's cores → oversubscription). `0`/`off` restores unbounded. |
| `GBENCH_TB_KEEP_JOBS_DIR` | *(unset)* | keep Harbor's jobs dir (`result.json` / `trial.log` / `agent/trajectory.json`) for diagnosis. A path keeps there; `1` keeps in a temp dir; unset keeps only on failure. |

## Leaderboard comparability

`leaderboard_comparable` is set on the result and is **True only** for a full run (no
`--eval-limit`) with every knob at its canonical value (`GBENCH_TB_TIMEOUT_MULTIPLIER=1.0`,
`GBENCH_TB_TURN_MAX_TOKENS=0`, `GBENCH_TB_PARSER=json`) and no lost trials. **The shipped defaults
(turn cap + `xml` parser + inflated timeout multiplier) are deliberately non-canonical** to make a
slow local endpoint measurable, so a default run reports `leaderboard_comparable: false` with the
active deviations listed in `leaderboard_comparable_reason`. This is intentional and honest. Set
all three canonical knobs for a leaderboard-comparable run. The canonical `terminal-bench-2-1` set
is **89 tasks**.
