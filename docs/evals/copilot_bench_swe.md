# CopilotBench SWE-Bench Verified Evaluation Setup Guide

## Overview
`copilot_bench_swe` runs **SWE-bench Verified** (`princeton-nlp/SWE-bench_Verified`, 500
human-validated instances): the model produces a patch for a real GitHub issue across
Python repositories (Django, SymPy, Flask, Astropy, …), and the patch is applied and the
repository's tests run inside the task's Docker image. An instance resolves iff those
tests pass; the reported metric is the resolved rate.

> **Not a long-context eval.** The suite name and the earlier "up to 256K long context"
> claim are misleading: prompts are issue text plus the harness's context, typically a few
> thousand tokens, and nothing here selects or measures long-context instances.

## Prerequisites
SWE-bench requires Docker to spin up environment containers for target repositories and run isolated pytest suites.

---

### 1. System Requirements (Docker Engine)
`pip install docker` only installs the Python SDK. The actual **Docker Engine daemon** must be installed on your Linux host:

```bash
# 1. Check if Docker is already installed (requires ~50GB free disk space for repo containers)
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

### 2. Python Dependencies
Install the required packages (or just `pip install gbench[evals]`, which pins these):
```bash
pip install 'swebench>=4.1,<5' datasets docker
```
NOTE: swebench is pinned `>=4.1,<5`. v5.0.0+ removed `swebench.harness.test_spec.test_spec` and
changed the TestSpec schema (`make_test_spec` raises `KeyError: 'image'` on `SWE-bench_Verified`), so
v5 cannot build this suite's specs. gbench targets the v4 layout (validated with 4.1.0).

---

## Running the Evaluation in gbench

### Full Benchmark Run (All Repos)
```bash
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals copilot_bench_swe \
       --eval-thinking \
       --sandboxes 16
```

### Fast Smoke Test (`--eval-limit`)
```bash
# Smoke test on 1 issue
gbench --evals-only \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it \
       --evals copilot_bench_swe \
       --eval-thinking \
       --eval-limit 1
```

### Concurrency Separation (`--sandboxes` vs `--batch-sizes`)
- `--batch-sizes 128`: Used for standard HTTP model inference.
- `--sandboxes 16` or `32`: Caps concurrent Docker container test environments spawned for SWE-bench git checkouts and pytests.
