# ui_control_osworld setup

> **Status: deregistered (roadmap-only, not runnable).** `ui_control_osworld` is not in the `SUITES`
> registry, so `--evals ui_control_osworld` / `--evals all` will not run it. The setup below is
> retained for the roadmap; a graded run needs `/dev/kvm` (see docs/evals_roadmap.md).

Canonical **OSWorld** ([github.com/xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)), an
**execution-based** desktop computer-use benchmark: **369 real tasks** over an Ubuntu-desktop VM. A
computer-use agent observes screenshots and emits `pyautogui` mouse/keyboard actions over many
steps; each task is scored by a **deterministic per-task evaluator** (getters read the final VM
state and compare to a reference), writing `result.txt` ∈ [0,1]. **No LLM judge → no GEMINI key.**

Pinned harness: commit `fc31a9049664292fcb35d6e501ee1dc839f2cf6d`.

## What gbench changes (everything else is upstream verbatim)
- Delegates to the pinned OSWorld harness inside a LOCAL orchestrator image
  (`gbench/docker/ui_control_osworld.Dockerfile`), driven **docker-out-of-docker**: OSWorld's
  `docker` provider (`DesktopEnv(provider_name="docker")`) spawns a **sibling QEMU-VM container**
  (`happysixd/osworld-docker`) per task on the host daemon, bind-mounting the Ubuntu qcow2.
- **Model wiring**: the agent (`mm_agents.PromptAgent`) runs host-side in the orchestrator and calls
  the served model over `/v1`. Its `call_llm` routes by model-name prefix; the `gpt` branch honours
  `OPENAI_BASE_URL`. Since vLLM rejects an unknown model id, gbench passes `--model gpt-4o` (a routing
  alias → the OpenAI-compat path), sets `OPENAI_BASE_URL` to the served endpoint, and monkeypatches
  the outgoing payload `model` field to the real served name.
- **DooD identity-mount** (the skillsbench/wildclaw pitfall): the provider bind-mounts the qcow2 via
  `os.path.abspath(path_to_vm)`, and `VMS_DIR` is `./docker_vm_data` relative to CWD, so the
  orchestrator chdir's to the identity-mounted `GBENCH_OSWORLD_VM_DIR` (which holds
  `docker_vm_data/Ubuntu.qcow2` + `results/`), and that path is identical on the host daemon.

## Metric
Headline **`accuracy`** = mean per-task success (`result.txt`) over **all selected tasks** (a
missing/errored task counts as 0); the per-domain breakdown, `n_selected`/`n_scored`/`n_missing` are
also reported. gbench never fabricates a scalar; a run that produces no summary hard-errors.

## Prerequisites (the suite HARD-ERRORS via `infra_required`, never skips / never a fake number)

### 1. A host with `/dev/kvm` (nested virtualization), the binding blocker
OSWorld's Ubuntu VM only becomes ready within its 300 s `/screenshot` timeout with **hardware
virtualization**. Without `/dev/kvm` (e.g. a GCP VM without nested-virt licensing) QEMU falls back to
TCG software emulation and the desktop will not boot in time → the suite hard-errors. Run on bare
metal or a nested-virt-enabled VM.

### 2. Build the LOCAL orchestrator image (gbench never pulls it)
```bash
docker build -t gbench-ui-control-osworld -f gbench/docker/ui_control_osworld.Dockerfile gbench/docker
```
Bakes the pinned OSWorld harness + its deps (torch/transformers/opencv/pyautogui) + the docker SDK +
the gbench adapter. (These deps hard-conflict with the vLLM stack, hence a container, not the gbench serving environment.)

### 3. Provision the Ubuntu VM disk (`GBENCH_OSWORLD_VM_DIR`)
Download + unzip the OSWorld Ubuntu qcow2 (~12 GB) into `docker_vm_data/`, and point
`GBENCH_OSWORLD_VM_DIR` at the parent (it is identity-mounted; also receives `results/`):
```bash
mkdir -p /srv/osworld/docker_vm_data && cd /srv/osworld/docker_vm_data
hf download xlangai/ubuntu_osworld Ubuntu.qcow2.zip --repo-type dataset --local-dir .
unzip Ubuntu.qcow2.zip          # -> Ubuntu.qcow2
export GBENCH_OSWORLD_VM_DIR=/srv/osworld
```

### 4. The task VM image + the served model
```bash
docker pull happysixd/osworld-docker    # the canonical OSWorld VM runner (loaded on the host daemon)
```
- A **served model at `/v1`** reachable from the orchestrator (with `--network host`, `127.0.0.1`).
  For screenshot tasks the model must be multimodal.

## Run
```bash
export GBENCH_OSWORLD_VM_DIR=/srv/osworld
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals ui_control_osworld
```

## Split topology: nested-virt CPU VM + a remote GPU-served model

**Use this when your GPU host cannot provide `/dev/kvm`.** GCP **accelerator machine families
(A2/A3/G2) do not support nested virtualization** (nor does E2), so an H100 box such as
`a3-highgpu-8g` can never expose `/dev/kvm`, and no container can conjure it (a container shares the
host kernel). OSWorld needs *no GPU* of its own - it is VM/CPU work that only needs a reachable model
endpoint - so run the orchestrator on a small **nested-virt-enabled CPU VM** and point it at the model
still served on the GPU box. Nothing in the runner changes; only the endpoint URL and the host differ.

**1. Serve the model on the GPU host so the CPU VM can reach it.** vLLM binds `127.0.0.1` by default;
start it with `--host 0.0.0.0` and open the port inside your VPC:
```bash
# on the GPU host (e.g. the a3-highgpu-8g box)
vllm serve google/gemma-4-26B-A4B-it --host 0.0.0.0 --port 8000   # ...your usual serving flags
# allow the CPU VM's subnet to reach it (adjust to your VPC/tags)
gcloud compute firewall-rules create allow-osworld-to-vllm \
  --network=<your-vpc> --direction=INGRESS --action=ALLOW \
  --rules=tcp:8000 --source-ranges=<cpu-vm-subnet-cidr>
```

**2. Create a nested-virt CPU VM** (Haswell+; N2 / N2D / C3 are eligible - NOT E2 / A-series). Put it
in the **same zone + VPC** as the GPU host so it reaches the model over the internal network:
```bash
gcloud compute instances create osworld-runner \
  --zone=<same-zone-as-gpu-host> \
  --machine-type=n2-standard-8 \
  --enable-nested-virtualization \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=150GB --boot-disk-type=pd-ssd \
  --network=<your-vpc> --subnet=<same-subnet>
```

**3. Confirm `/dev/kvm` exists on the new VM:**
```bash
sudo apt-get update && sudo apt-get install -y cpu-checker && sudo kvm-ok   # -> "KVM acceleration can be used"
ls -l /dev/kvm                                                              # -> present
```

**4. Provision the OSWorld prerequisites on that VM** exactly as in the sections above, on the CPU VM:
   install Docker (add your user to the `docker` group), install gbench (`pip install gbench` - a
   `--remote-endpoint` run needs neither vLLM nor a GPU, see the deps policy), then do prereq steps
   **2-4**: build the orchestrator image, download+unzip the Ubuntu qcow2 into
   `$GBENCH_OSWORLD_VM_DIR/docker_vm_data/`, and `docker pull happysixd/osworld-docker`.

**5. Run, pointing `--remote-endpoint` at the GPU host's internal IP** (the orchestrator runs with
`--network host`, so it reaches that IP directly):
```bash
export GBENCH_OSWORLD_VM_DIR=/srv/osworld
GPU_HOST_IP=$(gcloud compute instances describe <gpu-host-name> --zone=<zone> \
  --format='get(networkInterfaces[0].networkIP)')
gbench --evals-only \
       --remote-endpoint "http://${GPU_HOST_IP}:8000/v1" \
       --tokenizer google/gemma-4-26B-A4B-it \
       --evals ui_control_osworld
# smoke first: GBENCH_OSWORLD_LIMIT=3 (or --eval-limit 3)
```

**6. Tear down** the CPU VM when done (it bills while it exists):
`gcloud compute instances delete osworld-runner --zone=<zone>`.

> Notes. The model must be **multimodal** for the default `screenshot` observation type. If the GPU
> host is unreachable by IP, verify the firewall rule and that vLLM was started with `--host 0.0.0.0`
> (`curl http://${GPU_HOST_IP}:8000/v1/models` from the CPU VM should list the model). Everything else
> - reaping, the identity-mount, scoring - is identical to the single-host path.

## Environment knobs
| Env var | Default | Meaning |
|---|---|---|
| `GBENCH_UI_CONTROL_OSWORLD_IMAGE` | `gbench-ui-control-osworld` | orchestrator image tag |
| `GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE` | `happysixd/osworld-docker` | sibling VM image (host-loaded) |
| `GBENCH_OSWORLD_VM_DIR` | (required) | identity-mounted dir holding `docker_vm_data/Ubuntu.qcow2` + `results/` |
| `GBENCH_OSWORLD_MODEL_ENDPOINT` | (derived) | override the model `/v1` URL |
| `GBENCH_OSWORLD_DOMAIN` | (all) | single OSWorld domain, or `all` |
| `GBENCH_OSWORLD_LIMIT` | (none) | cap total tasks (smoke) |
| `GBENCH_OSWORLD_OBS_TYPE` | `screenshot` | `screenshot` / `a11y_tree` / `screenshot_a11y_tree` / `som` |
| `GBENCH_OSWORLD_MAX_STEPS` | 15 | per-task step budget |
| `GBENCH_OSWORLD_TEST_META` | `test_all.json` | manifest under `evaluation_examples/` |
| `GBENCH_UI_CONTROL_OSWORLD_TIMEOUT_S` | 172800 | orchestrator wall-clock cap (timeout → infra_required + sibling VMs reaped) |
| `GBENCH_UI_CONTROL_OSWORLD_TEMPERATURE` / `--temperature` | (not applied) | **no-op**, sampling follows the OSWorld/model default |

## What hard-errors (`infra_required`, never a skip / never a fake number)
- Docker CLI/daemon not reachable, or the orchestrator image not built.
- `/dev/kvm` absent (no hardware virtualization).
- `GBENCH_OSWORLD_VM_DIR` unset, or missing `docker_vm_data/Ubuntu.qcow2`.
- The `happysixd/osworld-docker` task image not on the host daemon.
- The orchestrator produced no summary (a harness failure, `status:error`, not 0%).

## Leaderboard comparability
Always **`leaderboard_comparable=False`**: a self-hosted agent + single trial, subset-capable, and
the official OSWorld leaderboard is maintainer-run. Scoring itself is the canonical deterministic
execution evaluator (no judge), so the per-task numbers are faithful; the headline is honest about
missing tasks (counted as 0).
