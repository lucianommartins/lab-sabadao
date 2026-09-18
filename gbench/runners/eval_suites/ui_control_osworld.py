# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: ui_control_osworld
# Description: OSWorld - execution-based desktop computer-use benchmark (369 tasks, deterministic verifier)

"""gbench native built-in runner for ui_control_osworld (Tool Use & Agentic Workflows).

Canonical OSWorld (github.com/xlang-ai/OSWorld): 369 real desktop tasks over an Ubuntu-desktop VM. A
computer-use agent observes screenshots and emits pyautogui actions over many steps; each task is
scored by a DETERMINISTIC per-task evaluator (getters read the final VM state, compare to a
reference) -> a binary/graded result.txt in [0,1]. NO LLM judge (so NO GEMINI dependency).

gbench DELEGATES to the pinned OSWorld harness inside a LOCAL orchestrator image
(gbench/docker/ui_control_osworld.Dockerfile), driven docker-OUT-of-docker: OSWorld's `docker`
provider (DesktopEnv(provider_name="docker")) spawns a SIBLING QEMU-VM container
(happysixd/osworld-docker) per task on the host daemon, bind-mounting the Ubuntu qcow2. The agent
(mm_agents PromptAgent) runs host-side in the orchestrator and calls the served model over /v1
(wired via OPENAI_BASE_URL + a model-name rewrite, since vLLM rejects the routing alias).

Headline `accuracy` = mean per-task success (result.txt) over ALL selected tasks (a missing/errored
task counts as 0); the per-domain breakdown is also reported. `leaderboard_comparable` is always
False (self-hosted agent, single trial, subset-capable; the OSWorld leaderboard is maintainer-run).

HARD-ERRORS (infra_required, never skips / never a fake number) if Docker, the orchestrator image,
`/dev/kvm` (OSWorld's VM only boots with hardware virtualization), the provisioned Ubuntu qcow2, the
happysixd/osworld-docker task image, or a reachable model endpoint are missing. See
docs/evals/ui_control_osworld.md.

Sampling: gbench does NOT pin a temperature for ui_control_osworld - the mm_agents PromptAgent samples
at the harness/model default and sends no reasoning/thinking field, so `--temperature` /
`GBENCH_UI_CONTROL_OSWORLD_TEMPERATURE` and `enable_thinking` are not applied here (any of these is a
silent no-op). The reasoning mode is the harness's own canonical protocol; gbench does not override it.

Concurrency: canonical OSWorld run.py runs tasks SEQUENTIALLY on a SINGLE DesktopEnv VM, so a per-task
concurrency/--sandboxes value does not apply (the parallel run_multienv*.py variants are agent-specific
and are not used here).

Sharding: this suite bypasses base.run_eval_suite, so it forwards GBENCH_SHARD ("I/N") into the
orchestrator container itself; the launcher (ui_control_osworld_run.py) consumes it by round-robin
selecting shard I of N over the DETERMINISTIC sorted (domain, example_id) task list (flat[I-1::N],
applied BEFORE OSWORLD_LIMIT, matching the native compose order) and rebuilds the --test_all_meta_path
manifest so the harness iterates exactly that subset. The reported n_selected/n_scored/n_missing and
per_domain therefore reflect the shard, is_full is False for any real shard, and leaderboard_comparable
stays False (a shard is a subset by construction). Missing/errored shard tasks still count as 0.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"
DOCS_URL = "docs/evals/ui_control_osworld.md"

_IMAGE_DEFAULT = "gbench-ui-control-osworld"
_TASK_IMAGE_DEFAULT = "happysixd/osworld-docker"
_FULL_TASK_COUNT = 369


def _image() -> str:
    return os.environ.get("GBENCH_UI_CONTROL_OSWORLD_IMAGE", _IMAGE_DEFAULT)


def _task_image() -> str:
    return os.environ.get("GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE", _TASK_IMAGE_DEFAULT)


def _vm_dir() -> str:
    return os.environ.get("GBENCH_OSWORLD_VM_DIR", "").strip()


def _docker_image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def check_ui_control_osworld_prerequisites() -> Tuple[bool, str]:
    """Docker + orchestrator image + /dev/kvm + provisioned qcow2 + task image. No judge/GEMINI."""
    image = _image()
    build = (f"Build the orchestrator LOCALLY (gbench never pulls it):\n"
             f"  docker build -t {image} -f gbench/docker/ui_control_osworld.Dockerfile gbench/docker\n"
             f"See " + DOCS_URL + " for the full provisioning recipe.")
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if not _docker_image_exists(image):
        return False, f"orchestrator image {image!r} not found. " + build

    if not os.path.exists("/dev/kvm"):
        return False, ("/dev/kvm is not present - OSWorld's Ubuntu VM only becomes ready within its "
                       "300s timeout with hardware virtualization (KVM). Run on a host with nested "
                       "virtualization enabled (bare metal or a nested-virt-licensed VM). This is the "
                       "binding external prerequisite. See " + DOCS_URL)
    vm_dir = _vm_dir()
    if not vm_dir:
        return False, ("GBENCH_OSWORLD_VM_DIR is unset. Provision the Ubuntu VM disk once "
                       "(docker_vm_data/Ubuntu.qcow2 from HF xlangai/ubuntu_osworld, ~12GB) and point "
                       "GBENCH_OSWORLD_VM_DIR at the directory containing docker_vm_data/. See " + DOCS_URL)
    if not os.path.isfile(os.path.join(vm_dir, "docker_vm_data", "Ubuntu.qcow2")):
        return False, (f"{vm_dir!r} has no docker_vm_data/Ubuntu.qcow2 - download + unzip the OSWorld "
                       f"Ubuntu qcow2 there (HF xlangai/ubuntu_osworld). See " + DOCS_URL)
    if not _docker_image_exists(_task_image()):
        return False, (f"the task VM image {_task_image()!r} is not on the host daemon. Provision it "
                       f"with `docker pull {_task_image()}` (the canonical OSWorld VM runner). See " + DOCS_URL)
    return True, ""


def _model_endpoint(base_url: str) -> str:
    """--network host -> the served model is at 127.0.0.1; just ensure a /v1 suffix."""
    override = os.environ.get("GBENCH_OSWORLD_MODEL_ENDPOINT")
    if override:
        return override
    ep = (base_url or "").rstrip("/")
    return ep if ep.endswith("/v1") else ep + "/v1"


def compute_ui_control_osworld_score(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not summary or not isinstance(summary.get("success_rate"), (int, float)):
        return None
    return {"success_rate": summary["success_rate"], "per_domain": summary.get("per_domain")}


def run_ui_control_osworld(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run canonical OSWorld via the pinned harness (docker-out-of-docker QEMU-VM siblings)."""
    # `enable_thinking` and `concurrency` are accepted for a uniform runner signature but are NOT
    # forwarded to the harness (verified against the pinned OSWorld ref):
    #  - reasoning mode: OSWorld's mm_agents PromptAgent sends a plain OpenAI-compat payload (model,
    #    messages, max_tokens, top_p, temperature) with NO reasoning/thinking field, so enable_thinking
    #    has nothing to act on. The agent's reasoning mode IS the canonical protocol; gbench does not
    #    override it (honest-record via the "sampling" field below, not a fabricated "thinking" record).
    #  - concurrency: canonical OSWorld run.py runs tasks strictly SEQUENTIALLY on a SINGLE DesktopEnv
    #    VM (nested for-loop, no multiprocessing; the parallel run_multienv*.py variants are
    #    agent-specific and are not used here), so a per-task concurrency/--sandboxes value does not
    #    apply (documented via the "execution" field below, not silently dropped).
    ok, reason = check_ui_control_osworld_prerequisites()
    if not ok:
        raise infra_required("ui_control_osworld", reason, DOCS_URL)

    vm_dir = _vm_dir()
    model_endpoint = _model_endpoint(base_url)
    domain = os.environ.get("GBENCH_OSWORLD_DOMAIN", "").strip()
    obs_type = os.environ.get("GBENCH_OSWORLD_OBS_TYPE", "screenshot").strip()
    max_steps = os.environ.get("GBENCH_OSWORLD_MAX_STEPS", "15").strip()
    test_meta = os.environ.get("GBENCH_OSWORLD_TEST_META", "test_all.json").strip()
    eff_limit = os.environ.get("GBENCH_OSWORLD_LIMIT", "").strip() or (str(limit) if limit else "")

    orch_name = "gbench_osworld_orch_" + os.path.basename(tempfile.mkdtemp(prefix="gbench_osworld_"))

    def _reap():
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)
        # sibling VM containers are named by the provider; reap by ancestor image as a backstop.
        ps = subprocess.run(["docker", "ps", "-aq", "--filter", f"ancestor={_task_image()}"],
                            capture_output=True, text=True)
        ids = [x for x in (ps.stdout or "").split() if x]
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)

    try:
        cmd = ["docker", "run", "--rm", "--name", orch_name, "--network", "host",
               "-v", "/var/run/docker.sock:/var/run/docker.sock",
               "-v", f"{vm_dir}:{vm_dir}:rw",                 # identity mount (qcow2 source for siblings)
               "-e", f"OSWORLD_WORKDIR={vm_dir}",             # docker_vm_data/ + results/ live here
               "-e", f"GBENCH_MODEL_BASE_URL={model_endpoint}",
               "-e", f"GBENCH_MODEL_NAME={model_name}",
               "-e", f"GBENCH_MODEL_API_KEY={os.environ.get('GBENCH_OSWORLD_MODEL_API_KEY', 'dummy')}",
               "-e", f"OSWORLD_OBS_TYPE={obs_type}",
               "-e", f"OSWORLD_MAX_STEPS={max_steps}",
               "-e", f"OSWORLD_TEST_META={test_meta}"]
        if domain:
            cmd += ["-e", f"OSWORLD_DOMAIN={domain}"]
        if eff_limit:
            cmd += ["-e", f"OSWORLD_LIMIT={eff_limit}"]
        # Forward the shard ONLY because the launcher consumes it to select an interleaved task subset
        # (rebuilds the --test_all_meta_path manifest); the harness then iterates exactly that subset.
        # The launcher applies the shard BEFORE OSWORLD_LIMIT, matching the native path's compose order.
        if os.environ.get("GBENCH_SHARD"):
            cmd += ["-e", f"GBENCH_SHARD={os.environ['GBENCH_SHARD']}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_UI_CONTROL_OSWORLD_TIMEOUT_S", str(48 * 60 * 60)))
        logger.info("ui_control_osworld: docker run %s (obs=%s, domain=%s, limit=%s)",
                    _image(), obs_type, domain or "all", eff_limit or "none")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "ui_control_osworld", f"the orchestrator exceeded "
                f"GBENCH_UI_CONTROL_OSWORLD_TIMEOUT_S ({timeout_s}s) and was killed; sibling VM "
                f"containers reaped. Raise the timeout or reduce the scope.", DOCS_URL) from exc

        summ_path = os.path.join(vm_dir, "ui_control_osworld_summary.json")
        if not os.path.exists(summ_path):
            raise infra_required(
                "ui_control_osworld", f"the orchestrator produced no summary (rc={proc.returncode}; a "
                f"harness failure, not a 0%). tail: {(proc.stderr or proc.stdout or '')[-800:]}", DOCS_URL)
        with open(summ_path, encoding="utf-8") as f:
            summary = json.load(f)
    finally:
        _reap()

    scored = compute_ui_control_osworld_score(summary)
    if scored is None:
        raise infra_required(
            "ui_control_osworld", "no tasks produced a result (all VM/infra failures? check /dev/kvm, "
            "the qcow2, the task image, and the model endpoint reachability).", DOCS_URL)

    n_selected = summary.get("n_selected") or 0
    n_scored = summary.get("n_scored") or 0
    n_missing = summary.get("n_missing") or 0
    # A sharded run covers a round-robin subset by construction, so it is never the full set.
    shard_spec = os.environ.get("GBENCH_SHARD", "").strip()
    sharded = bool(shard_spec) and shard_spec != "1/1"
    is_full = (not domain and not eff_limit and not sharded and n_selected >= _FULL_TASK_COUNT)

    reasons = ["self-hosted agent + single trial (the OSWorld leaderboard is maintainer-run)",
               f"observation_type={obs_type}"]
    if not is_full:
        reasons.append(f"subset (domain={domain or 'all'}, limit={eff_limit or 'none'}, "
                       f"shard={shard_spec or 'none'})")
    if n_missing:
        reasons.append(f"{n_missing} task(s) produced no result (counted as 0)")

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "ui_control_osworld",
        "model_name": model_name,
        "status": "success",
        "accuracy": round(scored["success_rate"] * 100.0, 2),   # headline = mean task success (%)
        "success_rate": round(scored["success_rate"], 4),
        "per_domain": scored.get("per_domain"),
        "n_selected": n_selected,
        "n_scored": n_scored,
        "n_missing": n_missing,
        "observation_type": obs_type,
        "scoring": "OSWorld deterministic per-task execution evaluators (result.txt in [0,1]); no LLM judge",
        "sampling": ("OSWorld harness owns sampling and reasoning mode: the mm_agents PromptAgent samples "
                     "at the harness default temperature and sends a plain OpenAI-compat payload with NO "
                     "reasoning/thinking field, so gbench does not pin --temperature or enable_thinking "
                     "here (either would be a silent no-op)"),
        "execution": ("sequential, single-VM: canonical OSWorld run.py drives one DesktopEnv VM through a "
                      "sequential per-task loop (no multiprocessing), so concurrency/--sandboxes does not "
                      "apply"),
        "raw_summary": summary,
        "metric": ("OSWorld success rate = mean of per-task result.txt (0-1) over ALL selected tasks "
                   "(a missing/errored task counts as 0). Agent = mm_agents PromptAgent (pyautogui "
                   "actions, screenshot obs) in a docker-provider Ubuntu VM, driven by the served "
                   "model via OPENAI_BASE_URL. Deterministic execution-based grading; no LLM judge."),
        "leaderboard_comparable": False,
        "leaderboard_comparable_reason": "; ".join(reasons),
    }
    return result
