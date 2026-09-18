# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: skillsbench
# Description: SkillsBench (benchflow-ai) - deterministic, execution-based agentic skills benchmark

"""gbench native built-in runner for skillsbench (Tool Use & Agentic Workflows).

Canonical SkillsBench (benchflow-ai/skillsbench): 87 tasks, each a self-contained per-task Docker
environment scored by a DETERMINISTIC pytest verifier (reward in [0,1] written to
/logs/verifier/reward.txt) - there is NO LLM judge anywhere. gbench DELEGATES the whole agentic
run to the bundled upstream BenchFlow runner inside a LOCAL orchestrator container
(gbench/docker/skillsbench.Dockerfile), run docker-OUT-of-docker: `bench eval run --backend docker`
builds + runs each TASK container as a sibling on the host daemon. The OpenCode agent runs inside
each task container, pointed at the gbench /v1 endpoint via an injected openai-compatible provider;
the task's verifier produces the reward.

Headline `accuracy` = mean reward over the tasks in the primary condition (default without-skills,
the model's raw capability). When both conditions run, the with-skills-vs-without lift (upstream's
signature metric) is also reported.

Because scoring is deterministic (no judge), a FULL run with the canonical OpenCode agent is in
principle leaderboard-comparable - but gbench runs a single trial (upstream averages 3) and may
cover a subset, so `leaderboard_comparable` is set True only for a full, both-condition run with the
canonical agent and otherwise False with the reason.

HARD-ERRORS (infra_required, never skips) if Docker, the locally-built image, or a reachable /v1
endpoint is missing. There is NO GEMINI dependency here (no judge). Heavy external provisioning
(Docker daemon, network egress for ~28 task builds + the agent install, a served model reachable
from task containers) - see docs/evals/skillsbench.md.

Sampling: gbench does NOT pin a temperature or reasoning mode for skillsbench - the delegated
BenchFlow agent (deepagents) samples at its own harness default (which is also how the leaderboard
runs each model), so `--temperature`/`--thinking`/`GBENCH_SKILLSBENCH_TEMPERATURE` are NOT forwarded
and would be silent no-ops; gbench does not record a temperature it never applied.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"
DOCS_URL = "docs/evals/skillsbench.md"

_IMAGE_DEFAULT = "gbench-skillsbench"
_FULL_TASK_COUNT = 87


def _image() -> str:
    return os.environ.get("GBENCH_SKILLSBENCH_IMAGE", _IMAGE_DEFAULT)


def check_skillsbench_prerequisites() -> Tuple[bool, str]:
    """Docker (with a reachable daemon) + the locally-built image. No judge/GEMINI needed."""
    image = _image()
    build = (f"Build the harness LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f gbench/docker/skillsbench.Dockerfile gbench/docker\n"
             f"It bundles pinned benchflow-ai/skillsbench + the BenchFlow runner. See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return False, f"image {image!r} not found. " + build
    return True, ""


def _task_reachable_endpoint(base_url: str) -> str:
    """Translate the gbench endpoint into a URL reachable FROM the sibling task containers.

    The orchestrator runs `--network host`, but the TASK containers BenchFlow spawns are on the
    default bridge, so 127.0.0.1 there is the container itself - they reach the host via the docker
    bridge gateway. `GBENCH_SKILLSBENCH_TASK_ENDPOINT` overrides the whole URL; else a
    localhost/127.0.0.1 host is rewritten to `GBENCH_SKILLSBENCH_TASK_HOST` (default 172.17.0.1).
    """
    override = os.environ.get("GBENCH_SKILLSBENCH_TASK_ENDPOINT")
    if override:
        return override
    ep = base_url.rstrip("/")
    if not ep.endswith("/v1"):
        ep += "/v1"
    host = os.environ.get("GBENCH_SKILLSBENCH_TASK_HOST", "172.17.0.1")
    return re.sub(r"://(127\.0\.0\.1|localhost)(?=[:/])", f"://{host}", ep)


def compute_skillsbench_score(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Roll the launcher's summary into a headline mean reward.

    Primary condition = without-skills if present, else the first condition with a mean reward.
    """
    conds = (summary or {}).get("conditions") or {}
    means = {c: v.get("mean_reward") for c, v in conds.items()
             if isinstance(v.get("mean_reward"), (int, float))}
    if not means:
        return None
    primary = "without-skills" if "without-skills" in means else next(iter(means))
    return {
        "primary_condition": primary,
        "mean_reward": means[primary],
        "condition_means": means,
        "with_skills_lift": summary.get("with_skills_lift"),
    }


def run_skillsbench(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical SkillsBench via the upstream BenchFlow runner (docker-out-of-docker) +
    deterministic per-task verifier."""
    ok, reason = check_skillsbench_prerequisites()
    if not ok:
        raise infra_required("skillsbench", reason, DOCS_URL)

    # gbench does NOT pin sampling for skillsbench: the delegated BenchFlow agent (deepagents) owns
    # temperature + reasoning mode, so `--temperature`/`--thinking` are intentionally NOT forwarded
    # to the harness (a passed value would be a silent no-op) and are not recorded as if applied.
    task_endpoint = _task_reachable_endpoint(base_url)
    conditions = os.environ.get("GBENCH_SKILLSBENCH_CONDITIONS", "without-skills")
    # The pinned BenchFlow (0.6.3) does not ship the leaderboard's `opencode` agent; its open-model
    # agents (deepagents/codex/...) read BENCHFLOW_PROVIDER_BASE_URL. Default to deepagents.
    agent = os.environ.get("GBENCH_SKILLSBENCH_AGENT", "deepagents")
    limit = kwargs.get("limit")

    # docker-out-of-docker: BenchFlow bind-mounts its working dirs into sibling task containers, so
    # the workdir must be an IDENTITY mount (same path on host + in the orchestrator) with TMPDIR
    # pointing there - otherwise the sibling sandbox gets empty mounts and no reward is collected.
    workdir = tempfile.mkdtemp(prefix="gbench_skillsbench_")
    os.chmod(workdir, 0o777)  # the container writes as its own uid
    orch_name = "gbench_skillsbench_" + os.path.basename(workdir)

    def _reap():
        # docker --rm only fires on a clean container EXIT; on a subprocess timeout the client is
        # killed while the orchestrator keeps running, so reap it by name in finally + on timeout.
        # (BenchFlow owns its per-task docker sandbox siblings, which it spawns with their own --rm;
        # gbench has no hook to label them, so their lifecycle stays with BenchFlow.)
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)

    try:
        cmd = ["docker", "run", "--rm", "--name", orch_name, "--network", "host",
               "-v", "/var/run/docker.sock:/var/run/docker.sock",
               "-v", f"{workdir}:{workdir}:rw",              # identity mount (DooD)
               "-e", f"SKILLSBENCH_WORKDIR={workdir}",
               "-e", f"GBENCH_MODEL_BASE_URL={task_endpoint}",
               "-e", f"GBENCH_MODEL_NAME={model_name}",
               "-e", f"GBENCH_MODEL_API_KEY={os.environ.get('GBENCH_SKILLSBENCH_API_KEY', 'dummy')}",
               "-e", f"SKILLSBENCH_CONDITIONS={conditions}",
               "-e", f"SKILLSBENCH_AGENT={agent}",
               "-e", f"SKILLSBENCH_CONCURRENCY={max(1, int(concurrency))}"]
        if limit and int(limit) > 0:
            cmd += ["-e", f"SKILLSBENCH_TASK_LIMIT={int(limit)}"]
        tasks_env = os.environ.get("GBENCH_SKILLSBENCH_TASKS")
        if tasks_env:
            cmd += ["-e", f"SKILLSBENCH_TASKS={tasks_env}"]
        # --shard I/N: the launcher (_task_list) consumes GBENCH_SHARD to select an interleaved
        # subset of the sorted task list (round-robin, applied before the limit), so forward it.
        if os.environ.get("GBENCH_SHARD"):
            cmd += ["-e", f"GBENCH_SHARD={os.environ['GBENCH_SHARD']}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_SKILLSBENCH_TIMEOUT_S", str(24 * 60 * 60)))
        logger.info("skillsbench: docker run %s (agent=%s, conditions=%s)", _image(), agent, conditions)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "skillsbench", f"the skillsbench container exceeded {timeout_s}s and was killed.",
                DOCS_URL) from exc

        summ_path = os.path.join(workdir, "skillsbench_summary.json")
        if not os.path.exists(summ_path):
            raise infra_required(
                "skillsbench",
                f"the container produced no summary (rc={proc.returncode}; a harness failure, not a "
                f"0%). tail: {(proc.stderr or proc.stdout or '')[-800:]}", DOCS_URL)
        with open(summ_path, encoding="utf-8") as f:
            summary = json.load(f)
    finally:
        _reap()
        shutil.rmtree(workdir, ignore_errors=True)

    scored = compute_skillsbench_score(summary)
    if scored is None:
        raise infra_required(
            "skillsbench", "no tasks produced a reward (all infra failures? check Docker egress + "
            "that the /v1 endpoint is reachable from task containers).", DOCS_URL)

    primary = scored["primary_condition"]
    n_tasks = (summary.get("conditions", {}).get(primary, {}) or {}).get("n_tasks")
    n_scored = (summary.get("conditions", {}).get(primary, {}) or {}).get("n_scored")
    both_conditions = len(scored["condition_means"]) >= 2
    # A sharded run covers only a subset by construction (n_tasks already reflects the shard size),
    # so it is never "full" regardless of coverage math.
    sharded = bool(os.environ.get("GBENCH_SHARD"))
    is_full = (not sharded) and (not limit or int(limit) <= 0) and (n_tasks or 0) >= _FULL_TASK_COUNT
    # Sampling is NOT part of the gate: gbench does not control temperature here (the agent harness
    # owns it, as on the leaderboard), so gating on a greedy value gbench never applied would be
    # meaningless. Comparability turns only on coverage, both-condition, and the canonical agent.
    leaderboard = bool(is_full and both_conditions and agent == "opencode")

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "skillsbench",
        "model_name": model_name,
        "status": "success",
        "accuracy": round(scored["mean_reward"] * 100.0, 2),  # headline = mean reward (%) of primary condition
        "mean_reward": round(scored["mean_reward"], 4),
        "primary_condition": primary,
        "condition_means": {k: round(v, 4) for k, v in scored["condition_means"].items()},
        "with_skills_lift": (round(scored["with_skills_lift"], 4)
                             if isinstance(scored.get("with_skills_lift"), (int, float)) else None),
        "n_tasks": n_tasks,
        "n_scored": n_scored,
        "agent": agent,
        "scoring": "deterministic per-task verifier (no LLM judge)",
        "sampling": ("deepagents (the delegated BenchFlow agent) owns sampling and reasoning mode; "
                     "gbench does not pin a temperature or forward --thinking (a passed value would "
                     "be a silent no-op), matching how the leaderboard runs each model under its own "
                     "harness defaults"),
        "raw_summary": summary,
        "metric": ("SkillsBench mean reward over per-task deterministic verifiers (reward in [0,1]); "
                   "headline is the primary condition, with the with-skills-vs-without lift when "
                   "both conditions run. Agent loop delegated to the upstream BenchFlow runner "
                   "(OpenCode agent); no LLM judge."),
        "leaderboard_comparable": leaderboard,
    }
    if not leaderboard:
        reasons = []
        if agent != "opencode":
            reasons.append(f"agent '{agent}' differs from the leaderboard's reference agent (opencode); "
                           f"the pinned BenchFlow 0.6.3 default is deepagents")
        if sharded:
            reasons.append(f"sharded run (GBENCH_SHARD={os.environ['GBENCH_SHARD']}) covers only a "
                           "subset of tasks")
        if not is_full:
            reasons.append("not the full 87-task set")
        if not both_conditions:
            reasons.append("only one condition run (upstream reports with-skills-vs-without)")
        reasons.append("single trial (upstream averages 3)")
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    else:
        result["leaderboard_comparable_reason"] = (
            "full both-condition run with the canonical OpenCode agent + deterministic verifier; "
            "gbench does not pin a temperature (the agent harness owns sampling, as on the "
            "leaderboard); note upstream averages 3 trials")
    return result
