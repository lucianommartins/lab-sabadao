# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: wildclawbench
# Description: WildClawBench (InternLM real-world long-horizon autonomous agent benchmark - 60 tasks)

"""gbench native built-in runner for wildclawbench (Tool Use & Agentic Workflows).

Canonical WildClawBench (github.com/internlm/WildClawBench): 60 hand-built long-horizon tasks across
6 categories (EN+ZH). Each task runs in its own Docker container from the prebaked image
`wildclawbench-ubuntu:v1.3` (a 13.5GB `docker load` from HF), with the OpenClaw agent inside; after
the agent finishes, the task's own `grade()` (programmatic checks + an LLM judge) is `docker exec`'d
for a per-metric 0-1 score. gbench DELEGATES the whole run to the pinned upstream harness
(`eval/run_batch.py`) inside a LOCAL orchestrator container (gbench/docker/wildclawbench.Dockerfile),
run docker-OUT-of-docker: run_batch builds/runs each TASK container as a sibling on the host daemon.

Two things gbench injects, VERBATIM everything else:
  * the OpenClaw agent is pointed at the gbench /v1 model endpoint via a `my_api.json` custom provider
    (baseUrl rewritten to a task-reachable host, like skillsbench);
  * the tasks' judge calls (all 43 judged tasks use `OpenAI(base_url=OPENROUTER_BASE_URL).chat...(
    model=JUDGE_MODEL)` and declare those keys in their `## Env`) are pointed at the gbench Gemini
    cascade proxy instead of OpenRouter/gpt-5.4. The judge PROMPTS are untouched.

Headline `accuracy` = the harness's own equal-weight global mean of per-task overall_score over ALL
selected tasks (a missing/errored task counts as 0) - i.e. run_batch's global_avg. This is NOT the
leaderboard's Overall Score, which follows a weighted multimodal/pure-text breakdown (approximated
separately by `weighted_overall`); the MM/pure-text means, avg time, and avg cost are also reported.
Because the headline is the equal-weight global mean (the leaderboard uses a weighted breakdown) and
a run here is a single trial, `leaderboard_comparable` is always False; grading uses gbench's standard
Gemini cascade by convention, which is a gbench grader choice, not a defect.

HARD-ERRORS (infra_required, never skips) if Docker, the orchestrator image, the provisioned host
checkout (with prepared workspace), the loaded task image, GEMINI_API_KEY, BRAVE_API_KEY (the
OpenClaw gateway won't boot without it), or a reachable model endpoint are missing. See
docs/evals/wildclawbench.md.

Sampling: gbench does NOT pin a temperature for wildclawbench - the OpenClaw agent samples at its
harness default (which is also how the leaderboard runs each model), so the usual `--temperature` /
`GBENCH_WILDCLAWBENCH_TEMPERATURE` overrides are NOT applied here (the OpenClaw harness owns
sampling; a value passed there would be a silent no-op, so gbench does not accept the illusion of
control). The judge cascade is pinned at 0.0. Concurrency: with the default host-network task
containers the OpenClaw gateway's fixed port forces serial execution (parallel=1);
GBENCH_WILDCLAWBENCH_TASK_NETWORK=bridge allows parallelism.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"
DOCS_URL = "docs/evals/wildclawbench.md"

_IMAGE_DEFAULT = "gbench-wildclawbench"
_TASK_IMAGE_DEFAULT = "wildclawbench-ubuntu:v1.3"
_FULL_TASK_COUNT = 60
_JUDGE_PORT_DEFAULT = 18791  # distinct from gaia2's 18790 so concurrent same-host judge proxies never collide


def _image() -> str:
    return os.environ.get("GBENCH_WILDCLAWBENCH_IMAGE", _IMAGE_DEFAULT)


def _task_image() -> str:
    return os.environ.get("GBENCH_WILDCLAWBENCH_TASK_IMAGE", _TASK_IMAGE_DEFAULT)


def _harness_dir() -> str:
    return prereqs_path("WildClawBench",
                        os.environ.get("GBENCH_WILDCLAWBENCH_HOST_DIR", "").strip()) or ""


def _docker_image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True).returncode == 0


# Live Gemini key ping lives in base.py (single source shared by every judge suite); the alias
# keeps the module-global name that this suite's key gate and the tests reference.
from .base import gemini_key_live_valid as _gemini_key_valid, suite_env, free_port


def _brave_key_valid(key: str) -> Tuple[bool, str]:
    """Live ping of the Brave Search API. (False, reason) ONLY on a definitive auth rejection
    (401/403/422); network/other is inconclusive -> (True, ...)."""
    url = "https://api.search.brave.com/res/v1/web/search?q=ping&count=1"
    req = urllib.request.Request(url, headers={"X-Subscription-Token": key,
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return (200 <= getattr(r, "status", 200) < 300), ""
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 422):
            return False, f"HTTP {e.code}"
        return True, f"inconclusive HTTP {e.code}"
    except Exception as e:
        return True, f"inconclusive {e}"


def check_wildclawbench_prerequisites() -> Tuple[bool, str]:
    """Docker + orchestrator image + provisioned host checkout + loaded task image + GEMINI key."""
    image = _image()
    build = (f"Build the orchestrator LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f gbench/docker/wildclawbench.Dockerfile gbench/docker\n"
             f"See " + DOCS_URL + " for the full provisioning recipe.")
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if not _docker_image_exists(image):
        return False, f"orchestrator image {image!r} not found. " + build

    harness = _harness_dir()
    if not harness:
        return False, ("GBENCH_WILDCLAWBENCH_HOST_DIR is unset. Provision a WildClawBench checkout "
                       "(clone the harness, `hf download` its workspace, run `bash script/prepare.sh`) "
                       "and point GBENCH_WILDCLAWBENCH_HOST_DIR at it. See " + DOCS_URL)
    if not os.path.isfile(os.path.join(harness, "eval", "run_batch.py")):
        return False, (f"GBENCH_WILDCLAWBENCH_HOST_DIR={harness!r} is not a WildClawBench checkout "
                       f"(eval/run_batch.py missing). See " + DOCS_URL)
    if not os.path.isdir(os.path.join(harness, "workspace")):
        return False, (f"the checkout at {harness!r} has no workspace/ (run the HF workspace "
                       f"download + `bash script/prepare.sh`). See " + DOCS_URL)

    if not _docker_image_exists(_task_image()):
        return False, (f"the task image {_task_image()!r} is not loaded on the host daemon. "
                       f"Download it from HF and `docker load -i wildclawbench-ubuntu_v1.3.tar`. "
                       f"See " + DOCS_URL)
    if not os.environ.get("GEMINI_API_KEY"):
        return False, ("GEMINI_API_KEY is unset - required for the WildClawBench judge cascade "
                       "(43 of 60 tasks grade with an LLM judge). See " + DOCS_URL)
    if not os.environ.get("BRAVE_API_KEY"):
        return False, ("BRAVE_API_KEY is unset - the OpenClaw gateway baked into "
                       "wildclawbench-ubuntu:v1.3 treats tools.web.search.apiKey as a REQUIRED "
                       "secret and refuses to start without it, so ALL tasks fail (not just the "
                       "Search & Retrieval ones). Get a free key at brave.com/search/api. See "
                       + DOCS_URL)

    # Live-validate the keys (a dummy/expired key otherwise only surfaces mid-run as judge/search
    # failures). Only a definitive auth rejection hard-errors; network flakiness is inconclusive and
    # passes. Set GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION=1 for air-gapped / offline provisioning.
    if not os.environ.get("GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION"):
        ok_g, why_g = _gemini_key_valid(os.environ["GEMINI_API_KEY"])
        if not ok_g:
            return False, (f"GEMINI_API_KEY was rejected by the Gemini API ({why_g}); the judge "
                           f"cascade cannot run. Check the key or set "
                           f"GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION=1 to bypass. See " + DOCS_URL)
        ok_b, why_b = _brave_key_valid(os.environ["BRAVE_API_KEY"])
        if not ok_b:
            return False, (f"BRAVE_API_KEY was rejected by the Brave Search API ({why_b}); OpenClaw's "
                           f"web search (and its gateway) need a valid key. Check the key or set "
                           f"GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION=1 to bypass. See " + DOCS_URL)
    return True, ""


def _task_reachable_endpoint(base_url: str, env_var: str, default_host: str = "172.17.0.1") -> str:
    """Translate a gbench URL into one reachable FROM the sibling task containers.

    run_batch's task containers are on the default bridge, so a localhost/127.0.0.1 host must be
    rewritten to the docker bridge gateway. `{env_var}` overrides the whole URL; else the host is
    swapped for `GBENCH_WILDCLAWBENCH_TASK_HOST` (default 172.17.0.1).
    """
    override = os.environ.get(env_var)
    if override:
        return override
    ep = (base_url or "").rstrip("/")
    if not ep.endswith("/v1"):
        ep += "/v1"
    host = os.environ.get("GBENCH_WILDCLAWBENCH_TASK_HOST", default_host)
    return re.sub(r"://(127\.0\.0\.1|localhost)(?=[:/])", f"://{host}", ep)


def compute_wildclawbench_score(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the headline mean and the modality breakdown out of the launcher summary."""
    if not summary or not isinstance(summary.get("global_mean"), (int, float)):
        return None
    return {
        "global_mean": summary["global_mean"],
        "multimodal_mean": summary.get("multimodal_mean"),
        "pure_text_mean": summary.get("pure_text_mean"),
        "weighted_overall": summary.get("weighted_overall"),
        "avg_time_min": summary.get("avg_time_min"),
        "avg_cost_usd": summary.get("avg_cost_usd"),
    }


def run_wildclawbench(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run canonical WildClawBench via the pinned InternLM harness (docker-out-of-docker).

    `limit` (--eval-limit N) caps the run to the first N selected tasks: upstream run_batch has no
    --limit flag, so it is forwarded as WILDCLAW_LIMIT and the launcher honours it by driving the
    harness single-task mode over the first N selected ids (the summary denominator is capped to
    match, so accuracy stays over exactly those N tasks).

    `--shard I/N` (GBENCH_SHARD, set by the gbench CLI) selects shard I of N over the deterministic
    sorted task-id list by round-robin (ids[i-1::n], matching sampling.shard_select), composed BEFORE
    the limit. Upstream run_batch has no --shard flag, so GBENCH_SHARD is forwarded into the container
    and the launcher drives single-task mode over exactly that sharded set, so run-set == score-set:
    the summary's n_tasks/accuracy cover only the shard, is_full is False (n_tasks < 60), and
    leaderboard_comparable stays False (a shard is a subset by construction).
    """
    ok, reason = check_wildclawbench_prerequisites()
    if not ok:
        raise infra_required("wildclawbench", reason, DOCS_URL)

    harness = _harness_dir()
    # Fixed default when free (deterministic single run); else an auto free port so two overlapping
    # wildclawbench (or wildclawbench + gaia2) runs do not collide on the host judge-proxy port. An
    # explicit GBENCH_WILDCLAWBENCH_JUDGE_PORT still forces a specific port.
    _env_judge_port = os.environ.get("GBENCH_WILDCLAWBENCH_JUDGE_PORT")
    judge_port = int(_env_judge_port) if _env_judge_port else free_port(_JUDGE_PORT_DEFAULT)
    # Task-container network. Default "host": the sibling task containers share the host net ns, so
    # they reach the model + judge proxy at 127.0.0.1 (the orchestrator is also --network host).
    # "bridge" = the upstream default; then endpoints are rewritten to the docker bridge gateway.
    task_network = os.environ.get("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "host").strip()
    if task_network == "host":
        ep = (base_url or "").rstrip("/")
        task_endpoint = os.environ.get("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT",
                                       ep if ep.endswith("/v1") else ep + "/v1")
        judge_endpoint = os.environ.get("GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT",
                                        f"http://127.0.0.1:{judge_port}/v1")
    else:
        task_endpoint = _task_reachable_endpoint(base_url, "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT")
        judge_endpoint = _task_reachable_endpoint(
            f"http://172.17.0.1:{judge_port}", "GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT")
    categories = os.environ.get("GBENCH_WILDCLAWBENCH_CATEGORIES", "")
    single_task = os.environ.get("GBENCH_WILDCLAWBENCH_TASK", "")
    parallel = int(os.environ.get("GBENCH_WILDCLAWBENCH_PARALLEL", str(max(1, int(concurrency)))))
    # In host-network mode every sibling task container shares the host net ns, so the OpenClaw
    # gateway's FIXED port (18789) would collide across parallel tasks -> only one binds, the rest
    # fail. Serialize. (bridge mode gives each container its own ns, so parallel is safe there.)
    if task_network == "host" and parallel > 1:
        logger.warning("wildclawbench: task-network=host -> forcing parallel=1 (the OpenClaw gateway "
                       "port is fixed and would collide across host-net siblings). Use "
                       "GBENCH_WILDCLAWBENCH_TASK_NETWORK=bridge for parallelism.")
        parallel = 1

    # docker-out-of-docker identity mounts (skillsbench pitfall): the harness dir (with workspace/)
    # is bind-mounted into sibling task containers, so it must be at an identity path; the workdir
    # holds output + summary and is TMPDIR.
    workdir = tempfile.mkdtemp(prefix="gbench_wildclawbench_")
    os.chmod(workdir, 0o777)
    orch_name = "gbench_wildclaw_orch_" + os.path.basename(workdir)
    run_label = "gbench_wildclaw_run=" + os.path.basename(workdir)

    def _reap():
        # Reap the orchestrator + any sibling task containers this run labelled (they are NOT --rm
        # and survive an orchestrator kill/timeout - see run_batch's per-task finally not running).
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)
        ps = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={run_label}"],
                            capture_output=True, text=True)
        ids = [x for x in (ps.stdout or "").split() if x]
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)

    try:
        cmd = ["docker", "run", "--rm", "--name", orch_name, "--network", "host",
               "-v", "/var/run/docker.sock:/var/run/docker.sock",
               "-v", f"{harness}:{harness}:rw",             # identity mount (DooD workspace source)
               "-v", f"{workdir}:{workdir}:rw",             # identity mount (output + summary)
               "-e", f"WILDCLAW_HARNESS_DIR={harness}",
               "-e", f"WILDCLAW_WORKDIR={workdir}",
               "-e", f"WILDCLAW_TASK_IMAGE={_task_image()}",
               "-e", f"WILDCLAW_TASK_LABEL={run_label}",
               "-e", f"WILDCLAW_JUDGE_PORT={judge_port}",
               "-e", f"WILDCLAW_JUDGE_ENDPOINT={judge_endpoint}",
               "-e", f"GBENCH_MODEL_BASE_URL={task_endpoint}",
               "-e", f"GBENCH_MODEL_NAME={model_name}",
               "-e", f"GBENCH_MODEL_API_KEY={os.environ.get('GBENCH_WILDCLAWBENCH_MODEL_API_KEY', 'dummy')}",
               "-e", f"GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY', '')}",
               "-e", f"WILDCLAW_TASK_NETWORK={task_network}",
               "-e", f"WILDCLAW_PARALLEL={parallel}"]
        # Propagate the judge-cascade knobs + BRAVE key (search tasks) into the orchestrator.
        for k in ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GBENCH_JUDGE_CASCADE_ROUNDS",
                  "GBENCH_JUDGE_BACKOFF", "GEMINI_OPENAI_BASE_URL", "BRAVE_API_KEY"):
            if os.environ.get(k):
                cmd += ["-e", f"{k}={os.environ[k]}"]
        # Host knob canonical GBENCH_WILDCLAWBENCH_JUDGE_REQUEST_TIMEOUT_S (legacy alias:
        # WILDCLAW_JUDGE_REQUEST_TIMEOUT_S); forward under the name the in-container judge proxy reads.
        _judge_timeout = suite_env("GBENCH_WILDCLAWBENCH_JUDGE_REQUEST_TIMEOUT_S",
                                   "WILDCLAW_JUDGE_REQUEST_TIMEOUT_S")
        if _judge_timeout:
            cmd += ["-e", f"WILDCLAW_JUDGE_REQUEST_TIMEOUT_S={_judge_timeout}"]
        if categories:
            cmd += ["-e", f"WILDCLAW_CATEGORIES={categories}"]
        if single_task:
            cmd += ["-e", f"WILDCLAW_TASK={single_task}"]
        # --eval-limit N: upstream run_batch has no --limit flag, so the launcher caps by driving the
        # harness single-task mode over the first N selected task ids (denominator capped to match).
        if limit is not None and int(limit) > 0:
            cmd += ["-e", f"WILDCLAW_LIMIT={int(limit)}"]
        # --shard I/N: upstream run_batch has no --shard flag either, so forward GBENCH_SHARD and let
        # the launcher select shard I of N over the deterministic sorted id list (ids[i-1::n], matching
        # sampling.shard_select) BEFORE the limit cap. The launcher drives single-task mode over that
        # sharded set, so run-set == score-set and the summary's n_tasks reflects the shard size.
        if os.environ.get("GBENCH_SHARD"):
            cmd += ["-e", f"GBENCH_SHARD={os.environ['GBENCH_SHARD']}"]
        if enable_thinking:
            cmd += ["-e", f"WILDCLAW_THINKING={os.environ.get('GBENCH_WILDCLAWBENCH_THINKING', 'high')}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_WILDCLAWBENCH_TIMEOUT_S", str(24 * 60 * 60)))
        logger.info("wildclawbench: docker run %s (parallel=%d, network=%s)", _image(), parallel, task_network)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "wildclawbench", f"the orchestrator exceeded GBENCH_WILDCLAWBENCH_TIMEOUT_S "
                f"({timeout_s}s) and was killed; sibling task containers were reaped. Raise the "
                f"timeout or reduce the task set.", DOCS_URL) from exc

        summ_path = os.path.join(workdir, "wildclawbench_summary.json")
        if not os.path.exists(summ_path):
            raise infra_required(
                "wildclawbench",
                f"the orchestrator produced no summary (rc={proc.returncode}; a harness failure, not "
                f"a 0%). tail: {(proc.stderr or proc.stdout or '')[-800:]}", DOCS_URL)
        with open(summ_path, encoding="utf-8") as f:
            summary = json.load(f)
    finally:
        _reap()
        shutil.rmtree(workdir, ignore_errors=True)

    scored = compute_wildclawbench_score(summary)
    if scored is None:
        raise infra_required(
            "wildclawbench", "no tasks produced a score (all infra failures? check the task image, "
            "the model endpoint reachability from task containers, and docker egress).", DOCS_URL)

    n_tasks = summary.get("n_tasks") or 0
    n_scored = summary.get("n_scored") or 0
    judge_fallback = summary.get("judge_fallback_tasks") or 0
    n_missing = summary.get("n_missing") or 0
    # A total wipeout (no task actually scored) yields global_mean=0.0 because missing tasks
    # count as 0 - that is a fabricated 0, not a real score. Hard-error instead of reporting
    # status=success accuracy=0.0.
    if n_scored == 0:
        raise infra_required(
            "wildclawbench",
            f"no task produced a real score (n_scored=0, {n_missing} missing of {n_tasks}); the "
            "global mean would be a fabricated 0.0. Check the task image, model-endpoint "
            "reachability from task containers, and docker egress.", DOCS_URL)
    n_grading_errors = summary.get("n_grading_errors") or 0
    is_full = (not single_task and not categories and n_tasks >= _FULL_TASK_COUNT)

    # The headline is the harness equal-weight global mean (the leaderboard Overall uses a weighted
    # MM/pure-text breakdown, approximated separately by weighted_overall) and a run here is a single
    # trial, so it is never leaderboard_comparable. Grading uses gbench's standard Gemini cascade by
    # convention (see `scoring`), which is a gbench grader choice, not a comparability defect.
    leaderboard = False
    reasons = ["headline is the harness equal-weight global mean; the leaderboard Overall follows a "
               "weighted MM/pure-text breakdown (approximated separately by weighted_overall)",
               "single trial (leaderboard audits multiple)"]
    if judge_fallback:
        reasons.append(f"{judge_fallback} task(s) fell back to deterministic grading (judge outage)")
    if n_missing:
        reasons.append(f"{n_missing} task(s) produced no score (counted as 0)")
    _shard_spec = os.environ.get("GBENCH_SHARD")
    if _shard_spec:
        reasons.append(f"sharded run (--shard {_shard_spec}): a non-overlapping subset of the "
                       f"full task set by construction")
    if not is_full:
        reasons.append("not the full 60-task set")

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "wildclawbench",
        "model_name": model_name,
        "thinking": enable_thinking,
        "status": "success",
        "accuracy": round(scored["global_mean"] * 100.0, 2),   # headline = harness global mean (%)
        "global_mean": round(scored["global_mean"], 4),
        "multimodal_mean": (round(scored["multimodal_mean"], 4)
                            if isinstance(scored.get("multimodal_mean"), (int, float)) else None),
        "pure_text_mean": (round(scored["pure_text_mean"], 4)
                           if isinstance(scored.get("pure_text_mean"), (int, float)) else None),
        "weighted_overall": (round(scored["weighted_overall"], 4)
                             if isinstance(scored.get("weighted_overall"), (int, float)) else None),
        "avg_time_min": (round(scored["avg_time_min"], 2)
                         if isinstance(scored.get("avg_time_min"), (int, float)) else None),
        "avg_cost_usd": (round(scored["avg_cost_usd"], 4)
                         if isinstance(scored.get("avg_cost_usd"), (int, float)) else None),
        "n_tasks": n_tasks,
        "n_scored": n_scored,
        "n_missing": n_missing,
        "n_grading_errors": n_grading_errors,
        "judge_fallback_tasks": judge_fallback,
        "scoring": "per-task programmatic checks + LLM judge (gbench Gemini cascade); harness global mean",
        "sampling": ("OpenClaw harness default (gbench does not pin a temperature for wildclawbench, "
                     "matching the leaderboard, which runs each model under its own harness defaults)"),
        "raw_summary": summary,
        "metric": ("WildClawBench headline = the harness equal-weight global mean of per-task "
                   "overall_score (0-1) over ALL selected tasks (a missing/errored task counts as 0), "
                   "i.e. run_batch's own global_avg - NOT the leaderboard's Overall Score, which "
                   "follows a weighted MM/pure-text breakdown (approximated by weighted_overall). "
                   "Agent = OpenClaw in the official wildclawbench-ubuntu:v1.3 image, pointed at the "
                   "gbench model endpoint; grading is the upstream per-task grade() (programmatic + "
                   "LLM judge). avg_cost is ~0 with a self-hosted endpoint (OpenRouter cost is "
                   "unavailable)."),
        "leaderboard_comparable": leaderboard,
        "leaderboard_comparable_reason": "; ".join(reasons),
    }
    return result
