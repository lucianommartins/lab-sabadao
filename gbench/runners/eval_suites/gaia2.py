# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: gaia2
# Description: GAIA2 / Meta Agents Research Environments (ARE) - stateful multi-turn agentic benchmark

"""gbench native built-in runner for gaia2 (Tool Use & Agentic Workflows).

Canonical GAIA2 (Meta Agents Research Environments,
github.com/facebookresearch/meta-agents-research-environments; PyPI
`meta-agents-research-environments`; HF dataset `meta-agents-research-environments/gaia2`) is a
STATEFUL, multi-turn, time-sensitive agentic benchmark: the agent acts inside Meta's ARE simulator,
issuing tool calls over many turns against evolving app state (contacts/calendar/email/files with
async events). Scoring is HYBRID - deterministic "hard validation" plus a LOAD-BEARING LLM judge for
soft/semantic checks.

gbench DELEGATES the whole run to the pinned `are-benchmark` CLI inside a LOCAL orchestrator image
(gbench/docker/gaia2.Dockerfile). Unlike skillsbench/wildclawbench there is NO docker-out-of-docker
and NO sibling task containers - ARE is a pure in-process Python simulator - so the orchestrator
just needs `--network host` to reach the served model. Two things gbench injects (everything else,
incl. the judge PROMPTS, is upstream verbatim):
  * the model-under-test is wired via LiteLLM's `local` provider (`--model openai/<served> --provider
    local --endpoint <gbench /v1>`);
  * gbench's standard Gemini cascade grades in place of ARE's default Llama-3.3-70B judge, via an
    in-container OpenAI-compat proxy (`--judge_provider local --judge_endpoint <proxy> --judge_model
    openai/gbench-cascade`).

Headline `accuracy` = the per-capability equal-weight **macro success rate** from ARE's own
benchmark_stats.json (the GAIA2 Overall); micro (per-scenario) rate + per-capability breakdown are
also reported. Because gbench defaults to a single-run `run` (not the 3-phase x3 `gaia2-run`),
`leaderboard_comparable` is False; grading uses gbench's standard Gemini cascade by convention
(the `--judge_model openai/gbench-cascade` proxy above), which is a gbench grader choice, not a defect.

HARD-ERRORS (infra_required, never skips) if Docker, the orchestrator image, GEMINI_API_KEY (the
load-bearing judge), or a reachable model endpoint are missing. The public cc-by-4.0 dataset is
baked into the image. See docs/evals/gaia2.md.

Sampling: gbench does NOT pin a temperature OR a reasoning/thinking mode for gaia2 -
the ARE agent samples and reasons at the harness/model default, so `--temperature` /
`GBENCH_GAIA2_TEMPERATURE` and the `enable_thinking` flag are not applied here (a value/flag there is
a silent no-op: nothing forwards it into the are-benchmark run). The judge cascade is pinned at 0.0.

Sharding: `--shard I/N` is NOT applicable to gaia2. This suite delegates scenario selection entirely
to the pinned `are-benchmark` CLI, whose only selection knob is `--limit` (first-N per config); ARE
streams the HF dataset and owns selection internally (huggingface_loader enumerates the stream and
breaks at `index > limit`), with no offset / scenario-id / shard API exposed on the CLI. An
interleaved, non-overlapping shard therefore cannot be expressed, so gbench does NOT forward
`GBENCH_SHARD` into the container (a no-op env would be a fabrication of shard coverage). When a shard
is requested the runner logs a one-time WARNING and runs unsharded (full / --limit as usual).
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Agentic Workflows"
DOCS_URL = "docs/evals/gaia2.md"

_IMAGE_DEFAULT = "gbench-gaia2"
_JUDGE_PORT_DEFAULT = 18790
# GAIA2's 5 capability configs (CapabilityTag.gaia2_capabilities()); demo/mini are smoke subsets.
_CAPABILITY_CONFIGS = ("adaptability", "ambiguity", "execution", "search", "time")


def _image() -> str:
    return os.environ.get("GBENCH_GAIA2_IMAGE", _IMAGE_DEFAULT)


def _docker_image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


# Live Gemini key ping lives in base.py (single source shared by every judge suite); the alias
# keeps the module-global name that this suite's prerequisite check and the tests reference.
from .base import gemini_key_live_valid as _gemini_key_valid, free_port


def check_gaia2_prerequisites() -> Tuple[bool, str]:
    """Docker + orchestrator image + GEMINI (the load-bearing judge). ARE + the dataset are baked."""
    image = _image()
    build = (f"Build the orchestrator LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f gbench/docker/gaia2.Dockerfile gbench/docker\n"
             f"It bundles the pinned ARE simulator + the public GAIA2 dataset. See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if not _docker_image_exists(image):
        return False, f"orchestrator image {image!r} not found. " + build
    if not os.environ.get("GEMINI_API_KEY"):
        return False, ("GEMINI_API_KEY is unset - GAIA2's soft/semantic validation is done by an LLM "
                       "judge (load-bearing; without it most scenarios score no_validation). gbench "
                       "swaps in its Gemini cascade. See " + DOCS_URL)
    if not os.environ.get("GBENCH_GAIA2_SKIP_KEY_VALIDATION"):
        ok_g, why_g = _gemini_key_valid(os.environ["GEMINI_API_KEY"])
        if not ok_g:
            return False, (f"GEMINI_API_KEY was rejected by the Gemini API ({why_g}); the judge "
                           f"cascade cannot run. Check the key or set "
                           f"GBENCH_GAIA2_SKIP_KEY_VALIDATION=1 to bypass. See " + DOCS_URL)
    return True, ""


def _task_reachable_endpoint(base_url: str) -> str:
    """The orchestrator runs --network host, so a served model on the host is reached at 127.0.0.1;
    just ensure a /v1 suffix. GBENCH_GAIA2_MODEL_ENDPOINT overrides the whole URL."""
    override = os.environ.get("GBENCH_GAIA2_MODEL_ENDPOINT")
    if override:
        return override
    ep = (base_url or "").rstrip("/")
    return ep if ep.endswith("/v1") else ep + "/v1"


def _agent_never_invoked(summary: Dict[str, Any]) -> bool:
    """True iff the run validated scenarios but the container positively reported that the agent was
    never invoked (agent-less `are-benchmark run`: no agent LLM usage in any per-run trace). Older
    summaries omit `agent_invoked` (-> None), so this returns False for them and never false-fires."""
    return bool(summary) and (summary.get("validated_runs") or 0) > 0 \
        and summary.get("agent_invoked") is False


def compute_gaia2_score(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Headline = macro (per-capability equal-weight) success rate; else micro if macro absent."""
    if not summary:
        return None
    macro = summary.get("macro_success_rate")
    micro = summary.get("micro_success_rate")
    head = macro if isinstance(macro, (int, float)) else micro
    if not isinstance(head, (int, float)):
        return None
    return {"headline": head, "macro_success_rate": macro, "micro_success_rate": micro}


def run_gaia2(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run canonical GAIA2 via the pinned Meta-ARE `are-benchmark` CLI (in-container, no DooD)."""
    ok, reason = check_gaia2_prerequisites()
    if not ok:
        raise infra_required("gaia2", reason, DOCS_URL)

    # --shard I/N is NOT applicable to gaia2 (see module docstring): the are-benchmark CLI only
    # supports --limit (first-N) and owns scenario selection internally (no offset/scenario-id/shard
    # API), so an interleaved non-overlapping shard cannot be expressed. Do NOT forward GBENCH_SHARD
    # into the container (a no-op env would fake shard coverage) - warn once and run unsharded.
    if os.environ.get("GBENCH_SHARD"):
        logger.warning(
            "gaia2: --shard %s is NOT applied - the are-benchmark CLI selects scenarios only via "
            "--limit (first-N per config) and owns selection internally (no offset/scenario-id/shard "
            "API), so an interleaved shard cannot be expressed here; running UNSHARDED. See %s.",
            os.environ["GBENCH_SHARD"], DOCS_URL)

    # Use the fixed default when free (deterministic for a single run); auto-pick a free port when it
    # is taken, so two concurrent/overlapping gaia2 (or gaia2 + wildclawbench) runs do not collide on
    # the host judge-proxy port. An explicit GBENCH_GAIA2_JUDGE_PORT still forces a specific port.
    _env_judge_port = os.environ.get("GBENCH_GAIA2_JUDGE_PORT")
    judge_port = int(_env_judge_port) if _env_judge_port else free_port(_JUDGE_PORT_DEFAULT)
    model_endpoint = _task_reachable_endpoint(base_url)
    mode = os.environ.get("GBENCH_GAIA2_MODE", "run").strip()          # run | gaia2-run
    single_config = os.environ.get("GBENCH_GAIA2_CONFIG", "").strip()  # narrows `run` (smoke: mini/demo)
    _explicit_limit = os.environ.get("GBENCH_GAIA2_LIMIT", "").strip()
    eff_limit = _explicit_limit or (str(limit) if limit else "")
    # ARE's --limit is applied PER capability config, and the default `run` covers all 5
    # capabilities (the "all-5-capabilities" config), so a raw --eval-limit N runs 5xN scenarios.
    # Distribute --eval-limit across the 5 capabilities (ceil) so the total stays ~N. Skipped when
    # the operator set GBENCH_GAIA2_LIMIT explicitly, or when a single GBENCH_GAIA2_CONFIG is
    # selected (one config -> no per-capability multiplier).
    _N_CAPABILITIES = 5
    if limit and not _explicit_limit and mode == "run" and not single_config:
        eff_limit = str(max(1, (int(limit) + _N_CAPABILITIES - 1) // _N_CAPABILITIES))
    if eff_limit and mode == "gaia2-run":
        # gaia2_run.py only applies --limit in the default `run` mode; the canonical 3-phase x3
        # `gaia2-run` builds its command without --limit, so the cap would be silently dropped.
        # Warn loudly (do not present a full run as a capped one) and point at the `run` mode.
        logger.warning(
            "gaia2: --eval-limit / GBENCH_GAIA2_LIMIT (%s) is NOT applied in gaia2-run mode "
            "(the canonical 3-phase x3 run takes no task cap) - the FULL benchmark will run. "
            "Use the default `run` mode (GBENCH_GAIA2_MODE=run) for a capped smoke run.", eff_limit)

    workdir = tempfile.mkdtemp(prefix="gbench_gaia2_")
    os.chmod(workdir, 0o777)
    orch_name = "gbench_gaia2_orch_" + os.path.basename(workdir)

    def _reap():
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)

    try:
        cmd = ["docker", "run", "--rm", "--name", orch_name, "--network", "host",
               "-v", f"{workdir}:{workdir}:rw",
               "-e", f"GAIA2_WORKDIR={workdir}",
               "-e", f"GBENCH_MODEL_BASE_URL={model_endpoint}",
               "-e", f"GBENCH_MODEL_NAME={model_name}",
               "-e", f"GBENCH_MODEL_API_KEY={os.environ.get('GBENCH_GAIA2_MODEL_API_KEY', 'dummy')}",
               "-e", f"GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY', '')}",
               "-e", f"GAIA2_JUDGE_PORT={judge_port}",
               "-e", f"GAIA2_MODE={mode}"]
        for env_key, cli_key in (("GBENCH_GAIA2_NUM_RUNS", "GAIA2_NUM_RUNS"),
                                 ("GBENCH_GAIA2_SCENARIO_TIMEOUT", "GAIA2_SCENARIO_TIMEOUT"),
                                 ("GBENCH_GAIA2_MAX_CONCURRENT", "GAIA2_MAX_CONCURRENT"),
                                 ("GBENCH_GAIA2_JUDGE_ENDPOINT", "GAIA2_JUDGE_ENDPOINT"),
                                 ("GBENCH_GAIA2_JUDGE_MODEL", "GAIA2_JUDGE_MODEL")):
            if os.environ.get(env_key):
                cmd += ["-e", f"{cli_key}={os.environ[env_key]}"]
        for k in ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GBENCH_JUDGE_CASCADE_ROUNDS",
                  "GBENCH_JUDGE_BACKOFF", "GEMINI_OPENAI_BASE_URL",
                  "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):  # optional: removes anon paths-info 429s
            if os.environ.get(k):
                cmd += ["-e", f"{k}={os.environ[k]}"]
        if single_config:
            cmd += ["-e", f"GAIA2_CONFIG={single_config}"]
        if eff_limit:
            cmd += ["-e", f"GAIA2_LIMIT={eff_limit}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_GAIA2_TIMEOUT_S", str(24 * 60 * 60)))
        logger.info("gaia2: docker run %s (mode=%s, config=%s)", _image(), mode, single_config or "all-5-capabilities")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "gaia2", f"the orchestrator exceeded GBENCH_GAIA2_TIMEOUT_S ({timeout_s}s) and was "
                f"killed. Raise the timeout or reduce the scope (GBENCH_GAIA2_CONFIG / GBENCH_GAIA2_LIMIT).",
                DOCS_URL) from exc

        summ_path = os.path.join(workdir, "gaia2_summary.json")
        if not os.path.exists(summ_path):
            raise infra_required(
                "gaia2", f"the orchestrator produced no summary (rc={proc.returncode}; a harness "
                f"failure, not a 0%). tail: {(proc.stderr or proc.stdout or '')[-800:]}", DOCS_URL)
        with open(summ_path, encoding="utf-8") as f:
            summary = json.load(f)
    finally:
        _reap()
        # Preserve the workdir (ARE per-scenario logs/traces) for audit when requested. By default it
        # is removed to avoid tempdir leaks, but then a 0% cannot be told apart from silent
        # misscoring (WS10 audit). Set GBENCH_GAIA2_KEEP_WORKDIR=1 to keep the per-scenario artifacts.
        if os.environ.get("GBENCH_GAIA2_KEEP_WORKDIR"):
            logger.info("[gaia2] GBENCH_GAIA2_KEEP_WORKDIR set: preserving %s for audit "
                        "(ARE per-scenario logs/traces).", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    # The model must actually have been invoked. `are-benchmark run` without --agent runs
    # agent-less and still returns a real macro=0.0; that is a harness no-op, not a 0% score, and
    # must never be published. The container reports agent_invoked=False only when it positively
    # found no agent LLM usage across the per-run traces (older summaries omit the key -> None ->
    # this guard stays silent, so it never false-fires on a legitimate run).
    if _agent_never_invoked(summary):
        raise infra_required(
            "gaia2", "the agent was never invoked (are-benchmark ran agent-less: no agent LLM usage "
            "in any per-run trace), so every scenario trivially failed. This is a harness bug, not a "
            "0% score. Ensure the run passes --agent (GAIA2_AGENT).", DOCS_URL)

    scored = compute_gaia2_score(summary)
    if scored is None:
        raise infra_required(
            "gaia2", "no scenarios produced a validated score (all no_validation? check the model "
            "endpoint reachability and the judge cascade / GEMINI_API_KEY).", DOCS_URL)

    total_scenarios = summary.get("total_scenarios") or 0
    no_validation = summary.get("no_validation_runs") or 0
    is_full = (mode == "gaia2-run") or (not single_config and not eff_limit)
    # Meta-ARE owns sampling AND reasoning mode: gbench forwards neither a temperature nor a
    # thinking/reasoning flag into the are-benchmark run, so recording either as applied would be a
    # fabrication. Report the single honest note instead (mirrors wildclawbench.py's convention).
    sampling_note = ("Meta-ARE harness default; gbench does not pin a temperature or a "
                     "reasoning/thinking mode for gaia2 - the ARE agent samples and reasons at the "
                     "harness/model default (a passed --temperature or enable_thinking is a silent "
                     "no-op, as nothing forwards it into the are-benchmark run)")

    leaderboard = False
    reasons = ["single-run `run` (canonical leaderboard uses the 3-phase gaia2-run x3 with Pass@k)"
               if mode != "gaia2-run" else "gbench single submission (leaderboard is maintainer-audited)"]
    if not is_full:
        reasons.append(f"subset (config={single_config or 'all'}, limit={eff_limit or 'none'})")
    if no_validation:
        reasons.append(f"{no_validation} run(s) scored no_validation (judge/endpoint issue)")

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "gaia2",
        "model_name": model_name,
        "status": "success",
        "accuracy": round(scored["headline"] * 100.0, 2),   # headline = macro success rate (%)
        "macro_success_rate": (round(scored["macro_success_rate"], 4)
                               if isinstance(scored.get("macro_success_rate"), (int, float)) else None),
        "micro_success_rate": (round(scored["micro_success_rate"], 4)
                               if isinstance(scored.get("micro_success_rate"), (int, float)) else None),
        "pass_at_k_percent": summary.get("pass_at_k_percent"),
        "pass_k_percent": summary.get("pass_k_percent"),
        "total_scenarios": total_scenarios,
        "no_validation_runs": no_validation,
        "exception_runs": summary.get("exception_runs"),
        "per_capability": summary.get("per_capability"),
        "mode": mode,
        "scoring": "ARE hybrid: deterministic hard-validation + LLM judge (gbench Gemini cascade)",
        "sampling": sampling_note,
        "raw_summary": summary,
        "metric": ("GAIA2 Overall = per-capability equal-weight macro success rate over ARE's own "
                   "benchmark_stats.json (deterministic hard-validation + LLM-judge soft-validation). "
                   "Agent runs in Meta's ARE simulator via are-benchmark, driven by the served model "
                   "through LiteLLM's OpenAI-compat provider. Default gbench mode is single-run `run` "
                   "over all 5 capability configs; the canonical 3-phase gaia2-run x3 is opt-in "
                   "(GBENCH_GAIA2_MODE=gaia2-run)."),
        "leaderboard_comparable": leaderboard,
        "leaderboard_comparable_reason": "; ".join(reasons),
    }
    return result
