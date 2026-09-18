# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: mcp_bench
# Description: MCP-Bench (Accenture) - live multi-server Model Context Protocol agent benchmark

"""gbench native built-in runner for mcp_bench (Tool Use & Function Calling).

Canonical MCP-Bench (Accenture/mcp-bench, arXiv:2508.20453): a live agentic benchmark over 28
real MCP servers (250 tools), where the model discovers tools, plans a multi-step trajectory,
invokes tools against the live servers, and grounds a final answer. Scoring = ComplexEval-style:
rule-based tool metrics (schema compliance, valid tool-name rate, execution success) + an LLM
judge scoring six sub-dimensions (task fulfillment, grounding, tool appropriateness, parameter
accuracy, dependency awareness, parallelism/efficiency), each run 5x with randomized order and
averaged (the upstream "judge stability" protocol). The headline is a 0-1 Overall Score = the
mean of four dimensions (schema understanding, task completion, tool usage, planning).

gbench DELEGATES the whole agent loop to the bundled upstream runner inside a LOCAL container
(gbench/docker/mcp_bench.Dockerfile bakes the runner + all 28 vendored servers). The launcher:

  * points the model-under-test at the gbench /v1 endpoint (openai_compatible provider);
  * injects gbench's ESTABLISHED Gemini cascade as the judge - upstream's exact 6-dimension
    prompts and 5x stability protocol are UNCHANGED; only the model underneath the judge changes
    from o4-mini to the Gemini cascade (base.judge_generate_cascade's model list/rounds/backoff),
    for consistency with every other gbench judged suite;
  * runs only the tasks whose servers are actually provisioned (offline servers always; key-gated
    servers if the key is set; network servers if egress works) and reports a manifest of what
    ran vs was dropped - "canonical-when-provisioned", never a silent skip or a fabricated number.

Because the judge is Gemini (the MCP-Bench leaderboard requires o4-mini) and a run may cover a
provisioned SUBSET, `leaderboard_comparable` is always False.

HARD-ERRORS (infra_required, never skips) if Docker, the locally-built image, or GEMINI_API_KEY
(the judge) is missing, or if the container produces no parseable result. Heavy external
provisioning - see docs/evals/mcp_bench.md.

Sampling: gbench does NOT pin a temperature or a thinking/reasoning mode for mcp_bench - the
delegated MCP-Bench agent loop (upstream benchmark.runner via the openai_compatible provider) samples
the model-under-test at its own harness default, so `--temperature` / `--thinking` /
`GBENCH_MCP_BENCH_TEMPERATURE` are NOT forwarded here (a value passed there would be a silent no-op,
so gbench does not claim the illusion of control). The Gemini cascade judge is pinned at 0.0.

Sharding: `--shard I/N` IS honoured. Each MCP-Bench task carries a stable, globally-unique
`task_id`, so this runner forwards `GBENCH_SHARD` into the container and the launcher selects an
interleaved subset - round-robin over the DETERMINISTIC sorted full task-id list (`ids[I-1::N]`,
matching sampling.shard_select) - BEFORE the provisioning filter and MCP_TASK_LIMIT, the same
compose order as the native path. A sharded run is a subset by construction, so `tasks_run` /
`subset_manifest.tasks_kept` cover only the shard, `is_full` is False, and `leaderboard_comparable`
stays False.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .swebench_common import infra_required
from .base import gemini_key_live_valid

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"
DOCS_URL = "docs/evals/mcp_bench.md"

_IMAGE_DEFAULT = "gbench-mcp-bench"
#: server API keys passed through to the container if present in the host env (all OPTIONAL;
#: their absence just narrows the provisioned server subset).
_SERVER_KEYS = ["GOOGLE_MAPS_API_KEY", "NCI_API_KEY", "HF_TOKEN", "NPS_API_KEY", "NASA_API_KEY"]


def _image() -> str:
    return os.environ.get("GBENCH_MCP_BENCH_IMAGE", _IMAGE_DEFAULT)


def check_mcp_bench_prerequisites() -> Tuple[bool, str]:
    """Docker + the locally-built image + GEMINI_API_KEY (the Gemini cascade judge)."""
    image = _image()
    build = (f"Build the harness LOCALLY (gbench never pulls):\n"
             f"  docker build -t {image} -f gbench/docker/mcp_bench.Dockerfile gbench/docker\n"
             f"It bundles pinned Accenture/mcp-bench + all 28 vendored MCP servers. See " + DOCS_URL)
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        return False, f"image {image!r} not found. " + build
    if not os.environ.get("GEMINI_API_KEY"):
        return False, ("GEMINI_API_KEY is not set - it powers gbench's Gemini cascade judge that "
                       "scores MCP-Bench's six judge sub-dimensions here.")
    _ok, _why = gemini_key_live_valid(os.environ["GEMINI_API_KEY"])
    if not _ok:
        return False, (f"GEMINI_API_KEY was rejected by the judge endpoint ({_why}); a valid key is "
                       "required for the Gemini cascade judge (fail-fast before the container run).")
    return True, ""


def _dim(metrics: Dict[str, Any], *keys: str) -> Optional[float]:
    """Mean of the given metric keys, skipping any that are missing/None. None if none present."""
    vals = [metrics[k] for k in keys if isinstance(metrics.get(k), (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def _overall_for_one(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute the four 0-1 dimensions + 0-1 Overall Score from one averaged-metrics dict.

    Judge dimensions are on a 1-10 scale upstream -> divided by 10 to sit on the 0-1 axis with
    the rule-based schema dimension (which is already 0-1). Overall = mean of the four dims.
    """
    schema = _dim(metrics, "input_schema_compliance", "valid_tool_name_rate")  # 0-1
    tc = metrics.get("task_completion_score")
    tu = metrics.get("tool_selection_score")
    pl = metrics.get("planning_effectiveness_and_efficiency_score")
    dims: Dict[str, Optional[float]] = {
        "schema_understanding": schema,
        "task_completion": (tc / 10.0) if isinstance(tc, (int, float)) else None,
        "tool_usage": (tu / 10.0) if isinstance(tu, (int, float)) else None,
        "planning_effectiveness": (pl / 10.0) if isinstance(pl, (int, float)) else None,
    }
    present = [v for v in dims.values() if v is not None]
    if not present:
        return None
    return {"dimensions": dims, "overall": sum(present) / len(present)}


def compute_overall_score(results_json: Any,
                          manifest: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Roll the container's output into the 0-1 Overall Score.

    `results_json` is either a flat averaged-metrics dict (single task file) or a
    {task_file: metrics} map (multi-file). Multi-file dimensions are task-weighted across files
    (weights from the manifest's per-file kept counts; equal weight if unavailable).
    """
    if not isinstance(results_json, dict) or not results_json:
        return None

    # Flat single-file metrics vs {file: metrics}.
    is_multi = all(isinstance(v, dict) for v in results_json.values()) and \
        any(("task_completion_score" in v or "input_schema_compliance" in v)
            for v in results_json.values() if isinstance(v, dict))
    if not is_multi:
        one = _overall_for_one(results_json)
        return {"overall": one["overall"], "dimensions": one["dimensions"],
                "per_file": None} if one else None

    per_file_counts = {}
    if manifest and isinstance(manifest.get("per_file"), dict):
        for fn, c in manifest["per_file"].items():
            per_file_counts[fn] = int(c.get("kept", 0)) if isinstance(c, dict) else 0

    dim_names = ["schema_understanding", "task_completion", "tool_usage", "planning_effectiveness"]
    weighted: Dict[str, List[Tuple[float, float]]] = {d: [] for d in dim_names}
    per_file_out = {}
    for path, metrics in results_json.items():
        if not isinstance(metrics, dict):
            continue
        one = _overall_for_one(metrics)
        if not one:
            continue
        base = os.path.basename(path)
        w = per_file_counts.get(base) or per_file_counts.get(base.replace("filtered_", "")) or 1
        per_file_out[base] = one
        for d in dim_names:
            v = one["dimensions"].get(d)
            if v is not None:
                weighted[d].append((v, float(w)))

    dim_means: Dict[str, Optional[float]] = {}
    for d in dim_names:
        pairs = weighted[d]
        if pairs:
            tot_w = sum(w for _, w in pairs)
            dim_means[d] = sum(v * w for v, w in pairs) / tot_w if tot_w else None
        else:
            dim_means[d] = None
    present = [v for v in dim_means.values() if v is not None]
    if not present:
        return None
    return {"overall": sum(present) / len(present), "dimensions": dim_means, "per_file": per_file_out}


def run_mcp_bench(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical MCP-Bench: upstream agent loop (28 servers, in-container) + gbench Gemini
    cascade judge, over the provisioned server subset."""
    ok, reason = check_mcp_bench_prerequisites()
    if not ok:
        raise infra_required("mcp_bench", reason, DOCS_URL)

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"

    workdir = tempfile.mkdtemp(prefix="gbench_mcp_bench_")
    os.chmod(workdir, 0o777)  # the container writes /out as its own uid
    orch_name = "gbench_mcp_bench_" + os.path.basename(workdir)

    def _reap():
        # docker --rm only fires on a clean container EXIT; on a subprocess timeout the client is
        # killed while the container keeps running, so reap the named container in finally + on
        # timeout to avoid leaking it across a long or multi-node run.
        subprocess.run(["docker", "rm", "-f", orch_name], capture_output=True)

    try:
        cmd = ["docker", "run", "--rm", "--name", orch_name, "--network", "host", "-v", f"{workdir}:/out:rw",
               "-e", f"MCP_ENDPOINT={endpoint}", "-e", f"MCP_MODEL={model_name}",
               "-e", f"GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY', '')}",
               "-e", "MCP_OUTPUT=/out/mcp_bench_results.json",
               "-e", "MCP_MANIFEST=/out/subset_manifest.json"]
        for key in _SERVER_KEYS:
            if os.environ.get(key):
                cmd += ["-e", f"{key}={os.environ[key]}"]
        for env_name, mcp_name in (("GBENCH_MCP_BENCH_DISTRACTION", "MCP_DISTRACTION_COUNT"),
                                   ("GBENCH_MCP_BENCH_DISABLE_STABILITY", "MCP_DISABLE_STABILITY"),
                                   ("GBENCH_MCP_BENCH_NO_SUBSET", "MCP_NO_SUBSET_FILTER"),
                                   ("GBENCH_MCP_BENCH_ASSUME_NETWORK", "MCP_ASSUME_NETWORK"),
                                   ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODELS"),
                                   ("GBENCH_JUDGE_MODEL", "GBENCH_JUDGE_MODEL")):
            v = os.environ.get(env_name)
            if v:
                cmd += ["-e", f"{mcp_name}={v}"]
        lim = kwargs.get("limit")
        if lim and int(lim) > 0:
            cmd += ["-e", f"MCP_TASK_LIMIT={int(lim)}"]
        # Forward the shard ONLY because the launcher consumes it: mcp_bench tasks carry stable,
        # globally-unique task_ids, so the launcher selects an interleaved subset (round-robin over
        # the sorted full task-id list) BEFORE MCP_TASK_LIMIT, matching the native path's compose
        # order. The manifest's tasks_kept then reflects only the shard.
        if os.environ.get("GBENCH_SHARD"):
            cmd += ["-e", f"GBENCH_SHARD={os.environ['GBENCH_SHARD']}"]
        cmd.append(_image())

        timeout_s = int(os.environ.get("GBENCH_MCP_BENCH_TIMEOUT_S", str(24 * 60 * 60)))
        logger.info("mcp_bench: docker run %s (endpoint=%s)", _image(), endpoint)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            _reap()
            raise infra_required(
                "mcp_bench", f"the mcp_bench container exceeded {timeout_s}s and was killed.",
                DOCS_URL) from exc
        if proc.returncode != 0:
            raise infra_required(
                "mcp_bench",
                f"the mcp_bench container failed (rc={proc.returncode}). "
                f"tail: {(proc.stderr or proc.stdout or '')[-800:]}", DOCS_URL)

        results_path = os.path.join(workdir, "mcp_bench_results.json")
        manifest_path = os.path.join(workdir, "subset_manifest.json")
        if not os.path.exists(results_path):
            raise infra_required(
                "mcp_bench", "the container produced no results file (a harness failure, not a 0%). "
                f"tail: {(proc.stderr or proc.stdout or '')[-500:]}", DOCS_URL)
        with open(results_path, encoding="utf-8") as f:
            results_json = json.load(f)
        manifest = None
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
    finally:
        _reap()
        shutil.rmtree(workdir, ignore_errors=True)

    if not results_json:
        # No tasks were runnable with the provisioned servers -> a provisioning gap, not a 0%.
        gap = ""
        if manifest and manifest.get("filtered"):
            gap = (f" No servers were provisioned (network_up={manifest.get('network_up')}, "
                   f"available={manifest.get('available_servers')}). Provide network egress and/or "
                   "server API keys.")
        raise infra_required("mcp_bench", "no MCP-Bench tasks could be run." + gap, DOCS_URL)

    scored = compute_overall_score(results_json, manifest)
    if scored is None:
        raise infra_required(
            "mcp_bench", "the container output had no scorable dimensions (all judge outages?).",
            DOCS_URL)

    tasks_run = manifest.get("tasks_kept") if manifest else None

    # A sharded run covers a round-robin subset of the task set by construction, so it is never the
    # full set; tasks_run already reflects only the shard's kept tasks (denominator honesty).
    shard_spec = os.environ.get("GBENCH_SHARD", "").strip()
    sharded = bool(shard_spec) and shard_spec != "1/1"
    subset_provisioned = bool(manifest and manifest.get("filtered") and
                              manifest.get("unavailable_servers"))
    limited = bool(lim and int(lim) > 0)
    is_full = not (sharded or subset_provisioned or limited)

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "mcp_bench",
        "model_name": model_name,
        "status": "success",
        "accuracy": round(scored["overall"] * 100.0, 2),   # headline = Overall Score (%)
        "overall_score": round(scored["overall"], 4),        # 0-1
        "dimensions": {k: (round(v, 4) if isinstance(v, (int, float)) else None)
                       for k, v in scored["dimensions"].items()},
        "per_file": scored.get("per_file"),
        "raw_metrics": results_json,
        "tasks_run": tasks_run,
        "is_full": is_full,
        "shard": shard_spec or None,
        "subset_manifest": manifest,
        "judge": "gbench-gemini-cascade",
        "sampling": ("MCP-Bench agent loop owns sampling; gbench does not pin a temperature or a "
                     "thinking/reasoning mode (a passed --temperature/--thinking would be a silent "
                     "no-op) - the delegated upstream runner samples the model-under-test at its own "
                     "default via the openai_compatible provider, as the MCP-Bench leaderboard does)."),
        "metric": ("MCP-Bench Overall Score (0-1) = mean of four dimensions (schema understanding "
                   "[rule-based], task completion, tool usage, planning effectiveness [Gemini-"
                   "judged, 6 sub-dimensions x 5 randomized-order stability runs]). Agent loop + "
                   "28 servers delegated to the upstream runner; judge is gbench's Gemini cascade "
                   "(upstream prompts + stability protocol verbatim; model differs from o4-mini)."),
        # A run here is a gbench-internal number, not a like-for-like MCP-Bench leaderboard entry:
        # gbench grades with its standard Gemini cascade (a gbench convention) where the leaderboard
        # is o4-mini-graded, and a run may cover only the provisioned server subset.
        "leaderboard_comparable": False,
        "leaderboard_comparable_reason": (
            "graded by gbench's standard Gemini cascade (a gbench convention); the published "
            "MCP-Bench leaderboard is o4-mini-graded, so this is a gbench-internal number"
            + ("; and only a provisioned server SUBSET was run" if subset_provisioned else "")
            + (f"; and only shard {shard_spec} (a round-robin subset by construction) was run"
               if sharded else "")),
    }
    return result
