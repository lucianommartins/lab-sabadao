# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: bigcodebench
# Description: BigCodeBench - calibrated pass@1 via the official BCB Docker harness

"""gbench native built-in runner for bigcodebench (Coding & Software Engineering).

Canonical BigCodeBench (bigcode/bigcodebench, instruct split) scored by calibrated
pass@1: the model's raw output is sanitized + calibrated (code_prompt prepended)
and executed against each task's unittest TestCase inside the official
`bigcodebench/bigcodebench-evaluate` Docker image. SANDBOX_EVAL. The image is
auto-pulled on first run (like putnam_formal's Lean image); skips cleanly if
Docker is unreachable or the pull fails.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_BIGCODEBENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/bigcodebench.md"
_IMAGE = "bigcodebench/bigcodebench-evaluate:latest"
_DATASETS = {"full": "bigcode/bigcodebench", "hard": "bigcode/bigcodebench-hard"}
_SPLIT = "v0.1.4"


def _ensure_docker_image(image: str, timeout: int = 1800) -> Tuple[bool, str]:
    """Return (ok, reason). Pull `image` if absent (first run); skip on failure."""
    try:
        if subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True).returncode == 0:
            return True, ""
        logger.info(f"Pulling Docker image '{image}' (first run; this can take a while)...")
        p = subprocess.run(["docker", "pull", image], capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return False, f"could not pull Docker image '{image}': {(p.stderr or '')[-200:]}"
        return True, ""
    except Exception as e:
        return False, f"could not ensure Docker image '{image}': {e}"


def check_bigcodebench_prerequisites() -> Tuple[bool, str]:
    """datasets + reachable Docker + the BCB evaluate image (auto-pulled on first run)."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed."
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=20)
        if r.returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    return _ensure_docker_image(_IMAGE)


def _load_bigcodebench_samples(
    limit: Optional[int] = None,
    subset: str = "full",
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load BigCodeBench (instruct prompts); raises on load/schema failure."""
    dataset_id = _DATASETS.get(subset, _DATASETS["full"])
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_id, split=_SPLIT)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for bigcodebench: {e}")
        raise RuntimeError(f"Could not load dataset for bigcodebench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for bigcodebench returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="bigcodebench")

    samples = []
    for item in rows:
        task_id = item.get("task_id")
        instruct = item.get("instruct_prompt")
        entry_point = item.get("entry_point")
        if not task_id or not instruct or not entry_point:
            raise RuntimeError(
                "bigcodebench: unexpected dataset schema "
                "(task_id/instruct_prompt/entry_point); refusing to fabricate sample data"
            )
        # Canonical: send instruct_prompt VERBATIM (no wrapper) to the chat model.
        messages = [{"role": "user", "content": str(instruct)}]
        samples.append((messages, task_id, {"category": "python", "task_id": task_id}))

    logger.info(f"Loaded {len(samples)} bigcodebench samples ('{dataset_id}' {_SPLIT}).")
    return samples


def _parse_bcb_results(workdir: str) -> Tuple[Dict[str, bool], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Parse the harness eval_results.json (per-task pass + per-test detail) + pass_at_k.json."""
    results: Dict[str, bool] = {}
    details: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Any] = {}
    for p in glob.glob(os.path.join(workdir, "**", "*_eval_results.json"), recursive=True):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for tid, entries in (data.get("eval") or {}).items():
            if isinstance(entries, list) and entries:
                e0 = entries[0]
                results[tid] = e0.get("status") == "pass"
                # Trajectory completeness: keep the harness's own status + per-test detail so a
                # failure is readable (which assertion / traceback) instead of a bare boolean.
                det = e0.get("details")
                details[tid] = {
                    "status": e0.get("status"),
                    "details": det if isinstance(det, (dict, str)) else None,
                }
    for p in glob.glob(os.path.join(workdir, "**", "*_pass_at_k.json"), recursive=True):
        try:
            with open(p, encoding="utf-8") as f:
                pk = json.load(f)
            if "pass@1" in pk:
                metrics["bigcodebench_pass_at_1"] = pk["pass@1"]
        except Exception:
            pass
    return results, details, metrics


def _make_scorer(subset: str, parallel: int, metrics: Dict[str, Any]):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        workdir = tempfile.mkdtemp(prefix="gbench_bcb_")
        # `mkdtemp` creates the directory 0700 owned by the HOST uid, but the image runs as
        # `bigcodebenchuser` (uid 1000). Bind-mounted at /app that uid cannot traverse it, so
        # every relative path resolution from the container's CWD raises PermissionError -
        # including the `Path("bigcode/bigcodebench/state.json").exists()` probe that
        # `datasets.load_dataset` does before it falls back to the Hub. That surfaced as
        # "bigcodebench: no eval results parsed" with the whole suite scoring 0.
        # Reproduced directly: a 0700 mount raises, the same mount at 0777 resolves and is
        # writable. The harness must also write its results back here, hence 0777 not 0755.
        os.chmod(workdir, 0o777)
        samples_path = os.path.join(workdir, "samples.jsonl")
        with open(samples_path, "w", encoding="utf-8") as f:
            for tr in sample_traces:
                tid = (tr.get("extra_payload") or {}).get("task_id")
                if tid:
                    # The harness asserts each row carries "completion" or "solution"
                    # (bigcodebench/data/utils.py::load_solutions). Writing "raw_solution"
                    # got past the sanitizer's file read and then died with
                    # "AssertionError: No completion or solution found in sample!", which
                    # surfaced as "harness produced no parsed results" and a 0/N score.
                    f.write(json.dumps({"task_id": tid,
                                        "solution": tr.get("response_text") or ""}) + "\n")
        os.chmod(samples_path, 0o644)   # a 077 umask would leave this unreadable to uid 1000

        mount = f"{workdir}:/app:rw"
        # The container's default CWD is not writable/traversable, and the harness resolves
        # its dataset through a RELATIVE path first
        # (`Path('bigcode/bigcodebench/state.json').exists()` in
        # bigcodebench/data/bigcodebench.py:29). That stat raised
        # `PermissionError: [Errno 13] ... 'bigcode/bigcodebench/state.json'` before any
        # evaluation ran, so the whole suite reported "harness produced no parsed results".
        # Run in the writable mount and keep the HF caches inside it too.
        workdir_env = ["-w", "/app",
                       "-e", "HF_HOME=/app/.hf",
                       "-e", "HF_DATASETS_CACHE=/app/.hf/datasets"]
        sanitize = ["docker", "run", "--rm", *workdir_env,
                    "--entrypoint", "bigcodebench.sanitize",
                    "-v", mount, _IMAGE, "--samples", "/app/samples.jsonl", "--calibrate"]
        # The evaluator asserts the samples file covers the WHOLE split:
        #   "Missing problems in samples. Expected 1140 problems, got 20"
        # so any --eval-limit run failed outright and reported 0/N. `--selective_evaluate`
        # takes a comma-separated task-id list and restricts the run to it, which is what a
        # subset run needs.
        task_ids = [(tr.get("extra_payload") or {}).get("task_id") for tr in sample_traces]
        selective = ",".join(t for t in task_ids if t)
        evaluate = ["docker", "run", "--rm", "-m", "16g", *workdir_env, "-v", mount, _IMAGE,
                    "--execution", "local", "--subset", subset, "--split", "instruct",
                    "--samples", "/app/samples-sanitized-calibrated.jsonl",
                    # Fire parses `--pass-k 1` as an int, and evaluate() iterates it:
                    #   TypeError: 'int' object is not iterable
                    # A trailing comma makes it the 1-tuple the signature expects.
                    "--no-gt", "--pass-k", "1,", "--parallel", str(max(1, parallel))]
        if selective:
            evaluate += ["--selective_evaluate", selective]

        def _run():
            s = subprocess.run(sanitize, capture_output=True, text=True)
            if s.returncode != 0:
                return s
            return subprocess.run(evaluate, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run)

        results, details, m = _parse_bcb_results(workdir)
        metrics.update(m)
        if not results:
            # CC6: the harness ran but produced nothing. Reporting every sample as simply
            # "incorrect" would surface as a clean success + 0%, which is indistinguishable
            # from a model that genuinely solved nothing.
            logger.error("bigcodebench: no eval results parsed. stderr tail: %s",
                         (proc.stderr or "")[-800:])
            metrics["bigcodebench_report"] = {
                "error": "harness produced no parsed results",
                "stderr_tail": (proc.stderr or "")[-400:],
            }
        for tr in sample_traces:
            tid = (tr.get("extra_payload") or {}).get("task_id")
            tr["is_correct"] = bool(results.get(tid, False))
            tr["status"] = "OK"
            if tid in details:
                tr["execution"] = details[tid]
    return _score


def run_bigcodebench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run BigCodeBench calibrated pass@1 (or skip if Docker/image unavailable)."""
    ok, reason = check_bigcodebench_prerequisites()
    if not ok:
        raise infra_required("bigcodebench", reason, DOCS_URL)
    subset = kwargs.get("subset", "full")
    samples = _load_bigcodebench_samples(limit=kwargs.get("limit"), subset=subset)
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="bigcodebench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(subset, concurrency, metrics),
        declared_scoring_mode="execution",  # sandboxed test execution, not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
        temperature=kwargs.get("temperature"),
    )
    result.update(metrics)
    if isinstance(metrics.get("bigcodebench_report"), dict) and metrics["bigcodebench_report"].get("error"):
        result["status"] = "error"
        result["error"] = metrics["bigcodebench_report"]["error"]
    return result
