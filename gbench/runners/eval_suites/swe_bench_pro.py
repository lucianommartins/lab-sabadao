# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: swe_bench_pro
# Description: SWE-bench Pro (ScaleAI) - resolved rate via the Pro-specific Docker harness

"""gbench native built-in runner for swe_bench_pro (Coding & Software Engineering).

Canonical SWE-bench Pro (ScaleAI/SWE-bench_Pro) scored by resolved-rate via the
Pro-specific Docker harness (scaleapi/SWE-bench_Pro-os `swe_bench_pro_eval.py` +
`jefzda/sweap-images`; vanilla swebench cannot score it). SANDBOX_EVAL. A real
run pulls tens-hundreds of GB of images, so it is gated behind an explicit opt-in
(GBENCH_SWE_BENCH_PRO_RUN=1; the bare SWE_BENCH_PRO_RUN still works as a deprecated
alias) and skips cleanly otherwise.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SWE_BENCH_PRO_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from . import swe_thread_cap
from .base import run_eval_suite, strip_thinking_tags, suite_env
from .sampling import stratified_sample
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_bench_pro.md"
_DATASET = "ScaleAI/SWE-bench_Pro"

#: The output-format instruction the canonical execution scorer REQUIRES. `_make_scorer`
#: grades by `extract_patch`, which pulls the model's patch from a ```diff block; a prompt
#: that does not ask for exactly that yields prose and an empty patch (a structural 0).
#: Exposed as a constant so plugin variants that reuse `_make_scorer` (e.g.
#: a planning-mode agentic plugin) share the SAME instruction and cannot drift from
#: the format their scorer parses. Thinking-mode runs still work: the plan goes in the
#: reasoning channel and `extract_patch` strips it before reading the diff.
PATCH_OUTPUT_INSTRUCTION = (
    "Output ONLY a single unified git diff (`diff --git a/... b/...`) that resolves "
    "the issue, inside a ```diff code block.")


def _harness_dir() -> Optional[str]:
    return prereqs_path("SWE-bench_Pro-os",
                        suite_env("GBENCH_SWE_BENCH_PRO_HARNESS_DIR", "SWE_BENCH_PRO_HARNESS_DIR"))


#: Columns `swe_bench_pro_eval.py` reads off the raw-sample frame (lines 96-99, 556-557).
_RAW_SAMPLE_COLUMNS = ("instance_id", "base_commit", "before_repo_set_cmd",
                       "selected_test_files_to_run", "fail_to_pass", "pass_to_pass")


def raw_sample_path() -> Optional[str]:
    """Path to the harness' `--raw_sample_path` table, generating it if the clone lacks it.

    `scaleapi/SWE-bench_Pro-os` documents `swe_bench_pro_full.csv` in its README but does
    not ship it, so a fresh clone always failed the prerequisite check and the suite
    skipped. The file is not privileged data: every column the harness reads is a column
    of the canonical HF dataset (`ScaleAI/SWE-bench_Pro`, test split), already stored in
    the string-of-python-literal form its `eval()` calls expect. So gbench writes it once
    into the harness directory (or `GBENCH_SWE_BENCH_PRO_RAW_SAMPLE` if set, or a temp file when
    the clone is read-only) instead of asking the operator to find it.

    Returns None when neither the file exists nor the dataset can be read.
    """
    explicit = suite_env("GBENCH_SWE_BENCH_PRO_RAW_SAMPLE", "SWE_BENCH_PRO_RAW_SAMPLE")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    hd = _harness_dir()
    if not hd:
        return None
    shipped = os.path.join(hd, "swe_bench_pro_full.csv")
    if os.path.isfile(shipped):
        return shipped

    try:
        from datasets import load_dataset
        df = load_dataset(_DATASET, split="test").to_pandas()
    except Exception as e:
        logger.warning("swe_bench_pro: cannot build the raw-sample table from %s (%s)",
                       _DATASET, e)
        return None
    missing = [c for c in _RAW_SAMPLE_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("swe_bench_pro: %s is missing the harness columns %s; the raw-sample "
                       "table cannot be generated.", _DATASET, missing)
        return None

    for target in (shipped, os.path.join(tempfile.gettempdir(), "swe_bench_pro_full.csv")):
        try:
            df.to_csv(target, index=False)
            logger.info("swe_bench_pro: wrote the raw-sample table (%d instances) to %s",
                        len(df), target)
            return target
        except OSError as e:
            logger.info("swe_bench_pro: could not write %s (%s)", target, e)
    return None


def check_swe_bench_pro_prerequisites() -> Tuple[bool, str]:
    """Docker + docker SDK + the Pro harness checkout + explicit opt-in (expensive)."""
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    try:
        import docker  # noqa: F401
    except ImportError:
        return False, "Python 'docker' SDK is not installed (pip install gbench[evals])."
    hd = _harness_dir()
    if not hd or not os.path.isfile(os.path.join(hd, "swe_bench_pro_eval.py")) \
            or not os.path.isdir(os.path.join(hd, "run_scripts")):
        return False, ("SWE-bench Pro harness not found: set GBENCH_SWE_BENCH_PRO_HARNESS_DIR to a clone of "
                       "scaleapi/SWE-bench_Pro-os (needs swe_bench_pro_eval.py and run_scripts/).")
    # The clone does not ship swe_bench_pro_full.csv; gbench builds it from the canonical
    # dataset. Only a genuinely unobtainable table is a prerequisite failure.
    if raw_sample_path() is None:
        return False, ("the harness raw-sample table is unavailable: "
                       f"{os.path.join(hd, 'swe_bench_pro_full.csv')} does not exist and it "
                       f"could not be generated from {_DATASET} (needs the dataset cached or "
                       "network access). Set GBENCH_SWE_BENCH_PRO_RAW_SAMPLE to point at your own copy.")
    if suite_env("GBENCH_SWE_BENCH_PRO_RUN", "SWE_BENCH_PRO_RUN") != "1":
        return False, ("SWE-bench Pro is gated: it pulls tens-hundreds of GB of images per run. "
                       "Set GBENCH_SWE_BENCH_PRO_RUN=1 to enable.")
    return True, ""


def _extract_patch(text: str) -> str:
    """Delegate to the shared extractor.

    This file used to carry its own copy, which only inspected the FIRST fenced block and
    otherwise took everything from `diff --git` to end-of-response - sweeping up the
    closing fence and the prose after it, so the harness rejected the patch as malformed
    ("patch: **** malformed patch at line 20: ```"). Three suites had the same copy.
    """
    from .swebench_common import extract_patch
    return extract_patch(strip_thinking_tags(text or ""))

def _load_swe_bench_pro_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load SWE-bench Pro tasks; raises on load/schema failure. Tests are looked up by the harness."""
    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET, split="test")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for swe_bench_pro: {e}")
        raise RuntimeError(f"Could not load dataset for swe_bench_pro: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for swe_bench_pro returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("repo"), seed="swe_bench_pro")

    samples = []
    for item in rows:
        instance_id = item.get("instance_id")
        repo = item.get("repo")
        problem = item.get("problem_statement")
        if not instance_id or not problem:
            raise RuntimeError(
                "swe_bench_pro: unexpected schema (instance_id/problem_statement); "
                "refusing to fabricate sample data")
        parts = [f"Repository: {repo}", "", f"Issue:\n{problem}"]
        if item.get("requirements"):
            parts += ["", f"Requirements:\n{item['requirements']}"]
        if item.get("interface"):
            parts += ["", f"Interface:\n{item['interface']}"]
        parts += ["", PATCH_OUTPUT_INSTRUCTION]
        messages = [{"role": "user", "content": "\n".join(parts)}]
        samples.append((messages, item.get("patch") or "",
                        {"category": str(item.get("repo_language") or repo), "instance_id": instance_id}))

    logger.info(f"Loaded {len(samples)} swe_bench_pro samples.")
    return samples


def _make_scorer(model_name: str, num_workers: int, metrics: Dict[str, Any]):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        hd = _harness_dir()
        workdir = tempfile.mkdtemp(prefix="gbench_swepro_")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        preds_path = os.path.join(workdir, "preds.json")
        prefix = "gbench__" + re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)[:40]

        preds = []
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            if iid:
                preds.append({"instance_id": iid, "patch": _extract_patch(tr.get("response_text") or ""),
                              "prefix": prefix})
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)

        cmd = [
            # sys.executable, NOT bare "python": the scorer imports pandas + docker (in the gbench
            # venv). Bare "python" resolves via PATH and, launched from a shell without that venv
            # first, silently ModuleNotFdErrors -> no eval_results.json -> every instance False.
            sys.executable, os.path.join(hd, "swe_bench_pro_eval.py"),
            f"--raw_sample_path={raw_sample_path()}",
            f"--patch_path={preds_path}", f"--output_dir={out_dir}",
            f"--scripts_dir={os.path.join(hd, 'run_scripts')}",
            f"--num_workers={max(1, num_workers)}",
            "--dockerhub_username=jefzda", "--use_local_docker",
        ]
        cmd, threads = swe_thread_cap.apply(cmd, num_workers, "swe_bench_pro")
        metrics["docker_thread_cap"] = threads or None

        def _run():
            return subprocess.run(cmd, cwd=hd, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run)

        results: Dict[str, bool] = {}
        report_path = os.path.join(out_dir, "eval_results.json")
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as f:
                results = {k: bool(v) for k, v in json.load(f).items()}
            # Count patches that were actually submittable. A looping model produces no
            # extractable diff, and "the model emitted no patch" is not "the patch did not
            # resolve the issue" - on the 2026-08-17 run this suite published 0/20 as a
            # `success`, indistinguishable from a genuine zero.
            empty = sum(1 for pr in preds if not (pr.get("patch") or "").strip())
            metrics["swe_bench_pro_report"] = {
                "total_instances": len(results),
                "resolved_instances": sum(results.values()),
                "empty_patch_instances": empty,
                "submitted_with_patch": len(preds) - empty,
            }
        else:
            logger.error("swe_bench_pro: eval_results.json not found. stderr tail: %s",
                         (proc.stderr or "")[-800:])
            metrics["swe_bench_pro_report"] = {"error": "harness report not produced"}

        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            tr["is_correct"] = bool(results.get(iid, False))
            tr["status"] = "OK"

        rep = metrics.get("swe_bench_pro_report") or {}
        if rep.get("error"):
            # No report at all (harness crash) -> promote to a run-level error, never a 0%.
            metrics["swe_bench_pro_error"] = rep["error"]
        elif rep.get("submitted_with_patch") == 0 and rep.get("total_instances"):
            metrics["swe_bench_pro_error"] = (
                "no extractable patch was produced for any of {} instances (all empty); "
                "this measures the generation, not the resolved rate".format(
                    rep["total_instances"]))
        elif rep.get("resolved_instances") == 0 and rep.get("submitted_with_patch"):
            # Patches WERE submitted but nothing resolved. The Pro harness collapses a failed
            # image pull / docker error into False (swe_bench_pro_eval.py returns None -> False),
            # so an all-images-failed run looks identical to a genuine 0%. If the harness stderr
            # shows those infra markers, it is a harness/infra failure, not a resolved rate.
            stderr = (proc.stderr or "")
            if ("Failed to pull or find image locally" in stderr
                    or "Error in eval_with_docker" in stderr):
                metrics["swe_bench_pro_error"] = (
                    "the Pro harness could not pull/run instance images (Docker/infra failure) so "
                    "0 of {} submitted patches were evaluated; this is a harness failure, not a 0% "
                    "resolved rate. stderr tail: {}".format(
                        rep.get("submitted_with_patch"), stderr[-400:]))
    return _score


def run_swe_bench_pro(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench Pro resolved-rate (or skip if harness/Docker/opt-in absent).

    Two modes (SWE-bench Pro is an AGENTIC task):
      * AGENTIC (opt-in `GBENCH_SWEBENCH_PRO_AGENTIC=1`): a mini-swe-agent rollout solves each
        issue inside the repo container, then patches are scored. This is the meaningful mode.
        Heavy (gated sweap-images, multi-turn) - validate a subset with --limit first.
      * single-turn (default): the model emits one diff from the issue with no repo access. That
        is structurally near-0 (patches can't apply) and measures generation, not SWE ability;
        kept as the default only for continuity. See docs/evals/swe_bench_pro.md.
    """
    # No-skip: missing Docker/harness/dataset - and the GBENCH_SWE_BENCH_PRO_RUN opt-in (a cost gate for
    # the tens-to-hundreds of GB of instance images) - HARD-ERROR rather than emit a fabricated
    # skip row. The reason string explains how to provision/opt in.
    ok, reason = check_swe_bench_pro_prerequisites()
    if not ok:
        raise infra_required("swe_bench_pro", reason, DOCS_URL)
    if os.getenv("GBENCH_SWEBENCH_PRO_AGENTIC") == "1":
        from .swe_bench_pro_agentic import execute_agentic
        return execute_agentic(model_name, base_url, concurrency=concurrency,
                               limit=kwargs.get("limit"), only_instance_ids=None,
                               eval_name="swe_bench_pro",
                               temperature=kwargs.get("temperature"),
                               max_output_tokens=kwargs.get("max_output_tokens"))
    samples = _load_swe_bench_pro_samples(limit=kwargs.get("limit"))
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="swe_bench_pro",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(model_name, concurrency, metrics),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature"),
    )
    result.update(metrics)
    if metrics.get("swe_bench_pro_error"):
        result["status"] = "error"
        result["error"] = metrics["swe_bench_pro_error"]
    # The published SWE-bench Pro leaderboard is the AGENTIC protocol over the full task set at
    # greedy decoding. This default single-turn path (one diff, no repo access) is structurally not
    # that number, so it is never leaderboard-comparable; a --eval-limit subset or a non-greedy
    # (--thinking) run compounds it.
    noncanon = ["single-turn generation, not the agentic SWE-bench Pro leaderboard protocol"]
    if kwargs.get("limit"):
        noncanon.append(f"subset run (--eval-limit {kwargs.get('limit')})")
    if result.get("temperature") not in (0.0, 0):
        noncanon.append(f"non-greedy temperature={result.get('temperature')}")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    return result
