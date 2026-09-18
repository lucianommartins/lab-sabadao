# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared SWE-bench execution-harness helpers.

Every SWE-bench-style suite (swe_bench_live, swe_bench_multilingual,
copilot_bench_swe, ...) resolves a GitHub issue by emitting one unified-diff
patch, then scores execution-based resolved-rate via the swebench Docker harness
(apply patch + run FAIL_TO_PASS/PASS_TO_PASS). This module holds the common
prereq check, dataset loader, patch extractor, and harness scorer so each suite
is a thin, consistent wrapper.
"""

import glob
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from . import swe_thread_cap
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)


def check_swebench_prereq(
    dataset: str, split: str, namespace: Optional[str], requires_fork: bool = False
) -> Tuple[bool, str]:
    """datasets + swebench + reachable Docker, and the harness can build a TestSpec."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed."
    try:
        from swebench.harness.run_evaluation import load_swebench_dataset  # noqa: F401
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: F401  (swebench < 5)
        except ImportError:
            from swebench.harness.run_evaluation import make_test_spec  # noqa: F401  (swebench >= 5 re-export)
    except ImportError:
        return False, "Python package 'swebench' is not installed (pip install gbench[evals])."
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=20)
        if r.returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception:
        return False, "Docker CLI/daemon is not available."
    try:
        from swebench.harness.run_evaluation import load_swebench_dataset
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec  # swebench < 5
        except ImportError:
            from swebench.harness.run_evaluation import make_test_spec       # swebench >= 5 re-export
        ds = load_swebench_dataset(dataset, split)
        try:
            make_test_spec(ds[0], namespace=namespace)
        except TypeError:
            make_test_spec(ds[0])
    except Exception as e:
        hint = ""
        if requires_fork:
            # The fork and upstream swebench are the SAME package name at different
            # versions, so only one can be installed at a time - and they support
            # different suites. Measured 2026-08-15 on this box:
            #     swebench 4.1.0 (upstream)   swe_bench_multilingual OK, swe_bench_live FAIL
            #     swebench 4.0.3 (Live fork)  swe_bench_live OK, swe_bench_multilingual FAIL
            #                                 (KeyError: 'parse_log_maven')
            # copilot_bench_swe works on both. Installing the fork into the main
            # environment therefore trades one suite for another, which is why this says
            # "separate virtualenv" rather than "pip install".
            hint = (" This suite needs the SWE-bench-Live harness fork "
                    "(github.com/SWE-bench-Live/SWE-bench-Live), which is the same "
                    "`swebench` package at an older version than the one upstream suites "
                    "need - installing it here would break swe_bench_multilingual. Install "
                    "it in a SEPARATE virtualenv and run this suite from there; see "
                    "docs/evals/swe_bench_live.md.")
        return False, (
            f"installed 'swebench' cannot build {dataset} test specs.{hint} "
            f"({type(e).__name__}: {str(e)[:100]})"
        )
    return True, ""


def extract_patch(text: str) -> str:
    """Extract a unified-diff patch from a model response.

    The old version looked only at the FIRST fenced block. When the model showed the
    current code first, or labelled the fence anything other than `diff`/`patch`, the
    `diff --git` check failed and it fell through to "everything from `diff --git` to the
    end of the response" - which swept up the closing fence and the prose after it. The
    harness then rejected the patch:

        patch: **** malformed patch at line 20: ```
        >>>>> Patch Apply Failed

    Measured on the 2026-08-17 run: EVERY non-empty patch across copilot_bench_swe (12),
    swe_bench_multilingual (8) and multi_swe_bench (6) carried a stray fence, so the
    resolved rate could only ever be ~0 regardless of the model.
    """
    if not text:
        return ""
    blocks = re.findall(r"```[^\n`]*\n([\s\S]*?)```", text)
    # 1) a block with the full git header
    for body in blocks:
        if "diff --git" in body:
            return _trim_after_diff(body)
    # 2) a block that is a unified diff WITHOUT the `diff --git` line. The model emits this
    #    constantly - `--- a/pylint/utils.py` / `+++ b/pylint/utils.py` / `@@ -115,7 +115,7 @@`
    #    - and it applies fine with `git apply -p1`, but requiring the header threw it away.
    #    Measured on 2026-08-17: 40 of the 47 "no extractable patch" samples across the four
    #    SWE suites were this, i.e. most of what looked like a model failure was extraction.
    for body in blocks:
        if _is_unified_diff(body):
            return _trim_after_diff(body)
    # 3) unfenced
    idx = text.find("diff --git")
    if idx != -1:
        return _trim_after_diff(text[idx:])
    m = _UNIFIED_START.search(text)
    if m and _HUNK.search(text[m.start():]):
        return _trim_after_diff(text[m.start():])
    return ""


#: A unified diff needs a source header and at least one hunk. `/dev/null` covers new files.
_UNIFIED_START = re.compile(r"^--- (?:a/|/dev/null)", re.M)
_HUNK = re.compile(r"^@@ -\d+", re.M)


def _is_unified_diff(body: str) -> bool:
    return bool(_UNIFIED_START.search(body) and _HUNK.search(body))


def _trim_after_diff(body: str) -> str:
    """Drop anything after the diff ends: a bare fence, or trailing prose."""
    lines = body.splitlines()
    out: List[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            break
        out.append(line)
    # trailing prose does not start with a diff marker; cut back to the last patch line
    while out and not re.match(r"^(diff --git|index |--- |\+\+\+ |@@|[+\- ]|new file|"
                               r"deleted file|old mode|new mode|similarity index|"
                               r"rename |Binary files|\\ No newline)", out[-1]):
        out.pop()
    return ("\n".join(out).strip() + "\n") if out else ""


def load_swe_samples(
    dataset: str, split: str, limit: Optional[int], eval_name: str
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load a SWE-bench-schema dataset into (messages, gold='', extra) samples; raises on failure."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset, split=split)
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for {eval_name}: {e}")
        raise RuntimeError(f"Could not load dataset for {eval_name}: {e}") from e

    if not rows:
        raise RuntimeError(f"{eval_name} returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("repo"), seed="swebench_common")

    samples = []
    for item in rows:
        repo = item.get("repo")
        instance_id = item.get("instance_id")
        problem = item.get("problem_statement")
        if not instance_id or not problem:
            raise RuntimeError(
                f"{eval_name}: unexpected dataset schema (instance_id/problem_statement); "
                "refusing to fabricate sample data"
            )
        prompt = (
            f"You are an expert software engineer resolving a GitHub issue in {repo}.\n\n"
            f"Issue ({instance_id}):\n{problem}\n\n"
            "Produce a single unified-diff patch in git format (`diff --git a/... b/...`) "
            "that resolves the issue. Output only the patch."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, "", {"category": str(repo), "instance_id": instance_id, "split": split}))

    logger.info(f"Loaded {len(samples)} {eval_name} samples from HF Hub ('{dataset}' {split}).")
    return samples


#: Substrings in a swebench `run_instance.log` that mean the MODEL's patch would not apply (the
#: harness DID run - it checked out the repo and tried to apply the diff). These are legitimate
#: unresolved outcomes (the model produced a bad diff), NOT infrastructure failures.
_PATCH_FAIL_SIGNATURES = (
    "patch apply failed",
    "malformed patch",
    "no file to patch",
    "can't find file to patch",
    "patch failed",
    "hunk failed",
    "hunk #",
    "hunks failed",
    "hunk ignored",
    "while trying to apply the patch",
    "saving rejects",
    "corrupt patch",
    "patch does not apply",
    "does not match index",
)


def _classify_swebench_errors(workdir: str, error_ids: List[str]) -> Tuple[set, set]:
    """Split swebench `error_ids` into (patch_apply_failed, harness_error).

    A patch-apply failure is a MODEL failure: the harness ran, checked out the repo, and the model's
    diff would not apply -> unresolved (0), the canonical outcome. A true harness error (docker/build/
    timeout/OOM, or a missing log we cannot inspect) means the instance could not be scored at all.
    We read each errored instance's `run_instance.log` and look for a patch-apply signature; anything
    else (including a missing log) is treated conservatively as a harness error."""
    patch_fail: set = set()
    harness_err: set = set()
    for iid in error_ids:
        logs = glob.glob(os.path.join(workdir, "logs", "run_evaluation", "*", "*", iid, "run_instance.log"))
        text = ""
        for lp in logs:
            try:
                with open(lp, encoding="utf-8", errors="ignore") as f:
                    text += f.read()
            except OSError:
                pass
        low = text.lower()
        if text and any(sig in low for sig in _PATCH_FAIL_SIGNATURES):
            patch_fail.add(iid)
        else:
            harness_err.add(iid)   # missing log or non-patch error -> conservative: infra
    return patch_fail, harness_err


def make_swebench_scorer(
    eval_name: str, model_name: str, dataset: str, split: str,
    namespace: Optional[str], max_workers: int, metrics: Dict[str, Any],
    harness_image: Optional[str] = None,
):
    """Async scorer: run the swebench harness over the model patches, mark resolved.

    `harness_image` (swe_bench_live only): run `swebench.harness.run_evaluation` inside that LOCAL
    image instead of the host interpreter, so the SWE-bench-Live fork of `swebench` stays isolated
    from the upstream `swebench` the main env uses (they are the same package name at incompatible
    versions and cannot coexist). None => the host interpreter (upstream swebench)."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        model_tag = "gbench__" + re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)[:48]
        run_id = "gbench_" + re.sub(r"[^A-Za-z0-9_.-]", "_", f"{eval_name}_{model_name}")[:40]

        preds: Dict[str, Dict[str, str]] = {}
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            if not iid:
                continue
            preds[iid] = {
                "instance_id": iid,
                "model_name_or_path": model_tag,
                "model_patch": extract_patch(tr.get("response_text") or ""),
            }

        workdir = tempfile.mkdtemp(prefix="gbench_swe_")
        metrics["swebench_workdir"] = workdir   # so execute_swebench can classify errored instances
        preds_path = os.path.join(workdir, "preds.jsonl")
        with open(preds_path, "w", encoding="utf-8") as f:
            for p in preds.values():
                f.write(json.dumps(p) + "\n")

        harness_args = [
            "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset, "--split", split,
            "--predictions_path", preds_path, "--run_id", run_id,
            "--max_workers", str(max(1, max_workers)), "--cache_level", "env",
            "--instance_ids", *list(preds.keys()),
        ]
        if namespace:
            harness_args += ["--namespace", namespace]

        if harness_image:
            # Run the harness INSIDE the isolated fork image (swe_bench_live). It spawns per-instance
            # task containers on the HOST daemon via the mounted socket (docker-out-of-docker); the
            # workdir is identity-mounted so the report path it writes matches what we read here, and
            # the HF cache is mounted so it loads the dataset without a re-download.
            hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            threads = swe_thread_cap.resolve_threads(max_workers)
            cmd = ["docker", "run", "--rm", "--network", "host",
                   "-v", "/var/run/docker.sock:/var/run/docker.sock",
                   "-v", f"{workdir}:{workdir}:rw", "-w", workdir]
            if os.path.isdir(hf_home):
                cmd += ["-v", f"{hf_home}:{hf_home}:rw", "-e", f"HF_HOME={hf_home}"]
            if os.environ.get("HF_TOKEN"):
                cmd += ["-e", f"HF_TOKEN={os.environ['HF_TOKEN']}"]
            if threads and threads > 0:
                for _tvar in swe_thread_cap.THREAD_VARS:
                    cmd += ["-e", f"{_tvar}={threads}"]
            cmd += [harness_image, "python", *harness_args]
            metrics["harness_image"] = harness_image
        else:
            # Without the thread cap every task container sees all host cores and oversubscribes
            # itself into a spin-wait; see swe_thread_cap for the measurement.
            cmd, threads = swe_thread_cap.apply([sys.executable, *harness_args], max_workers, eval_name)
        metrics["docker_thread_cap"] = threads or None

        def _run():
            return subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run)

        report_path = os.path.join(workdir, f"{model_tag}.{run_id}.json")
        resolved: set = set()
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                rep = json.load(f)
            resolved = set(rep.get("resolved_ids", []))
            metrics["swebench_report"] = {
                k: rep.get(k) for k in (
                    "total_instances", "submitted_instances", "completed_instances",
                    "resolved_instances", "unresolved_instances", "error_instances",
                    "empty_patch_instances",
                    # error_ids MUST be carried through: execute_swebench feeds it to
                    # _classify_swebench_errors to split patch-apply failures (model 0) from
                    # true harness/infra errors. Without it the split silently saw [] and the
                    # infra guard could never fire, so an all-infra-errored run reported a fake
                    # "success, 0%" instead of status=error.
                    "error_ids",
                )
            }
        else:
            logger.error(
                "%s: harness report not found (%s). stderr tail: %s",
                eval_name, report_path, (proc.stderr or "")[-800:],
            )
            metrics["swebench_report"] = {"error": "harness report not produced"}

        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            tr["is_correct"] = iid in resolved
            tr["status"] = "OK"
    return _score


def execute_swebench(
    eval_name: str, model_name: str, base_url: str, concurrency: int,
    dataset: str, split: str, namespace: Optional[str], **kwargs,
) -> Dict[str, Any]:
    """Load samples, generate patches, and score via the harness. Assumes prereqs pass."""
    samples = load_swe_samples(dataset, split, kwargs.get("limit"), eval_name)
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name=eval_name,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=make_swebench_scorer(
            eval_name, model_name, dataset, split, namespace, concurrency, metrics,
            harness_image=kwargs.get("harness_image")),
        thinking=kwargs.get("enable_thinking", False),
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature"),
    )
    result.update(metrics)
    # leaderboard_comparable honesty (shared by swe_bench_multilingual / copilot_bench_swe /
    # swe_bench_live). The published SWE-bench resolved rate is the FULL test split at greedy
    # decoding; a --eval-limit subset, a non-greedy (e.g. --thinking) run, or a partially
    # evaluated harness (below) is not that number. Gate on full-dataset + greedy only - NOT on the
    # harness flavour, since swe_bench_live legitimately runs the SWE-bench-Live fork and is still
    # canonical for its own leaderboard.
    noncanon: list = []
    # This shared runner prompts the model single-turn with the issue text and no
    # repository access; the published SWE-bench numbers use an agentic/retrieval
    # scaffold, so the resolved rate here is not directly leaderboard-comparable.
    noncanon.append("single-turn generation with no repository access (not the "
                    "agentic/retrieval SWE-bench protocol)")
    if kwargs.get("limit"):
        noncanon.append(f"subset run (--eval-limit {kwargs.get('limit')})")
    if result.get("temperature") not in (0.0, 0):
        noncanon.append(
            f"non-greedy temperature={result.get('temperature')} (leaderboard is greedy 0.0)")
    # CC6: a harness that did not actually evaluate is an error, not a 0% score.
    #
    # The original guard only fired when the report carried an `error` key. On the
    # 2026-08-17 run the reports were well-formed and said the work never happened:
    #   swe_bench_multilingual  completed 0/20   (9 errored, 11 empty patches)
    #   copilot_bench_swe       completed 1/20   (11 errored, 8 empty patches)
    # Both were published as ordinary 0% scores, which is exactly the fake number this
    # whole audit exists to prevent - "the model resolved nothing" and "the harness never
    # ran" are indistinguishable to a reader.
    report = metrics.get('swebench_report')
    if isinstance(report, dict):
        if report.get('error'):
            result['status'] = 'error'
            result['error'] = report['error']
            noncanon.append("harness error (no valid resolved rate)")
        else:
            completed = report.get('completed_instances') or 0
            total = report.get('total_instances') or result.get('total_questions') or 0
            empty = report.get('empty_patch_instances') or 0
            # Split swebench's error_instances into MODEL failures (the diff would not apply - the
            # harness ran, checked out the repo, and the model's patch was bad -> unresolved/0, the
            # canonical outcome) vs TRUE harness errors (docker/build/timeout/OOM -> the instance
            # could not be scored). Only the latter is an "infrastructure failure"; a non-applying
            # patch is an honest 0, not an error, so it must not hide the model's real resolved rate.
            patch_fail_ids, harness_err_ids = _classify_swebench_errors(
                metrics.get('swebench_workdir') or "", report.get('error_ids') or [])
            report['patch_apply_failed_instances'] = len(patch_fail_ids)
            report['harness_error_instances'] = len(harness_err_ids)
            # instances with a model-attributable verdict: resolved/unresolved (completed) + empty
            # patch (model emitted none) + patch-apply failure (model diff bad). All count as scored.
            model_scoreable = completed + empty + len(patch_fail_ids)
            if total:
                if harness_err_ids and model_scoreable == 0:
                    # nothing was model-scoreable AND real infra errors -> genuine infra failure
                    result['status'] = 'error'
                    result['error'] = (
                        "the harness could not score any of {} instances ({} true harness errors "
                        "docker/build/timeout, {} empty patches); this is a "
                        "harness/infrastructure failure, not a 0% resolved rate"
                    ).format(total, len(harness_err_ids), empty)
                    noncanon.append("harness infrastructure failure (no scoreable instances)")
                elif harness_err_ids:
                    # some instances hit TRUE infra errors; the resolved rate is over the rest.
                    result['partial_evaluation'] = {
                        "scoreable_instances": model_scoreable, "total_instances": total,
                        "harness_error_instances": len(harness_err_ids),
                        "patch_apply_failed_instances": len(patch_fail_ids),
                        "empty_patch_instances": empty,
                    }
                    noncanon.append(
                        f"partial: {len(harness_err_ids)} of {total} instance(s) hit harness errors "
                        f"(not model-scoreable)")
                    logger.warning(
                        "[%s] %d of %d instances hit TRUE harness errors (not scoreable); %d "
                        "patch-apply failures + %d empty patches count as unresolved (model).",
                        eval_name, len(harness_err_ids), total, len(patch_fail_ids), empty)
                elif completed < total:
                    # every non-completed instance is a MODEL failure (patch-apply / empty patch), so
                    # the resolved rate over `total` is the honest number - nothing to flag.
                    logger.info(
                        "[%s] %d/%d resolved; %d patch-apply failures + %d empty patches count as "
                        "unresolved (model failures, not infrastructure).", eval_name,
                        report.get('resolved_instances', 0), total, len(patch_fail_ids), empty)
    # Set the honest comparability flag last (after the partial/error checks). Error rows keep
    # whatever status was set above; the flag simply records that the number is not comparable.
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    return result


def skipped_result(eval_name: str, model_name: str, reason: str, docs_url: str) -> Dict[str, Any]:
    """Standard skip dict."""
    msg = f"[SKIP] {eval_name} skipped: {reason} See '{docs_url}' for setup instructions."
    logger.warning(msg)
    print(f"\n{msg}")
    return {
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "status": "skipped",
        "total_questions": 0,
        "correct_answers": 0,
        "accuracy": 0.0,
        "skip_reason": f"{reason} (See {docs_url})",
    }


def infra_required(eval_name: str, reason: str, docs_url: str) -> RuntimeError:
    """Return a hard error for a canonical suite whose required infra is absent.

    The no-skip policy (2026-09): a faithful eval that cannot run because an
    out-of-process prerequisite is missing - Docker, an execution harness, a live
    external API/DB/VM, the official agent-environment, or a required judge/API key -
    must NOT silently skip nor emit a fabricated/partial number. It raises instead.
    `run_eval_suites` (evals.py) isolates the raise into a per-suite `status:"error"`
    row carrying this message and continues the sweep, so one un-provisioned suite
    never aborts the rest. Idiom at the call site: `raise infra_required(...)`.
    """
    msg = (f"[INFRA REQUIRED] {eval_name} cannot run and must not be skipped: {reason} "
           f"Provision the prerequisites and re-run - see '{docs_url}'.")
    logger.error(msg)
    return RuntimeError(msg)


def prereqs_path(subdir: str, explicit: Optional[str] = None) -> Optional[str]:
    """Resolve a suite's local prerequisite (a checkout / data dir / file), with a shared fallback.

    Precedence:
      1. `explicit` - the suite's own env var (already stripped by the caller), if set;
      2. `$GBENCH_PREREQS_DIR/<subdir>` when GBENCH_PREREQS_DIR is set AND that path exists on disk;
      3. None.

    This is purely ADDITIVE: it returns None (never a fabricated path) when nothing resolves, so the
    caller's existing hard-error / infra_required check fires unchanged. The `os.path.exists` guard
    means a wrong or empty prereqs tree still hard-errors exactly as before - the fallback only kicks
    in when the operator has actually provisioned the standard tree. `<subdir>` is the suite-specific
    path under the prereqs root (may be nested, e.g. "PurpleLlama/CodeShield").
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    root = os.environ.get("GBENCH_PREREQS_DIR")
    if not root:
        return None
    cand = os.path.join(os.path.expanduser(root), subdir)
    return cand if os.path.exists(cand) else None


def partial_run(eval_name: str, reason: str, docs_url: str,
                **detail: Any) -> Dict[str, Any]:
    """Record a run that PROCEEDED but measured less than the benchmark defines.

    The sibling of `skipped_result`, for the case where a missing prerequisite removes
    part of the benchmark instead of all of it. A bare `logger.warning` is not enough:
    the log is ephemeral and the result JSON is what anyone reads later, so a partial
    run was indistinguishable from a complete one. On 2026-08-19 `aider_polyglot` dropped
    all its Java exercises for want of `gradle` and published a 5-language score with
    nothing in the JSON marking it partial.

    Returns fields to merge into the suite's result. Always sets
    `leaderboard_comparable=False`, because a benchmark measured over a subset of what it
    defines is not the benchmark.
    """
    msg = (f"[PARTIAL] {eval_name}: {reason} "
           f"See '{docs_url}' for the prerequisites and how to install them.")
    logger.warning(msg)
    out: Dict[str, Any] = {
        "partial_run": True,
        "partial_reason": f"{reason} (See {docs_url})",
        "leaderboard_comparable": False,
    }
    out.update(detail)
    return out
