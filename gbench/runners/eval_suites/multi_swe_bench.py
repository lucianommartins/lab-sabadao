# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: multi_swe_bench
# Description: Multi-SWE-bench (ByteDance, 7 languages) - resolved rate via its own harness

"""gbench native built-in runner for multi_swe_bench (Coding & Software Engineering).

Canonical Multi-SWE-bench (ByteDance-Seed/Multi-SWE-bench) scored by execution-based
resolved-rate via the project's OWN harness (`multi_swe_bench.harness.run_evaluation`;
vanilla swebench cannot score it - different prediction schema and test fields).
SANDBOX_EVAL. Skips cleanly if the harness/Docker are absent.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MULTI_SWE_BENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import importlib.util
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
from .base import run_eval_suite, strip_thinking_tags
from .sampling import stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/multi_swe_bench.md"
_REPO = "ByteDance-Seed/Multi-SWE-bench"
_SRC_FILES: Dict[str, str] = {}  # instance_id -> local dataset jsonl path (for the scorer)


def check_multi_swe_bench_prerequisites() -> Tuple[bool, str]:
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    try:
        # find_spec raises (not returns None) when the PARENT package is absent.
        if importlib.util.find_spec("multi_swe_bench.harness.run_evaluation") is None:
            raise ModuleNotFoundError
    except ModuleNotFoundError:
        return False, "Python package 'multi_swe_bench' is not installed (pip install multi-swe-bench)."
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

def _load_multi_swe_bench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load Multi-SWE-bench instances via list_repo_files (load_dataset is broken for this repo)."""
    global _SRC_FILES
    _SRC_FILES = {}
    try:
        from huggingface_hub import HfApi, hf_hub_download
        files = HfApi().list_repo_files(_REPO, repo_type="dataset")
        jsonl_files = sorted(f for f in files if f.endswith("_dataset.jsonl"))
    except Exception as e:
        logger.error(f"Failed to list dataset for multi_swe_bench: {e}")
        raise RuntimeError(f"Could not load dataset for multi_swe_bench: {e}") from e

    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]] = []
    for jf in jsonl_files:
        lang = jf.split("/")[0] if "/" in jf else "unknown"
        local_p = hf_hub_download(repo_id=_REPO, filename=jf, repo_type="dataset")
        with open(local_p, encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                item = json.loads(line)
                org, repo, number = item.get("org"), item.get("repo"), item.get("number")
                if not org or not repo or number is None:
                    raise RuntimeError(
                        "multi_swe_bench: unexpected schema (org/repo/number); "
                        "refusing to fabricate sample data"
                    )
                instance_id = item.get("instance_id") or f"{org}__{repo}-{number}"
                _SRC_FILES[instance_id] = local_p
                title = str(item.get("title") or "").strip()
                body = str(item.get("body") or "").strip()
                prompt = (
                    f"Repository: {org}/{repo} ({lang})\n"
                    f"Base commit: {(item.get('base') or {}).get('sha', '')}\n\n"
                    f"Issue: {title}\n\n{body}\n\n"
                    "Output ONLY a unified git diff (`diff --git a/... b/...`), rooted at the "
                    "repository top, that resolves the issue."
                )
                samples.append(([{"role": "user", "content": prompt}],
                                item.get("fix_patch") or "",
                                {"category": lang, "instance_id": instance_id,
                                 "org": org, "repo": repo, "number": number}))

    if not samples:
        raise RuntimeError("multi_swe_bench returned empty rows")
    # Stratified across LANGUAGES, not a contiguous head (audit RC-1). The loader used to
    # `break` as soon as it had `limit` rows while walking `sorted(jsonl_files)`; `c/...`
    # sorts first, so `--eval-limit 20` returned 20 C instances (all facebook/zstd) and
    # never opened java/go/rust/kotlin/cpp/js/ts. A *Multi*-SWE-bench score measured on one
    # language is not the benchmark. Measured 2026-08-17: 1 distinct category in 20 samples.
    before = len(samples)
    samples = stratified_sample(
        samples, limit,
        key_fn=lambda s: (s[2] or {}).get("category"),
        seed="multi_swe_bench")
    langs = sorted({(s[2] or {}).get("category") for s in samples})
    logger.info("Loaded %d multi_swe_bench samples from %d instances across %d language(s): %s",
                len(samples), before, len(langs), ", ".join(str(l) for l in langs))
    return samples


def _make_scorer(model_name: str, max_workers: int, metrics: Dict[str, Any]):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        workdir = tempfile.mkdtemp(prefix="gbench_mswe_")
        preds_path = os.path.join(workdir, "preds.jsonl")
        subset_path = os.path.join(workdir, "dataset.jsonl")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # predictions: {org, repo, number, fix_patch}
        wanted = set()
        with open(preds_path, "w", encoding="utf-8") as pf:
            for tr in sample_traces:
                e = tr.get("extra_payload") or {}
                iid = e.get("instance_id")
                if not iid:
                    continue
                wanted.add(iid)
                pf.write(json.dumps({"org": e["org"], "repo": e["repo"],
                                     "number": e["number"],
                                     "fix_patch": _extract_patch(tr.get("response_text") or "")}) + "\n")

        # dataset subset: pull the exact rows we ran from the cached source files
        seen_files = set(_SRC_FILES.get(i) for i in wanted if _SRC_FILES.get(i))
        with open(subset_path, "w", encoding="utf-8") as sf:
            for src in seen_files:
                for line in open(src, encoding="utf-8"):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    iid = row.get("instance_id") or f"{row.get('org')}__{row.get('repo')}-{row.get('number')}"
                    if iid in wanted:
                        sf.write(line)

        # `repo_dir` is the ONE field the harness requires that has no argparse default
        # (every other one -- log_level, max_workers_build_image, stop_on_error, ... --
        # is filled in for us). Omitting it made the harness die in about a second with
        #     ValueError: Invalid repo_dir: None
        # on every run since the suite was written, so it has never produced a score.
        # `_check_repo_dir` also requires the directory to EXIST, hence the makedirs.
        #
        # It lives outside `workdir` on purpose: workdir is a fresh mkdtemp per run, and
        # with `need_clone: True` the harness clones every repo in the sample into
        # repo_dir. Putting it in the temp dir would re-clone Java/Rust/Go/C++/TS repos
        # from scratch on every single run.
        repo_dir = os.getenv("GBENCH_MULTI_SWE_REPO_DIR") or os.path.join(
            os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
            "gbench", "multi_swe_bench", "repos")
        try:
            os.makedirs(repo_dir, exist_ok=True)
        except OSError as e:
            logger.error("multi_swe_bench: cannot create repo_dir %s (%s); set "
                         "GBENCH_MULTI_SWE_REPO_DIR to a writable path.", repo_dir, e)
        if os.path.isdir(repo_dir) and not os.listdir(repo_dir):
            logger.warning(
                "multi_swe_bench: %s is empty, so this run will CLONE every repo in the "
                "sample and build one Docker image per instance (Java/Rust/Go/C++/TS). "
                "The first run is slow and disk-hungry; later runs reuse the clones.",
                repo_dir)

        config = {
            "mode": "evaluation", "workdir": workdir, "output_dir": out_dir,
            "log_dir": os.path.join(workdir, "logs"),
            "patch_files": [preds_path], "dataset_files": [subset_path],
            "repo_dir": repo_dir,
            "force_build": False, "need_clone": True, "clear_env": True,
            "max_workers": max(1, max_workers),
        }
        cfg_path = os.path.join(workdir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as cf:
            json.dump(config, cf)

        cmd, threads = swe_thread_cap.apply(
            [sys.executable, "-m", "multi_swe_bench.harness.run_evaluation", "--config", cfg_path],
            max_workers, "multi_swe_bench")
        metrics["docker_thread_cap"] = threads or None

        def _run():
            return subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run)

        resolved_ids: List[str] = []
        report = None
        for root, _, fs in os.walk(workdir):
            if "final_report.json" in fs:
                with open(os.path.join(root, "final_report.json"), encoding="utf-8") as f:
                    report = json.load(f)
                resolved_ids = report.get("resolved_ids", [])
                break
        if report is None:
            logger.error("multi_swe_bench: final_report.json not found. stderr tail: %s",
                         (proc.stderr or "")[-800:])
            metrics["multi_swe_bench_report"] = {
                "error": "harness report not produced",
                "stderr_tail": (proc.stderr or "")[-400:],
            }
        else:
            metrics["multi_swe_bench_report"] = {
                k: report.get(k) for k in ("total_instances", "resolved_instances",
                                           "unresolved_instances", "error_instances",
                                           "empty_patch_instances")
            }

        def _resolved(e: Dict[str, Any]) -> bool:
            if e.get("instance_id") in resolved_ids:
                return True
            # spelling of ids varies; fall back to matching repo + number. Match the
            # number as a BOUNDED token (not a substring) so issue #5 never matches
            # "...-15" / "...-50", and tolerate org/repo vs org__repo spelling.
            rep, num = str(e.get("repo")), str(e.get("number"))
            rep_variants = {rep, rep.replace("/", "__")}
            num_re = re.compile(rf"(?<!\d){re.escape(num)}(?!\d)")
            return any(num_re.search(str(r)) and any(rv in str(r) for rv in rep_variants)
                       for r in resolved_ids)

        for tr in sample_traces:
            tr["is_correct"] = _resolved(tr.get("extra_payload") or {})
            tr["status"] = "OK"
    return _score


def run_multi_swe_bench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Multi-SWE-bench resolved-rate (or skip if the harness/Docker are unavailable)."""
    ok, reason = check_multi_swe_bench_prerequisites()
    if not ok:
        raise infra_required("multi_swe_bench", reason, DOCS_URL)
    samples = _load_multi_swe_bench_samples(limit=kwargs.get("limit"))
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="multi_swe_bench",
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
    # CC6: a harness that did not actually evaluate is an error, not a 0% score.
    # Measured 2026-08-17: the harness died in 1 second with
    #   ValueError: Invalid repo_dir: None
    # (the config written above omits `repo_dir`, which the harness requires and
    # requires to exist), and this suite still published "20 questions, 0 correct,
    # 0.00%" into the results table. A reader cannot tell that apart from a model
    # that resolved nothing - which is exactly the fake number this audit exists to
    # prevent. Same guard as execute_swebench in swebench_common.
    report = metrics.get("multi_swe_bench_report")
    if isinstance(report, dict) and report.get("error"):
        result["status"] = "error"
        result["error"] = (
            "the Multi-SWE-bench harness produced no report, so nothing was "
            "evaluated; this is a harness/infrastructure failure, not a 0% "
            "resolved rate ({})".format(report.get("stderr_tail", "").strip()[-200:]
                                        or report["error"])
        )
    # Same class as the swe_bench family: a single-turn issue-text -> git-diff generation with NO
    # repository access, on a language subset (and honoring --eval-limit), whereas the published
    # Multi-SWE-bench number uses an agentic repo scaffold over the full 7-language set. Never
    # directly comparable.
    noncanon = ["single-turn generation with no repository access (the published Multi-SWE-bench "
                "number uses an agentic repo scaffold over the full language set)"]
    if kwargs.get("limit"):
        noncanon.append(f"--eval-limit subset ({int(kwargs['limit'])} tasks)")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    return result
