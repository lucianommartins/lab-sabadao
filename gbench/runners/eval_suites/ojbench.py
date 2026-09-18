# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: ojbench
# Description: OJBench (NOI/ICPC) - online-judge Pass@1 via the official ojbench + DMOJ sandbox

"""gbench native built-in runner for ojbench (Code & Competitive Programming).

Canonical OJBench (He-Ren/OJBench_testdata): the model emits a full stdin/stdout
solution; correctness is online-judge Pass@1 (Accepted iff ALL testcases pass
within per-problem time/memory limits), computed by the official `ojbench` library
over the DMOJ sandbox. SANDBOX_EVAL. Skips cleanly unless ojbench + DMOJ
judge-server + PyPy3 + g++ + the testdata (GBENCH_OJBENCH_TESTDATA; the bare
OJBENCH_TESTDATA still works as a deprecated alias) are all present.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_OJBENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, strip_thinking_tags, suite_env
from .sampling import stratified_sample
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/ojbench.md"
_IMAGE_DEFAULT = "gbench-ojbench"


def _image() -> str:
    return os.environ.get("GBENCH_OJBENCH_IMAGE", _IMAGE_DEFAULT)


def _testdata_dir() -> Optional[str]:
    return prereqs_path("OJBench_testdata", suite_env("GBENCH_OJBENCH_TESTDATA", "OJBENCH_TESTDATA"))


def check_ojbench_prerequisites() -> Tuple[bool, str]:
    """Docker + the locally-built gbench-ojbench image (which carries ojbench + DMOJ judge-server +
    PyPy3 + g++ on Python 3.11) + the NOI/ICPC testdata (bind-mounted read-only at run time).

    The judge is containerized because DMOJ's cptbox does not build on the 3.12 serving env
    (PyLongObject.ob_digit was removed in 3.12); on the image's Python 3.11 it compiles cleanly.
    """
    build = (f"Build it LOCALLY (gbench never pulls):\n"
             f"  docker build -t {_image()} -f docker/ojbench.Dockerfile docker\nSee {DOCS_URL}.")
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", _image()], capture_output=True).returncode != 0:
        return False, f"image {_image()!r} not found. " + build
    td = _testdata_dir()
    if not td or not os.path.isdir(os.path.join(td, "NOI")) or not os.path.isdir(os.path.join(td, "ICPC")):
        return False, ("OJBench testdata not found: set GBENCH_OJBENCH_TESTDATA (legacy alias "
                       "OJBENCH_TESTDATA still works) to a snapshot of "
                       "He-Ren/OJBench_testdata containing NOI/ and ICPC/ (7.85 GB, "
                       "`hf download He-Ren/OJBench_testdata --repo-type dataset`).")
    return True, ""


def _load_ojbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load OJBench prompts (full.jsonl); prompt sent verbatim; raises on load/schema failure."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="He-Ren/OJBench_testdata",
                               filename="prompts/full.jsonl", repo_type="dataset")
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except Exception as e:
        logger.error(f"Failed to load dataset for ojbench: {e}")
        raise RuntimeError(f"Could not load dataset for ojbench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for ojbench returned empty rows")
    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="ojbench")

    samples = []
    for item in rows:
        pid = item.get("id")
        prompt = item.get("prompt")
        lang = item.get("language")
        dataset = item.get("dataset")
        difficulty = item.get("difficulty")
        if pid is None or not prompt or not lang:
            raise RuntimeError(
                "ojbench: unexpected schema (id/prompt/language); refusing to fabricate sample data")
        # Canonical: the prompt already embeds the response-format constraint; send verbatim.
        messages = [{"role": "user", "content": str(prompt)}]
        samples.append((messages, pid, {
            "category": f"{dataset}_{difficulty}",
            "row": {"id": pid, "dataset": dataset, "language": lang, "difficulty": difficulty},
        }))

    logger.info(f"Loaded {len(samples)} ojbench samples.")
    return samples


def _judge_in_container(records: List[Dict[str, Any]], num_workers: int) -> List[Dict[str, Any]]:
    """Judge the generated submissions inside gbench-ojbench. The DMOJ sandbox (cptbox) uses
    ptrace + seccomp, so the run needs SYS_PTRACE and an unconfined seccomp profile (Docker's
    default profile blocks installing the inner seccomp filter). Records in / results out via a
    bind-mounted work dir; the 7.85 GB testdata is bind-mounted read-only."""
    if not records:
        return []
    td = os.path.abspath(_testdata_dir())
    workdir = tempfile.mkdtemp(prefix="gbench_ojbench_")
    os.chmod(workdir, 0o777)                        # the container writes results as its own uid
    name = "gbench_ojbench_" + os.path.basename(workdir)

    def _reap():
        # docker --rm only fires on a clean EXIT; reap the named container on timeout too.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    with open(os.path.join(workdir, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    cmd = ["docker", "run", "--rm", "--name", name,
           "--security-opt", "seccomp=unconfined", "--cap-add", "SYS_PTRACE",
           "-v", f"{workdir}:/work",
           "-v", f"{td}:/testdata:ro",
           "-e", "OJBENCH_TESTDATA=/testdata",
           "-e", f"OJBENCH_WORKERS={max(1, num_workers)}",
           _image()]
    # Bounded, configurable timeout so a stuck submission can NEVER hang the whole run (a compiled
    # solution's DMOJ sandbox execution can deadlock under Docker). Scales with the batch size;
    # GBENCH_OJBENCH_JUDGE_TIMEOUT_S overrides for large runs.
    _env_to = suite_env("GBENCH_OJBENCH_JUDGE_TIMEOUT_S", default="")
    timeout_s = int(_env_to) if _env_to else max(1200, 180 * len(records))
    logger.info("ojbench: judging %d submissions in %s (timeout %ds)", len(records), _image(), timeout_s)
    proc = None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ojbench: judge container exceeded {timeout_s}s and was killed - a compiled submission's "
            f"DMOJ cptbox sandbox can deadlock under Docker. Raise GBENCH_OJBENCH_JUDGE_TIMEOUT_S if "
            f"this was a large but healthy run; see {DOCS_URL}.") from e
    finally:
        _reap()
    results_path = os.path.join(workdir, "results.jsonl")
    if not os.path.isfile(results_path):
        raise RuntimeError(
            "ojbench: the judge container produced no results.jsonl (judging failed) - this is a "
            "harness/infra failure, not a 0%. Last output:\n"
            + ((proc.stderr or proc.stdout or "") if proc else "")[-1200:])
    with open(results_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_scorer(num_workers: int):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio

        records = []
        for tr in sample_traces:
            resp = tr.get("response_text")
            row = (tr.get("extra_payload") or {}).get("row") or {}
            if resp and row.get("id") is not None:
                records.append({**row, "content": strip_thinking_tags(resp)})

        results = await asyncio.to_thread(_judge_in_container, records, num_workers)

        # A result counts ONLY if the judge actually returned a verdict (`is_passed`). A submitted
        # program that comes back without one was NOT graded - e.g. the DMOJ worker crashed on that
        # problem (a wrong judge-server build with no Problem.cases() returns records unchanged) -
        # which is a harness failure, NOT a model 0. Scoring it False silently turns an ungraded
        # run into a fake 0% (measured 2026-09-11).
        by_key = {(r.get("id"), r.get("language")): bool(r.get("is_passed"))
                  for r in (results or []) if isinstance(r, dict) and "is_passed" in r}
        if records and not by_key:
            # Nothing was graded at all: never report this as a clean 0%.
            raise infra_required(
                "ojbench",
                "the judge container returned NO graded verdicts (`is_passed` missing on every "
                "submission) - the DMOJ judge could not score any program (commonly a wrong "
                "judge-server build: OJBench needs the commit its README pins, whose Problem.cases() "
                "the PyPI dmoj lacks). This is a judge/infra failure, not a 0%% resolved rate.",
                DOCS_URL)
        for tr in sample_traces:
            row = (tr.get("extra_payload") or {}).get("row") or {}
            key = (row.get("id"), row.get("language"))
            if key in by_key:
                tr["is_correct"] = by_key[key]           # genuinely judged
                tr["status"] = "OK"
            elif tr.get("response_text") and row.get("id") is not None:
                # submitted but the judge returned no verdict for it -> harness failure: exclude
                # from the numerator/denominator (like a judge outage) rather than score it wrong.
                tr["is_correct"] = False
                tr["status"] = "OK"
                tr["scoring_excluded"] = True
                tr["scoring_note"] = "judge returned no verdict for this submission"
            else:
                tr["is_correct"] = False              # empty/no model answer: a genuine non-answer
                tr["status"] = "OK"
    return _score


def run_ojbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run OJBench online-judge Pass@1 (or skip if the judge/testdata are unavailable)."""
    ok, reason = check_ojbench_prerequisites()
    if not ok:
        raise infra_required("ojbench", reason, DOCS_URL)
    samples = _load_ojbench_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="ojbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(concurrency),
        declared_scoring_mode="execution",  # sandboxed judge (test execution), not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature"),
    )
