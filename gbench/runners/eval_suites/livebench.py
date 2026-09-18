# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: livebench
# Description: LiveBench (Abacus/NYU monthly-refreshed, contamination-limited; 6 core categories)

"""gbench native built-in runner for livebench (General Knowledge & Reasoning).

Canonical LiveBench (White et al.; livebench.ai; github.com/LiveBench/LiveBench) scores 6
core categories - coding, data_analysis, instruction_following, language, math, reasoning -
with its OWN ground-truth judges (no LLM judge), including real code EXECUTION for the coding
category. Its package + coding-execution deps (tensorflow/numba/opencv/...) would churn a
torch/vLLM serving venv, so gbench runs LiveBench's pipeline inside an isolated Docker image
(`gbench-livebench`, built from docker/livebench.Dockerfile): it invokes LiveBench's
`run_livebench.py` (inference via --api-base against the gbench-served model, then grading)
and reads the resulting `all_groups.csv` (per-category + `average`).

The `agentic_coding` 7th category is excluded (LiveBench's default benchmark set already
excludes it; it needs a separate ~150GB Multi-SWE-Bench container harness).

This suite hard-errors (never silently skips) if Docker or the `gbench-livebench` image is
absent. Because it shells out to LiveBench's own scorers, it does NOT use run_eval_suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with `--temperature`, or for this suite
alone with `GBENCH_LIVEBENCH_TEMPERATURE`, which takes precedence over both. LLM-judge grading
is pinned at 0.0 and is not affected by either (LiveBench uses no LLM judge).
"""

import csv
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from .base import resolve_temperature
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "General Knowledge & Reasoning"
DOCS_URL = "docs/evals/livebench.md"

# The 6 core categories (LiveBench run_livebench.py DEFAULT_BENCHMARKS; agentic_coding excluded).
_CATEGORIES = ["coding", "data_analysis", "instruction_following", "language", "math", "reasoning"]
# Approx number of sub-tasks across the 6 core categories at a recent release (coding 2 +
# data_analysis 3 + instruction_following 4 + language 3 + math 3 + reasoning 3). Used only to
# spread --eval-limit across tasks (LiveBench's --question-end is PER TASK); the real count varies
# by release, so this is a smoke estimate, not a correctness input.
_EST_CORE_TASKS = 18
_IMAGE_DEFAULT = "gbench-livebench"


def _image() -> str:
    return os.environ.get("GBENCH_LIVEBENCH_IMAGE", _IMAGE_DEFAULT)


def _categories() -> List[str]:
    """The 6 core categories, or a subset via GBENCH_LIVEBENCH_CATEGORIES (space/comma list)."""
    raw = (os.environ.get("GBENCH_LIVEBENCH_CATEGORIES") or "").replace(",", " ").split()
    return raw or list(_CATEGORIES)


def _check_prereqs(image: str) -> None:
    build = (f"Build it (context = your LiveBench checkout with LFS *.json pulled):\n"
             f"  docker build -t {image} -f docker/livebench.Dockerfile "
             f"/path/to/LiveBench")
    if not shutil.which("docker"):
        raise infra_required("livebench", "docker CLI not found. " + build, DOCS_URL)
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise infra_required("livebench", "docker daemon not reachable. " + build, DOCS_URL)
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        raise infra_required("livebench", f"image {image!r} not found. " + build, DOCS_URL)


def _build_command(image: str, outdir: str, model_id: str, display: str, base_url: str,
                   concurrency: int, temperature: float, release: str, extra: str) -> List[str]:
    """The `docker run` that runs LiveBench's own inference+grading+report inside the image."""
    # LiveBench's per-category bench-name path truncates a category at the first
    # underscore (common.py: split_bench_name[1].split('_')[0]), so "live_bench/data_analysis"
    # resolves to the non-existent HF dataset "livebench/data" and the whole run crashes
    # ("No objects to concatenate"). The whole-suite name "live_bench" uses the correct
    # code path (it iterates LIVE_BENCH_CATEGORIES by full name), so use it whenever the
    # full default set is requested. A subset that includes a multi-word category can't be
    # isolated by bench-name upstream; warn rather than silently emit a broken name.
    cats = _categories()
    if set(cats) == set(_CATEGORIES):
        benches = "live_bench"
    else:
        multiword = [c for c in cats if "_" in c]
        if multiword:
            logger.warning(
                "livebench: categories %s contain an underscore; LiveBench's per-category "
                "loader truncates them to a non-existent HF dataset (e.g. data_analysis -> "
                "livebench/data). Run the full default set (uses the whole-suite path) or "
                "select only single-word categories.", multiword)
        benches = " ".join(f"live_bench/{c}" for c in cats)
    rel = f"--livebench-release-option {release} " if release else ""
    # --mode single runs inference+grading in one process (no tmux, unlike sequential/parallel).
    # --ignore-missing-judgments lets show_livebench_result score whatever categories completed
    # instead of dropping the model for lacking every task (a full run has them all anyway).
    inner = (
        "python run_livebench.py "
        f"--model {model_id} --model-display-name {display} "
        f"--api-base {base_url} --api-key EMPTY --use-litellm "
        f"--bench-name {benches} --question-source huggingface --mode single "
        f"--parallel-requests {concurrency} --parallel-grading {concurrency} "
        f"--force-temperature {temperature} {rel}{extra} "
        f"&& python show_livebench_result.py --model-list {display} --bench-name live_bench "
        f"--ignore-missing-judgments {rel}"
        "&& cp -f all_groups.csv all_tasks.csv /out/ 2>/dev/null || true"
    )
    return ["docker", "run", "--rm", "--network", "host",
            "-v", f"{outdir}:/out", image, "bash", "-lc", inner]


def _parse_all_groups(path: str, display: str) -> Dict[str, float]:
    """Read LiveBench's all_groups.csv (index=model, columns=categories + 'average')."""
    scores: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        cols = header[1:]  # first col is the (unnamed) model index
        for row in reader:
            if not row:
                continue
            # LiveBench lowercases the model slug in its outputs -> match case-insensitively.
            row_slug = re.sub(r"[^A-Za-z0-9._-]", "-", row[0]).lower()
            if row[0].strip().lower() == display.lower() or row_slug == display.lower():
                for name, val in zip(cols, row[1:]):
                    try:
                        scores[name] = float(val)
                    except (TypeError, ValueError):
                        pass
                break
    return scores


def run_livebench(model_name: str, base_url: str, concurrency: int,
                  enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run LiveBench's 6 core categories inside the gbench-livebench image; report the average."""
    image = _image()
    _check_prereqs(image)

    model_id = os.environ.get("GBENCH_LIVEBENCH_MODEL") or model_name
    display = re.sub(r"[^A-Za-z0-9._-]", "-", model_id).lower()  # LiveBench lowercases the slug
    release = os.environ.get("GBENCH_LIVEBENCH_RELEASE", "").strip()
    extra = os.environ.get("GBENCH_LIVEBENCH_EXTRA_ARGS", "").strip()
    # Honor --eval-limit by subsetting each task (LiveBench's --question-end), unless the
    # operator already set an explicit range in EXTRA_ARGS.
    limit = kwargs.get("limit")
    if limit and "--question-end" not in extra and "--question-id" not in extra:
        # --question-end is PER TASK, and the core suite has ~_EST_CORE_TASKS tasks across the 6
        # categories, so a raw --question-end=limit runs limit x tasks. Distribute --eval-limit
        # across the tasks (round) so the total stays ~limit, floored at 1/task (LiveBench has no
        # cap below one question per task, so ~n_tasks is the practical minimum for a full-suite run).
        _per_task = max(1, round(int(limit) / _EST_CORE_TASKS))
        extra = (extra + f" --question-end {_per_task}").strip()
    temperature, _src = resolve_temperature("livebench", kwargs.get("temperature"), thinking=enable_thinking)

    outdir = tempfile.mkdtemp(prefix="gbench_livebench_")
    cmd = _build_command(image, outdir, model_id, display, base_url, concurrency,
                         temperature, release, extra)
    logger.info("livebench: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    groups = os.path.join(outdir, "all_groups.csv")
    if not os.path.isfile(groups):
        raise RuntimeError(
            "livebench: LiveBench produced no all_groups.csv (inference/grading failed) - this "
            "is a harness/infra failure, not a 0%. Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1200:])

    cats = _categories()
    scores = _parse_all_groups(groups, display)
    dims = {c: scores[c] for c in cats if c in scores}
    if not dims:
        raise RuntimeError(
            "livebench: no categories were scored for release "
            f"'{release or 'latest'}' - nothing to report (not a 0%). LiveBench is a moving "
            "benchmark: the non-agentic `coding` category is DEPRECATED after 2025-04-02, so at "
            "the latest release it (and possibly others) have no active questions. Pin "
            "GBENCH_LIVEBENCH_RELEASE to a release where your categories are active (e.g. "
            "2025-04-02 for all 6 core categories), or narrow GBENCH_LIVEBENCH_CATEGORIES.")

    overall = scores.get("average")
    if overall is None:  # single active category: no 'average' column -> use it directly
        overall = round(sum(dims.values()) / len(dims), 2)

    missing = [c for c in cats if c not in dims]  # e.g. coding at a post-2025-04-02 release
    default_cats = set(cats) == set(_CATEGORIES)
    return {
        "benchmark_type": "eval",
        "eval_name": "livebench",
        "model_name": model_name,
        "status": "success" if not missing else "completed_with_errors",
        "accuracy": round(float(overall), 2),
        "total_questions": len(dims),      # categories scored; per-task detail is in LiveBench's CSVs
        "correct_answers": 0,
        "dimension_scores": dims,
        "livebench_release": release or "latest",
        "categories_missing": missing,     # active-category set is release-dependent
        "metric": ("LiveBench average over the active core categories for the release (equal "
                   "weight), each scored by LiveBench's own ground-truth judges (no LLM judge); "
                   "agentic_coding excluded"),
        # Comparable to LiveBench's leaderboard when it's the FULL question set for a PINNED,
        # reproducible release with all default categories active. NOT gated on thinking: the
        # LiveBench leaderboard includes reasoning/thinking models, so a --thinking run is still
        # a faithful measurement (of the thinking model).
        "leaderboard_comparable": bool(release and default_cats and not missing
                                       and not limit),
    }
