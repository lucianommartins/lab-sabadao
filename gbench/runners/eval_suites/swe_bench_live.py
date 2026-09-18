# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: swe_bench_live
# Description: SWE-bench-Live - continuously-updated real GitHub issue resolution (resolved rate)

"""gbench native built-in runner for swe_bench_live (Coding & Software Engineering).

Canonical SWE-bench-Live scored by the SWE-bench-Live fork of the swebench Docker harness
(per-instance DockerHub images under namespace 'starryzhang'), resolved rate.

The fork is the `swebench` PACKAGE at a version incompatible with the upstream `swebench` the main
env uses (swe_bench_multilingual / copilot_bench_swe), and only one can be installed at a time. So
gbench ISOLATES the fork in a LOCAL image (gbench/docker/swe_bench_live.Dockerfile) instead of a
separate host virtualenv: the model rollout (patch generation) runs in the gbench process against
the served endpoint (it needs only `datasets`, not `swebench`), and only the scoring step
(`swebench.harness.run_evaluation`) runs inside the fork image, docker-out-of-docker (it spawns the
per-instance task containers on the host daemon). HARD-ERRORS (infra_required, never skips) if
Docker or the locally-built fork image is absent.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SWE_BENCH_LIVE_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import os
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

from .swebench_common import load_swe_samples, execute_swebench, infra_required

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_bench_live.md"
_DATASET = "SWE-bench-Live/SWE-bench-Live"
_NAMESPACE = "starryzhang"
_DEFAULT_SPLIT = "lite"
_IMAGE_DEFAULT = "gbench-swe-bench-live"


def _image() -> str:
    return os.environ.get("GBENCH_SWE_BENCH_LIVE_IMAGE", _IMAGE_DEFAULT)


def check_swe_bench_live_prerequisites() -> Tuple[bool, str]:
    """datasets (rollout) + Docker + the LOCAL fork image (the fork lives in the image, not the host
    env, so the main env keeps upstream swebench for swe_bench_multilingual)."""
    build = (f"Build the fork image LOCALLY (gbench never pulls):\n"
             f"  docker build -t {_image()} -f gbench/docker/swe_bench_live.Dockerfile gbench/docker\n"
             f"It isolates the SWE-bench-Live fork of `swebench` from the main env's upstream "
             f"`swebench`. See " + DOCS_URL)
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed (pip install gbench[evals])."
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + build
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + build
    if subprocess.run(["docker", "image", "inspect", _image()],
                      capture_output=True).returncode != 0:
        return False, f"the swe_bench_live fork image {_image()!r} is not built. " + build
    return True, ""


def _load_swe_bench_live_samples(limit: Optional[int] = None, split: str = _DEFAULT_SPLIT):
    return load_swe_samples(_DATASET, split, limit, "swe_bench_live")


def run_swe_bench_live(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench-Live: rollout in-process, score inside the isolated fork image."""
    ok, reason = check_swe_bench_live_prerequisites()
    if not ok:
        raise infra_required("swe_bench_live", reason, DOCS_URL)
    kwargs["enable_thinking"] = enable_thinking
    kwargs["harness_image"] = _image()
    return execute_swebench(
        "swe_bench_live", model_name, base_url, concurrency,
        _DATASET, kwargs.get("split", _DEFAULT_SPLIT), _NAMESPACE, **kwargs,
    )
