# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: swe_bench_multilingual
# Description: SWE-bench Multilingual (300 issues across 9 non-Python languages, resolved rate)

"""gbench native built-in runner for swe_bench_multilingual (Coding & Software Engineering).

Canonical SWE-bench Multilingual scored by execution-based resolved-rate via the
VANILLA swebench Docker harness (namespace 'swebench'; per-language tests run inside
the prebuilt instance images, so the host toolchains are irrelevant). Thin wrapper
over swebench_common; SANDBOX_EVAL. Hard-errors (infra_required, never skips) if
swebench/Docker/dataset are absent.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

from typing import Any, Dict, Optional
from .swebench_common import (
    check_swebench_prereq, load_swe_samples, execute_swebench, infra_required,
)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_bench_multilingual.md"
_DATASET = "SWE-bench/SWE-bench_Multilingual"
_SPLIT = "test"
_NAMESPACE = "swebench"


def check_swe_bench_multilingual_prerequisites():
    return check_swebench_prereq(_DATASET, _SPLIT, _NAMESPACE, requires_fork=False)


def _load_swe_bench_multilingual_samples(limit: Optional[int] = None, split: str = _SPLIT):
    return load_swe_samples(_DATASET, split, limit, "swe_bench_multilingual")


def run_swe_bench_multilingual(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench Multilingual (or skip if swebench/Docker are unavailable)."""
    ok, reason = check_swe_bench_multilingual_prerequisites()
    if not ok:
        # No-skip policy: hard-error when swebench/Docker/dataset are unavailable (matches the
        # sibling copilot_bench_swe / swe_bench_live suites), never a fabricated skip row.
        raise infra_required("swe_bench_multilingual", reason, DOCS_URL)
    kwargs["enable_thinking"] = enable_thinking
    return execute_swebench(
        "swe_bench_multilingual", model_name, base_url, concurrency,
        _DATASET, kwargs.get("split", _SPLIT), _NAMESPACE, **kwargs,
    )
