# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: copilot_bench_swe
# Description: SWE-bench Verified (resolved rate via the vanilla swebench Docker harness)

"""gbench native built-in runner for copilot_bench_swe (Coding & Software Engineering).

Autonomous GitHub-issue resolution on SWE-bench Verified, scored by execution-based
resolved-rate via the VANILLA swebench Docker harness (namespace 'swebench'). Thin
wrapper over swebench_common; SANDBOX_EVAL. Skips cleanly if swebench/Docker absent.
(The '256k long-context' framing is a prompt-only extension; canonical resolved-rate
is prompt-agnostic, so the plain issue prompt is used.)

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_COPILOT_BENCH_SWE_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

from typing import Any, Dict, Optional
from .swebench_common import (
    check_swebench_prereq, load_swe_samples, execute_swebench, infra_required,
)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/copilot_bench_swe.md"
_DATASET = "princeton-nlp/SWE-bench_Verified"
_SPLIT = "test"
_NAMESPACE = "swebench"


def check_copilot_bench_swe_prerequisites():
    return check_swebench_prereq(_DATASET, _SPLIT, _NAMESPACE, requires_fork=False)


def _load_copilot_bench_swe_dataset(limit: Optional[int] = None, split: str = _SPLIT):
    return load_swe_samples(_DATASET, split, limit, "copilot_bench_swe")


def run_copilot_bench_swe(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench Verified (or skip if swebench/Docker are unavailable)."""
    ok, reason = check_copilot_bench_swe_prerequisites()
    if not ok:
        raise infra_required("copilot_bench_swe", reason, DOCS_URL)
    kwargs["enable_thinking"] = enable_thinking
    return execute_swebench(
        "copilot_bench_swe", model_name, base_url, concurrency,
        _DATASET, kwargs.get("split", _SPLIT), _NAMESPACE, **kwargs,
    )
