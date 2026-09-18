# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: hmmt
# Description: HMMT (Harvard-MIT Mathematics Tournament Competition Math Benchmark)

"""gbench native built-in runner for hmmt (STEM & Scientific Reasoning).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_HMMT_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample
from .extraction_common import anchored_span
from .math_equiv import math_equivalent, last_boxed

logger = logging.getLogger(__name__)

PILLAR = "STEM & Scientific Reasoning"

# HMMT answers are exact competition values (integers, fractions, closed forms). The
# canonical MathArena protocol tells the model where to put the final answer so the grader
# can extract it deterministically; without it the model's answer is buried in prose and
# the boxed-extraction path never fires.
_ANSWER_INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def _load_hmmt_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load hmmt benchmark dataset directly from HF Hub (matharena/hmmt)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('MathArena/hmmt_feb_2025', split='train')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for hmmt: {e}")
        raise RuntimeError(f"Could not load dataset for hmmt: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for hmmt returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("category"), seed="hmmt")

    samples = []
    for item in rows:
        prompt = item.get("problem")
        gold = item.get("answer")
        if not prompt or gold is None:
            raise RuntimeError(
                "hmmt: unexpected dataset schema (missing 'problem'/'answer'); "
                "refusing to fabricate sample data"
            )
        prompt = str(prompt)
        gold = str(gold).strip()
        cat = item.get("category", "number_theory")

        messages = [{"role": "user", "content": prompt + _ANSWER_INSTRUCTION}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} hmmt samples.")
    return samples


def _extract_hmmt_answer(response_text: str) -> Optional[str]:
    """The model's stated answer: the last ``\\boxed{}`` else the last explicit anchor span.

    Extraction is ANCHORED, never "any number in the text", so an incidental figure in the
    reasoning ("we test 5 cases ... the answer is 12") cannot be credited as the answer.
    """
    box = last_boxed(response_text)
    if box is not None:
        return box.strip()
    span = anchored_span(response_text)
    if span:
        return span.strip().strip("$").strip().rstrip(".").strip()
    return None


def _eval_hmmt(response_text: str, gold_target: str) -> bool:
    """Grade a response by canonical symbolic/numeric equivalence to the gold answer.

    HMMT answers are exact competition values (integers, fractions, radicals, closed forms).
    The previous string/single-number matcher scored ``0.5`` != ``1/2`` and
    ``\\frac{\\sqrt2}{2}`` != ``\\frac{1}{\\sqrt2}`` as wrong. We extract the model's anchored
    answer, then compare with math_verify (+ a sympy fallback) - the MathArena/leaderboard
    grader. run_hmmt has already guaranteed the backend is present (require_backend),
    so this never silently downgrades.
    """
    if not response_text or not str(gold_target).strip():
        return False
    candidate = _extract_hmmt_answer(response_text)
    if candidate is None:
        return False
    return math_equivalent(candidate, gold_target)


def run_hmmt(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute hmmt native built-in evaluation suite."""
    from .math_equiv import require_backend
    require_backend("hmmt")   # hard-errors (infra_required) if math_verify is absent
    samples = _load_hmmt_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="hmmt",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_hmmt,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
    )
