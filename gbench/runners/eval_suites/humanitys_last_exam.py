# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: humanitys_last_exam
# Description: Humanity's Last Exam (HLE - 2,500 frontier expert-level STEM & scientific reasoning questions)

"""gbench native built-in runner for humanitys_last_exam (STEM & Scientific Reasoning).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_HUMANITYS_LAST_EXAM_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict, judge_generate_cascade
from .sampling import stratified_sample
from .dataset_utils import build_image_message

logger = logging.getLogger(__name__)

PILLAR = "STEM & Scientific Reasoning"

# HLE asks the model to state a confidence so a Calibration Error can be reported alongside
# accuracy (both are canonical HLE headline metrics).
_CONFIDENCE_INSTRUCTION = (
    "\n\nAfter your answer, state your confidence that it is correct as a percentage on its "
    "own line, exactly: 'Confidence: X%' (0-100)."
)


def _load_humanitys_last_exam_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load humanitys_last_exam benchmark dataset directly from HF Hub (cais/hle)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('cais/hle', split='test')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for humanitys_last_exam: {e}")
        raise RuntimeError(f"Could not load dataset for humanitys_last_exam: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for humanitys_last_exam returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("category"), seed="humanitys_last_exam")

    samples = []
    for item in rows:
        prompt = item.get("question") or item.get("prompt")
        gold = item.get("answer") or item.get("target")
        if not prompt or gold is None:
            raise RuntimeError(
                "humanitys_last_exam: unexpected dataset schema (missing 'question'/'answer'); "
                "refusing to fabricate sample data"
            )
        gold = str(gold).strip()
        cat = item.get("category", "Mathematics")

        # cais/hle carries an `image` data-URI for a ~14% multimodal subset; it was never
        # read, so those questions were silently evaluated text-only (unanswerable). Attach it.
        messages, _attached = build_image_message(str(prompt) + _CONFIDENCE_INSTRUCTION,
                                                  item.get("image"))
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} humanitys_last_exam samples.")
    return samples


def _parse_confidence(text: str) -> Optional[float]:
    """The model's stated confidence in [0, 1], or None if it gave none."""
    m = re.findall(r"confidence\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?", text or "", re.IGNORECASE)
    if not m:
        return None
    c = float(m[-1])
    c = c / 100.0 if c > 1.0 else c
    return max(0.0, min(1.0, c))


def _calibration_error(result: Dict[str, Any]) -> Tuple[Optional[float], int]:
    """RMS calibration error over 10 confidence bins (canonical HLE calibration metric)."""
    import math
    from .metrics import scorable_traces
    pairs = []
    for t in scorable_traces(result):
        conf = _parse_confidence(t.get("response_text"))
        if conf is None:
            continue
        pairs.append((conf, 1.0 if t.get("is_correct") else 0.0))
    if not pairs:
        return None, 0
    bins: List[List[Tuple[float, float]]] = [[] for _ in range(10)]
    for conf, corr in pairs:
        bins[min(9, int(conf * 10))].append((conf, corr))
    n = len(pairs)
    sq = 0.0
    for b in bins:
        if not b:
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        acc = sum(a for _, a in b) / len(b)
        sq += (len(b) / n) * (acc - mean_conf) ** 2
    return math.sqrt(sq), n


def _eval_humanitys_last_exam(response_text: str, gold_target: str) -> bool:
    """Fast deterministic evaluation for multiple choice and closed-form math in HLE."""
    if not response_text:
        return False
    resp = response_text.strip()
    gold = str(gold_target).strip()
    if not gold:
        return False
    def _norm(s):
        return re.sub(r"[\s$,]|\\left|\\right|\\!|\\,", "", str(s)).strip().rstrip(".").lower()

    # CC4: deterministic pass only on an EXTRACTED final answer that equals the gold;
    # bare containment used to credit any response mentioning the gold in passing, before
    # the judge ever ran.
    match_boxed = re.findall(r"\\boxed\{([^\}]+)\}", resp)
    if match_boxed and _norm(match_boxed[-1]) == _norm(gold):
        return True
    match_fa = re.search(r"(?:Final Answer|Answer)\s*:\s*(.+)", resp, re.IGNORECASE)
    if match_fa and _norm(match_fa.group(1).split("\n")[0]) == _norm(gold):
        return True
    if _norm(resp) == _norm(gold):
        return True
    return False


async def _async_judge_humanitys_last_exam(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical CAIS Humanity's Last Exam 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local fallback for HLE.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_humanitys_last_exam(cleaned, gold)
            trace["status"] = "OK"
        return

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        gold = str(trace.get("gold_answer") or "").strip()

        # Fast deterministic pass
        if _eval_humanitys_last_exam(str(resp_text or ""), gold):
            trace["is_correct"] = True
            trace["judge_grade"] = "CORRECT"
            trace["status"] = "OK"
            pbar.update(1)
            return

        if not resp_text:
            trace["is_correct"] = False
            trace["judge_grade"] = "FAILED"
            pbar.update(1)
            return

        messages = trace.get("messages", [])
        question = messages[0].get("content", "") if messages else ""

        prompt = (
            "You are an expert evaluator for Humanity's Last Exam (HLE).\n"
            "Evaluate whether the candidate response correctly and accurately solves the expert question matching the gold target answer.\n\n"
            f"Question:\n{question}\n\n"
            f"Gold Target Answer:\n{gold}\n\n"
            f"Candidate Model Response:\n{resp_text}\n\n"
            "Output exactly:\n"
            "Grade: CORRECT / INCORRECT"
        )

        async with semaphore:
            text, _judge_used = await judge_generate_cascade(prompt)
        if text is None:
            # Judge cascade exhausted (infra outage): excluded from accuracy,
            # NOT scored INCORRECT. base.run_eval_suite drops it from pass/fail.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            trace["status"] = "OK"
            pbar.update(1)
            return

        grade_str = (text or "").strip().upper()
        is_corr = parse_grade_verdict(grade_str)
        trace["is_correct"] = is_corr
        trace["judge_grade"] = grade_str
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [HLE]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_humanitys_last_exam(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute humanitys_last_exam native built-in evaluation suite."""
    skip = gemini_required_skip("humanitys_last_exam", model_name)
    if skip is not None:
        return skip
    samples = _load_humanitys_last_exam_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="humanitys_last_exam",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_humanitys_last_exam,
        async_eval_fn=_async_judge_humanitys_last_exam,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
    # Accuracy stays the headline; HLE also reports Calibration Error (RMS over confidence bins).
    cerr, n_conf = _calibration_error(result)
    if cerr is not None:
        result["calibration_error"] = round(cerr * 100.0, 2)
        result["calibration_samples"] = n_conf
    result["metric"] = "judge accuracy (headline) + calibration_error (RMS, canonical HLE)"
    return result
