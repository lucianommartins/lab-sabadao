# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: frames
# Description: Google FRAMES (Factuality, Retrieval, And Multi-hop Evaluation Suite - 824 questions)

"""gbench native built-in runner for frames (Factuality & RAG).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_FRAMES_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict, judge_config, judge_generate_cascade
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Factuality & RAG"


def _load_frames_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load FRAMES from HF Hub (google/frames-benchmark); raises on load/schema failure (no fabricated fallback)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('google/frames-benchmark', split='test')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for frames: {e}")
        raise RuntimeError(f"Could not load dataset for frames: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for frames returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("reasoning_types"), seed="frames")

    samples = []
    for item in rows:
        # google/frames-benchmark columns are capitalised: 'Prompt' / 'Answer'.
        prompt = item.get("Prompt")
        gold = item.get("Answer")
        if not prompt or gold is None:
            raise RuntimeError(
                "frames: unexpected dataset schema (missing 'Prompt'/'Answer'); "
                "refusing to fabricate sample data"
            )
        cat = str(item.get("reasoning_types") or "multi_hop")
        messages = [{"role": "user", "content": str(prompt).strip()}]
        samples.append((messages, str(gold).strip(), {"category": cat}))

    logger.info(f"Loaded {len(samples)} frames samples.")
    return samples


def _eval_frames(response_text: str, gold_target: str) -> bool:
    """Fallback local evaluation if GEMINI_API_KEY is not available (unit tests/offline)."""
    if not response_text:
        return False
    resp = response_text.strip()
    gold = str(gold_target).strip()
    if not gold:
        return False
    if gold.lower() in resp.lower():
        return True
    return False


async def _async_judge_frames(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical Google FRAMES 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local fallback for FRAMES.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_frames(cleaned, gold)
            trace["scoring_mode"] = "judge_fallback"
            trace["status"] = "OK"
        return

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        if not resp_text:
            trace["is_correct"] = False
            trace["judge_grade"] = "FAILED"
            pbar.update(1)
            return

        messages = trace.get("messages", [])
        question = messages[0].get("content", "") if messages else ""
        gold = str(trace.get("gold_answer") or "")

        # Canonical FRAMES grader rubric (paper appendix): decide whether the ground truth
        # answer is PRESENT in the prediction, judging substance over exact wording.
        prompt = (
            "===Task===\n"
            "I need your help evaluating an answer provided by an LLM against a ground truth "
            "answer. Determine if the ground truth answer is present in the LLM's response.\n\n"
            "===Instructions===\n"
            "1. Carefully compare the Predicted Answer with the Ground Truth Answer.\n"
            "2. Consider the substance of the answers - look for equivalent information or a "
            "correct answer. Do not focus on exact wording unless the exact wording is crucial "
            "to the meaning.\n"
            "3. Base your decision on whether the meaning and the vital facts of the Ground "
            "Truth Answer are present in the Predicted Answer.\n\n"
            "===Input Data===\n"
            f"- Question: {question}\n"
            f"- Predicted Answer: {resp_text}\n"
            f"- Ground Truth Answer: {gold}\n\n"
            "===Output Format===\n"
            "Grade: CORRECT (the ground truth is present) / INCORRECT (it is not)"
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

        grade_str = text.strip().upper()
        is_corr = parse_grade_verdict(grade_str)
        trace["is_correct"] = is_corr
        trace["judge_grade"] = grade_str
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [FRAMES]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_frames(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute frames native built-in evaluation suite."""
    skip = gemini_required_skip("frames", model_name)
    if skip is not None:
        return skip
    samples = _load_frames_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="frames",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_frames,
        async_eval_fn=_async_judge_frames,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
