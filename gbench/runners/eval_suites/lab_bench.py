# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: lab_bench
# Description: LAB-Bench ProtocolQA (Language Agent Biology Benchmark for Wet-Lab Protocols)

"""gbench native built-in runner for lab_bench (Reasoning & Knowledge).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_LAB_BENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import csv
import hashlib
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Reasoning & Knowledge"


def _build_mcq_prompt(protocol: str, question: str, ideal: str,
                      distractors: List[str]) -> Tuple[str, str]:
    """Build the multiple-choice prompt and its gold letter.

    The concluding instruction names the ANSWER FORMAT only; it must never contain
    the gold letter itself (that would leak the answer into the prompt and invalidate
    the score). Options are shuffled DETERMINISTICALLY per item (seeded on the question),
    not sorted: a plain alphabetical sort puts the correct answer in a position determined
    by its own text (distractors are often shorter), a positional bias gpqa also fixed.
    """
    options = [ideal] + list(distractors)
    order = list(range(len(options)))
    random.Random(hashlib.sha256(question.encode("utf-8")).hexdigest()).shuffle(order)
    options_shuffled = [options[i] for i in order]
    choice_letters = [chr(65 + i) for i in range(len(options_shuffled))]
    choice_str = "\n".join(f"{choice_letters[i]}. {opt}" for i, opt in enumerate(options_shuffled))
    gold_idx = options_shuffled.index(ideal) if ideal in options_shuffled else 0
    gold_letter = choice_letters[gold_idx] if gold_idx < len(choice_letters) else "A"
    prompt = (
        f"[Wet-Lab Protocol Context]\n{protocol}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choice_str}\n\n"
        f"Conclude with a single line 'Final Answer: X', where X is the letter "
        f"(A, B, C, ...) of the correct choice."
    )
    return prompt, gold_letter


def _load_lab_bench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load LAB-Bench benchmark dataset directly from HF Hub (futurehouse/lab-bench)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("futurehouse/lab-bench", "ProtocolQA", split="train")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for lab_bench: {e}")
        raise RuntimeError(f"Could not load dataset for lab_bench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for lab_bench returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("subtask"), seed="lab_bench")

    samples = []
    for item in rows:
        protocol = item.get("protocol", "").strip()
        question = item.get("question", "").strip()
        ideal = item.get("ideal", "").strip()
        distractors = item.get("distractors", [])
        subtask = item.get("subtask", "protocolqa")

        prompt, gold_letter = _build_mcq_prompt(protocol, question, ideal, distractors)
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_letter, {"category": subtask, "ideal": ideal}))

    logger.info(f"Loaded {len(samples)} lab_bench samples.")
    return samples


def _eval_lab_bench(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against correct multiple-choice letter."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip().upper()

    # Tolerate wrappers the model puts around the letter on the "Final Answer" line,
    # e.g. "Final Answer: <C>", "**C**", "(C)", '"C"' - skip non-letters up to the
    # first letter on that line (bounded by newline so it can't run into the body).
    match_fa = re.search(r"Final Answer:\s*[^A-Za-z\n]*([A-Za-z])", resp, re.IGNORECASE)
    if match_fa and match_fa.group(1).upper() == gold:
        return True

    match_boxed = re.search(r"\\boxed\{([A-Za-z])\}", resp, re.IGNORECASE)
    if match_boxed and match_boxed.group(1).upper() == gold:
        return True

    lines = [line.strip() for line in resp.split("\n") if line.strip()]
    if lines:
        last_line = lines[-1]
        m = re.search(r"(?:answer|choice|option)\s*(?:is|:)?\s*([A-Za-z])\b", last_line, re.IGNORECASE)
        if m and m.group(1).upper() == gold:
            return True

    return f" {gold} " in f" {resp} " or f"({gold})" in resp or f"**{gold}**" in resp


def run_lab_bench(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native LAB_BENCH evaluation benchmark."""
    samples = _load_lab_bench_samples(limit=limit)
    result = run_eval_suite(
        eval_name="lab_bench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_lab_bench,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
    # gbench runs only the ProtocolQA task (1 of ~8 LAB-Bench tasks), so the headline is a task
    # subscore, not the full LAB-Bench number.
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "ProtocolQA task only (1 of ~8 LAB-Bench tasks); a task subscore, not the full benchmark")
    return result
