# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: mmlu_redux
# Description: MMLU-Redux (Cleaned-Label Multi-Subject Academic Benchmark across 57 subjects)

"""gbench native built-in runner for mmlu_redux (General Knowledge & Reasoning).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MMLU_REDUX_TEMPERATURE`,
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

logger = logging.getLogger(__name__)

PILLAR = "General Knowledge & Reasoning"

OPTION_LETTERS = ["A", "B", "C", "D"]

# The 30 subject configs that make up edinburgh-dawg/mmlu-redux (100 rows each,
# 3,000 total). The canonical Redux set is all subjects, not a single one.
MMLU_REDUX_SUBJECTS = [
    "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "conceptual_physics", "econometrics",
    "electrical_engineering", "formal_logic", "global_facts",
    "high_school_chemistry", "high_school_geography",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "high_school_us_history",
    "human_aging", "logical_fallacies", "machine_learning", "miscellaneous",
    "philosophy", "professional_accounting", "professional_law",
    "public_relations", "virology",
]


def _load_mmlu_redux_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MMLU-Redux (all 30 subjects) from HF Hub (edinburgh-dawg/mmlu-redux).

    Columns: 'question' (str), 'choices' (list[str] of 4), 'answer' (int 0-3).
    Raises on load/schema failure (no fabricated fallback).
    """
    try:
        from datasets import load_dataset
        subject_rows = []
        for subject in MMLU_REDUX_SUBJECTS:
            ds = load_dataset("edinburgh-dawg/mmlu-redux", subject, split="test")
            for item in ds:
                subject_rows.append((subject, item))
    except Exception as e:
        logger.error(f"Failed to load dataset for mmlu_redux: {e}")
        raise RuntimeError(f"Could not load dataset for mmlu_redux: {e}") from e

    if not subject_rows:
        raise RuntimeError("Dataset for mmlu_redux returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    subject_rows = stratified_sample(subject_rows, limit, lambda r: r[0], seed="mmlu_redux")

    samples = []
    dropped = 0
    for subject, item in subject_rows:
        question = item.get("question")
        choices = item.get("choices")
        answer_idx = item.get("answer")

        # MMLU-Redux exists to CORRECT MMLU's labels; using the original `answer` for every
        # row scores against the errors it documents. Honour the annotations:
        #   * unanswerable/ambiguous rows are dropped (no valid key exists);
        #   * wrong-groundtruth rows use the corrected answer.
        err = str(item.get("error_type") or "").strip().lower().replace(" ", "_")
        if err in ("no_correct_answer", "bad_question_clarity", "multiple_correct_answers"):
            dropped += 1
            continue
        if err in ("wrong_groundtruth", "expert"):
            corrected = item.get("correct_answer")
            if corrected is not None and str(corrected).strip() != "":
                c = str(corrected).strip()
                if c.isdigit():
                    answer_idx = int(c)
                elif len(c) == 1 and c.upper() in OPTION_LETTERS:
                    answer_idx = OPTION_LETTERS.index(c.upper())
                elif isinstance(choices, list) and c in choices:
                    answer_idx = choices.index(c)
        if (not question or not isinstance(choices, list) or len(choices) < 2
                or not isinstance(answer_idx, int) or not 0 <= answer_idx < len(choices)):
            raise RuntimeError(
                "mmlu_redux: unexpected dataset schema "
                "(need 'question' str, 'choices' list, 'answer' int index); "
                "refusing to fabricate sample data"
            )
        gold = OPTION_LETTERS[answer_idx]
        options = "\n".join(
            f"({OPTION_LETTERS[i]}) {choice}" for i, choice in enumerate(choices)
        )
        prompt = (
            f"{question}\n{options}\n\n"
            "Conclude with 'The answer is (X)'."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": subject}))

    logger.info(f"Loaded {len(samples)} mmlu_redux samples "
                f"(dropped {dropped} rows Redux marks unanswerable/ambiguous).")
    return samples


def _eval_mmlu_redux(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against gold target using 100% deterministic letter matching."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip().upper()

    if not gold:
        return False

    # 1. Boxed letter match
    match_boxed = re.findall(r"\\boxed\{([A-D])\}", resp, re.IGNORECASE)
    if match_boxed and match_boxed[-1].upper() == gold:
        return True

    # 2. Final Answer match
    match_fa = re.search(r"(?:Final Answer|Answer|Choice)\s*:\s*(?:Option\s*)?([A-D])\b", resp, re.IGNORECASE)
    if match_fa and match_fa.group(1).upper() == gold:
        return True

    # 3. Last stand-alone letter
    letters = re.findall(r"\b([A-D])\b", resp)
    if letters and letters[-1].upper() == gold:
        return True

    return False


def run_mmlu_redux(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute mmlu_redux native built-in evaluation suite."""
    samples = _load_mmlu_redux_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="mmlu_redux",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_mmlu_redux,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
