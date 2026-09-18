# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: simpleqa
# Description: OpenAI SimpleQA (Factuality & Hallucination Benchmark - 4,326 short questions)

"""gbench native built-in runner for simpleqa (Factuality & Knowledge).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SIMPLEQA_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import ast
import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL, parse_grade_verdict, judge_generate_cascade
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Factuality & Knowledge"

# Canonical SimpleQA grader rubric (OpenAI simple-evals). The distinguishing rules are:
# CORRECT = fully contains the gold with no contradiction; INCORRECT = contradicts the gold;
# NOT_ATTEMPTED = hedges / gives no definitive answer and does not contradict the gold.
_GRADER_TEMPLATE = """You are grading a predicted answer against a gold target for a factual question. Assign exactly one grade.

Rules:
- CORRECT: the predicted answer fully contains the gold target with no contradicting statement. Hedging ("possibly X") is fine if X is stated and correct. Minor formatting/spelling/paraphrase differences are fine. For a numeric gold, the number must match to the gold's precision.
- INCORRECT: the predicted answer contains any statement that contradicts the gold target (a wrong name, date, number, etc.). A wrong answer is INCORRECT even if hedged.
- NOT_ATTEMPTED: the gold target is not given and is not contradicted - the model refused, said it does not know, asked for clarification, or gave only related-but-incomplete information.

Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}

Respond in this format:
Grade: CORRECT / INCORRECT / NOT_ATTEMPTED"""


def _classify_grade(grade_text: str) -> Optional[str]:
    """The three-way SimpleQA verdict from a judge reply (last token wins)."""
    hits = re.findall(r"NOT[_ ]?ATTEMPTED|INCORRECT|CORRECT", (grade_text or "").upper())
    if not hits:
        return None
    return "NOT_ATTEMPTED" if hits[-1].replace(" ", "_") == "NOT_ATTEMPTED" else hits[-1]


def _simpleqa_topic(metadata: Any) -> Optional[str]:
    """Topic out of SimpleQA's `metadata` column, which is a stringified Python dict."""
    if isinstance(metadata, dict):
        topic = metadata.get("topic")
        return str(topic).strip() if topic else None
    text = str(metadata or "").strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        if isinstance(value, dict) and value.get("topic"):
            return str(value["topic"]).strip()
    m = re.search(r"['\"]topic['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
    return m.group(1).strip() if m else None


def _load_simpleqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load simpleqa benchmark dataset directly from HF Hub (openai/simple-evals)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('basicv8vc/SimpleQA', split='test')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for simpleqa: {e}")
        raise RuntimeError(f"Could not load dataset for simpleqa: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for simpleqa returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("metadata"), seed="simpleqa")

    samples = []
    for item in rows:
        prompt = item.get("problem") or item.get("question")
        gold = item.get("answer") or item.get("target")
        if not prompt or gold is None:
            raise RuntimeError(
                "simpleqa: unexpected dataset schema (missing 'problem'/'answer'); "
                "refusing to fabricate sample data"
            )
        gold = str(gold).strip()
        # SimpleQA has no top-level `category`; the topic lives inside `metadata`, which
        # the row stores as a stringified dict. The default therefore applied to every
        # row and the per-category table reported one bucket, "geography", for the whole
        # benchmark.
        cat = item.get("category") or _simpleqa_topic(item.get("metadata")) or "uncategorized"

        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} simpleqa samples.")
    return samples


def _eval_simpleqa(response_text: str, gold_target: str) -> bool:
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


async def _async_judge_simpleqa(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 64,
) -> None:
    """Canonical OpenAI SimpleQA 100% LLM Judge executed in parallel post-generation."""
    from tqdm import tqdm
    import asyncio

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using local string fallback for SimpleQA.")
        for trace in sample_traces:
            cleaned = str(trace.get("response_text") or "").strip()
            gold = str(trace.get("gold_answer") or "").strip()
            trace["is_correct"] = _eval_simpleqa(cleaned, gold)
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

        prompt = _GRADER_TEMPLATE.format(question=question, target=gold, predicted_answer=resp_text)

        async with semaphore:
            text, _judge_used = await judge_generate_cascade(prompt)
        if text is None:
            # Judge cascade exhausted (infra outage): excluded from accuracy,
            # NOT scored 0. base.run_eval_suite drops it from pass/fail accuracy.
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

    with tqdm(total=len(sample_traces), desc="Judging [SIMPLEQA]") as pbar:
        tasks = [_judge_single(trace, pbar) for trace in sample_traces]
        await asyncio.gather(*tasks)


def run_simpleqa(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute simpleqa native built-in evaluation suite."""
    skip = gemini_required_skip("simpleqa", model_name)
    if skip is not None:
        return skip
    samples = _load_simpleqa_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="simpleqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_simpleqa,
        async_eval_fn=_async_judge_simpleqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 512),
    )
    # Canonical SimpleQA: NOT_ATTEMPTED is its OWN bucket (base's accuracy collapses it into
    # incorrect). Report the three-way split, accuracy_given_attempted (= correct / attempted),
    # and the headline F-score = harmonic mean of overall-correct and correct-given-attempted.
    c = i = na = 0
    for t in result.get("sample_traces", []) or []:
        g = str(t.get("judge_grade") or "")
        if g == "JUDGE_OUTAGE":
            continue
        verdict = _classify_grade(g)
        if verdict is None:                       # no-key fallback path: only is_correct
            verdict = "CORRECT" if t.get("is_correct") else "INCORRECT"
        if verdict == "CORRECT":
            c += 1
        elif verdict == "NOT_ATTEMPTED":
            na += 1
        else:
            i += 1
    n = c + i + na
    attempted = c + i
    overall = (c / n) if n else 0.0
    given_attempted = (c / attempted) if attempted else 0.0
    f_score = (2 * overall * given_attempted / (overall + given_attempted)
               if (overall + given_attempted) else 0.0)
    result["correct"] = round(overall * 100.0, 2)
    result["not_attempted"] = round(na / n * 100.0, 2) if n else 0.0
    result["incorrect"] = round(i / n * 100.0, 2) if n else 0.0
    result["accuracy_given_attempted"] = round(given_attempted * 100.0, 2)
    result["f_score"] = round(f_score * 100.0, 2)
    result["correct_at_pass_1"] = result.get("accuracy")   # base's correct/scored
    result["metric"] = ("SimpleQA F-score = harmonic mean of overall-correct and "
                        "correct-given-attempted (canonical); see correct / incorrect / "
                        "not_attempted for the three-way split")
    if n:
        result["accuracy"] = result["f_score"]
    return result
