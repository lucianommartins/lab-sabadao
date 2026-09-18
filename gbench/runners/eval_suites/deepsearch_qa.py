# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: deepsearch_qa
# Description: DeepSearchQA (xbench/DeepSearch-2510 - Autonomous Web Search & Information Retrieval Benchmark)

"""gbench native built-in runner for deepsearch_qa (Agentic & Web Research).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_DEEPSEARCH_QA_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from .search_tool import (WEB_SEARCH_TOOL, backend_tally, execute_tool,
                          reset_backend_tally, search_available,
                          search_backend_name, unavailable_reason)
from .swebench_common import infra_required
from .base import (run_eval_suite, DEFAULT_JUDGE_MODEL, parse_grade_verdict,
                   judge_generate_cascade, gemini_key_live_valid)
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Agentic & Web Research"


def _xor_decrypt(data: bytes, key: str) -> str:
    """XOR decrypt benchmark data with canary key."""
    key_bytes = key.encode("utf-8")
    k_len = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % k_len] for i in range(len(data))]).decode("utf-8")


def _load_deepsearch_qa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load DeepSearchQA benchmark dataset directly from HF Hub (xbench/DeepSearch-2510)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("xbench/DeepSearch-2510", split="train")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for deepsearch_qa: {e}")
        raise RuntimeError(f"Could not load dataset for deepsearch_qa: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for deepsearch_qa returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="deepsearch_qa")

    samples = []
    for item in rows:
        canary = str(item.get("canary") or "")
        raw_prompt = item.get("prompt", "")
        raw_answer = item.get("answer", "")
        task_id = str(item.get("id", "ds_task"))

        if canary and raw_prompt:
            try:
                question = _xor_decrypt(base64.b64decode(raw_prompt), canary)
                gold = _xor_decrypt(base64.b64decode(raw_answer), canary)
            except Exception:
                question = str(raw_prompt)
                gold = str(raw_answer)
        else:
            question = str(raw_prompt)
            gold = str(raw_answer)

        prompt = (
            f"[DeepSearch Autonomous Research Question #{task_id}]\n"
            f"{question}\n\n"
            "Search, synthesize factual findings, and provide the concise factual answer.\n"
            "Conclude with: Final Answer: <answer>"
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold.strip(), {"category": "deepsearch_web"}))

    logger.info(f"Loaded {len(samples)} deepsearch_qa samples.")
    return samples


async def _async_judge_deepsearch_qa(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 32,
) -> None:
    """Canonical DeepSearchQA LLM-judge grading via gbench's Gemini judge cascade.

    The dataset's short factual answers can be phrased many ways, so grading is judge-only.
    GEMINI_API_KEY is guaranteed present here: run_deepsearch_qa hard-errors up front if the
    search/judge backend is unavailable, so there is no string-match fallback. Binary
    CORRECT/INCORRECT; a judge-cascade outage is recorded as JUDGE_OUTAGE (excluded from the
    denominator by run_eval_suite), never silently scored wrong.
    """
    import asyncio
    from tqdm import tqdm

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        resp_text = trace.get("response_text")
        if not resp_text:
            trace["is_correct"] = False
            trace["judge_grade"] = "FAILED"
            pbar.update(1)
            return
        msgs = trace.get("messages", [])
        question = msgs[0].get("content", "") if msgs else ""
        gold = str(trace.get("gold_answer") or "")
        prompt = (
            "You are grading a model's answer to a factual research question against the gold "
            "answer. Grade CORRECT if the model's final answer matches the gold in meaning "
            "(ignore phrasing, extra detail, or citations); INCORRECT if it gives a different "
            "or contradicting answer, or fails to answer.\n\n"
            f"Question: {question}\n"
            f"Gold Answer: {gold}\n"
            f"Model Response:\n{resp_text}\n\n"
            "Respond in this format:\nGrade: CORRECT / INCORRECT"
        )
        async with semaphore:
            text, _judge_used = await judge_generate_cascade(prompt)
        if text is None:
            trace["judge_grade"] = "JUDGE_OUTAGE"
            trace["status"] = "OK"
            pbar.update(1)
            return
        grade_str = (text or "").strip().upper()
        trace["is_correct"] = parse_grade_verdict(grade_str)
        trace["judge_grade"] = grade_str
        trace["status"] = "OK"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [DEEPSEARCH_QA]") as pbar:
        await asyncio.gather(*[_judge_single(t, pbar) for t in sample_traces])


def run_deepsearch_qa(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native DeepSearchQA evaluation benchmark (canonical LLM judge over search results)."""
    # deepsearch_qa is an AUTONOMOUS web-research benchmark: it requires a live search backend,
    # and the same GEMINI_API_KEY powers the canonical LLM judge. Without it the suite cannot be
    # run canonically - hard-error (never skip), checked before any dataset load or generation.
    if not search_available():
        raise infra_required("deepsearch_qa", unavailable_reason(), "docs/evals/deepsearch_qa.md")
    # Live auth ping so a present-but-invalid/expired key fails fast here, not after a full run of
    # search + generation turns every judge call into an outage.
    _ok, _why = gemini_key_live_valid(os.environ["GEMINI_API_KEY"])
    if not _ok:
        raise infra_required(
            "deepsearch_qa",
            f"GEMINI_API_KEY was rejected by the search/judge endpoint ({_why}); a valid key is "
            "required for Google-Search grounding and the LLM judge.",
            "docs/evals/deepsearch_qa.md")
    reset_backend_tally()   # per-run, not per-process (a sweep runs many search suites)
    samples = _load_deepsearch_qa_samples(limit=limit)
    extra_payload = dict(kwargs.get("extra_payload") or {})
    extra_payload.setdefault("tools", [WEB_SEARCH_TOOL])

    result = run_eval_suite(
        eval_name="deepsearch_qa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_judge_deepsearch_qa,   # judge-only (string-match fallback removed)
        thinking=enable_thinking,
        extra_payload=extra_payload,
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        tool_executor=execute_tool,
    )
    # Search-only, not the browsing agent the public leaderboard uses: say so on the result
    # so the number is never read as leaderboard-comparable.
    result["search_backend"] = search_backend_name()
    result["search_backend_calls"] = backend_tally()
    result["leaderboard_comparable"] = False
    return result
