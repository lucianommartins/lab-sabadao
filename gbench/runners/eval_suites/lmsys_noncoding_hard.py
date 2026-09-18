# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native LMSYS / WildBench Hard Non-Coding Reasoning evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_LMSYS_NONCODING_HARD_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import (run_eval_suite, gemini_required_skip, DEFAULT_JUDGE_MODEL,
                   judge_generate_cascade)
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

#: WildBench primary_tag for coding tasks - excluded from a NON-coding suite.
_CODING_TAGS = {"Coding & Debugging", "coding & debugging", "Coding", "coding"}


def _load_lmsys_noncoding_hard_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]]:
    """Load hard non-coding prompts from canonical HF dataset ('WildEval/WildBench', config 'v2-hard')."""
    from datasets import load_dataset

    ds = load_dataset("WildEval/WildBench", "v2-hard", split="test")
    raw_samples = list(ds)
    # The suite is NON-coding: WildBench-hard is ~33% "Coding & Debugging"; keeping those
    # made the reported number not a non-coding measure. Filter before sampling so the limit
    # still spans the non-coding tags.
    raw_samples = [r for r in raw_samples if str((r or {}).get("primary_tag")) not in _CODING_TAGS]
    raw_samples = limit_dataset(raw_samples, limit, 'primary_tag', seed="lmsys_noncoding_hard")
    logger.info(f"Loaded {len(raw_samples)} Hard Non-Coding samples from HF Hub ('WildEval/WildBench').")

    samples = []
    for item in raw_samples:
        conv = item.get("conversation_input", [])
        if not conv:
            continue
        messages = [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in conv]
        checklist = item.get("checklist", [])
        tag = item.get("primary_tag", "general")
        samples.append((messages, checklist, {"category": tag}))

    return samples


def _eval_lmsys_noncoding_hard(response_text: str, checklist: List[str]) -> bool:
    """No-key deterministic fallback: checklist keyword overlap (labelled judge_fallback).

    This is NOT the canonical metric - WildBench WB-Score is an LLM judge (see
    _async_judge_lmsys). It stands in only when no judge key is available.
    """
    if not response_text or len(response_text.strip()) < 30:
        return False
    text = response_text.lower()
    if not checklist:
        return False
    hits = 0
    for criterion in checklist:
        words = [w for w in re.findall(r"\w+", criterion.lower()) if len(w) > 4]
        if not words:
            hits += 1
            continue
        if sum(1 for w in words if w in text) / len(words) >= 0.4:
            hits += 1
    return (hits / len(checklist)) >= 0.5


def _conversation_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages or "")
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


async def _async_judge_lmsys(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 32,
) -> None:
    """Canonical WildBench WB-Score: an LLM judge rates the response 1-10 against the checklist.

    Replaces the keyword-overlap heuristic (a binary pass/fail). Each trace gets a float
    ``wb_score`` in [0, 1] = (score - 1) / 9; run_lmsys_noncoding_hard reports the mean.
    """
    import asyncio
    from tqdm import tqdm

    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not set; using checklist-heuristic fallback for lmsys.")
        for trace in sample_traces:
            trace["is_correct"] = _eval_lmsys_noncoding_hard(
                str(trace.get("response_text") or ""), trace.get("gold_answer") or [])
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
        query = _conversation_text(trace.get("messages", []))
        checklist = trace.get("gold_answer") or []
        checklist_text = "\n".join(f"- {c}" for c in checklist) if checklist else "(none)"
        prompt = (
            "Act as an impartial judge and evaluate the quality of the AI response to the "
            "user query below, using the checklist as a guide.\n\n"
            f"# User Query\n{query}\n\n"
            f"# Evaluation Checklist\n{checklist_text}\n\n"
            f"# AI Response\n{resp_text}\n\n"
            "Rate the response 1-10 (1-2 terrible, 3-4 poor, 5-6 fair, 7-8 good, 9-10 "
            "excellent) for helpfulness, correctness, completeness and adherence to the "
            "checklist.\nOutput exactly:\nScore: <a single integer from 1 to 10>"
        )
        async with semaphore:
            text, _judge_used = await judge_generate_cascade(prompt)
        if text is None:
            trace["judge_grade"] = "JUDGE_OUTAGE"
            pbar.update(1)
            return
        m = re.search(r"score\s*[:=]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if not m:
            trace["judge_grade"] = "JUDGE_OUTAGE"     # unparseable judge reply -> excluded
            pbar.update(1)
            return
        score = max(1.0, min(10.0, float(m.group(1))))
        # Canonical WildBench per-example ADJUSTED score = (Y-5)*2, range -8..+10 (allenai/WildBench
        # _create_tables.py). The reported WB-Score is the mean of these * 10 = (mean_raw-5)*20.
        trace["wb_score"] = round((score - 5.0) * 2.0, 4)
        trace["wb_score_raw"] = score
        trace["is_correct"] = score >= 7.0            # secondary pass_rate only
        trace["judge_grade"] = f"SCORE_{int(round(score))}"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [LMSYS_WB]") as pbar:
        await asyncio.gather(*[_judge_single(t, pbar) for t in sample_traces])


def run_lmsys_noncoding_hard(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Hard Non-Coding suite; headline = mean WB-Score (canonical WildBench)."""
    from .metrics import finalize_mean_metric
    skip = gemini_required_skip("lmsys_noncoding_hard", model_name)
    if skip is not None:
        return skip
    samples = _load_lmsys_noncoding_hard_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="lmsys_noncoding_hard",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_judge_lmsys,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    result = finalize_mean_metric(
        result,
        metric_name="WB-Score = (mean_raw-5)*20, canonical WildBench range -80..+100",
        score_key="wb_score",
        secondary_key="pass_rate_at_7",
        # Headline = mean(adjusted wb_score) * 10 = (mean_raw-5)*20, the canonical WildBench WB-Score.
        scale=10.0,
        # A failed/empty response is the worst raw score (1 -> (1-5)*2 = -8), not an outage: count it
        # rather than dropping it, so a model that times out on hard items cannot inflate its mean.
        # JUDGE_OUTAGE (judge-side infra) is still excluded by finalize_mean_metric.
        failure_score=-8.0,
    )
    # This is the NON-CODING subset of WildBench-hard by design (coding tags filtered out); the
    # published WildBench WB-Score leaderboard scores the full set, so a run here is a gbench-internal
    # number rather than a like-for-like leaderboard entry. Grading uses gbench's standard Gemini
    # cascade by convention (a gbench grader choice, not a defect).
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "non-coding subset of WildBench-hard by design (the published WildBench leaderboard scores "
        "the full set); graded by gbench's standard Gemini cascade (a gbench convention)")
    return result
