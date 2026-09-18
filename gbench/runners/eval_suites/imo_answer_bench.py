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

"""Native IMO-AnswerBench (Google DeepMind IMO-Bench) evaluation suite.

Dataset: `OpenEvals/IMO-AnswerBench` (CC-BY-4.0) - 400 olympiad problems with a short
final answer, evenly split across Algebra / Combinatorics / Geometry / Number theory.

This suite previously loaded `AI-MO/NuminaMath-CoT` and filtered it for rows mentioning
"IMO". NuminaMath-CoT is a chain-of-thought *training* corpus of ~860k rows: the filtered
subset was not the benchmark, its size depended on a substring match, and any model
trained on NuminaMath had seen the items. The real benchmark is loaded here instead.

Scoring follows IMO-Bench's **AnswerAutoGrader**: the model's final answer must be
*semantically equivalent* to the reference, with no partial credit. Answers are LaTeX
expressions, sets and tuples, so equality is decided in two stages - normalized textual
equality first, then an LLM equivalence judge for the rest.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_IMO_ANSWER_BENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, DEFAULT_JUDGE_MODEL, judge_generate_cascade, gemini_required_skip
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

PILLAR = "Mathematics & Proofs"
DOCS_URL = "docs/evals/imo_answer_bench.md"
_DATASET = "OpenEvals/IMO-AnswerBench"


def _load_imo_answer_bench_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load IMO-AnswerBench from HF Hub; raises rather than substituting another corpus."""
    from datasets import load_dataset

    try:
        ds = load_dataset(_DATASET, split="train")
    except Exception as e:
        logger.error("Failed to load %s: %s", _DATASET, e)
        raise RuntimeError(f"Could not load dataset for imo_answer_bench: {e}") from e

    # Stratified across the four categories, not a contiguous head (audit RC-1):
    # the 400 problems are stored grouped, so a head was 20 Algebra questions.
    ds = limit_dataset(ds, limit, "Category", seed="imo_answer_bench")
    raw_samples = list(ds)
    if not raw_samples:
        raise RuntimeError("imo_answer_bench: dataset returned no rows")
    logger.info("Loaded %d IMO-AnswerBench problems from '%s'.", len(raw_samples), _DATASET)

    samples = []
    for item in raw_samples:
        problem = str(item.get("Problem") or "").strip()
        gold = str(item.get("Short Answer") or "").strip()
        if not problem or not gold:
            continue
        prompt = f"Solve the following olympiad problem:\n\n{problem}\n\n"
        if enable_thinking:
            prompt += ("Reason step by step, then present the final exact answer in the "
                       "format: 'Final Answer: \\boxed{answer}'.")
        else:
            prompt += ("Present the final exact answer in the format: "
                       "'Final Answer: \\boxed{answer}'.")
        samples.append(([{"role": "user", "content": prompt}], gold, {
            "category": str(item.get("Category") or "unknown"),
            "subcategory": str(item.get("Subcategory") or ""),
            "problem_id": str(item.get("Problem ID") or ""),
        }))
    return samples


def _extract_answer(response_text: str) -> str:
    """The model's stated final answer: \\boxed{} -> "Final Answer:" -> last non-empty line."""
    text = str(response_text or "").strip()
    # \boxed{} may nest one brace level (\boxed{\frac{1}{2}}), so match balanced-ish.
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxed:
        return boxed[-1].strip()
    anchored = re.findall(r"Final Answer\s*:?\s*(.+)", text, re.IGNORECASE)
    if anchored:
        return anchored[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _normalize_math(text: str) -> str:
    """Strip presentation-only LaTeX so `$3$`, `3` and `\\(3\\)` compare equal."""
    t = str(text or "").strip()
    t = re.sub(r"\\(left|right|displaystyle|text|mathrm|,|;|!|quad|qquad)\b", "", t)
    t = t.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    t = re.sub(r"[\$\s]", "", t)
    t = t.rstrip(".")
    return t.lower()


def _eval_imo_answer_bench(response_text: str, gold_answer: str) -> bool:
    """Deterministic half of AnswerAutoGrader: normalized equality of the stated answer.

    Strictly equality, not containment. The previous version accepted `gold in cand`, so a
    response listing several candidate answers passed as long as the right one appeared
    among them - and with golds as short as `3` that is easy to hit by accident.
    Semantically-equivalent-but-differently-written answers are settled by the judge.
    """
    if not response_text or not gold_answer:
        return False
    pred = _normalize_math(_extract_answer(response_text))
    gold = _normalize_math(gold_answer)
    return bool(gold) and pred == gold


_GRADER_PROMPT = """You are the AnswerAutoGrader for IMO-AnswerBench.

# PROBLEM
{problem}

# REFERENCE ANSWER
{gold}

# MODEL'S FINAL ANSWER
{pred}

Decide whether the model's final answer is mathematically EQUIVALENT to the reference.
Equivalent formatting (e.g. `1/2` vs `\\frac{{1}}{{2}}`, a set written in another order,
an algebraically identical expression) counts as correct. A different value, a partial
answer, a missing case, or extra spurious cases counts as incorrect. There is no partial
credit.

Reply with exactly one word: EQUIVALENT or DIFFERENT."""


async def _async_grade_imo(sample_traces: List[Dict[str, Any]],
                           judge_model: str = DEFAULT_JUDGE_MODEL,
                           judge_concurrency: int = 32) -> None:
    """AnswerAutoGrader: exact-match first, LLM equivalence judge for the remainder."""
    from tqdm import tqdm

    pending = []
    for trace in sample_traces:
        response = str(trace.get("response_text") or "")
        gold = str(trace.get("gold_answer") or "")
        if not response:
            trace["is_correct"] = False
            trace["judge_grade"] = "no_response"
        elif _eval_imo_answer_bench(response, gold):
            trace["is_correct"] = True
            trace["judge_grade"] = "exact"
        else:
            pending.append(trace)

    if not pending:
        return
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Defensive: run_imo_answer_bench's gemini_required_skip gate already hard-errors on an
        # absent/invalid key, so this is unreachable in normal flow. Never silently downgrade to
        # textual-only scoring (that emits a depressed number masquerading as the judge metric).
        from .swebench_common import infra_required
        raise infra_required(
            "imo_answer_bench",
            "requires GEMINI_API_KEY for canonical AnswerAutoGrader grading",
            "docs/evals/imo_answer_bench.md")

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _grade(trace: Dict[str, Any], pbar: tqdm) -> None:
        messages = trace.get("messages") or []
        problem = messages[0].get("content", "") if messages else ""
        prompt = _GRADER_PROMPT.format(
            problem=problem[:6000],
            gold=str(trace.get("gold_answer") or ""),
            pred=_extract_answer(str(trace.get("response_text") or ""))[:2000])
        async with semaphore:
            text, _judge_used = await judge_generate_cascade(prompt)
        if text is None:
            # Judge cascade exhausted (infra outage): excluded from accuracy by
            # base.run_eval_suite, NOT scored wrong.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            pbar.update(1)
            return
        upper = text.strip().upper()
        if "EQUIVALENT" in upper:
            verdict = True
        elif "DIFFERENT" in upper:
            verdict = False
        else:
            verdict = None
        if verdict is None:
            # Judge answered but gave neither verdict (unparseable): wrong, NOT an outage.
            trace["is_correct"] = False
            trace["judge_grade"] = "judge_error"
        else:
            trace["is_correct"] = verdict
            trace["judge_grade"] = "equivalent" if verdict else "different"
        pbar.update(1)

    with tqdm(total=len(pending), desc="Grading [IMO_ANSWER_BENCH]") as pbar:
        await asyncio.gather(*[_grade(t, pbar) for t in pending])


def run_imo_answer_bench(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run IMO-AnswerBench (400 problems) under AnswerAutoGrader semantics."""
    # Fail-fast: the AnswerAutoGrader judge is load-bearing. A missing/invalid key must hard-error
    # BEFORE the generation run, never downgrade to textual-only scoring (which would publish a
    # depressed number labelled as the canonical judge metric).
    gemini_required_skip("imo_answer_bench", model_name)
    samples = _load_imo_answer_bench_samples(
        enable_thinking=enable_thinking,
        limit=kwargs.get("limit"),
    )
    result = run_eval_suite(
        eval_name="imo_answer_bench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_grade_imo,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 16384 if enable_thinking else 4096),
    )
    result["dataset"] = _DATASET
    result["metric"] = "AnswerAutoGrader equivalence (no partial credit)"
    return result
