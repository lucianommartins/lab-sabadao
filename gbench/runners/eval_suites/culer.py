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

"""Native CULER (Code-RULER / Long-Context Code Retrieval) evaluation suite.

Evaluates the CODE domain of LongBench-v2 (long-context multiple-choice QA over multi-file
codebases). LongBench-v2 samples reach ~2M chars, so the context is left-cropped to ~128K
tokens to fit the server window; a truncated run is therefore NOT directly comparable to a
full-context LongBench-v2 result (the suite flags leaderboard_comparable=False).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_CULER_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, suite_env
from .sampling import limit_dataset

logger = logging.getLogger(__name__)


def _load_culer_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load CULER code retrieval samples from canonical HF Hub dataset ('zai-org/LongBench-v2')."""
    from datasets import load_dataset

    ds = load_dataset("zai-org/LongBench-v2", split="train")
    max_context_chars = 400000  # ~110k-128k tokens to safely fit within 131k server limit

    # CULER is the CODE slice of LongBench-v2. Loading the split unfiltered meant only
    # ~10% of the scored rows were code and the rest were single/multi-document QA, so the
    # number was not a code-retrieval number at all. `GBENCH_CULER_ALL_DOMAINS=1` (legacy
    # alias `CULER_ALL_DOMAINS=1` still works) restores the old unfiltered behaviour for a
    # full LongBench-v2 run.
    if not suite_env("GBENCH_CULER_ALL_DOMAINS", "CULER_ALL_DOMAINS"):
        def _is_code(row):
            return "code" in str(row.get("domain") or "").lower()
        before = len(ds)
        ds = ds.filter(_is_code) if hasattr(ds, "filter") else [r for r in ds if _is_code(r)]
        logger.info("[culer] filtered LongBench-v2 to the code domain: %d/%d rows "
                    "(set GBENCH_CULER_ALL_DOMAINS=1 to score every domain)", len(ds), before)

    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, "domain", seed="culer")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} CULER code retrieval samples from HF Hub ('zai-org/LongBench-v2').")

    samples = []
    for item in raw_samples:
        context = item.get("context", "")
        q_text = item.get("question", "")
        gold_ans = str(item.get("answer", "")).strip().upper()
        domain = item.get("domain", "code")

        c_a = item.get("choice_A", "")
        c_b = item.get("choice_B", "")
        c_c = item.get("choice_C", "")
        c_d = item.get("choice_D", "")
        options_str = f"(A) {c_a}\n(B) {c_b}\n(C) {c_c}\n(D) {c_d}"

        # Left-crop context to fit 128k budget
        if len(context) > max_context_chars:
            context = context[-max_context_chars:]

        prompt = (
            f"=== LONG CONTEXT REPOSITORY / DOCUMENT ===\n{context}\n\n"
            f"=== QUESTION ===\n{q_text}\n\n"
            f"Options:\n{options_str}\n\n"
            "Output only the correct option letter in the format: 'Answer: (X)'."
        )
        messages = [{"role": "user", "content": prompt}]
        # No per-sample max_tokens: a per-sample cap OVERRIDES the sovereign --max-output-tokens (and
        # truncated the reasoning to "---" under --thinking, a false 0%). The run's budget - or
        # base.py's thinking-aware default when unset - governs output length.
        samples.append((messages, gold_ans, {"category": domain}))

    return samples


def _eval_culer(response_text: str, gold_answer: str) -> bool:
    """Validate predicted answer against gold target needle/option."""
    if not response_text or not gold_answer:
        return False

    resp = response_text.strip()
    gold = gold_answer.strip().upper()

    match = re.search(r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-D])\)?", resp, re.IGNORECASE)
    if match:
        return match.group(1).upper() == gold

    tokens = re.findall(r"\b([A-D])\b", resp)
    if tokens:
        return tokens[-1].upper() == gold

    return False


def run_culer(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native CULER (LongBench-v2 code-domain MCQA) evaluation suite."""
    samples = _load_culer_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="culer",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_culer,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),   # sovereign; base.py defaults when unset
        temperature=kwargs.get("temperature"),
    )
    # LongBench-v2 samples exceed the ~128k-token crop, so context is truncated to fit the
    # server window; the score is not directly comparable to a full-context LongBench-v2 run.
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "context left-cropped to ~128k tokens (LongBench-v2 samples reach ~2M chars)")
    return result
