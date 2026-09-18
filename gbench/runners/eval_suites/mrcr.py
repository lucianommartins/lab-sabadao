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

"""Native MRCR (Needle in a Haystack / Long-Context Retrieval) evaluation suite.

NOTE ON SERVER CONTEXT LENGTH REQUIREMENTS:
The server serving the model must support a maximum context length of at least 140,000 tokens
(--max-model-len 140000) to safely evaluate full 131K MRCR long-context prompts without HTTP 400
context overflow errors.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MRCR_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import csv
import difflib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .metrics import finalize_mean_metric

logger = logging.getLogger(__name__)

def _load_mrcr_samples(limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MRCR long-context retrieval samples from canonical HF Hub dataset ('openai/mrcr')."""
    import json
    from datasets import load_dataset

    ds = load_dataset("openai/mrcr", split="train")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, "n_needles", seed="mrcr")
    raw_samples = list(ds)
    samples = []
    for item in raw_samples:
        prompt_raw = item["prompt"]
        if isinstance(prompt_raw, str) and prompt_raw.strip().startswith("["):
            try:
                messages = json.loads(prompt_raw)
            except Exception:
                messages = [{"role": "user", "content": prompt_raw}]
        else:
            messages = [{"role": "user", "content": str(prompt_raw)}]

        ans = str(item.get("answer", ""))
        n_needles = item.get("n_needles", 8)
        category = f"{n_needles}_needles"
        canary = str(item.get("random_string_to_prepend", "") or "")
        # The canary is half the metric (the response must START with it), so it has to
        # reach the scorer. It used to be recorded in the sample metadata and never read.
        samples.append((messages, {"answer": ans, "canary": canary},
                        {"category": category, "canary": canary}))

    logger.info(f"Loaded {len(samples)} MRCR samples from HF Hub ('openai/mrcr').")
    return samples


def grade_mrcr(response_text: str, gold_target: Any) -> float:
    """Canonical MRCR grade in [0, 1] (openai/mrcr `grade`).

        if not response.startswith(random_string_to_prepend): return 0
        response = response.removeprefix(random_string_to_prepend)
        answer   = answer.removeprefix(random_string_to_prepend)
        return SequenceMatcher(None, response, answer).ratio()

    Two things the previous implementation did not do: it never applied the canary gate
    (the dataset's `random_string_to_prepend` was loaded into the sample metadata and left
    unread), and it graded with case-insensitive containment instead of the sequence
    ratio - so a response that merely contained the needle anywhere scored the same as a
    verbatim reproduction, and a response that ignored the required prefix was not
    penalised at all.
    """
    if isinstance(gold_target, dict):
        answer = str(gold_target.get("answer") or "")
        canary = str(gold_target.get("canary") or "")
    else:
        answer, canary = str(gold_target or ""), ""
    response = str(response_text or "")
    if not response or not answer:
        return 0.0
    if canary:
        if not response.startswith(canary):
            return 0.0
        response = response[len(canary):]
        if answer.startswith(canary):
            answer = answer[len(canary):]
    # Canonical openai/mrcr grade() uses SequenceMatcher's default (autojunk=True); forcing
    # autojunk=False changed the ratio on the long, block-duplicated MRCR strings and made
    # the number non-comparable to published MRCR.
    return float(difflib.SequenceMatcher(None, response, answer).ratio())


def _eval_mrcr(response_text: str, gold_target: Any) -> bool:
    """Harness pass/fail: a verbatim reproduction. The reported metric is the mean grade."""
    return grade_mrcr(response_text, gold_target) >= 0.99


def _get_server_max_model_len(base_url: str) -> Optional[int]:
    """Query /v1/models from server to obtain max_model_len if reported."""
    import json
    import urllib.request

    try:
        url = f"{base_url.rstrip('/')}/models"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models and "max_model_len" in models[0]:
                return int(models[0]["max_model_len"])
    except Exception as e:
        logger.debug(f"Could not query max_model_len from {base_url}: {e}")
    return None


def run_mrcr(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native MRCR long-context evaluation suite."""
    max_len = _get_server_max_model_len(base_url)
    if max_len is not None:
        logger.info(f"Server reported max_model_len: {max_len}")
        if max_len < 140000:
            logger.warning(
                f"Server max_model_len ({max_len}) is below recommended 140,000 tokens for MRCR 131K. "
                f"Long-context prompts may fail with HTTP 400 context overflow."
            )

    samples = _load_mrcr_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="mrcr",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_mrcr,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    # MRCR's canonical score is the AVERAGE sequence-match ratio, not a pass rate. A failed
    # or empty response (e.g. an over-context 131K prompt that 400s) is a 0, NOT a sample to
    # drop from the mean - dropping the hard ones silently inflated the score.
    return finalize_mean_metric(
        result,
        metric_name="mean SequenceMatcher ratio after the canary-prefix gate (canonical MRCR)",
        scorer=lambda t: grade_mrcr(t.get("response_text"), t.get("gold_answer")),
        stash_key="mrcr_grade",
        secondary_key="exact_match_rate",
        failure_score=0.0,
    )
