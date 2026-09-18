# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: infographicvqa
# Description: InfographicVQA (Visual Question Answering on Infographic Posters Benchmark)

"""Native InfographicVQA (Visual Question Answering on Infographics) evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_INFOGRAPHICVQA_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .dataset_utils import extract_lossless_image_b64
from .metrics import finalize_mean_metric

logger = logging.getLogger(__name__)


def _load_infographicvqa_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load InfographicVQA samples directly from canonical HF Hub dataset ('mm-eval/InfographicVQA')."""
    from datasets import load_dataset

    ds = load_dataset("mm-eval/InfographicVQA", split="validation")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, None, seed="infographicvqa")
    samples = []
    for item in ds:
        q_text = ""
        msgs = item.get("messages", [])
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                msgs = []
        if isinstance(msgs, list) and msgs:
            q_text = msgs[0].get("question", "")
        elif isinstance(item.get("question"), str):
            q_text = item["question"]

        gold_answers = item.get("answer", [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        q_type = item.get("question_type") or "general"
        media_list = item.get("media", [])
        if not media_list:
            continue

        img_obj = media_list[0]
        b64_str = extract_lossless_image_b64(img_obj)
        if not b64_str:
            continue

        prompt = (
            f"Question: {q_text}\n"
            "Analyze the infographic and output only the direct short answer text found in the graphic."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        samples.append((messages, gold_answers, {"category": q_type}))

    logger.info(f"Loaded {len(samples)} InfographicVQA samples with direct lossless image bytes.")
    return samples


def _eval_infographicvqa(response_text: str, gold_answers: Any):
    """Score with the canonical metric: ANLS >= 0.5 (the InfographicVQA metric).

    Replaces a bidirectional substring test (gold in pred or pred in gold), which
    credited any verbose answer that merely mentioned the gold.
    """
    from .vqa_common import eval_anls, extract_short_answer
    if not response_text or gold_answers is None:
        return False
    return eval_anls(extract_short_answer(response_text), gold_answers)

def run_infographicvqa(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native InfographicVQA evaluation suite."""
    from .vqa_common import anls_score, extract_short_answer
    samples = _load_infographicvqa_samples(enable_thinking=enable_thinking, limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="infographicvqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_infographicvqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
    # Canonical InfographicVQA headline is MEAN ANLS, not the share clearing 0.5. Failed/empty
    # responses score 0 ANLS (all questions count), so they are not dropped from the mean.
    return finalize_mean_metric(
        result,
        metric_name="mean ANLS (canonical InfographicVQA)",
        scorer=lambda t: anls_score(extract_short_answer(t.get("response_text") or ""),
                                    t.get("gold_answer")),
        stash_key="anls",
        secondary_key="pass_rate_at_0.5",
        failure_score=0.0,
    )
