# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: chartqa
# Description: ChartQA (Plot and Chart Visual Reasoning Benchmark)

"""Native ChartQA (Plot and Visual Reasoning) evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_CHARTQA_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import base64
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .dataset_utils import extract_lossless_image_b64

logger = logging.getLogger(__name__)


def _load_chartqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load ChartQA plot reasoning samples with direct lossless raw image byte decoding."""
    from datasets import load_dataset, Image

    try:
        ds = load_dataset("ahmed-masry/ChartQA", split="test")
    except Exception:
        ds = load_dataset("HuggingFaceM4/ChartQA", split="test")

    try:
        ds = ds.cast_column("image", Image(decode=False))
    except Exception:
        pass

    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, None, seed="chartqa")
    samples = []
    for item in ds:
        q_text = item.get("query", "") or item.get("question", "")
        label = item.get("label", [""])
        img_obj = item.get("image")

        b64_str = extract_lossless_image_b64(img_obj)
        if not b64_str:
            continue

        prompt = (
            f"Question: {q_text}\n"
            "Analyze the chart image and answer the question. Give only the direct short answer or number."
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
        samples.append((messages, label, {"category": "chart_reasoning"}))

    logger.info(f"Loaded {len(samples)} ChartQA samples with direct lossless PNG bytes.")
    return samples


def _eval_chartqa(response_text: str, gold_answers: Any):
    """Score with the canonical metric: relaxed accuracy: numeric within 5%, else exact match.

    Replaces a bidirectional substring test (gold in pred or pred in gold), which
    credited any verbose answer that merely mentioned the gold.
    """
    from .vqa_common import eval_relaxed, extract_short_answer
    if not response_text or gold_answers is None:
        return False
    return eval_relaxed(extract_short_answer(response_text), gold_answers)

def run_chartqa(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native ChartQA evaluation suite."""
    samples = _load_chartqa_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="chartqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_chartqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
