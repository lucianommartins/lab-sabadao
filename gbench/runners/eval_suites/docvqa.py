# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: docvqa
# Description: DocVQA (Document Visual Question Answering Benchmark)

"""Native DocVQA (Document Visual Question Answering) evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_DOCVQA_TEMPERATURE`,
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
from .metrics import finalize_mean_metric

logger = logging.getLogger(__name__)


def _load_docvqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load DocVQA document samples with direct lossless raw image byte decoding."""
    from datasets import load_dataset, Image

    try:
        ds = load_dataset("lmms-lab-encoder/DocVQA", "DocVQA", split="validation")
    except Exception as e:
        # NO silent fallback to hf-internal-testing/fixtures_docvqa: that is a 5-sample unit
        # test fixture, and reporting a score over it under the name 'docvqa' is a fabricated
        # headline. Fail loudly instead.
        raise RuntimeError(
            f"docvqa: could not load lmms-lab-encoder/DocVQA validation split ({e}); refusing "
            "to fall back to the 5-sample unit-test fixture and report it as 'docvqa'.") from e

    try:
        ds = ds.cast_column("image", Image(decode=False))
    except Exception:
        pass

    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, None, seed="docvqa")
    samples = []
    for item in ds:
        q_text = item.get("question", "")
        answers = item.get("answers", [""])
        doc_type = item.get("data_type", "document")
        img_obj = item.get("image")

        b64_str = extract_lossless_image_b64(img_obj)
        if not b64_str:
            continue

        prompt = (
            f"Question: {q_text}\n"
            "Extract the exact answer directly from the document image. Output only the short answer text."
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
        samples.append((messages, answers, {"category": doc_type}))

    logger.info(f"Loaded {len(samples)} DocVQA samples with direct lossless PNG bytes.")
    return samples


def _eval_docvqa(response_text: str, gold_answers: Any):
    """Score with the canonical metric: ANLS >= 0.5 (the DocVQA metric).

    Replaces a bidirectional substring test (gold in pred or pred in gold), which
    credited any verbose answer that merely mentioned the gold.
    """
    from .vqa_common import eval_anls, extract_short_answer
    if not response_text or gold_answers is None:
        return False
    return eval_anls(extract_short_answer(response_text), gold_answers)

def run_docvqa(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native DocVQA document understanding evaluation suite."""
    from .vqa_common import anls_score, extract_short_answer
    samples = _load_docvqa_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="docvqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_docvqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
    # Canonical DocVQA headline is MEAN ANLS (the continuous per-question score), not the
    # share clearing the 0.5 threshold. A failed/empty response scores 0 ANLS (all questions
    # count), so it is not dropped from the mean.
    return finalize_mean_metric(
        result,
        metric_name="mean ANLS (canonical DocVQA)",
        scorer=lambda t: anls_score(extract_short_answer(t.get("response_text") or ""),
                                    t.get("gold_answer")),
        stash_key="anls",
        secondary_key="pass_rate_at_0.5",
        failure_score=0.0,
    )
