# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: textvqa
# Description: TextVQA (Visual Question Answering on Text in Images Benchmark)

"""Native TextVQA (Visual Question Answering on Text in Images) evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_TEXTVQA_TEMPERATURE`,
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


def _load_textvqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load TextVQA samples directly from canonical HF Hub dataset ('lmms-lab/textvqa')."""
    from datasets import load_dataset, Image

    ds = load_dataset("lmms-lab/textvqa", split="validation")
    try:
        ds = ds.cast_column("image", Image(decode=False))
    except Exception:
        pass

    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, None, seed="textvqa")
    samples = []
    for item in ds:
        q_text = item.get("question", "")
        answers = item.get("answers", [""])
        img_obj = item.get("image")

        b64_str = extract_lossless_image_b64(img_obj)
        if not b64_str:
            continue

        prompt = f"Question: {q_text}\nAnswer with only the short text string found in the image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        samples.append((messages, answers, {"category": "scene_ocr"}))

    logger.info(f"Loaded {len(samples)} TextVQA samples with direct lossless image bytes.")
    return samples


def _eval_textvqa(response_text: str, gold_answers: Any):
    """Score with the canonical metric: VQA accuracy min(#annotators/3, 1) >= 0.5.

    Replaces a bidirectional substring test (gold in pred or pred in gold), which
    credited any verbose answer that merely mentioned the gold.
    """
    from .vqa_common import eval_vqa, extract_short_answer
    if not response_text or gold_answers is None:
        return False
    return eval_vqa(extract_short_answer(response_text), gold_answers)

def run_textvqa(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native TextVQA evaluation suite."""
    samples = _load_textvqa_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="textvqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_textvqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),   # sovereign; base.py defaults when unset
    )
    return _promote_soft_headline(result)


def _promote_soft_headline(result: Dict[str, Any]) -> Dict[str, Any]:
    """Promote the canonical TextVQA MEAN soft VQA-accuracy (mean of min(#annotators-match/3, 1))
    to the headline. The binary pass-rate (fraction clearing >=0.5) under-reports and drops
    defensible partial credit, so recompute the soft mean per trace and keep the binary form as
    `binary_pass_rate`. Mutates and returns `result`."""
    from .vqa_common import vqa_accuracy, extract_short_answer
    soft = []
    for tr in result.get("sample_traces", []):
        gold = tr.get("gold_answer")
        if gold is None:
            continue
        s = vqa_accuracy(extract_short_answer(tr.get("response_text") or ""), gold)
        tr["vqa_soft_score"] = round(s, 4)
        soft.append(s)
    if soft:
        result["binary_pass_rate"] = result.get("accuracy")
        result["accuracy"] = round(sum(soft) / len(soft) * 100.0, 2)
        result["metric"] = ("TextVQA mean soft accuracy = mean of min(#annotator-match/3, 1) x100 "
                            "(canonical); binary_pass_rate = fraction of samples with soft>=0.5")
    return result
