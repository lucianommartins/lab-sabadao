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

"""Native CausalBench (corr2cause) Causal Graph & Reasoning evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_CAUSALBENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)


def _load_causalbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load causal reasoning benchmark samples from canonical HF dataset ('causal-nlp/corr2cause')."""
    from datasets import load_dataset

    ds = load_dataset("causal-nlp/corr2cause", split="test")
    # Stratified, not a contiguous head (audit RC-1). The column is 'num_variables' (the old
    # 'num_vars' key does not exist, so stratification was a silent no-op).
    ds = limit_dataset(ds, limit, "num_variables", seed="causalbench")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} CausalBench samples from HF Hub ('causal-nlp/corr2cause').")

    samples = []
    for item in raw_samples:
        text = item.get("input", "").strip()
        label_int = item.get("label", 0)
        gold_ans = "Yes" if label_int == 1 else "No"
        num_vars = str(item.get("num_variables", 2))

        prompt = (
            f"{text}\n\n"
            "Based strictly on the statistical relations and causal rules given above, is the hypothesis valid? "
            "Answer with only 'Yes' or 'No'."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_ans, {"category": f"{num_vars}_vars"}))

    return samples


def _eval_causalbench(response_text: str, gold_answer: str) -> bool:
    """Yes/No causal verdict (CC7).

    Previously `\\b(yes|true|valid)\\b` was searched over the whole response, so
    "No, this is not valid" matched the positive branch via "valid" and the opposite
    verdict was never checked. Now the anchored/last verdict decides and an ambiguous
    response (neither verdict present) is incorrect rather than guessed.
    """
    from .extraction_common import binary_verdict
    if not response_text:
        return False
    verdict = binary_verdict(response_text)
    if verdict is None:
        return False
    return verdict is (str(gold_answer).strip().lower() in ("yes", "true", "valid", "1"))

def run_causalbench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native CausalBench evaluation suite.

    Headline is corr2cause's canonical positive-class F1, not plain accuracy: the test set is
    ~85% majority-"No", so accuracy is a trivial-baseline number. F1 is recomputed over all
    traces from the predicted vs gold verdict.
    """
    samples = _load_causalbench_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="causalbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_causalbench,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    from .extraction_common import binary_verdict
    tp = fp = fn = 0
    for t in result.get("sample_traces", []):
        gold_yes = str(t.get("gold_answer", "")).strip().lower() in ("yes", "true", "valid", "1")
        pred_yes = (binary_verdict(t.get("response_text") or "") is True)
        if pred_yes and gold_yes:
            tp += 1
        elif pred_yes and not gold_yes:
            fp += 1
        elif (not pred_yes) and gold_yes:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    result["raw_accuracy"] = result.get("accuracy")   # plain acc kept for reference
    result["accuracy"] = round(f1 * 100.0, 2)          # headline = canonical positive-class F1
    result["causalbench_report"] = {
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }
    # Native reimplementation with a custom prompt (not the upstream corr2cause template); the F1
    # metric itself is canonical.
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "native reimplementation with a custom prompt (not the upstream corr2cause template)")
    return result
