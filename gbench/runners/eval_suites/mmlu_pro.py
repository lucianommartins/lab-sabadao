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

"""Native MMLU-Pro zero-shot / CoT evaluation suite (10 options A-J).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MMLU_PRO_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def _format_options(choices: Any) -> str:
    return "\n".join(f"({let}) {choice}"
                     for let, choice in zip(OPTION_LETTERS[:len(choices or [])], choices or []))


def _few_shot_prefixes(n_shot: int) -> Dict[str, str]:
    """Canonical MMLU-Pro few-shot CoT exemplars, keyed by category.

    The `validation` split is exactly the official exemplar set - 5 questions per
    category, each with a worked `cot_content` answer - so few-shot prompting uses the
    same demonstrations as the published protocol rather than sampled test items.

    `--eval-n-shot` used to be accepted, gated to this suite, and recorded on the result
    as `n_shot`, but no loader ever read it: every run was 0-shot and labelled otherwise.
    """
    if n_shot <= 0:
        return {}
    from datasets import load_dataset
    try:
        dev = list(load_dataset("TIGER-Lab/MMLU-Pro", split="validation"))
    except Exception as e:
        logger.warning("[mmlu_pro] --eval-n-shot requested but the validation split could "
                       "not be loaded (%s); running 0-shot", e)
        return {}

    by_category: Dict[str, List[str]] = {}
    for item in dev:
        category = str(item.get("category", ""))
        if len(by_category.setdefault(category, [])) >= n_shot:
            continue
        cot = str(item.get("cot_content") or "").strip()
        cot = re.sub(r"^A:\s*", "", cot)
        if not cot:
            continue
        by_category[category].append(
            f"Question: {item.get('question', '')}\n\n"
            f"{_format_options(item.get('options') or [])}\n\n"
            f"Answer: {cot}")
    return {c: "\n\n".join(shots) + "\n\n" for c, shots in by_category.items() if shots}


def _load_mmlu_pro_samples(enable_thinking: bool = False, limit: Optional[int] = None,
                           n_shot: int = 0) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MMLU-Pro samples directly from canonical HF Hub dataset ('TIGER-Lab/MMLU-Pro')."""
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, "category", seed="mmlu_pro")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} MMLU-Pro samples from HF Hub ('TIGER-Lab/MMLU-Pro').")

    few_shot = _few_shot_prefixes(n_shot)
    if few_shot:
        logger.info("[mmlu_pro] prepending %d-shot CoT exemplars for %d categories",
                    n_shot, len(few_shot))

    samples = []
    for item in raw_samples:
        q_text = item.get("question", "") or item.get("problem", "")
        choices = item.get("options") or item.get("choices", [])
        ans = item.get("answer")
        if ans is None:
            ans = item.get("answer_index", 0)
        if isinstance(ans, str) and ans.strip().upper() in OPTION_LETTERS:
            gold_letter = ans.strip().upper()
        else:
            gold_idx = int(ans)
            gold_letter = OPTION_LETTERS[gold_idx]

        options_str = _format_options(choices)
        category = str(item.get("category", ""))
        if enable_thinking:
            prompt = (
                f"Question: {q_text}\n\n{options_str}\n\n"
                "Let's think step by step and then output the correct option letter in the format: 'Answer: (X)'.\n"
                "Answer:"
            )
        else:
            prompt = (
                f"Question: {q_text}\n\n{options_str}\n\n"
                "Answer with only the correct option letter in the format: 'Answer: (X)'.\n"
                "Answer:"
            )
        shots = few_shot.get(category)
        if shots:
            prompt = (f"The following are multiple choice questions (with answers) about "
                      f"{category}.\n\n{shots}{prompt}")
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_letter, {"category": category}))
    return samples


def _eval_mmlu_pro(response_text: str, gold_letter: str) -> bool:
    """Check if predicted answer matches gold MMLU-Pro letter (A-J)."""
    # LAST match, not first, at every tier - TIGER-Lab's own extractor writes
    # `.*[aA]nswer:\s*\(?([A-J])\)?` whose leading `.*` is GREEDY, so it lands on the
    # final occurrence. Under CoT the model can state an answer mid-reasoning and then
    # revise it ("Answer: B ... wait ... Answer: D"); first-match grades the abandoned one.
    # Measured 2026-08-20 the two agreed on 12,013/12,032 and scored an identical 82.58%,
    # because the prompt ends with "Answer:" and the model complies once - the divergence
    # appears mainly when the suite is run with thinking ON.
    text = response_text.strip().upper()
    hits = re.findall(r"ANSWER:\s*\(?([ABCDEFGHIJ])\)?", text)
    if not hits:
        m = re.match(r"\(?([ABCDEFGHIJ])\)?", text)
        hits = [m.group(1)] if m else []
    if not hits:
        # Bare standalone letter, last one. On uppercased text `\b[A-J]\b` also matches
        # the article "A", so taking the FIRST reliably grabbed a word from the reasoning.
        hits = re.findall(r"\b([ABCDEFGHIJ])\b", text)
    return bool(hits) and hits[-1] == gold_letter


def run_mmlu_pro(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native MMLU-Pro evaluation suite (canonical 5-shot CoT by default; `--eval-n-shot N` overrides, 0 = 0-shot)."""
    _ns = kwargs.get("eval_n_shot")
    if _ns is None:
        _ns = kwargs.get("n_shot")
    n_shot = 5 if _ns is None else int(_ns)   # canonical 5-shot CoT default; explicit 0 respected
    samples = _load_mmlu_pro_samples(enable_thinking=enable_thinking, limit=kwargs.get("limit"),
                                     n_shot=n_shot)
    return run_eval_suite(
        eval_name="mmlu_pro",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_mmlu_pro,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
