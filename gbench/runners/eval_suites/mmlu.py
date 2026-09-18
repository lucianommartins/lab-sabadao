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

"""Native MMLU zero-shot / CoT evaluation suite (57 subjects, 4 choices A-D).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MMLU_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

OPTION_LETTERS = ["A", "B", "C", "D"]


def _gold_letter(ans: Any, choices: Any) -> str:
    if isinstance(ans, str) and ans.strip().upper() in OPTION_LETTERS:
        return ans.strip().upper()
    gold_idx = int(ans)
    return OPTION_LETTERS[gold_idx] if 0 <= gold_idx < len(OPTION_LETTERS) else "A"


def _format_options(choices: Any) -> str:
    return "\n".join(f"({let}) {choice}"
                     for let, choice in zip(OPTION_LETTERS[:len(choices or [])], choices or []))


def _few_shot_prefixes(n_shot: int) -> Dict[str, str]:
    """Canonical MMLU few-shot exemplars, keyed by subject.

    Classic MMLU is a 5-shot benchmark: the `dev` split ships exactly 5 demonstrations per
    subject, prepended before each test question. `--eval-n-shot` used to be rejected by the
    CLI for this suite, so every MMLU run was 0-shot despite the docs advertising 5-shot.
    """
    if n_shot <= 0:
        return {}
    from datasets import load_dataset
    try:
        dev = list(load_dataset("cais/mmlu", "all", split="dev"))
    except Exception as e:
        logger.warning("[mmlu] --eval-n-shot requested but the dev split could not be "
                       "loaded (%s); running 0-shot", e)
        return {}
    by_subject: Dict[str, List[str]] = {}
    for item in dev:
        subject = str(item.get("subject", ""))
        if len(by_subject.setdefault(subject, [])) >= n_shot:
            continue
        choices = item.get("choices", [])
        letter = _gold_letter(item.get("answer", 0), choices)
        by_subject[subject].append(
            f"Question: {str(item.get('question', '')).strip()}\n\n"
            f"{_format_options(choices)}\n\nAnswer: ({letter})")
    return {s: "\n\n".join(shots) + "\n\n" for s, shots in by_subject.items() if shots}


def _load_mmlu_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
    n_shot: int = 0,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load canonical MMLU samples directly from HF Hub ('cais/mmlu', 'all', split='test')."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    # Stratified, not a contiguous head: the rows are stored grouped by
    # `subject`, so `[:limit]` returned a single category (audit RC-1).
    ds = limit_dataset(ds, limit, "subject", seed="mmlu")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} MMLU samples from HF Hub ('cais/mmlu').")

    few_shot = _few_shot_prefixes(n_shot)
    if few_shot:
        logger.info("[mmlu] prepending %d-shot exemplars for %d subjects",
                    n_shot, len(few_shot))

    samples = []
    for item in raw_samples:
        q_text = str(item.get("question", "")).strip()
        choices = item.get("choices", [])
        subject = str(item.get("subject", "general")).strip()
        gold_letter = _gold_letter(item.get("answer", 0), choices)

        options_str = _format_options(choices)
        if enable_thinking:
            prompt = (
                f"Subject: {subject.replace('_', ' ').title()}\n"
                f"Question: {q_text}\n\n{options_str}\n\n"
                "Let's think step by step and then output the correct option letter in the format: 'Final Answer: (X)'.\n"
                "Answer:"
            )
        else:
            prompt = (
                f"Subject: {subject.replace('_', ' ').title()}\n"
                f"Question: {q_text}\n\n{options_str}\n\n"
                "Answer with only the correct option letter in the format: 'Final Answer: (X)'.\n"
                "Answer:"
            )
        shots = few_shot.get(subject)
        if shots:
            prompt = (f"The following are multiple choice questions (with answers) about "
                      f"{subject.replace('_', ' ')}.\n\n{shots}{prompt}")
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_letter, {"category": subject}))
    return samples


def _eval_mmlu(response_text: str, gold_letter: str) -> bool:
    """Check if predicted answer matches gold MMLU letter (A-D)."""
    text = response_text.strip().upper()
    # Anchored forms first.
    for pat in (r"FINAL\s+ANSWER:\s*\(?([ABCD])\)?", r"ANSWER:\s*\(?([ABCD])\)?",
                r"\\BOXED\{\(?([ABCD])\)?\}"):
        m = re.findall(pat, text)
        if m:
            return m[-1] == gold_letter
    # Whole response is just the letter.
    if text.strip().strip("().") in ("A", "B", "C", "D"):
        return text.strip().strip("().") == gold_letter
    # CC7: fall back to the LAST standalone option letter, not the FIRST. `^\(?([ABCD])`
    # matched the leading "A" of a sentence like "A careful analysis shows C", and
    # `\b([ABCD])\b` took the first such letter anywhere - both credited wrong answers.
    standalone = re.findall(r"\b([ABCD])\b", text)
    if standalone:
        return standalone[-1] == gold_letter
    return False


def run_mmlu(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native MMLU evaluation suite (canonical 5-shot by default; `--eval-n-shot N` overrides, 0 = 0-shot)."""
    _ns = kwargs.get("eval_n_shot")
    if _ns is None:
        _ns = kwargs.get("n_shot")
    n_shot = 5 if _ns is None else int(_ns)   # canonical 5-shot default; explicit 0 respected
    samples = _load_mmlu_samples(enable_thinking=enable_thinking, limit=kwargs.get("limit"),
                                 n_shot=n_shot)
    return run_eval_suite(
        eval_name="mmlu",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_mmlu,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
