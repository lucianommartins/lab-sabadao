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

"""Native GPQA Diamond zero-shot evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_GPQA_DIAMOND_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import hashlib
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

PROMPT_FOOTER = (
    "Try to reason about the question step by step. Don't give a final"
    " answer without reasoning. Output the final answer in the format"
    " 'Final Answer: (X)' where X is the correct letter choice. Answer:"
)
OPTION_LETTERS = "ABCD"

def _shuffled_options(options, key: str):
    """Deterministic per-item shuffle of the answer choices.

    `sorted(options)` put the choices in alphabetical order, so the correct answer landed
    in a position determined by its own text - and, because the distractors are usually
    shorter or numerically smaller, disproportionately often in the same slot. Canonical
    GPQA shuffles per item; seeding on the question keeps that reproducible across runs.
    """
    order = list(range(len(options)))
    random.Random(hashlib.sha256(key.encode("utf-8")).hexdigest()).shuffle(order)
    return [options[i] for i in order]


def _load_gpqa_samples(enable_thinking: bool, limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load GPQA Diamond samples directly from canonical HF Hub dataset ('Idavidrein/gpqa')."""
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    # `limit > 0` matters: `--eval-limit 0` means "no limit" everywhere else, but here it
    # took `range(0)` and produced an empty suite that then reported 0 questions.
    # Stratified, not a contiguous head: the rows are stored grouped by
    # `Subdomain`, so `[:limit]` returned a single category (audit RC-1).
    ds = limit_dataset(ds, limit, "Subdomain", seed="gpqa_diamond")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} GPQA Diamond samples from HF Hub ('Idavidrein/gpqa').")

    samples = []
    for item in raw_samples:
        question = item["Question"]
        options = [
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
            item["Correct Answer"],
        ]
        sorted_options = _shuffled_options(options, str(question))
        gold_idx = sorted_options.index(item["Correct Answer"])
        gold_letter = OPTION_LETTERS[gold_idx]

        options_text = "\n".join(
            f"({letter}) {opt}" for letter, opt in zip(OPTION_LETTERS, sorted_options)
        )
        prompt = f"Question: {question}\n\n{options_text}\n\n{PROMPT_FOOTER}"
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_letter, {"category": str(item.get("Subdomain", item.get("High-level domain", "")))}))
    return samples


def _eval_gpqa(response_text: str, gold_letter: str) -> bool:
    """Extract the answer letter from the model's response and check match."""
    response = response_text.strip()
    choices = ["A", "B", "C", "D"]

    # Direct single letter
    if response.upper() in choices:
        pred = response.upper()
    else:
        # "Final Answer: (X)" or "Final Answer: X" pattern - take last match
        matches = re.findall(r"(?i)final\s+answer\s*[:=]\s*\(?([A-D])\)?", response)
        if matches:
            pred = matches[-1].upper()
        else:
            # "Answer: X" or "Answer: (X)" pattern - take last
            matches = re.findall(r"(?i)answer\s*[:=]\s*\(?([A-D])\)?", response)
            if matches:
                pred = matches[-1].upper()
            else:
                # (X) pattern - take last
                matches = re.findall(r"\(([A-D])\)", response)
                if matches:
                    pred = matches[-1].upper()
                else:
                    # Standalone letter on last lines
                    pred = ""
                    for line in reversed(response.split("\n")):
                        line = line.strip().rstrip(".)")
                        if line.upper() in choices:
                            pred = line.upper()
                            break

    return pred == gold_letter.upper()


def run_gpqa_diamond(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native GPQA Diamond evaluation suite."""
    samples = _load_gpqa_samples(enable_thinking, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="gpqa_diamond",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_gpqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
