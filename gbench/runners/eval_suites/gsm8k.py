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

"""Native GSM8K math reasoning evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_GSM8K_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .extraction_common import final_number

logger = logging.getLogger(__name__)


def _extract_gold_num(gold_str: str) -> str:
    """Extract gold number after '#### ' in GSM8K answer."""
    if "#### " in gold_str:
        return gold_str.split("#### ")[-1].strip().replace(",", "")
    return re.sub(r"[^\d.-]", "", gold_str)


# Canonical GSM8K 8-shot chain-of-thought exemplars (Wei et al. 2022, arXiv:2201.11903), the fixed
# set pinned verbatim in EleutherAI lm-evaluation-harness `gsm8k_cot` (gsm8k-cot.yaml). Stored as
# (question, cot_reasoning, answer). The load-bearing canonical piece is the exemplar CONTENT; a
# prior version drew the exemplars from the first 8 TRAIN rows (a valid 8-shot CoT, but not THE
# canonical published set, and formatting/content measurably move GSM8K accuracy).
_GSM8K_COT_EXEMPLARS: List[Tuple[str, str, str]] = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they "
     "are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there "
     "must have been 21 - 15 = 6.", "6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking "
     "lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.", "5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left "
     "in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After "
     "eating 35, they had 74 - 35 = 39.", "39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many "
     "lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny "
     "20 - 12 = 8.", "8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys "
     "does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more "
     "toys. 5 + 4 = 9.", "9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from "
     "monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = "
     "20 computers were added. 9 + 20 is 29.", "29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How "
     "many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After "
     "losing 2 more, he had 35 - 2 = 33 golf balls.", "33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has "
     "23 - 15 dollars left. 23 - 15 is 8.", "8"),
]


def _gsm8k_fewshot_prefix(n_shot: int) -> str:
    """GSM8K chain-of-thought few-shot prefix from the canonical fixed 8 exemplars.

    Uses the fixed Wei et al. 2022 exemplars (pinned in lm-eval-harness `gsm8k_cot`), formatted in
    gbench's own "Question: ... / Final Answer: N" protocol (level-a fidelity: canonical exemplar
    CONTENT + gbench answer protocol and tolerant extractor). Canonical GSM8K is 8-shot CoT; `n_shot`
    selects how many of the fixed exemplars to use (0 = 0-shot). Only 8 canonical exemplars exist, so
    a larger request is capped at 8."""
    if n_shot <= 0:
        return ""
    if n_shot > len(_GSM8K_COT_EXEMPLARS):
        logger.warning(
            "gsm8k: %d-shot requested but only %d canonical exemplars exist; using %d.",
            n_shot, len(_GSM8K_COT_EXEMPLARS), len(_GSM8K_COT_EXEMPLARS))
    parts = [
        f"Question: {q}\n\n{cot}\nFinal Answer: {a}"
        for q, cot, a in _GSM8K_COT_EXEMPLARS[:n_shot]
    ]
    return "\n\n".join(parts) + "\n\n"


def _load_gsm8k_samples(
    limit: Optional[int] = None, n_shot: int = 8,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load GSM8K test set directly from canonical HF Hub dataset ('openai/gsm8k').

    Canonical protocol is 8-shot CoT; `n_shot` selects the exemplar count (0 = 0-shot).
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, None, seed="gsm8k")
    raw_samples = list(ds)
    prefix = _gsm8k_fewshot_prefix(n_shot)
    logger.info(f"Loaded {len(raw_samples)} GSM8K samples from HF Hub ('openai/gsm8k'); {n_shot}-shot CoT.")

    samples = []
    for item in raw_samples:
        question = item["question"]
        gold_num = _extract_gold_num(item["answer"])

        prompt = (
            f"{prefix}"
            f"Question: {question}\n\n"
            "Let's think step by step and finish your answer with 'Final Answer: X' where X is the number."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_num, {}))
    return samples


def _eval_gsm8k(response_text: str, gold_num: str) -> bool:
    """Check the model's stated final number against the gold.

    Extraction is shared (audit CC7): \\boxed{} and every "Final Answer:"/"the answer is"
    phrasing are honoured before the last-number fallback, and the LAST anchor wins. The
    previous version recognised one exact phrasing, so a boxed answer - or a second,
    corrected "Final Answer:" line - fell through to the bare last number in the response.
    """
    pred = final_number(response_text)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold_num)) < 1e-4
    except ValueError:
        return pred == str(gold_num).strip()


def run_gsm8k(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native GSM8K math reasoning evaluation suite (canonical 8-shot CoT; `--eval-n-shot N` overrides, 0 = 0-shot)."""
    _ns = kwargs.get("eval_n_shot")
    if _ns is None:
        _ns = kwargs.get("n_shot")
    n_shot = 8 if _ns is None else int(_ns)
    samples = _load_gsm8k_samples(limit=kwargs.get("limit"), n_shot=n_shot)
    result = run_eval_suite(
        eval_name="gsm8k",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_gsm8k,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    # Level-a fidelity: the canonical fixed 8-shot CoT exemplars are pinned, but gbench keeps its own
    # "Final Answer:" answer protocol (not lm-eval gsm8k_cot's "Q:/A:" + "The answer is N." + stop
    # "Q:"), and a thinking run deviates from the published greedy protocol - so a run here is not
    # strictly like-for-like with published gsm8k_cot numbers.
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "canonical fixed 8-shot CoT exemplars, but gbench's own 'Final Answer:' answer protocol "
        "(not lm-eval gsm8k_cot's 'Q:/A:' + 'The answer is N.' + stop 'Q:'); not strictly comparable "
        "to published gsm8k_cot numbers")
    return result
