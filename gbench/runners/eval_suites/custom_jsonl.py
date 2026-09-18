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

"""Generic JSONL Evaluation Suite runner for zero-code custom benchmark evaluation."""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from .base import run_eval_suite

logger = logging.getLogger(__name__)


def _load_custom_jsonl_samples(
    jsonl_path: Union[str, Path],
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load evaluation samples from a user-provided JSONL file."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom JSONL evaluation dataset not found: {path}")

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # 1. Format messages
            if "messages" in item and isinstance(item["messages"], list):
                messages = item["messages"]
            elif "prompt" in item:
                messages = [{"role": "user", "content": item["prompt"]}]
            elif "question" in item:
                messages = [{"role": "user", "content": item["question"]}]
            elif "input" in item:
                messages = [{"role": "user", "content": item["input"]}]
            else:
                logger.warning(f"Sample at line {idx+1} missing prompt/messages. Skipping.")
                continue

            # 2. Extract gold answer & eval_type.
            # First PRESENT key wins, by presence not truthiness: an `or` chain dropped a
            # legitimately falsy gold (0, 0.0, "", False) through to the next key and left
            # the row with gold=None, i.e. unscorable (always wrong).
            gold = None
            for _gk in ("gold", "answer", "gold_answer", "target"):
                if _gk in item:
                    gold = item[_gk]
                    break
            eval_type = item.get("eval_type", "contains").lower().strip()
            category = str(item.get("category") or item.get("domain") or "default")

            extra = {
                "eval_type": eval_type,
                "category": category,
            }
            # The scorer only receives (response, gold), so a per-row `eval_type` recorded
            # in the sample metadata never reached it: every row was graded with every
            # strategy at once, and `"eval_type": "exact"` still passed on containment.
            samples.append((messages, {_ET_KEY: eval_type, "gold": gold}, extra))
            if limit is not None and len(samples) >= limit:
                break

    logger.info(f"Loaded {len(samples)} custom evaluation samples from {path.name}")
    return samples


_ET_KEY = "__gbench_eval_type"

#: `eval_type` values a row may declare. `contains` stays the default so existing files
#: score exactly as before; the others narrow the check to what the row asked for.
EVAL_TYPES = ("contains", "exact", "numeric", "multiple_choice")


def _numeric_match(resp: str, gold_str: str) -> bool:
    try:
        gold_num = float(re.sub(r"[^\d.eE+-]", "", gold_str))
    except (ValueError, TypeError):
        return False
    for found in re.findall(r"[-+]?\d*\.?\d+", resp):
        try:
            if abs(float(found) - gold_num) < 1e-4:
                return True
        except ValueError:
            continue
    return False


def _mc_match(resp: str, gold_str: str) -> bool:
    if not (len(gold_str) == 1 and gold_str.isalpha()):
        return False
    from .extraction_common import last_mc_letter
    picked = last_mc_letter(resp, letters="ABCDEFGHIJ")
    return bool(picked) and picked.lower() == gold_str.lower()


def _eval_custom_jsonl(response_text: str, gold_answer: Any) -> bool:
    """Score one row with the strategy the row declares in `eval_type`.

    Every strategy used to be tried on every row and any hit passed, so a row marked
    `"eval_type": "exact"` still passed on a substring, and a short gold such as `4` could
    be matched by an unrelated number anywhere in the response.
    """
    eval_type = "contains"
    if isinstance(gold_answer, dict) and _ET_KEY in gold_answer:
        eval_type = str(gold_answer.get(_ET_KEY) or "contains").lower().strip()
        gold_answer = gold_answer.get("gold")

    if not response_text or gold_answer is None:
        return False
    resp = response_text.strip()
    gold_str = str(gold_answer).strip()
    if not gold_str:
        return False

    if eval_type == "exact":
        return resp.lower() == gold_str.lower()
    if eval_type == "numeric":
        return _numeric_match(resp, gold_str)
    if eval_type in ("multiple_choice", "mc", "letter"):
        return _mc_match(resp, gold_str)

    # Default `contains`: exact, then containment, then the type-appropriate check.
    if resp.lower() == gold_str.lower() or gold_str.lower() in resp.lower():
        return True
    if _mc_match(resp, gold_str):
        return True
    return _numeric_match(resp, gold_str)


def run_custom_jsonl(
    jsonl_path: Union[str, Path],
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native evaluation on a user-provided JSONL benchmark file."""
    path = Path(jsonl_path)
    eval_name = path.stem.lower()
    samples = _load_custom_jsonl_samples(path, limit=kwargs.get("limit"))

    return run_eval_suite(
        eval_name=eval_name,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_custom_jsonl,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
