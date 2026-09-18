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

"""Native Berkeley Function Calling Leaderboard (BFCL) evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_BFCL_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .fc_common import (parse_tool_calls, parse_raw_tool_calls, score_possible_answer,
                        score_tool_call, normalize_tool_name)

logger = logging.getLogger(__name__)


def _extract_user_text(q_obj: Any) -> str:
    """The actual USER question from a BFCL `question` field.

    BFCL nests turns as a list of message lists. Taking ``q_obj[0][0].content`` grabbed the
    FIRST message of the first turn regardless of role, so on live rows whose turn 0 is
    [system, user] it fed the model the SYSTEM prompt and discarded the user's question.
    """
    text = ""
    if isinstance(q_obj, list):
        for turn in q_obj:
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        text += (msg.get("content") or "") + "\n"
            elif isinstance(turn, dict) and turn.get("role") == "user":
                text += (turn.get("content") or "") + "\n"
            elif isinstance(turn, str):
                text += turn + "\n"
    elif isinstance(q_obj, str):
        text = q_obj
    return text.strip()

SUPPORTED_BFCL_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "java",
    "javascript",
    "sql",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]


def _load_bfcl_samples(categories: Optional[str]) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load BFCL train/test samples directly from canonical Gorilla HF Hub repository."""
    import json
    from huggingface_hub import hf_hub_download

    cats_to_load = (
        SUPPORTED_BFCL_CATEGORIES
        if not categories or categories == "all"
        else [c.strip() for c in categories.split(",") if c.strip()]
    )

    raw_samples = []
    for cat in cats_to_load:
        try:
            fpath_q = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"BFCL_v3_{cat}.json", repo_type="dataset")
            fpath_a = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset")

            ans_map = {}
            with open(fpath_a, "r", encoding="utf-8") as fa:
                for line in fa:
                    if line.strip():
                        obj = json.loads(line)
                        ans_map[obj.get("id")] = obj.get("ground_truth")

            with open(fpath_q, "r", encoding="utf-8") as fq:
                for line in fq:
                    if line.strip():
                        item = json.loads(line)
                        item["ground_truth"] = ans_map.get(item.get("id"))
                        item["test_category"] = cat
                        raw_samples.append(item)
        except Exception as cat_err:
            logger.warning(f"Could not load BFCL category '{cat}': {cat_err}")

    logger.info(f"Loaded {len(raw_samples)} BFCL samples across {len(cats_to_load)} categories from HF Hub.")

    allowed_cats = set(categories.split(",")) if categories and categories != "all" else None

    samples = []
    for item in raw_samples:
        cat = item.get("test_category", "simple_python")
        if allowed_cats and cat not in allowed_cats:
            continue

        user_content = _extract_user_text(item["question"]) or str(item["question"])

        tools = []
        for fn in item.get("function", []):
            tools.append({"type": "function", "function": fn})

        messages = [{"role": "user", "content": user_content}]
        gold_call = item.get("ground_truth")

        samples.append((messages, gold_call, {"tools": tools} if tools else {}))
    return samples


def _eval_bfcl(response_text: str, gold_call: Any, tool_calls: Any = None) -> bool:
    """Structurally match the emitted call against BFCL's `possible_answer` gold.

    Gold is a list of {func_name: {param: [accepted values]}}. Previously this only asked
    whether the function name and one accepted value appeared ANYWHERE in the response
    text, so a model could pass by echoing the prompt's own API list without emitting a
    call, and wrong arguments elsewhere in the prose still counted.

    `tool_calls` is the STRUCTURED OpenAI tool-call array base.py captured for this reply.
    When present we score off it (parse_raw_tool_calls) instead of re-parsing base.py's lossy
    text render `name({json}) name(a=1)` -- that round-trip drops spaces in unquoted values,
    splits commas inside unquoted strings, truncates namespaced function names, and swallows
    the JSON object as a positional arg, scoring correct calls as wrong. Falls back to the text
    parse when no structured calls are available (offline re-scoring of old traces).
    """
    if not gold_call:
        return False
    structured = parse_raw_tool_calls(tool_calls) if tool_calls else None
    if structured is None and not response_text:
        return False
    specs = gold_call if isinstance(gold_call, list) else (
        [gold_call] if isinstance(gold_call, dict) else None)
    if specs is None:
        return score_tool_call(response_text, gold_call, require_args=True)
    # Structural possible_answer match (name + accepted values + no unexpected params), scored
    # off the clean structured calls when available.
    if not score_possible_answer(response_text, specs, calls=structured):
        return False
    # Bijective at the function level: reject hallucinated extra calls to functions outside
    # the gold set, so parallel/multiple must match the gold call set rather than a superset.
    gold_names = {normalize_tool_name(fn) for spec in specs
                  if isinstance(spec, dict) for fn in spec}
    called = structured if structured is not None else parse_tool_calls(response_text)
    called_names = {normalize_tool_name(n) for n, _ in called}
    return called_names <= gold_names

def run_bfcl(
    model_name: str,
    base_url: str,
    concurrency: int,
    eval_categories: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native BFCL evaluation suite."""
    samples = _load_bfcl_samples(eval_categories)
    return run_eval_suite(
        eval_name="bfcl",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_bfcl,
        thinking=kwargs.get("enable_thinking", False),
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
