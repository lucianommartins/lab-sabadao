# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: bfcl_v3_live
# Description: BFCL v3 LIVE single-turn function calling + irrelevance abstention

"""gbench native built-in runner for bfcl_v3_live (Tool Use & Function Calling).

This suite loads the BFCL **v3 LIVE** single-turn subsets (live_simple, live_parallel,
live_multiple, live_parallel_multiple) plus live_irrelevance abstention - roughly the
"Live" 10% slice of the BFCL v4 leaderboard. It was previously mis-named
`bfcl_v4_agentic`, which implied the v4 agentic track (web search / memory / format
sensitivity); that track is now a separate suite, `bfcl_v4_agentic`.
Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_BFCL_V3_LIVE_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample
from .fc_common import parse_tool_calls, score_possible_answer, normalize_tool_name

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"


def _load_bfcl_v3_live_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load canonical Berkeley Function Calling Leaderboard (BFCL) dataset from HF Hub."""
    from huggingface_hub import hf_hub_download

    subsets = [
        ("BFCL_v3_live_simple.json", "possible_answer/BFCL_v3_live_simple.json", "live_simple"),
        ("BFCL_v3_live_parallel.json", "possible_answer/BFCL_v3_live_parallel.json", "live_parallel"),
        ("BFCL_v3_live_multiple.json", "possible_answer/BFCL_v3_live_multiple.json", "live_multiple"),
        ("BFCL_v3_live_parallel_multiple.json", "possible_answer/BFCL_v3_live_parallel_multiple.json", "live_parallel_multiple"),
        ("BFCL_v3_live_irrelevance.json", None, "live_irrelevance"),
    ]

    all_raw_samples = []
    for q_filename, a_filename, cat_name in subsets:
        try:
            q_path = hf_hub_download(
                repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                filename=q_filename,
                repo_type="dataset",
            )
            with open(q_path, "r", encoding="utf-8") as f:
                q_items = [json.loads(line) for line in f if line.strip()]

            a_items = []
            if a_filename:
                a_path = hf_hub_download(
                    repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                    filename=a_filename,
                    repo_type="dataset",
                )
                with open(a_path, "r", encoding="utf-8") as f:
                    a_items = [json.loads(line) for line in f if line.strip()]

            for idx, q_obj in enumerate(q_items):
                ans_obj = a_items[idx] if idx < len(a_items) else None
                all_raw_samples.append((q_obj, ans_obj, cat_name))
        except Exception as e:
            logger.warning(f"Could not load BFCL subset {q_filename}: {e}")

    if not all_raw_samples:
        raise RuntimeError("No BFCL samples loaded from gorilla-llm/Berkeley-Function-Calling-Leaderboard")

    # Stratified, not a contiguous head (audit RC-1).
    all_raw_samples = stratified_sample(all_raw_samples, limit, None, seed="bfcl_v3_live")

    samples = []
    for q_obj, ans_obj, cat_name in all_raw_samples:
        tools = q_obj.get("function") or []
        question_data = q_obj.get("question") or []

        # Extract user prompt from question
        user_text = ""
        if isinstance(question_data, list):
            for turn in question_data:
                if isinstance(turn, list):
                    for msg in turn:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            user_text += msg.get("content", "") + "\n"
                elif isinstance(turn, dict) and turn.get("role") == "user":
                    user_text += turn.get("content", "") + "\n"
                elif isinstance(turn, str):
                    user_text += turn + "\n"
        elif isinstance(question_data, str):
            user_text = question_data

        user_text = user_text.strip() or "Execute the appropriate tool for the request."

        # Canonical system prompt with available tool definitions
        tools_str = json.dumps(tools, indent=2) if tools else "[]"
        system_msg = (
            "You are an expert function calling assistant. You have access to the following tools:\n"
            f"{tools_str}\n\n"
            "If a function should be called, respond with the function call as JSON in the format:\n"
            '{"name": "function_name", "arguments": {"param1": "value1", ...}}\n'
            "If multiple functions should be called, respond with a JSON list of function calls:\n"
            '[{"name": "func1", "arguments": {...}}, {"name": "func2", "arguments": {...}}]\n'
            "If no function is suitable or needed to answer the user request, answer directly without calling any tools."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_text},
        ]

        ground_truth = (ans_obj.get("ground_truth") if isinstance(ans_obj, dict) else []) or []
        # Declare the tools on the REQUEST as well as rendering them into the system prompt.
        #
        # Measured against the live endpoint: with the schemas only in the prompt text, the
        # model correctly emits `[{"name": ..., "arguments": {...}}]`, but the server runs
        # `--enable-auto-tool-choice --tool-call-parser`, which recognises that shape and
        # lifts it out of `content`. With no `tools` declared on the request there is
        # nowhere for it to go, so it is DISCARDED: `completion_tokens=40, content_chars=0,
        # tool_calls=0`. That is the 12/20 "empty responses" on the 2026-08-15 sweep - the
        # model answered every time and the answer was thrown away.
        #
        # Declaring them routes the extraction into `tool_calls`, which the trace and
        # `_eval_bfcl_v3_live` already read.
        meta = {"category": cat_name}
        declared = [{"type": "function", "function": fn}
                    for fn in tools if isinstance(fn, dict) and fn.get("name")]
        if declared:
            meta["tools"] = declared
        samples.append((messages, ground_truth, meta))

    logger.info(f"Loaded {len(samples)} canonical BFCL v3 Live samples.")
    return samples


def _eval_bfcl_v3_live(response_text: str, gold_ground_truth: Any) -> bool:
    """Structural BFCL `possible_answer` check (was a lowercased substring scan).

    The old scorer credited a response whenever the gold function name and one accepted
    value appeared ANYWHERE in the text - true of a model that merely echoed the prompt's
    tool list without emitting a call. This parses the response into (name, args) tool calls
    and applies BFCL's possible_answer semantics.
    """
    if not response_text:
        return False
    calls = parse_tool_calls(response_text)

    # Irrelevance / abstention: gold is empty -> the model must NOT emit a tool call.
    if not gold_ground_truth:
        return len(calls) == 0
    if not isinstance(gold_ground_truth, list):
        return False

    # Every gold function must be called with an accepted value for each required parameter.
    if not score_possible_answer(response_text, gold_ground_truth):
        return False
    # Bijective at the function level: reject hallucinated extra calls to functions outside
    # the gold set, so a parallel/multiple answer must match the gold call set, not a
    # superset. (Name-set comparison is robust to base.py rendering one call as two tokens.)
    gold_names = {normalize_tool_name(fn) for spec in gold_ground_truth
                  if isinstance(spec, dict) for fn in spec}
    called_names = {normalize_tool_name(n) for n, _ in calls}
    return called_names <= gold_names


def run_bfcl_v3_live(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute bfcl_v3_live native built-in evaluation suite."""
    samples = _load_bfcl_v3_live_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="bfcl_v3_live",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_bfcl_v3_live,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
    # gbench runs only the BFCL "Live" single-turn slice (~10% of the BFCL leaderboard), so the
    # headline is the Live-category subscore, not the full BFCL overall number.
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "BFCL 'Live' single-turn slice only (~10% of BFCL); a category subscore, not BFCL overall")
    return result
