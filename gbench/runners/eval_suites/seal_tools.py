# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: seal_tools
# Description: Seal-Tools (NLPCC Multi-Tool Single-Turn and Nested Function Calling Benchmark)

"""gbench native built-in runner for seal_tools (Tool Use & Function Calling).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SEAL_TOOLS_TEMPERATURE`,
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
from .fc_common import (gold_call_dicts, parse_tool_calls, score_exact_call_set,
                        tool_items, param_items, prf_counts, prf_from_counts)
from .metrics import scorable_traces

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"


def _load_seal_tools_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load seal_tools benchmark dataset directly from HF Hub (seal-tools/seal-tools)."""
    rows = []
    try:
        from datasets import load_dataset
        # The TEST split is the evaluation set (1,354 rows). The train split (12,022) is
        # training data, and its `domain` column is the literal string "train" for every
        # row - so scoring it also produced a single meaningless category. On test,
        # `domain` is in-domain / out-domain (700 / 654), the dimension the benchmark
        # actually reports.
        ds = load_dataset('casey-martin/Seal-Tools', split='test')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for seal_tools: {e}")
        raise RuntimeError(f"Could not load dataset for seal_tools: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for seal_tools returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("domain"), seed="seal_tools")

    samples = []
    for item in rows:
        convs = item.get("conversations") or []
        prompt = ""
        gold = ""
        for c in convs:
            role = str(c.get("from", "")).lower()
            val = str(c.get("value", "")).strip()
            if role in ("human", "user") and not prompt:
                prompt = val
            elif role in ("gpt", "assistant") and not gold:
                gold = val

        cat = str(item.get("domain") or "tool_use")
        messages = [{"role": "user", "content": prompt or "Generate tool calls."}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} seal_tools samples.")
    return samples


def _eval_seal_tools(response_text: str, gold_target: str) -> bool:
    """Binary pass = the WHOLE gold call set is reproduced exactly (secondary pass_rate).

    The old scorer parsed only the FIRST gold call (parse_gold_call), so on the ~78% of
    Seal-Tools rows that are multi-call it ignored every call after the first. This scores
    the entire set; the headline metric (micro Tool/Parameter F1) is computed in run_seal_tools.
    """
    if not response_text or not str(gold_target).strip():
        return False
    gold_calls = gold_call_dicts(gold_target)
    if not gold_calls:
        return False
    return bool(score_exact_call_set(gold_calls, parse_tool_calls(response_text))["exact"])


def run_seal_tools(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute seal_tools; headline is micro Parameter-level F1 (canonical Seal-Tools)."""
    samples = _load_seal_tools_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="seal_tools",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_seal_tools,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
    # Canonical Seal-Tools reports Format ACC + micro Tool P/R/F1 + micro Parameter P/R/F1,
    # pooled across the whole set - not a per-sample all-or-nothing boolean. Pool per-sample
    # (tp, fp, fn) counts over the traces (P/R/F1 are corpus-level, not averageable).
    tool = [0, 0, 0]
    param = [0, 0, 0]
    fmt_ok = n = 0
    for t in scorable_traces(result):
        gold_calls = gold_call_dicts(t.get("gold_answer"))
        pred_calls = parse_tool_calls(t.get("response_text") or "")
        tc = prf_counts(tool_items(gold_calls), tool_items(pred_calls))
        pc = prf_counts(param_items(gold_calls), param_items(pred_calls))
        t["seal_tool_prf"] = tc
        t["seal_param_prf"] = pc
        for i in range(3):
            tool[i] += tc[i]
            param[i] += pc[i]
        fmt_ok += 1 if pred_calls else 0
        n += 1
    tool_prf = prf_from_counts(*tool)
    param_prf = prf_from_counts(*param)
    result["exact_match_rate"] = result.get("accuracy")
    result["format_accuracy"] = round(fmt_ok / n * 100.0, 2) if n else 0.0
    for name, prf in (("tool", tool_prf), ("parameter", param_prf)):
        result[f"{name}_precision"] = round(prf["precision"] * 100.0, 2)
        result[f"{name}_recall"] = round(prf["recall"] * 100.0, 2)
        result[f"{name}_f1"] = round(prf["f1"] * 100.0, 2)
    result["metric"] = ("micro Parameter-level F1 (canonical Seal-Tools headline); "
                        "tool_f1 / format_accuracy / exact_match_rate also reported")
    if n:
        result["accuracy"] = round(param_prf["f1"] * 100.0, 2)
    return result
