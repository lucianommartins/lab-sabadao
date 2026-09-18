# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: gorilla_apibench
# Description: Gorilla APIBench (UC Berkeley API Selection Benchmark) - HuggingFace-hub subset only (1 of the 3 API hubs; TorchHub/TensorFlowHub not loaded)

"""gbench native built-in runner for gorilla_apibench (Tool Use & Function Calling).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_GORILLA_APIBENCH_TEMPERATURE`,
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

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"


def _load_gorilla_apibench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load gorilla_apibench benchmark dataset directly from HF Hub (gorilla-llm/APIBench)."""
    rows = []
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="gorilla-llm/APIBench",
            filename="huggingface_eval.json",
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load dataset for gorilla_apibench: {e}")
        raise RuntimeError(f"Could not load dataset for gorilla_apibench: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for gorilla_apibench returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("category"), seed="gorilla_apibench")

    samples = []
    for item in rows:
        code_text = item.get("code") or ""
        if "###Instruction:" in code_text:
            prompt = code_text.split("###Output:")[0].replace("###Instruction:", "").strip()
        else:
            prompt = code_text.strip()

        gold = str(item.get("api_call") or (item.get("api_data") or {}).get("api_name") or "").strip()
        if not prompt or not gold:
            raise RuntimeError(
                "gorilla_apibench: unexpected dataset schema (missing code/api_call); "
                "refusing to fabricate sample data"
            )
        cat = str((item.get("api_data") or {}).get("domain") or item.get("provider") or "huggingface")

        # APIBench is scored by AST-matching the generated **api_call** against the gold
        # call. The previous instruction, "Respond with: Tool: <model_name_or_api_call>",
        # invited a prose tool NAME - the 2026-08-15 sweep answered "Tool: OpenAI Whisper
        # API" against a gold of
        # `WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')` and
        # scored a structural 0/20. Ask for the call itself, in the canonical
        # `<<<api_call>>>` shape Gorilla uses.
        messages = [{"role": "user", "content": (
            f"Task: {prompt}\n\n"
            f"Write the single Python API call from the {cat} domain that performs this "
            "task. Give the call exactly as it would appear in code, including the model "
            "or checkpoint identifier and its arguments.\n\n"
            "Answer on one line in this form:\n"
            "<<<api_call>>>: AutoModel.from_pretrained('org/checkpoint-name')"
        )}]
        samples.append((messages, gold, {"category": cat}))

    logger.info(f"Loaded {len(samples)} gorilla_apibench samples.")
    return samples


def _eval_gorilla_apibench(response_text: str, gold_target: str) -> bool:
    """Evaluate candidate response against gold target: deterministic AST match on the gold
    API call, falling back to case-insensitive substring containment when the AST parse misses."""
    if not response_text:
        return False

    resp = response_text.strip()
    gold = str(gold_target).strip()

    if not gold:
        return False

    # Canonical APIBench does AST matching of the api_call. Compare the parsed call
    # (name + arguments) rather than counting how many gold tokens appear in the text:
    # a 60% token overlap is reachable by quoting the prompt back.
    from .fc_common import parse_gold_call, parse_tool_calls, call_matches
    g = parse_gold_call(gold)
    if g:
        if call_matches(g, parse_tool_calls(resp), require_args=bool(g[1])):
            return True
        # Textual fallback for a call our parser could not extract from the response.
        # It must still require the ARGUMENTS, not just the callable: the checkpoint id is
        # the answer in APIBench, so a name-only check credited
        # `...from_pretrained('openai/whisper-large')` against a gold of `whisper-tiny`.
        if g[0].lower() not in resp.lower():
            return False
        gold_args = [str(v) for v in (g[1] or {}).values() if str(v).strip()]
        return all(a.lower() in resp.lower() for a in gold_args)
    ident = re.findall(r"[A-Za-z0-9_\-/\.]{6,}", gold)
    return bool(ident) and all(t.lower() in resp.lower() for t in ident[:2])


def run_gorilla_apibench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute gorilla_apibench native built-in evaluation suite."""
    samples = _load_gorilla_apibench_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="gorilla_apibench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_gorilla_apibench,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
    # gbench runs only the HuggingFace API hub (1 of APIBench's 3 hubs: HF/Torch/TensorHub), so the
    # headline is a hub subset, not the full APIBench number.
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "HuggingFace hub only (1 of APIBench's 3 hubs); a subset, not the full benchmark")
    return result
