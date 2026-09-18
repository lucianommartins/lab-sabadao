# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: cruxeval
# Description: CRUXEval (MIT/Meta Code Reasoning & I/O Execution Simulation Benchmark)

"""gbench native built-in runner for cruxeval (Code Reasoning & Execution).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_CRUXEVAL_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import ast
import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Code Reasoning & Execution"

# Canonical CRUXEval-O (output prediction) prompt (cruxeval-org/cruxeval prompts.py): the
# model completes `assert f(input) == ??` inside [ANSWER]/[/ANSWER] tags, primed by one
# worked example. The prior free-text "What does f(input) return? Final Answer:" template
# diverged from the published protocol.
_CRUXEVAL_INSTRUCTIONS = (
    "You are given a Python function and an assert statement containing an input to the "
    "function. Complete the assert statement with a literal (no unsimplified expressions, no "
    "function calls) containing the output when executing the provided code on the given "
    "input, even if the function is incorrect or incomplete. Do NOT output any extra "
    "information."
)
_CRUXEVAL_ONESHOT = (
    "[PYTHON]\ndef f(n):\n    return n\nassert f(17) == 17\n[/PYTHON]\n"
    "[ANSWER]\nassert f(17) == 17\n[/ANSWER]"
)


def _make_cruxeval_prompt(code: str, inp: str, thinking: bool) -> str:
    task = f"[PYTHON]\n{str(code).strip()}\nassert f({str(inp).strip()}) == ??\n[/PYTHON]\n"
    if thinking:
        head = (_CRUXEVAL_INSTRUCTIONS + " Think step by step, then surround the completed "
                "assert statement with [ANSWER] and [/ANSWER] tags.")
        return f"{head}\n\n{_CRUXEVAL_ONESHOT}\n\n{task}"
    # direct: prime with the example and open the [ANSWER] tag for immediate completion.
    head = (_CRUXEVAL_INSTRUCTIONS + " Provide the full assert statement with the correct "
            "output in [ANSWER] and [/ANSWER] tags, following the example.")
    return f"{head}\n\n{_CRUXEVAL_ONESHOT}\n\n{task}[ANSWER]\n"


def _load_cruxeval_samples(
    limit: Optional[int] = None,
    enable_thinking: bool = False,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load CRUXEval from HF Hub (cruxeval-org/cruxeval); raises on load/schema failure (no fabricated fallback)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset('cruxeval-org/cruxeval', split='test')
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for cruxeval: {e}")
        raise RuntimeError(f"Could not load dataset for cruxeval: {e}") from e

    if not rows:
        raise RuntimeError(f"Dataset for cruxeval returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="cruxeval")

    samples = []
    for item in rows:
        # cruxeval-org/cruxeval columns: 'code' (a function named f), 'input', 'output'.
        # This is the canonical CRUXEval-O (output prediction) task: given code + input,
        # predict the exact output of f(input).
        code = item.get("code")
        inp = item.get("input")
        out = item.get("output")
        if code is None or inp is None or out is None:
            raise RuntimeError(
                "cruxeval: unexpected dataset schema (missing 'code'/'input'/'output'); "
                "refusing to fabricate sample data"
            )
        prompt = _make_cruxeval_prompt(code, inp, enable_thinking)
        gold = str(out).strip()
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold, {"category": "output_prediction"}))

    logger.info(f"Loaded {len(samples)} cruxeval samples.")
    return samples


def _extract_cruxeval_answer(response_text: str) -> Optional[str]:
    """The literal the model states as its answer, in the canonical CRUXEval format.

    Order: the last ``[ANSWER]...[/ANSWER]`` block (canonical) -> \\boxed{} -> a "Final
    Answer:"/"Output:"/"==>" anchor -> the last non-empty line. Inside an [ANSWER] block the
    RHS of ``assert f(...) == <literal>`` is the answer. Backticks/fencing are stripped;
    quotes are NOT: the gold is a Python literal, so `'0'` (str) and `0` (int) must not be
    conflated.
    """
    if not response_text:
        return None
    resp = response_text.strip()

    # Canonical: the content of the LAST [ANSWER] block (closing tag optional).
    blocks = re.findall(r"\[ANSWER\](.*?)\[/ANSWER\]", resp, re.DOTALL | re.IGNORECASE)
    block = blocks[-1] if blocks else None
    if block is None and re.search(r"\[ANSWER\]", resp, re.IGNORECASE):
        block = re.split(r"\[ANSWER\]", resp, flags=re.IGNORECASE)[-1]
    if block is not None:
        assert_rhs = re.search(r"assert\s+f\(.*?\)\s*==\s*(.+)", block)
        candidate = assert_rhs.group(1) if assert_rhs else block
        return candidate.strip().rstrip(";").strip().strip("`").strip()

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", resp)
    if boxed:
        return boxed[-1].strip().strip("`").strip()

    anchored = re.findall(r"(?:Final Answer|Output|==>)\s*:?\s*(.+)", resp, re.IGNORECASE)
    if anchored:
        return anchored[-1].strip().strip("`").strip()

    lines = [line.strip() for line in resp.splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1]
    if last.startswith("```"):                      # a bare fenced block: use its contents
        body = [line for line in lines if not line.startswith("```")]
        last = body[-1] if body else last
    return last.strip("`").strip()


def _eval_cruxeval(response_text: str, gold_target: str) -> bool:
    """Compare the model's stated output literal with the gold literal.

    Two leniencies are deliberately gone (audit 3A):
      * bare containment (`gold in resp`) credited any response that happened to include a
        short literal - golds such as `0`, `True` or `[]` occur in almost every response;
      * case-insensitive matching, which equates the Python literals `true` and `True`.
    Equality is either textual (exact, after fencing is stripped) or, when both sides parse
    as Python literals, structural - so `[1, 2]` and `[1,2]` agree but `'0'` and `0` do not.
    """
    gold = str(gold_target).strip()
    if not gold or not response_text:
        return False

    pred = _extract_cruxeval_answer(response_text)
    if pred is None:
        return False
    if pred == gold:
        return True

    try:
        gold_value = ast.literal_eval(gold)
        pred_value = ast.literal_eval(pred)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return False
    # bool is an int subclass in Python: True must not equal 1.
    if isinstance(gold_value, bool) != isinstance(pred_value, bool):
        return False
    return type(gold_value) is type(pred_value) and gold_value == pred_value


def run_cruxeval(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute cruxeval native built-in evaluation suite."""
    samples = _load_cruxeval_samples(limit=kwargs.get("limit"), enable_thinking=enable_thinking)
    return run_eval_suite(
        eval_name="cruxeval",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_cruxeval,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 2048),
    )
