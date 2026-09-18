# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: nexus_function_calling
# Description: NexusRaven API Evaluation (single-call function calling, AST match)
# NOTE: Nexusflow/NexusRaven_API_evaluation is licensed CC-BY-NC-4.0 (non-commercial).

"""gbench native built-in runner for nexus_function_calling (Tool Use & Function Calling).

Canonical NexusRaven single-call benchmark: each query is presented with the
Python function signatures available to it (rendered from the API list), and the
model must emit ONE Python function call. Scored by AST-equivalence (function
name + args, order-independent, falsy args dropped) against the gold call.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_NEXUS_FUNCTION_CALLING_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"

_REPO = "Nexusflow/NexusRaven_API_evaluation"


def _render_tool(api_row: Dict[str, Any]) -> str:
    """Render one API definition as a Python signature + docstring (Raven format)."""
    name = api_row["name"]
    args = api_row.get("args_dicts") or []
    params, arg_docs = [], []
    for a in args:
        pname = a.get("name")
        ptype = a.get("type") or "None"
        required = a.get("required")
        default = a.get("default")
        sig = pname
        if ptype and ptype != "None":
            sig += f": {ptype}"
        if not required:
            d = default if (default not in (None, "", "None")) else "None"
            sig += f" = {d}"
        params.append(sig)
        arg_docs.append(f"- {pname} ({ptype}): {a.get('description', '')}")
    doc = api_row.get("description", "")
    if arg_docs:
        doc += "\n" + "\n".join(arg_docs)
    return f'Function:\ndef {name}({", ".join(params)}):\n"""\n{doc}\n"""\n'


def _load_nexus_function_calling_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load NexusRaven standardized queries + API list; raises on load/schema failure."""
    try:
        from datasets import load_dataset
        sq = list(load_dataset(_REPO, "standardized_queries", split="train"))
        al = list(load_dataset(_REPO, "standardized_api_list", split="train"))
    except Exception as e:
        logger.error(f"Failed to load dataset for nexus_function_calling: {e}")
        raise RuntimeError(
            f"Could not load dataset for nexus_function_calling: {e}"
        ) from e

    if not sq or not al:
        raise RuntimeError("nexus_function_calling returned empty rows")

    api_index = {(r["dataset"], r["name"]): r for r in al}

    samples = []
    for row in sq:
        dataset = row["dataset"]
        query = row["prompt"]
        gold_name = row["python_function_name"]
        gold_args = json.loads(row["python_args_dict"])
        context_fns = row.get("context_functions") or []
        if not query or not gold_name or not context_fns:
            raise RuntimeError(
                "nexus_function_calling: unexpected dataset schema; "
                "refusing to fabricate sample data"
            )

        tool_blocks = []
        for fn in context_fns:
            api_row = api_index.get((dataset, fn))
            if api_row is None:
                raise RuntimeError(
                    f"nexus_function_calling: unresolved function '{fn}' in '{dataset}'"
                )
            tool_blocks.append(_render_tool(api_row))
        tools_block = "\n".join(tool_blocks)

        gold_row = api_index.get((dataset, gold_name))
        gold_params = [a.get("name") for a in (gold_row.get("args_dicts") or [])] if gold_row else []

        content = (
            f"{tools_block}\nUser Query: {query}\n\n"
            "Respond with a single Python function call and nothing else."
        )
        gold = json.dumps({"name": gold_name, "args": gold_args, "params": gold_params})
        messages = [{"role": "user", "content": content}]
        samples.append((messages, gold, {"category": dataset}))

    # Stratified on each sample's own category, not a contiguous head (RC-1).
    samples = stratified_sample(
        samples, limit, lambda s: (s[2] or {}).get("category"), seed="nexus_function_calling")
    logger.info(f"Loaded {len(samples)} nexus_function_calling samples from HF Hub.")
    return samples


def _extract_call(text: str) -> Optional[str]:
    """Extract the Python call string from the model output (canonical Raven rule)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^Call:\s*", "", t)
    t = t.replace("```python", "").replace("```", "")
    m = re.search(r"\w+\(.*\)", t, re.DOTALL)
    return m.group(0) if m else None


def _eval_nexus_function_calling(response_text: str, gold: str) -> bool:
    """AST-equivalence match: function name + args (order-independent, falsy dropped)."""
    try:
        g = json.loads(gold)
        gname = g["name"]
        gargs = {k: v for k, v in g["args"].items() if v}
        gparams = g.get("params") or []
    except Exception:
        return False

    call = _extract_call(response_text)
    if not call:
        return False
    try:
        tree = ast.parse(call, mode="eval")
    except Exception:
        return False

    node = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            node = n
            break
    if node is None or not isinstance(node.func, ast.Name):
        return False

    pname = node.func.id
    pargs: Dict[str, Any] = {}
    try:
        for idx, a in enumerate(node.args):  # positional -> param name via signature order
            if idx < len(gparams):
                pargs[gparams[idx]] = ast.literal_eval(a)
        for kw in node.keywords:
            if kw.arg is None:
                continue
            pargs[kw.arg] = ast.literal_eval(kw.value)
    except Exception:
        return False
    pargs = {k: v for k, v in pargs.items() if v}
    return pname == gname and pargs == gargs


def run_nexus_function_calling(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute nexus_function_calling native built-in evaluation suite."""
    samples = _load_nexus_function_calling_samples(limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="nexus_function_calling",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_nexus_function_calling,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 1024),
    )
