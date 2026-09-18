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

"""Native τ³-bench (tau3) Knowledge-Retrieval Agentic evaluation suite.

τ³-bench is the current generation of the sierra-research/tau2-bench line. Its defining new
*text* domain is `banking_knowledge`: a retrieval-augmented (RAG) customer-service domain over
a ~700-document knowledge base. gbench's `tau3` therefore runs the `banking_knowledge`
environment (was previously mislabelled as "telecom", which is actually a τ² domain already
covered by the `tau2` suite). See docs/evals/tau3.md.

The single-turn `_eval_tau3` / `_load_tau3_samples` helpers below are a legacy tool-call
diagnostic (telecom-derived) kept only for tests; they are unrelated to the banking_knowledge
environment path that produces the real score.

Sampling: this suite asks for temperature 1.0 (the reference tau2 config, 1.0/0.95/top_k 64). The global default is
1.0, the model's shipped `generation_config.json`. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_TAU3_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import json
import logging

from .sampling import stratified_sample
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from .swebench_common import infra_required
from .tau_common import env_requested, run_tau_env

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/tau3.md"

# Legacy single-turn diagnostic source (telecom); NOT the banking_knowledge env path.
SIERRA_TAU3_TELECOM_URL = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/telecom/tasks.json"


def _load_tau3_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]]:
    """Load canonical Sierra Tau Telecom scenarios."""
    try:
        req = urllib.request.Request(SIERRA_TAU3_TELECOM_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as res:
            raw_tasks = json.loads(res.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to load Sierra Tau Telecom tasks: {e}")
        raise RuntimeError(f"Could not load canonical Sierra Tau Telecom tasks: {e}") from e

    if not raw_tasks:
        raise RuntimeError("Sierra Tau Telecom returned 0 tasks.")

    # Stratified, not a contiguous head (audit RC-1).
    raw_tasks = stratified_sample(raw_tasks, limit, None, seed="tau3")

    samples = []
    for t in raw_tasks:
        scenario = t.get("user_scenario") or {}
        instructions = (scenario.get("instructions") or {}).get("task_instructions") or t.get("description") or ""
        eval_criteria = t.get("evaluation_criteria") or {}
        actions = eval_criteria.get("actions") or []
        assertions = eval_criteria.get("nl_assertions") or []

        prompt = (
            "You are an expert telecom customer service AI agent.\n\n"
            f"Customer Request & Context:\n{instructions}\n\n"
            "Respond by providing the appropriate tool calls or response to resolve the customer's request. "
            "Format tool invocations in standard JSON with 'name' and 'arguments'."
        )

        messages = [{"role": "user", "content": prompt}]
        gold = {
            "domain": "telecom",
            "actions": actions,
            "assertions": assertions,
            "task_id": t.get("id"),
        }
        samples.append((messages, gold, {"category": "telecom_v3"}))

    logger.info(f"Loaded {len(samples)} canonical Tau-bench Telecom scenarios.")
    return samples


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract all top-level JSON objects from free text (for tool-call detection)."""
    objs: List[Dict[str, Any]] = []
    dec = json.JSONDecoder()
    i, n = 0, len(text or "")
    while i < n:
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
                if isinstance(obj, dict):
                    objs.append(obj)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return objs


def _required_calls(gold: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Normalize gold into a list of required (tool_name, args) calls."""
    calls: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(gold, dict) and gold.get("actions"):
        for a in gold["actions"]:
            if isinstance(a, dict):
                name = a.get("name") or a.get("action") or a.get("tool")
                args = a.get("arguments") or a.get("kwargs") or a.get("args") or {}
                if name:
                    calls.append((str(name), args if isinstance(args, dict) else {}))
    elif isinstance(gold, dict) and gold.get("tool"):
        calls.append((str(gold["tool"]), gold.get("args") or {}))
    return calls


def _call_present(req_name: str, req_args: Dict[str, Any], objs: List[Dict[str, Any]]) -> bool:
    for o in objs:
        name = o.get("name") or o.get("tool")
        args = o.get("arguments") or o.get("args") or {}
        if str(name) == req_name and isinstance(args, dict):
            if all(str(args.get(k)) == str(v) for k, v in req_args.items()):
                return True
    return False


def _eval_tau3(response_text: str, gold_scenario: Dict[str, Any]) -> bool:
    """Strict required-action match: every gold tool call must appear (name + args).

    NOTE: canonical tau-bench (v3/telecom) is a multi-turn *environment* benchmark
    (final state check + nl_assertion LLM judge). This single-turn harness scores
    only the required tool actions; assertion-only tasks score False. Full fidelity
    needs the sierra-research/tau2-bench environment simulator.
    """
    if not response_text:
        return False
    required = _required_calls(gold_scenario)
    if not required:
        return False
    objs = _extract_json_objects(response_text)
    return all(_call_present(n, a, objs) for n, a in required)


def run_tau3(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run the τ³-bench `banking_knowledge` (RAG) evaluation suite.

    banking_knowledge is a multi-turn *environment* benchmark with a retrieval backend
    (BM25 + dense embeddings + shell). With TAU2_ENV_RUN=1, the tau2-bench simulator
    installed, `rank-bm25`, and an embeddings key (GEMINI_API_KEY works via Google's
    OpenAI-compatible endpoint by default), this runs the canonical simulator. Against a
    plain single-turn /v1 endpoint it CANNOT be scored, so it reports a clean skip rather
    than a misleading 0%. See docs/evals/tau3.md for the retrieval knobs (TAU2_RETRIEVAL_CONFIG,
    TAU2_EMBED_MODEL, ...).
    """
    if env_requested():
        return run_tau_env("tau3", ["banking_knowledge"], model_name, base_url, concurrency,
                           kwargs.get("limit"), DOCS_URL, enable_thinking=enable_thinking)
    raise infra_required(
        "tau3",
        "tau3 (τ³-bench banking_knowledge) is a multi-turn retrieval-augmented environment "
        "benchmark and cannot be scored against a single-turn /v1 endpoint (the agent must "
        "retrieve over a knowledge base and act on environment state the model never sees). "
        "Set TAU2_ENV_RUN=1 to run the full simulator",
        DOCS_URL)
