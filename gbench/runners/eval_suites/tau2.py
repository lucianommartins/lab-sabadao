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

"""Native Tau-bench v2 Multi-Domain Agentic Tool Calling evaluation suite.

Sampling: this suite asks for temperature 1.0 (the reference tau2 config, 1.0/0.95/top_k 64). The global default is
1.0, the model's shipped `generation_config.json`. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_TAU2_TEMPERATURE`,
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

DOCS_URL = "docs/evals/tau2.md"

SIERRA_TAU2_DOMAINS = {
    "airline": "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/airline/tasks.json",
    "retail": "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/retail/tasks.json",
    "telecom": "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains/telecom/tasks.json",
}


def _load_tau2_samples(
    domain_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]]:
    """Load canonical Sierra Tau2-bench tasks across airline, retail, and telecom domains."""
    domains = [domain_filter.lower()] if domain_filter else ["airline", "retail", "telecom"]
    raw_tasks_by_domain = []

    for d in domains:
        url = SIERRA_TAU2_DOMAINS.get(d)
        if not url:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as res:
                domain_tasks = json.loads(res.read().decode("utf-8"))
            for t in domain_tasks:
                raw_tasks_by_domain.append((t, d))
        except Exception as e:
            logger.warning(f"Could not load Sierra Tau2 tasks for domain {d}: {e}")

    if not raw_tasks_by_domain:
        raise RuntimeError("Failed to load canonical Sierra Tau2 tasks.")

    # Stratified, not a contiguous head (audit RC-1).
    raw_tasks_by_domain = stratified_sample(raw_tasks_by_domain, limit, None, seed="tau2")

    samples = []
    for t, domain in raw_tasks_by_domain:
        scenario = t.get("user_scenario") or {}
        instructions = (scenario.get("instructions") or {}).get("task_instructions") or t.get("description") or ""
        eval_criteria = t.get("evaluation_criteria") or {}
        actions = eval_criteria.get("actions") or []
        assertions = eval_criteria.get("nl_assertions") or []

        prompt = (
            f"You are an expert customer service AI agent operating in the {domain.upper()} domain.\n\n"
            f"Customer Request & Context:\n{instructions}\n\n"
            "Respond by providing the appropriate tool calls or response to resolve the customer's request. "
            "Format tool invocations in standard JSON with 'name' and 'arguments'."
        )

        messages = [{"role": "user", "content": prompt}]
        gold = {
            "domain": domain,
            "actions": actions,
            "assertions": assertions,
            "task_id": t.get("id"),
        }
        samples.append((messages, gold, {"category": domain}))

    logger.info(f"Loaded {len(samples)} canonical Tau2 tasks across {len(domains)} domains.")
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


def _eval_tau2(response_text: str, gold_scenario: Dict[str, Any]) -> bool:
    """Strict required-action match: every gold tool call must appear (name + args).

    NOTE: canonical tau2-bench is a multi-turn *environment* benchmark (final
    DB-state check + nl_assertion LLM judge). This single-turn harness scores only
    the required tool actions; assertion-only tasks (no required actions) cannot be
    verified deterministically here and score False. Full fidelity needs the
    sierra-research/tau2-bench environment simulator.
    """
    if not response_text:
        return False
    required = _required_calls(gold_scenario)
    if not required:
        return False
    objs = _extract_json_objects(response_text)
    return all(_call_present(n, a, objs) for n, a in required)


def run_tau2(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Tau-bench v2 Multi-Domain Agentic Tool Calling evaluation suite.

    tau2-bench is a multi-turn *environment* benchmark. With TAU2_ENV_RUN=1 and the
    tau2-bench simulator installed, this runs the canonical simulator. Against a plain
    single-turn /v1 endpoint it CANNOT be scored (assertion-only tasks carry no actions
    to match, and gold action arguments depend on environment state the model never
    sees), so it hard-errors (infra_required) rather than emit a misleading 0%. The
    single-turn strict-action proxy remains available as `_eval_tau2` for diagnostics/tests.
    """
    if env_requested():
        domain = kwargs.get("domain")
        domains = [domain.lower()] if domain else ["airline", "retail", "telecom"]
        return run_tau_env("tau2", domains, model_name, base_url, concurrency,
                           kwargs.get("limit"), DOCS_URL, enable_thinking=enable_thinking)
    raise infra_required(
        "tau2",
        "tau2-bench is a multi-turn environment benchmark and cannot be scored against a "
        "single-turn /v1 endpoint (assertion-only tasks have no actions to match; gold "
        "action arguments require environment state the model never sees). Set "
        "TAU2_ENV_RUN=1 to run the full simulator",
        DOCS_URL)
