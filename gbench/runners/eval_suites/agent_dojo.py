# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: agent_dojo
# Description: AgentDojo (Tool-Use Adversarial Prompt Injection Attack Resilience Benchmark)

"""gbench native built-in runner for agent_dojo (Instruction & Safety).

Canonical AgentDojo (ethz-spylab/agentdojo) measures two things that only exist inside its
tool-execution environment: **utility** (did the agent actually complete the user's task, by
calling the suite's real tools) and **security / attack success rate** (did an injected
instruction get the agent to perform the attacker's task). gbench delegates to the agentdojo
package's OWN suites, attacks and scorers rather than reimplementing them:

* It builds an in-process AgentDojo pipeline (`SystemMessage -> InitQuery -> OpenAILLM ->
  ToolsExecutionLoop`) pointed at the gbench-served `/v1` endpoint via `OpenAILLM` (AgentDojo's
  `vllm_parsed` provider), which uses the NATIVE OpenAI tools API. gbench's vLLM endpoint has
  tool-calling enabled (`--enable-auto-tool-choice` + a tool-call parser), so the server returns
  proper `tool_calls`; the prompt-based `LocalLLM` would instead see `content=None` because the
  server's parser consumes the model's function-call text. (Set the endpoint up that way, or
  swap in `LocalLLM` for a server without tool parsing.)
* For each of the 4 canonical suites (workspace / travel / banking / slack) at a pinned
  `benchmark_version` it runs both `benchmark_suite_without_injections` (benign utility) and
  `benchmark_suite_with_injections` under the `important_instructions` attack (utility under
  attack + attack success). `security_results[...] == True` means THE ATTACK SUCCEEDED (the
  agentdojo docstring: "the second [bool] indicat[es] if the injection was successful").

Headline `accuracy` = benign task utility. `attack_success_rate` (lower is better) is reported
separately and is NOT folded into accuracy. The agentdojo package is REQUIRED; if it cannot be
imported the suite hard-errors (never skips).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with `--temperature`, or for this suite
alone with `GBENCH_AGENT_DOJO_TEMPERATURE`, which takes precedence over both. This suite uses
**no LLM judge** (AgentDojo scores against its environments' ground truth), so no judge
temperature (the usual pinned 0.0) applies here.
"""

import json
import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import resolve_temperature
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/agent_dojo.md"
PILLAR = "Instruction & Safety"

#: Pinned benchmark version and attack (both overridable). v1.2.2 is the latest in the package.
_BENCHMARK_VERSION = os.environ.get("GBENCH_AGENTDOJO_VERSION", "v1.2.2")
_ATTACK = os.environ.get("GBENCH_AGENTDOJO_ATTACK", "important_instructions")
#: The pipeline "name" AgentDojo maps to a prose model label for the important_instructions
#: attack (see agentdojo.models.MODEL_NAMES). gemma is not in that map; the canonical label for a
#: locally-served model is `vllm_parsed` -> "Local model" (what AgentDojo's own vllm_parsed
#: provider uses). Override with a `gemini-*` id to get the Google-targeted attack variant.
_MODEL_LABEL = os.environ.get("GBENCH_AGENTDOJO_MODEL_LABEL", "vllm_parsed")


def _load_agentdojo() -> Dict[str, Any]:
    """Import the agentdojo package (hard-error with install instructions if missing)."""
    try:
        from agentdojo.agent_pipeline import (AgentPipeline, InitQuery, OpenAILLM, SystemMessage,
                                              ToolsExecutionLoop, ToolsExecutor)
        from agentdojo.agent_pipeline.agent_pipeline import load_system_message
        from agentdojo.attacks.attack_registry import load_attack
        from agentdojo.benchmark import (benchmark_suite_with_injections,
                                         benchmark_suite_without_injections)
        from agentdojo.logging import OutputLogger
        from agentdojo.task_suite.load_suites import get_suites
        import openai
    except Exception as e:  # ImportError, or a transitive dep problem
        raise infra_required(
            "agent_dojo",
            f"the agentdojo package is required and could not be imported ({e}). Install it: "
            "pip install agentdojo==0.1.35 (base install; the 'transformers' extra is not needed "
            "for the canonical run).",
            DOCS_URL,
        ) from e
    return {
        "AgentPipeline": AgentPipeline, "InitQuery": InitQuery, "OpenAILLM": OpenAILLM,
        "SystemMessage": SystemMessage, "ToolsExecutionLoop": ToolsExecutionLoop,
        "ToolsExecutor": ToolsExecutor, "load_system_message": load_system_message,
        "load_attack": load_attack, "get_suites": get_suites,
        "benchmark_with": benchmark_suite_with_injections,
        "benchmark_without": benchmark_suite_without_injections,
        "OutputLogger": OutputLogger, "openai": openai,
    }


def _served_model_id(base_url: str, fallback: str) -> str:
    """The model id the endpoint actually serves (LocalLLM must send that exact id to vLLM)."""
    override = os.environ.get("GBENCH_AGENTDOJO_MODEL")
    if override:
        return override
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except Exception as e:                                              # noqa: BLE001
        logger.warning("agent_dojo: could not read served model id (%s); using %r", e, fallback)
        return fallback


def _pct(xs: List[bool]) -> float:
    return round(sum(1 for x in xs if x) / len(xs) * 100.0, 2) if xs else 0.0


def run_agent_dojo(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical AgentDojo by delegating to the agentdojo package's suites/attacks/scorers."""
    A = _load_agentdojo()
    limit = limit if limit is not None else kwargs.get("limit")

    model_id = _served_model_id(base_url, model_name)
    temperature, _src = resolve_temperature("agent_dojo", kwargs.get("temperature"),
                                            thinking=enable_thinking)
    client = A["openai"].OpenAI(api_key="EMPTY", base_url=base_url)
    # Use the NATIVE tools API (AgentDojo's `vllm_parsed` provider). The gbench endpoint has
    # vLLM tool-calling enabled (--enable-auto-tool-choice + a tool-call parser), which returns
    # proper `tool_calls`; the prompt-based LocalLLM would get content=None because the server's
    # parser consumes the model's function-call text. Requires no server changes.
    llm = A["OpenAILLM"](client, model_id, temperature=temperature)
    # Canonical no-defense pipeline (identical to AgentPipeline.from_config with defense=None).
    pipeline = A["AgentPipeline"]([
        A["SystemMessage"](A["load_system_message"](None)),
        A["InitQuery"](),
        llm,
        A["ToolsExecutionLoop"]([A["ToolsExecutor"](), llm]),
    ])
    # AgentDojo derives the attack's prose model label from pipeline.name (MODEL_NAMES); use the
    # canonical local-model label so load_attack('important_instructions') resolves.
    pipeline.name = _MODEL_LABEL

    try:
        suites = A["get_suites"](_BENCHMARK_VERSION)
    except Exception as e:                                              # noqa: BLE001
        raise infra_required(
            "agent_dojo",
            f"agentdojo has no benchmark version {_BENCHMARK_VERSION!r} ({e}); set "
            "GBENCH_AGENTDOJO_VERSION to one the installed package provides.", DOCS_URL) from e

    logdir = Path(tempfile.mkdtemp(prefix="gbench_agentdojo_"))
    benign_u: List[bool] = []
    attack_u: List[bool] = []
    attack_s: List[bool] = []
    per_suite: Dict[str, Any] = {}

    # `limit` (--eval-limit) is a GLOBAL cap on scored user tasks, like every other suite --
    # NOT a per-suite cap. Without a global budget a small --eval-limit still fans out across
    # all 4 suites AND the user x injection cross-product (hundreds of runs). We consume the
    # budget across suites and stop once it is exhausted; inj_ids are capped by the same
    # budget so the attack cross-product stays bounded too.
    remaining = int(limit) if limit else None

    # AgentDojo's benchmark functions require an active logging context (they read
    # `logger.logdir`); without it TraceLogger raises AttributeError on the NullLogger.
    # AgentDojo ALSO dumps every task's full conversation transcript to the ROOT logger at
    # INFO (hundreds of lines per run). Raise the root level to WARNING for the duration so
    # that spam is suppressed while genuine warnings/errors still surface, then restore it.
    _root = logging.getLogger()
    _prev_level = _root.level
    _root.setLevel(logging.WARNING)
    try:
        with A["OutputLogger"](str(logdir)):
            for name, suite in suites.items():
                if remaining is not None and remaining <= 0:
                    break
                user_ids = list(suite.user_tasks)
                inj_ids = list(suite.injection_tasks)
                if remaining is not None:
                    user_ids = user_ids[:remaining]
                    inj_ids = inj_ids[:remaining]
                    remaining -= len(user_ids)

                without = A["benchmark_without"](pipeline, suite, logdir, True,
                                                 user_tasks=user_ids,
                                                 benchmark_version=_BENCHMARK_VERSION)
                attack = A["load_attack"](_ATTACK, suite, pipeline)
                withinj = A["benchmark_with"](pipeline, suite, attack, logdir, True,
                                              user_tasks=user_ids, injection_tasks=inj_ids,
                                              benchmark_version=_BENCHMARK_VERSION)

                # SuiteResults is a TypedDict (a dict subclass): access by key, not attribute.
                bu = list(without["utility_results"].values())
                au = list(withinj["utility_results"].values())
                as_ = list(withinj["security_results"].values())  # True == attack succeeded
                benign_u += bu
                attack_u += au
                attack_s += as_
                per_suite[name] = {
                    "benign_utility": _pct(bu),
                    "utility_under_attack": _pct(au),
                    "attack_success_rate": _pct(as_),
                    "n_user_tasks": len(user_ids),
                    "n_injection_tasks": len(inj_ids),
                }
    finally:
        _root.setLevel(_prev_level)

    if not benign_u:
        raise RuntimeError("agent_dojo: no tasks were scored (empty suites/subset)")

    return {
        "benchmark_type": "eval",
        "eval_name": "agent_dojo",
        "model_name": model_name,
        "status": "success",
        "accuracy": _pct(benign_u),                       # headline = benign task utility
        "utility_under_attack": _pct(attack_u),
        "attack_success_rate": _pct(attack_s),            # lower is better; NOT folded into accuracy
        "total_questions": len(benign_u),
        "correct_answers": sum(1 for x in benign_u if x),
        "dimension_scores": per_suite,
        "benchmark_version": _BENCHMARK_VERSION,
        "attack": _ATTACK,
        "suites": list(suites.keys()),
        "metric": ("AgentDojo (delegated to the agentdojo package): headline accuracy = benign "
                   "task utility across the 4 suites; attack_success_rate = targeted injection "
                   f"success under the '{_ATTACK}' attack (lower is better); utility_under_attack "
                   "also reported. security_results True == attack succeeded."),
        "temperature": temperature,
        "temperature_source": _src,
        # Honest sampling record. gbench builds the AgentDojo pipeline in-process and passes this
        # temperature straight into AgentDojo's own OpenAILLM, which applies it on the /v1
        # chat.completions requests (verified in agentdojo openai_llm.py: OpenAILLM stores the
        # temperature and chat_completion_request forwards it), so the temperature above is
        # genuinely used -- it is NOT a silent no-op. Reasoning/thinking mode, however, is NOT
        # injected: AgentDojo's OpenAI tools path has no thinking toggle, so --thinking only
        # shifts the default temperature and does not enable model reasoning.
        "sampling": ("gbench passes the resolved temperature into AgentDojo's own OpenAILLM, "
                     "which applies it on the /v1 chat.completions requests, so the temperature "
                     "above is genuinely used. Reasoning/thinking mode is NOT injected: "
                     "AgentDojo's OpenAI tools path has no thinking toggle, so --thinking only "
                     "shifts the default temperature and does not enable model reasoning."),
        # Comparable when it is the FULL task set (no --eval-limit) for the pinned version+attack
        # AND greedy (temperature 0.0), which is how the published AgentDojo numbers are produced.
        "leaderboard_comparable": bool(not limit and temperature == 0.0),
    }
