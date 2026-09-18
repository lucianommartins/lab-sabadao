# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Cascading Gemini judge for the gbench mcp_bench container. Runs INSIDE the
# gbench-mcp-bench image; drop-in for MCP-Bench's own `LLMProvider` (duck-typed).
#
# It preserves upstream MCP-Bench's judge PROCEDURE completely - the same 6 sub-dimension
# prompts, the same 5x randomized-order stability averaging (that all lives in the upstream
# evaluator and is untouched) - and swaps ONLY the model call underneath: instead of a single
# Azure o4-mini, it uses gbench's ESTABLISHED Gemini cascade (the same model list, rounds and
# backoff as base.judge_generate_cascade), reached through Gemini's OpenAI-compatible endpoint
# so no google-genai dependency is needed in the container.
#
# Because the judge model is Gemini (not the canonical o4-mini the MCP-Bench leaderboard
# requires), a run scored this way is never leaderboard_comparable - by design, for
# consistency with every other gbench judged suite.

import asyncio
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mcp_bench.cascade_judge")

# gbench's default judge cascade (kept in sync with base._DEFAULT_JUDGE_CASCADE). Overridable by
# the SAME env knobs gbench uses, so an operator moves both implementations with one setting.
_DEFAULT_JUDGE_CASCADE = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                          "gemini-3-flash-preview", "gemini-2.5-flash"]
_GEMINI_OPENAI_BASE = os.environ.get(
    "GEMINI_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")


def _cascade() -> List[str]:
    """Mirror base.judge_cascade(): GBENCH_JUDGE_MODELS (csv) > GBENCH_JUDGE_MODEL (single) > default."""
    raw = (os.environ.get("GBENCH_JUDGE_MODELS") or "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    single = (os.environ.get("GBENCH_JUDGE_MODEL") or "").strip()
    if single:
        return [single]
    return list(_DEFAULT_JUDGE_CASCADE)


def _rounds() -> int:
    return max(1, int(os.environ.get("GBENCH_JUDGE_CASCADE_ROUNDS", "3")))


def _backoff() -> float:
    return float(os.environ.get("GBENCH_JUDGE_BACKOFF", "1.0"))


class CascadeGeminiJudge:
    """Duck-typed replacement for MCP-Bench's LLMProvider used only as the judge.

    Contract the upstream evaluator depends on (verified against llm/provider.py):
      * ``async get_completion(system_prompt, user_prompt, max_tokens, return_usage=False)``
        returning a ``str`` (or ``(str, usage_dict)`` when ``return_usage=True``);
      * attributes ``deployment_name`` / ``provider_type``;
      * ``clean_and_parse_json(raw)`` (the evaluator parses the judge JSON through it).
    Raises on total outage - matching LLMProvider, which raises; the evaluator's stability
    loop already catches per-iteration failures and skips them.
    """

    def __init__(self, deployment_name: str = "gemini-cascade") -> None:
        self.deployment_name = deployment_name
        self.provider_type = "openai_compatible"
        self.client = None  # unused: get_completion is overridden
        self._async_client = None
        self._cascade = _cascade()
        self._rounds = _rounds()
        self._backoff = _backoff()

    def _client_lazy(self):
        if self._async_client is None:
            from openai import AsyncOpenAI  # container ships `openai`
            key = os.environ.get("GEMINI_API_KEY", "")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set - the mcp_bench Gemini cascade judge "
                                   "cannot run.")
            self._async_client = AsyncOpenAI(api_key=key, base_url=_GEMINI_OPENAI_BASE)
        return self._async_client

    async def get_completion(self, system_prompt: str, user_prompt: str, max_tokens: int,
                             return_usage: bool = False) -> Any:
        client = self._client_lazy()
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        last_err: Optional[Exception] = None
        # Structurally identical to base.judge_generate_cascade: try every model each round;
        # only back off (burst) BETWEEN rounds after a whole-cascade pass fails; temp pinned 0.0.
        for rnd in range(self._rounds):
            for model in self._cascade:
                try:
                    resp = await client.chat.completions.create(
                        model=model, messages=messages, max_tokens=max_tokens, temperature=0.0)
                    content = resp.choices[0].message.content
                    if content is not None and content.strip() != "":
                        if return_usage:
                            usage = getattr(resp, "usage", None)
                            usage_dict = {
                                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                            }
                            return content.strip(), usage_dict
                        return content.strip()
                except Exception as e:  # 429/network/timeout/model-missing -> next model
                    last_err = e
                    logger.debug("mcp_bench judge model %s failed: %s", model, e)
            if rnd < self._rounds - 1:
                await asyncio.sleep(self._backoff * (2 ** rnd) + random.uniform(0, self._backoff))
        raise RuntimeError(f"JUDGE_OUTAGE: all Gemini cascade models failed ({last_err})")

    # Ported verbatim in spirit from LLMProvider.clean_and_parse_json (provider.py:210): strip
    # ```json fences, then json / json_repair. The evaluator calls this on the judge output.
    def clean_and_parse_json(self, raw_json: str) -> Any:
        if raw_json is None:
            return None
        s = raw_json.strip()
        if s.startswith("```"):
            s = s.split("```", 2)
            s = s[1] if len(s) > 1 else raw_json
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:]
            s = s.strip().rstrip("`").strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            from json_repair import loads as _repair_loads
            return _repair_loads(s)
        except Exception:
            # last resort: extract the outermost object
            import re
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None
