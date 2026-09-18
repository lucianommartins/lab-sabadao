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

"""A `web_search` tool the agentic suites can actually call, backed by Gemini grounding.

`gaia` and `deepsearch_qa` are web-research benchmarks: their questions are answerable only
by looking things up. Run without any tool they score a structural 0 - on the 2026-08-15
sweep GAIA replied "I do not have access to the internet. Therefore, I cannot watch the
video" and both suites returned 0/20. That is not a measurement of the model.

This offers the same Google-Search grounding `bfcl_v4_agentic` already uses (same
`GEMINI_API_KEY`, same endpoint), exposed as an OpenAI-style tool so the evaluated model
drives the search itself.

Comparability: canonical GAIA leaderboard entries use a full browsing agent (page fetch,
file download, code execution). This is search-only, so a gbench GAIA number is **not**
leaderboard-comparable; the result records the backend it used.

Two backend facts, both measured live on 2026-08-20. Read them before quoting any number
from a search-dependent suite.

**Google Search grounding has its own quota, separate per-model from the RPM/TPM a project
dashboard reports.** The identical request returns HTTP 200 without `google_search` and can
be HTTP 429 with it, and the 429 is per-model - so one model being quota'd says nothing about
the next. A grounding 429 cannot be diagnosed from the model's RPM/TPM graph. This is exactly
why the search backend is a MODEL CASCADE (below), not a single model.

**Grounding uses a MODEL CASCADE, not a scraper fallback.** Grounding quota is per-model, so
a per-model 429 (which a concurrent burst hits) is recovered by trying the next model, each
with its own quota. Default order: gemini-3.7-flash -> 3.6-flash -> 3.5-flash ->
3-flash-preview -> 2.5-flash (override with GBENCH_SEARCH_MODELS). This replaced the
DuckDuckGo fallback, which on this box is captcha-blocked (HTTP 202 anomaly page, 0/12 gold
recall) and returned blind snippets that also masked grounding failures - a second real
grounding model is strictly better. `search_backend_calls` records which model served each
call; check it before believing a number.

**High --sandboxes is throttle-safe.** The 2026-08-21 sweep ran the search suites at
--sandboxes 32; the request burst 429'd every cascade model at once and browsecomp / gaia /
mcp_atlas each recorded 100%% all-grounding-models-failed (the grounding path itself was fine
- verified live at concurrency 1 and 16 the same day). Two guards make any fan-out safe:
(1) a GLOBAL token-bucket rate limiter (`GBENCH_SEARCH_MAX_RPS`, default 8) that every
grounding request passes through, so the total request rate self-limits no matter how many
sandboxes call at once; and (2) when the whole cascade fails one pass, a bounded
backoff-with-jitter retry of the entire cascade (`GBENCH_SEARCH_CASCADE_ROUNDS`, default 3)
that lets a transient burst drain. Together they turn what was a fake zero into a completed
lookup. Tune the RPS down if you still see all-fail in the tally, up for a paid key.

**Deliberately snippet-only - NO `url_context` (anti-contamination).** The Gemini API's
`url_context` tool would let this fetch full page bodies (deeper retrieval), but it was
evaluated 2026-08-21 and **rejected**: these are public benchmarks whose question->answer
pairs live on the web (HF dataset viewers, GitHub, papers, "solved" write-ups), and fetching
a page that hosts the answer key is contamination - cheating, not capability. url_context
amplifies that over snippets (it ingests the whole page). If ever reconsidered it must ship
WITH a fetch-domain blocklist, a canary/exact-answer check, and per-URL logging - never bare.
See docs/gbench-search-grounding.md (root docs, not this public repo).
"""

import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: How many grounding chunks to hand back per query.
MAX_RESULTS = int(os.environ.get("GBENCH_SEARCH_MAX_RESULTS", "8"))

#: The tool as the evaluated model sees it.
WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web and return ranked results with titles, URLs and snippets. "
            "Use it whenever the answer depends on information you do not already know, "
            "and call it repeatedly to refine or cross-check. Prefer specific queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
}


#: Grounding-model cascade, tried in order until one succeeds. Grounding quota is PER MODEL
#: (verified 2026-08-21 on one key: 2.5-flash 429 while 3.7/3.6/3.5-flash + 3-flash-preview
#: all 200 for the same grounded request), so a per-model 429 - which a concurrent burst
#: hits (mcp_atlas: 33%% of searches got no data) - is recovered by the next model's separate
#: quota. This REPLACED the DuckDuckGo fallback: DDG is captcha-blocked on this box and
#: returns blind snippets, whereas a second grounding model is a real, equivalent backend.
#: Override with GBENCH_SEARCH_MODELS (comma list); the legacy singular GBENCH_SEARCH_MODEL
#: still works as a one-model cascade.
_DEFAULT_SEARCH_CASCADE = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                           "gemini-3-flash-preview", "gemini-2.5-flash"]

#: Attempts per grounding call for TRANSIENT errors, and the base backoff. 429 is not
#: retried on the same model - the cascade moves to the next model instead (its own quota),
#: and if the WHOLE cascade fails, search_with_fallback backs off and retries the cascade
#: (see _SEARCH_CASCADE_ROUNDS). That is where a burst-429 is recovered; retrying the same
#: throttled model in place would only add latency.
_SEARCH_ATTEMPTS = int(os.environ.get("GBENCH_SEARCH_ATTEMPTS", "3"))
_SEARCH_BACKOFF = float(os.environ.get("GBENCH_SEARCH_BACKOFF", "1.0"))


def _cascade_rounds() -> int:
    """How many times to retry the WHOLE cascade when every model fails one pass.

    A synchronized all-model failure is the signature of a rate burst (the 2026-08-21 sweep
    ran the search suites at --sandboxes 32 and 429'd every cascade model at once, recording
    100%% all-grounding-models-failed). Backing off and re-running the cascade lets the burst
    drain and the lookup complete. Read at call time so a sweep/test can tune it live.
    """
    return max(1, int(os.environ.get("GBENCH_SEARCH_CASCADE_ROUNDS", "3")))


class _RateLimiter:
    """Thread-safe token bucket that caps the GLOBAL grounding request rate.

    The grounding rate limit is per-model, but the burst that trips it is global: a high
    --sandboxes fan-out fires many concurrent `search_with_fallback` calls, and when their
    combined request rate exceeds the limit every cascade model 429s at once (browsecomp /
    gaia / mcp_atlas each recorded 100%% failure on 2026-08-21). Capping the total request
    rate here makes ANY --sandboxes safe: excess callers block until a token frees, so the
    request rate self-limits regardless of fan-out. `rps <= 0` disables it. clock/sleep are
    injectable so the behaviour is testable without real waits.
    """

    def __init__(self, rps: float, burst: float, *,
                 clock=time.monotonic, sleep=time.sleep):
        self.rps = float(rps)
        self.capacity = max(1.0, float(burst))
        self.tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rps <= 0:
            return
        while True:
            with self._lock:
                now = self._clock()
                self.tokens = min(self.capacity,
                                  self.tokens + (now - self._updated) * self.rps)
                self._updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rps
            self._sleep(wait)


def _build_rate_limiter() -> _RateLimiter:
    rps = float(os.environ.get("GBENCH_SEARCH_MAX_RPS", "8"))
    burst = float(os.environ.get("GBENCH_SEARCH_MAX_BURST", str(max(1, int(rps)))))
    return _RateLimiter(rps, burst)


#: The global limiter every grounding HTTP request passes through. Built from the environment
#: at import; call reset_rate_limiter() after changing GBENCH_SEARCH_MAX_RPS/_BURST.
_RATE_LIMITER = _build_rate_limiter()


def reset_rate_limiter() -> None:
    """Rebuild the global limiter from the current environment (after an env change / tests)."""
    global _RATE_LIMITER
    _RATE_LIMITER = _build_rate_limiter()


def _search_cascade() -> List[str]:
    """The ordered list of grounding models to try."""
    plural = os.environ.get("GBENCH_SEARCH_MODELS", "").strip()
    if plural:
        return [m.strip() for m in plural.split(",") if m.strip()]
    single = os.environ.get("GBENCH_SEARCH_MODEL", "").strip()
    if single:
        return [single]
    return list(_DEFAULT_SEARCH_CASCADE)


def search_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def unavailable_reason() -> str:
    return ("web search requires GEMINI_API_KEY (Google-Search grounding across the model "
            "cascade). Without it, every question that needs a lookup is unanswerable and "
            "the suite skips rather than report a structural 0%")


def _is_failure(results: List[Dict[str, str]]) -> bool:
    """True when grounding reported a failure (429/network/etc) rather than an answer.

    "no results" is a real answer and must NOT advance the cascade - re-running a genuinely
    empty query on another model just burns quota and can invent a match. Grounding is
    trustworthy when it says empty.
    """
    return bool(results) and results[0].get("title") == "error"


import itertools

_KEY_LOCK = threading.Lock()
_KEY_CYCLE: Optional[Any] = None


def get_gemini_api_key() -> str:
    """Get next Gemini API key (supports comma-separated round-robin)."""
    global _KEY_CYCLE
    raw = (os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not raw:
        return ""
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if len(keys) == 1:
        return keys[0]
    with _KEY_LOCK:
        if _KEY_CYCLE is None:
            _KEY_CYCLE = itertools.cycle(keys)
        return next(_KEY_CYCLE)


def gemini_search(query: str, max_results: int = MAX_RESULTS,
                  model: Optional[str] = None) -> List[Dict[str, str]]:
    """Google-Search-grounded results as [{title, url, snippet}], from one grounding model."""
    import urllib.request

    key = get_gemini_api_key()
    if not key:
        return [{"title": "error", "url": "", "snippet": "GEMINI_API_KEY is not set"}]
    if not model:
        model = _search_cascade()[0]
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    payload = {"contents": [{"parts": [{"text": str(query)}]}],
               "tools": [{"google_search": {}}]}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    # Retry transient errors, but NOT 429 on the SAME model. Measured 2026-08-20: 2 of 12
    # grounding calls failed with a transient error and both succeeded on re-issue; without a
    # retry a one-off blip became a permanent "the web has no answer" for that sample. 429 is
    # excluded here on purpose - a throttled model does not clear in-place, so the cascade
    # moves to the next model (its own quota) and, if every model is throttled at once,
    # search_with_fallback backs off and retries the whole cascade. Every attempt first takes
    # a token from the global rate limiter, so a high --sandboxes fan-out cannot burst past
    # the grounding rate limit in the first place.
    data = None
    for attempt in range(_SEARCH_ATTEMPTS):
        _RATE_LIMITER.acquire()
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except Exception as e:
            if getattr(e, "code", None) == 429 or attempt == _SEARCH_ATTEMPTS - 1:
                # A failed lookup is reported to the model as a failed lookup, not as "no
                # results" - otherwise the model concludes the fact does not exist.
                return [{"title": "error", "url": "",
                         "snippet": f"search backend failed: {e}"}]
            # transient (network / 5xx): exponential backoff with jitter, then retry.
            time.sleep(_SEARCH_BACKOFF * (2 ** attempt) + random.uniform(0, _SEARCH_BACKOFF))

    cand = (data.get("candidates") or [{}])[0]
    gm = cand.get("groundingMetadata") or {}
    chunks = gm.get("groundingChunks") or []

    bodies: Dict[int, List[str]] = {}
    for sup in gm.get("groundingSupports") or []:
        text = ((sup.get("segment") or {}).get("text") or "").strip()
        for idx in sup.get("groundingChunkIndices") or []:
            if text:
                bodies.setdefault(int(idx), []).append(text)

    answer = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
    results: List[Dict[str, str]] = []
    for i, ch in enumerate(chunks[:max_results]):
        web = ch.get("web") or {}
        results.append({
            "title": web.get("title") or web.get("domain") or "result",
            "url": web.get("uri") or "",
            "snippet": " ".join(bodies.get(i, [])) or answer[:500],
        })
    if not results and answer:
        results = [{"title": "grounded-answer", "url": "", "snippet": answer[:1500]}]
    if not results:
        results = [{"title": "no results", "url": "", "snippet": "The search returned nothing."}]
    return results


async def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Tool executor for `run_eval_suite(tool_executor=...)`. Returns the tool message."""
    import asyncio
    if name != "web_search":
        return json.dumps({"error": f"unknown tool '{name}'"})
    query = str((args or {}).get("query") or "").strip()
    if not query:
        return json.dumps({"error": "web_search requires a non-empty 'query'"})
    results, backend = await asyncio.to_thread(search_with_fallback, query)
    return json.dumps({"query": query, "results": results, "backend": backend},
                      ensure_ascii=False)


def search_with_fallback(query: str,
                         max_results: int = MAX_RESULTS) -> Tuple[List[Dict[str, str]], str]:
    """Ground the query, cascading across gemini models until one succeeds.

    Google-Search grounding quota is PER MODEL (measured 2026-08-20/21: on one key,
    gemini-2.5-flash returned 429 while gemini-3.7/3.6/3.5-flash and gemini-3-flash-preview
    all returned 200 for the identical grounded request). So a per-model 429 - which a
    concurrent burst hits easily (mcp_atlas: 299 429s, 33%% of its searches got no data) - is
    recovered by trying the NEXT model, each with its own quota. This replaced the DuckDuckGo
    fallback, which on this box is captcha-blocked and returns blind snippets; a second real
    grounding model is strictly better than a broken scraper. The backend that actually
    served (the model id) is returned and tallied so a run never silently claims grounding it
    did not get.
    """
    def _tallied(results, backend):
        _BACKEND_TALLY[backend] = _BACKEND_TALLY.get(backend, 0) + 1
        return results, backend

    if os.environ.get("GEMINI_API_KEY"):
        cascade = _search_cascade()
        rounds = _cascade_rounds()
        last = None
        for rnd in range(rounds):
            for model in cascade:
                results = gemini_search(query, max_results, model=model)
                if not _is_failure(results):
                    return _tallied(results, f"gemini:{model}")
                last = results
            # Every model failed this pass. A single model's 429 is handled above by moving to
            # the next model, so reaching here means they were ALL failing at once - the
            # signature of a rate burst (or a transient network-wide fault). Back off with
            # jitter and retry the whole cascade: the burst drains while we wait, which is what
            # turns a --sandboxes-32 all-fail into a completed lookup instead of a fake zero.
            if rnd < rounds - 1:
                time.sleep(_SEARCH_BACKOFF * (2 ** rnd) + random.uniform(0, _SEARCH_BACKOFF))
        # Still failing after every round: quotas genuinely out, or a real outage.
        return _tallied(last or [{"title": "error", "url": "",
                                  "snippet": "all grounding models failed"}],
                        "all-grounding-models-failed")
    # No key: nothing to ground with. The suite skips rather than serve blind results.
    return _tallied([{"title": "error", "url": "", "snippet": unavailable_reason()}],
                    "none")


#: Which backend actually served each call this process. `search_backend_name()` reports
#: what was CONFIGURED; on 2026-08-20 that read `gemini-google-search-grounding` on a run
#: where every single lookup had 429'd. Only a tally of what answered can tell those apart.
_BACKEND_TALLY: Dict[str, int] = {}


def backend_tally() -> Dict[str, int]:
    """Calls served per backend since the last reset, e.g. `{"gemini:gemini-3.7-flash": 812}`."""
    return dict(_BACKEND_TALLY)


def reset_backend_tally() -> None:
    _BACKEND_TALLY.clear()


def search_backend_name() -> Optional[str]:
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini-grounding-cascade[" + ",".join(_search_cascade()) + "]"
    return None
