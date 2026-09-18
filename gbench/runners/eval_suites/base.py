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

"""Base async execution engine for self-contained evaluation benchmark suites.

Handles concurrent HTTP dispatching to OpenAI-compatible server, real-time tqdm
progress logging, error retries, and standardized metric calculations.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import itertools
import json
import logging
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from .sampling import stratified_sample, parse_shard, shard_select


@functools.lru_cache(maxsize=None)
def _eval_fn_accepts_tool_calls(fn: Callable) -> bool:
    """True if `fn` declares a `tool_calls` parameter (or **kwargs). Function-calling suites opt in
    to being scored off the STRUCTURED tool calls (the text render is lossy); every other 2-arg
    eval_fn is called unchanged."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "tool_calls" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _apply_eval_fn(fn: Optional[Callable], pred: Any, gold: Any, tool_calls: Any) -> bool:
    """Call `fn(pred, gold)`, additionally passing the structured `tool_calls` when `fn` accepts it."""
    if fn is None:
        return False
    if _eval_fn_accepts_tool_calls(fn):
        return fn(pred, gold, tool_calls=tool_calls)
    return fn(pred, gold)


class Reply(NamedTuple):
    """One model response, with the metadata needed to tell WHY it looks the way it does."""
    text: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    completion_tokens: int = 0
    stop_reason: Optional[Any] = None
    #: Observed per-sequence decode rate for this request. Every timeout in the harness is
    #: sized from an ASSUMED rate; this is the measured one, so the assumption can be
    #: checked on the box that is actually running rather than the one it was tuned on.
    decode_tok_s: Optional[float] = None


#: A 20-gram repeated this many times in a response that ran to the token cap is taken as
#: degenerate repetition rather than a long answer. Calibrated on the 2026-08-15 run:
#: looping responses showed 12-21 repeats, ordinary long outputs (patches, programs) stay
#: in the low single digits because real code rarely repeats a 20-word window verbatim.
REPETITION_NGRAM = 20
REPETITION_THRESHOLD = 8


def repetition_run(text: Optional[str]) -> int:
    """Most times any 20-word window repeats. 0 when the text is too short to judge."""
    words = str(text or "").split()
    if len(words) < REPETITION_NGRAM * 3:
        return 0
    from collections import Counter
    grams = Counter(" ".join(words[i:i + REPETITION_NGRAM])
                    for i in range(len(words) - REPETITION_NGRAM))
    return grams.most_common(1)[0][1] if grams else 0


def repetition_onset(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Where degenerate repetition begins, and what the model is stuck on.

    Reports the first occurrence of the most-repeated 20-word window, so a run can be
    correlated against prompt shape: does the model degenerate immediately (a prompt it
    cannot start), or only deep into a long derivation (a search it cannot terminate)?

    * `onset_word` / `onset_frac` - where the cycling text first appears, absolute and as a
      fraction of the whole response
    * `period_words` - distance between consecutive repeats, i.e. the size of the cycle
    * `cycle_preview` - what it is repeating, so the cause is legible without opening traces
    """
    words = str(text or "").split()
    if len(words) < REPETITION_NGRAM * 3:
        return None
    positions: Dict[str, List[int]] = {}
    for i in range(len(words) - REPETITION_NGRAM):
        positions.setdefault(" ".join(words[i:i + REPETITION_NGRAM]), []).append(i)
    gram, hits = max(positions.items(), key=lambda kv: len(kv[1]))
    if len(hits) < REPETITION_THRESHOLD:
        return None
    return {
        "onset_word": hits[0],
        "onset_frac": round(hits[0] / len(words), 3),
        "period_words": (hits[1] - hits[0]) if len(hits) > 1 else None,
        "repeat_count": len(hits),
        "total_words": len(words),
        "cycle_preview": gram[:160],
    }


#: A capped response this many times longer than the suite's own healthy median is
#: reclassified `truncated` -> `non_convergent`. 0 disables.
NON_CONVERGENT_LENGTH_RATIO = float(os.environ.get("GBENCH_NON_CONVERGENT_RATIO", "4.0"))


def reclassify_non_convergent(sample_traces: List[Dict[str, Any]]) -> int:
    """Relabel capped responses that are far longer than this suite's healthy ones.

    `repetition_run` detects a 20-gram repeating verbatim. It cannot see a model that keeps
    re-deriving the same quantity with different algebra, which is what long-form reasoning
    failure actually looks like. Measured on AIME 2026-08-20: 7 samples capped at exactly
    16,384 tokens against a healthy median of 1,987 (8.2x), `repetition_run` 1-2 against a
    threshold of 8, tails reading "Let me re-calculate everything. Is it possible m=37 and
    n=...?" and 70 enumerated case markers on one of them.

    The distinction is operational, not cosmetic: `truncated` tells the operator to raise
    --max-output-tokens. For these it buys more of the same at more wall-clock. Calibrated
    against the run's OWN healthy median rather than a constant, so it transfers to suites
    whose answers are legitimately long.
    """
    if NON_CONVERGENT_LENGTH_RATIO <= 0:
        return 0
    healthy = [t.get("completion_tokens") or 0 for t in sample_traces
               if t.get("health") == "ok"]
    healthy = [n for n in healthy if n > 0]
    if len(healthy) < 8:
        return 0
    import statistics
    cutoff = statistics.median(healthy) * NON_CONVERGENT_LENGTH_RATIO
    n = 0
    for t in sample_traces:
        if t.get("health") == "truncated" and (t.get("completion_tokens") or 0) >= cutoff:
            t["health"] = "non_convergent"
            n += 1
    return n


def classify_reply(reply: "Reply") -> str:
    """`ok` | `request_failed` | `truncated` | `empty_reasoned` | `empty`.

    Without this every one of these looked like an ordinary wrong answer:
      * `truncated`       - hit the output-token budget mid-answer. In the 2026-08-15
                            sweep this alone was codeforces' IOI 0/6 and 10/20 of
                            swe_bench_multilingual.
      * `empty_reasoned`  - the model spent its whole budget in the reasoning channel and
                            emitted no `content` (gemma-4 with `--reasoning-parser gemma4`).
      * `empty`           - the server returned 200 with nothing at all.
    """
    if reply.text is None:
        return "request_timeout" if (reply.error or "").startswith("timeout") else "request_failed"
    has_output = bool(str(reply.text).strip()) or bool(reply.tool_calls)
    if reply.finish_reason == "length":
        # A response that hit the cap is only "truncated" if it was going somewhere. When
        # the model is stuck re-deriving the same step, more budget buys more of the same:
        # measured on 2026-08-15, gpqa_diamond produced 163,976 chars with a 20-gram
        # repeated 20x and ended on "Step 375: Let's try k=-283", and arc_agi re-enumerated
        # grid rows with "Wait, the rows are 1, 2,". Telling the operator to raise
        # --max-output-tokens there is wrong advice, so name it differently.
        return "looping" if repetition_run(reply.text) >= REPETITION_THRESHOLD else "truncated"
    if not has_output:
        if (reply.reasoning or "").strip():
            return "empty_reasoned"
        # The server generated tokens and returned none of them. On gemma-4 this is the
        # model emitting a tool call followed by `<|tool_response>` (token 50, a configured
        # stop token): vLLM's tool-call parser lifts the call out of `content`, and when the
        # request declared no `tools` there is nowhere to put it, so it is DISCARDED.
        # Measured live: completion_tokens=34, content=null, tool_calls=[], stop_reason=50.
        # That is a suite/serving mismatch, not a model that said nothing - scoring it as an
        # ordinary wrong answer hid 28 samples across bfcl_v3_live / mcp_atlas / skillsbench.
        if reply.completion_tokens:
            return "output_discarded"
        return "empty"
    return "ok"

#: Set to 0 to disable the one-shot recovery of a forced-final turn whose answer the
#: tool-call parser discarded.
RETRY_DISCARDED_FINAL = os.environ.get("GBENCH_RETRY_DISCARDED_FINAL", "1") != "0"


async def _retry_discarded_final(*, reply, convo, session, api_url, model_name, payload,
                                 semaphore, budget, temperature):
    """Re-ask once when a forced-final turn came back `output_discarded`.

    The final turn withdraws `tools` so the model has to commit to an answer. gemma-4
    answers that by emitting a tool call anyway, terminated by `<|tool_response>` (token 50,
    a configured stop token); vLLM's parser lifts the call out of `content`, finds no
    `tools` on the request to bind it to, and drops it - so a turn that generated real
    tokens returns `content=null`. Nothing downstream can recover it, and the sample is
    graded as though the model said nothing.

    Returns the recovered reply, or the original if recovery is off, unnecessary or failed
    (never raises, and never returns something emptier than what it was handed).
    """
    if not RETRY_DISCARDED_FINAL or classify_reply(reply) != "output_discarded":
        return reply
    convo.append({"role": "user", "content": (
        "That response could not be read because it was a tool call. Tools are disabled "
        "and no tool call can succeed. Reply with plain text only - no function call, no "
        "tool syntax - stating your final answer.")})
    final_max, oversize = clamp_to_context(convo, budget)
    if oversize:
        return reply
    try:
        retry = await _send_single_request(
            session=session, api_url=api_url, model_name=model_name, messages=convo,
            extra_payload=payload, semaphore=semaphore, pbar=None,
            max_output_tokens=final_max, temperature=temperature, thinking=False)
    except Exception as e:                                                  # noqa: BLE001
        logger.debug("discarded-final retry failed: %s", e)
        return reply
    return retry if (retry.text or "").strip() else reply


#: Judge / grader model for every LLM-graded built-in suite, overridable with
#: `GBENCH_JUDGE_MODEL`. Previously each suite hardcoded its own: 13 sat on
#: `gemini-2.5-flash`, 4 on `gemini-3.5-flash`, and the plugin engine on
#: `gemini-3.6-flash` - so "the judge" meant three different models depending on which
#: suite you looked at, and no single knob moved them.
DEFAULT_JUDGE_MODEL = os.environ.get("GBENCH_JUDGE_MODEL", "gemini-3.6-flash")

#: Run-level generation knobs, set once by the runner and consulted by `run_eval_suite`
#: for anything a suite did not pass explicitly.
#:
#: `evals.py` hands every suite the same kwargs dict, but a suite only honours a knob if
#: it forwards it - and 107 of 122 never forwarded `temperature`, so `--temperature`
#: (documented as applying "across all benchmarks") silently did nothing for them
#: (audit RC-2). Resolving here means a knob cannot be lost by omission; a suite that
#: passes a value explicitly still wins.
_RUN_KNOBS: Dict[str, Any] = {}

#: Minimum output-token budget for EVERY suite, and the per-suite exceptions to it.
#:
#: Set from measurement, not from judgement about which benchmark is "long-form". Across
#: 32 suites measured on 2026-08-17, the largest *healthy* completion - one that finished
#: on its own with no degenerate repetition - was 8,020 tokens (hmmt). Second 5,730
#: (livebench), third 5,239 (putnam); everything else under 3,600. Every *looping*
#: response, by contrast, ran 10,035-56,201 tokens. The two populations do not overlap.
#:
#: 16384 is therefore 2x the largest genuine answer ever observed and below every observed
#: loop: it cannot truncate a real answer, and it caps a spiral at a quarter of what a
#: 65536 budget costs. A per-suite table encoding "this one writes patches, that one writes
#: letters" was tried and got hmmt, codeforces and putnam wrong.
#:
#: EXCEPTIONS go in the dict below, and only with a measurement: if a suite genuinely needs
#: more, the run reports `genuinely_truncated` (distinct from `looping`), which is the
#: signal to raise it. Nothing is here yet because nothing has earned it - the 65536 tier
#: previously rested on a 114,292-char copilot_bench_swe response recorded before
#: repetition detection existed, so it was never established whether that was a patch or a
#: loop.
DEFAULT_MIN_OUTPUT_TOKENS = 16384

#: Floor when `--thinking` is on. A reasoning trace and the answer share one `max_tokens`,
#: so the non-thinking floor truncates mid-reasoning: on the 2026-08-18 smoke run `aime`
#: lost 3/3 responses to the cap and scored 0%, which is a measurement artefact rather than
#: a model result. Canonical AIME budgets are 32,768 (DeepSeek-R1, open-r1, Evalchemy) to
#: 64K (R1-0528); 32,768 is the common floor.
#:
#: This exists because `_run_suite_async` already had a `32768 if thinking` default that
#: could never fire: `run_eval_suite` applies the floor FIRST, so `max_output_tokens` was
#: never None by the time that branch was reached. Keep the two in agreement.
#:
#: Deliberately a plain constant, like DEFAULT_MIN_OUTPUT_TOKENS above: the value follows
#: from `--thinking`, and `--max-output-tokens` already overrides both. A third knob would
#: only add a way for the budget to disagree with the mode it was chosen for.
THINKING_MIN_OUTPUT_TOKENS = 32768
SUITE_MIN_OUTPUT_TOKENS: Dict[str, int] = {}

#: Ceiling on the per-turn budget for suites that run a TOOL LOOP - the mirror image of the
#: floors above, and the only place a floor is not what a suite needs.
#:
#: `max_output_tokens` is sent on EVERY turn, so in a loop it does not bound the run, it
#: licenses that much output per turn. Measured on browsecomp 2026-08-18 at the operator's
#: `--max-output-tokens 65536`: 3/3 samples spent the entire 65,536 in one turn enumerating
#: candidates ("Maybe the movie is "The Blind Side"? No." on repeat) and never emitted an
#: answer. Forcing `tool_choice=required` was tried first and REFUTED the obvious
#: explanation: the model then made 5/2/4 real searches, got real grounded results, and
#: still burned 70k+ tokens per sample without answering. The turn length is the problem,
#: not tool availability.
#:
#: Re-running the same 3 questions at 8192/turn: 12 clean search rounds with ZERO truncated
#: turns on the sample that had managed one, 13.1k tokens/sample (5.4x less) and 257s
#: (4.0x faster). 4096 was also tried and is too tight - the model cannot finish reasoning
#: and emit the call, so every turn truncates and tool calls drop from 12/0/4 to 2/1/0.
#:
#: This is a no-op for every other tool-loop suite: the largest turn any of them produced
#: on the same run was mcp_atlas at 3,014 tokens (gaia 1,063; other tool-loop suites were
#: lower), so the ceiling sits 2.7x above the observed maximum. It deliberately does
#: NOT apply to single-shot suites, where a long turn IS the answer (aime spends 31,252).
TOOL_LOOP_MAX_OUTPUT_TOKENS = 8192


def tool_loop_turn_ceiling(has_tool_loop: bool, max_output_tokens: Optional[int]) -> Optional[int]:
    """The capped per-turn budget, or None to leave `max_output_tokens` alone."""
    if not has_tool_loop or not max_output_tokens:
        return None
    ceiling = TOOL_LOOP_MAX_OUTPUT_TOKENS
    raw = os.environ.get("GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS")
    if raw is not None:
        try:
            ceiling = max(0, int(raw))
        except ValueError:
            logger.warning(
                "GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS=%r is not an integer; ignoring", raw)
    if not ceiling or max_output_tokens <= ceiling:
        return None
    return ceiling


def set_run_knobs(**knobs: Any) -> None:
    """Record run-level generation knobs (called by the runner, once per suite)."""
    _RUN_KNOBS.update({k: v for k, v in knobs.items() if v is not None})


#: Executors are `(name, args)`. One that also accepts a third parameter is handed the
#: sample index, so a stateful environment can keep per-sample state instead of sharing it
#: across concurrently-running samples. Cached because it is consulted per tool call.
_EXECUTOR_ARITY: Dict[int, bool] = {}


def _executor_wants_session(fn: Callable) -> bool:
    key = id(fn)
    if key not in _EXECUTOR_ARITY:
        try:
            import inspect
            params = list(inspect.signature(fn).parameters.values())
            positional = [p for p in params
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            _EXECUTOR_ARITY[key] = len(positional) >= 3
        except (TypeError, ValueError):
            _EXECUTOR_ARITY[key] = False
    return _EXECUTOR_ARITY[key]


def get_run_knob(name: str, default: Any = None) -> Any:
    return _RUN_KNOBS.get(name, default)


#: Default sampling temperature for EVERY suite.
#:
#: 1.0 is what Google ships this model with - `generation_config.json` carries
#: `do_sample true, temperature 1.0, top_k 64, top_p 0.95`, and the Gemma 4 model card
#: prescribes the same "across all use cases". vLLM would apply it by default; gbench used
#: to override it to 0.0 on every request.
#:
#: The reason it changed is measured, not doctrinal. Replaying the 2026-08-17
#: copilot_bench_swe prompts, 4 attempts each, only the temperature differing:
#:
#:     prompt set          T=0.0        T=1.0     T=1.0 +top_p .95/top_k 64
#:     10 that looped      31/40 (78%)  16/40 (40%)  15/40 (38%)
#:     10 that did not     10/40 (25%)  -            3/40  (8%)
#:
#: Paired per-prompt, 13 of 14 discordant prompts improved (one-sided sign test
#: p = 0.0009), 0 of 10 loopers got worse, and the healthy control improved too - so the
#: reduction is not bought by degrading answers that already worked. `top_k`/`top_p` add
#: nothing detectable on top of the temperature (40% vs 38% at n=40).
#:
#: Caveats worth keeping in view: 38% residual looping means this is a mitigation, not a
#: cure; and the experiment measured DEGENERATION, not accuracy - a suite whose canonical
#: protocol is greedy (BigCodeBench, IFEval, RULER, the ANLS/relaxed-accuracy vision
#: suites) is no longer leaderboard-comparable at 1.0. Set `--temperature 0.0`, or the
#: per-suite env var below, to restore it.
#:
#: The resolved DEFAULT is THINK-AWARE (project baseline): a no-think run defaults to
#: `NOTHINK_TEMPERATURE` (0.0 - greedy, canonical-comparable) and a `--thinking` run to
#: `DEFAULT_TEMPERATURE` (1.0 - the shipped config; the reasoning pass makes looping far
#: less of a concern than the 0.0 no-think figure above). Both are overridden by
#: `--temperature` or the per-suite env var. NOTE the tradeoff: a no-think run at 0.0
#: carries the higher degenerate-repetition rate measured above - surfaced by the
#: looping/non-convergent metrics, not silently absorbed.
DEFAULT_TEMPERATURE = 1.0     #: --thinking default (Gemma 4's shipped generation_config)
NOTHINK_TEMPERATURE = 0.0     #: no-think default (greedy; canonical-comparable)

#: Grading temperature for the LLM judge. Deliberately NOT tied to DEFAULT_TEMPERATURE or
#: to `--temperature`: those describe the model under test, and a grader that drifts with
#: the run's sampling knob would make two runs of the same answers disagree.
#:
#: Until 2026-08-17 all 15 judge call sites passed no config at all, so grading ran at the
#: Gemini API default of 1.0 - an undocumented source of run-to-run variance sitting on top
#: of every judge-scored suite.
JUDGE_TEMPERATURE = 0.0


def judge_config(**overrides: Any) -> Any:
    """`GenerateContentConfig` for judge calls: deterministic grading.

    Returns None if google-genai is unavailable, which callers pass through harmlessly
    (`config=None` is the SDK's own default).
    """
    try:
        from google.genai import types
    except ImportError:
        return None
    return types.GenerateContentConfig(temperature=JUDGE_TEMPERATURE, **overrides)


# --------------------------------------------------------------------------- #
# Judge cascade - mirrors search_tool.search_with_fallback. Google-Search grounding
# quota is per-model (a 429 on one Gemini version 200s on another), so the search suites
# cascade across versions and recover bursts by retrying the whole cascade. The judge hits
# the same per-model quota, so it gets the same treatment: try each version in order, move
# to the next on any failure, back off and retry the whole cascade on a synchronized
# all-model failure, and only when EVERY model fails EVERY round declare a JUDGE_OUTAGE
# (excluded from accuracy) rather than scoring the sample wrong.
# --------------------------------------------------------------------------- #
#: Ordered Gemini judge models. Same order as search_tool's grounding cascade. Override with
#: GBENCH_JUDGE_MODELS (comma list); the legacy singular GBENCH_JUDGE_MODEL still works as a
#: one-model cascade.
_DEFAULT_JUDGE_CASCADE = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
                         "gemini-3-flash-preview", "gemini-2.5-flash"]
_JUDGE_BACKOFF = float(os.environ.get("GBENCH_JUDGE_BACKOFF", "1.0"))


def judge_cascade() -> List[str]:
    """The ordered list of Gemini judge models to try (see the module note above)."""
    plural = os.environ.get("GBENCH_JUDGE_MODELS", "").strip()
    if plural:
        return [m.strip() for m in plural.split(",") if m.strip()]
    single = os.environ.get("GBENCH_JUDGE_MODEL", "").strip()
    if single:
        return [single]
    return list(_DEFAULT_JUDGE_CASCADE)


def _judge_cascade_rounds() -> int:
    """How many times to retry the WHOLE cascade when every model fails one pass (burst)."""
    return max(1, int(os.environ.get("GBENCH_JUDGE_CASCADE_ROUNDS", "3")))


def gemini_api_keys() -> List[str]:
    """All configured Gemini API keys, in order, de-duplicated.

    Reads GEMINI_API_KEYS (preferred) or GEMINI_API_KEY and splits on comma OR semicolon, so a key
    LIST works uniformly for the judge, the liveness gate, and search grounding - not just search.
    A single key returns a 1-element list.
    """
    raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or ""
    seen: set = set()
    out: List[str] = []
    for k in re.split(r"[;,]", raw):
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


_JUDGE_CLIENTS = None
_JUDGE_CLIENT_CYCLE = None


def _judge_clients():
    """Lazily-built google-genai clients, ONE per configured key, cached. Rotated across judge calls
    (see `_next_judge_client`) so a multi-key `GEMINI_API_KEY` spreads per-key rate limits instead of
    hammering (and failing on) a single key."""
    global _JUDGE_CLIENTS, _JUDGE_CLIENT_CYCLE
    if _JUDGE_CLIENTS is None:
        from google import genai
        _JUDGE_CLIENTS = [genai.Client(api_key=k) for k in gemini_api_keys()]
        _JUDGE_CLIENT_CYCLE = itertools.cycle(_JUDGE_CLIENTS) if _JUDGE_CLIENTS else None
    return _JUDGE_CLIENTS


def _next_judge_client():
    """The next judge client in round-robin order (spreads load across keys), or None if no key."""
    _judge_clients()
    return next(_JUDGE_CLIENT_CYCLE) if _JUDGE_CLIENT_CYCLE else None


async def judge_generate_cascade(contents: Any, *, config: Any = None) -> Tuple[Optional[str], str]:
    """Grade `contents` via the Gemini judge cascade.

    Returns `(text, model_used)` from the first model that answers, or `(None, "JUDGE_OUTAGE")`
    when every model fails every round. On the latter the caller MUST set
    `trace["judge_grade"] = "JUDGE_OUTAGE"` (so base.run_eval_suite excludes it from the
    accuracy denominator) rather than scoring the sample wrong. Grading stays deterministic
    (JUDGE_TEMPERATURE) via `judge_config()`.
    """
    if not gemini_api_keys():
        return None, "JUDGE_OUTAGE"
    try:
        if not _judge_clients():
            return None, "JUDGE_OUTAGE"
    except ImportError:
        return None, "JUDGE_OUTAGE"
    cfg = config if config is not None else judge_config()
    cascade = judge_cascade()
    rounds = _judge_cascade_rounds()
    for rnd in range(rounds):
        client = _next_judge_client()   # rotate key per round + across calls -> spreads rate limits
        for model in cascade:
            try:
                resp = await client.aio.models.generate_content(
                    model=model, contents=contents, config=cfg)
                text = getattr(resp, "text", None)
                if text is not None:
                    return text, model
            except Exception as e:  # noqa: BLE001 - 429/network/timeout: move to next model
                logger.debug("judge model %s failed: %s", model, e)
        if rnd < rounds - 1:  # every model failed this pass -> burst; back off and retry all
            await asyncio.sleep(_JUDGE_BACKOFF * (2 ** rnd) + random.uniform(0, _JUDGE_BACKOFF))
    return None, "JUDGE_OUTAGE"


def temperature_env_var(eval_name: str) -> str:
    """Per-suite temperature override, e.g. `GBENCH_COPILOT_BENCH_SWE_TEMPERATURE`."""
    slug = re.sub(r"[^A-Z0-9]+", "_", str(eval_name).upper()).strip("_")
    return f"GBENCH_{slug}_TEMPERATURE"


def free_port(preferred: Optional[int] = None) -> int:
    """Return an available localhost TCP port. If `preferred` is given and currently free, return it
    (so a single run stays deterministic); otherwise return an OS-assigned free port. Used so a local
    judge proxy / helper server does not collide with a fixed default when two runs overlap. There is
    a small TOCTOU window (the port could be taken between check and bind), acceptable for avoiding a
    fixed-default clash; an OS-assigned port is effectively never re-handed immediately."""
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", int(preferred)))
                return int(preferred)
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_LEGACY_ENV_WARNED: set = set()


def suite_env(canonical: str, *legacy: str, default: Optional[str] = None) -> Optional[str]:
    """Read a per-suite env knob by its canonical ``GBENCH_<SUITE>_<KNOB>`` name, falling back to any
    legacy bare-prefixed aliases (kept for back-compat). Warns once per legacy name when it is used.

    Convention: every per-suite knob is ``GBENCH_<SUITE>_<KNOB>``. Older suites shipped bare-prefixed
    names (e.g. ``TAU2_NUM_TRIALS``, ``SCICODE_EVAL_TIMEOUT_S``); those still work as deprecated
    aliases so nothing breaks, but the canonical name takes precedence and is what the docs and
    ``--help`` advertise. Third-party/standard vars (HF_TOKEN, OPENAI_API_KEY, GEMINI_API_KEY, ...)
    are NOT suite knobs and are read directly, not through this helper.
    """
    v = os.environ.get(canonical)
    if v is not None:
        return v
    for name in legacy:
        v = os.environ.get(name)
        if v is not None:
            if name not in _LEGACY_ENV_WARNED:
                _LEGACY_ENV_WARNED.add(name)
                logging.getLogger(__name__).warning(
                    "env var %s is a deprecated alias; use %s instead.", name, canonical)
            return v
    return default


def resolve_temperature(eval_name: str, suite_value: Optional[float] = None,
                        *, thinking: bool = False) -> Tuple[float, str]:
    """Resolve the temperature for `eval_name`, returning (value, where_it_came_from).

    Precedence, highest first:
      1. `GBENCH_<EVAL>_TEMPERATURE`  - per-suite env override, for tuning one eval
         without disturbing the rest of a sweep. Beats everything, including the CLI.
      2. `--temperature`             - the operator was explicit for this whole run.
      3. the value the suite passed  - a suite that encodes its own protocol.
      4. the think-aware default     - `DEFAULT_TEMPERATURE` (1.0) when `thinking`,
         else `NOTHINK_TEMPERATURE` (0.0). This is the project baseline; pass the run's
         `thinking` flag so every suite defaults consistently.

    Recording the source matters as much as the value: a number produced at 1.0 is not
    comparable with a greedy leaderboard entry, and the result must be able to say so.
    """
    var = temperature_env_var(eval_name)
    raw = (os.environ.get(var) or "").strip()
    if raw:
        try:
            return float(raw), f"env:{var}"
        except ValueError:
            logger.warning("%s=%r is not a number; ignoring it.", var, raw)
    knob = get_run_knob("temperature")
    if knob is not None:
        return float(knob), "cli:--temperature"
    if suite_value is not None:
        return float(suite_value), "suite"
    return (DEFAULT_TEMPERATURE if thinking else NOTHINK_TEMPERATURE), "default"


#: Pessimistic per-sequence decode rate used to size the HTTP timeout. Under heavy
#: concurrency each sequence gets a small share of aggregate throughput, so a long
#: generation legitimately takes a long time.
MIN_DECODE_TOK_S = float(os.environ.get("GBENCH_MIN_DECODE_TOK_S", "8"))
REQUEST_TIMEOUT_FLOOR_S = int(os.environ.get("GBENCH_REQUEST_TIMEOUT_S", "1200"))


#: Decoding penalties, OFF by default. The 2026-08-15 run showed the model hitting the
#: token cap while repeating itself (a 20-gram recurring up to 379 times), which a penalty
#: can suppress - but canonical benchmark protocol is greedy with no penalty, so switching
#: one on silently would make every number non-comparable with published results and with
#: gbench's own earlier runs. Opt in per run and the value is recorded on the result.
#:
#: Note it does NOT address unbounded enumeration ("Step 375: Let's try k=-283"), which is
#: search rather than repetition and shows a 20-gram count of 2.
def decoding_penalties() -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, env in (("repetition_penalty", "GBENCH_REPETITION_PENALTY"),
                     ("frequency_penalty", "GBENCH_FREQUENCY_PENALTY"),
                     ("presence_penalty", "GBENCH_PRESENCE_PENALTY")):
        raw = os.environ.get(env)
        if raw:
            try:
                out[key] = float(raw)
            except ValueError:
                logger.warning("%s=%r is not a number; ignoring", env, raw)
    return out



def normalize_function_calls(tool_calls: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The model's emitted calls as `[{"name": str, "arguments": dict}]`.

    Recorded so a tool-use suite can be scored OFFLINE later. Some tool-use suites are graded
    by comparing the emitted FunctionCall AST against a golden one; when the golden is not
    in the export yet, the model's side of that comparison is still worth banking - it means
    scoring, once targets arrive, is a diff over stored JSON rather than a re-run (no GPU,
    no re-sampling, no temperature variance between the two halves).

    The wire form nests the call under `function` and carries `arguments` as a JSON STRING,
    which is not diffable. Arguments that will not parse are kept verbatim under
    `arguments_raw` rather than dropped - a malformed argument list is itself a result.
    """
    out: List[Dict[str, Any]] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = fn.get("name")
        if not name:
            continue
        raw = fn.get("arguments")
        entry: Dict[str, Any] = {"name": name}
        if isinstance(raw, dict):
            entry["arguments"] = raw
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                entry["arguments"] = parsed if isinstance(parsed, dict) else {"_value": parsed}
            except ValueError:
                entry["arguments"] = None
                entry["arguments_raw"] = raw[:4000]
        else:
            entry["arguments"] = {}
        out.append(entry)
    return out


def request_timeout_s(max_output_tokens: Optional[int]) -> int:
    """How long one request may take, given how much it is allowed to generate.

    A fixed 1200 s was fine at 8192 tokens and wrong at 65536: at 256-way concurrency each
    sequence gets a small slice of aggregate decode throughput, so a long answer needs well
    over an hour. The request would time out, retry three times and land as a bare
    `request_failed` - i.e. the campaign would silently drop exactly its longest answers,
    which are the ones the big budgets exist to capture.

    Scales with the token budget rather than the concurrency because the budget is what
    bounds the work; `GBENCH_MIN_DECODE_TOK_S` tunes the assumed floor rate.
    """
    budget = max_output_tokens or 8192
    return max(REQUEST_TIMEOUT_FLOOR_S,
               int(REQUEST_TIMEOUT_FLOOR_S + budget / max(1.0, MIN_DECODE_TOK_S)))


#: chars-per-token used only to pre-clamp `max_tokens`; deliberately low (i.e. it
#: OVER-estimates the prompt) so the estimate errs toward a smaller, safe request.
_CHARS_PER_TOKEN = 3.0


#: A whole image is a fixed, small number of soft tokens (gemma-4 tops out at 1120), not a
#: function of its base64 length. Counting the payload as text made a 2 MB document page
#: look like ~2.9 MILLION tokens and blocked the request as over-context: on the
#: 2026-08-17 run that silently dropped 56 requests across 10 vision/long-context suites
#: (omnidocbench 9/20, screenspot 9/20, mrcr 7/20, ...). Budget the maximum.
IMAGE_SOFT_TOKENS = int(os.environ.get("GBENCH_IMAGE_SOFT_TOKENS", "1120"))


def _estimate_prompt_tokens(messages: Any) -> Optional[int]:
    """Rough prompt size in tokens, counting images as soft tokens rather than base64."""
    import json as _json
    if isinstance(messages, str):
        return int(len(messages) / _CHARS_PER_TOKEN)
    if not isinstance(messages, list):
        return None
    chars, images = 0, 0
    try:
        for m in messages:
            content = (m or {}).get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        chars += len(str(part)); continue
                    if part.get("type") in ("image_url", "image", "input_image"):
                        images += 1
                    else:
                        chars += len(_json.dumps(part))
            elif content is not None:
                chars += len(_json.dumps(content))
            # tool calls / names / role overhead
            for k in ("tool_calls", "name", "tool_call_id"):
                if isinstance(m, dict) and m.get(k):
                    chars += len(_json.dumps(m[k]))
    except Exception:
        return None
    return int(chars / _CHARS_PER_TOKEN) + images * IMAGE_SOFT_TOKENS


def clamp_to_context(messages: Any, max_tokens: Optional[int],
                     max_model_len: Optional[int] = None) -> Tuple[Optional[int], Optional[str]]:
    """Shrink `max_tokens` so `prompt + max_tokens` fits the server's context window.

    vLLM rejects a request whose prompt+max_tokens exceeds max_model_len. The old code
    only recovered *reactively*, by parsing the 400 and retrying - which costs a round
    trip per request and, when the prompt alone is over the limit, simply retried three
    times and recorded `request_failed` with no reason at all. On the 2026-08-15 sweep
    that was mrcr: 6 rows with 1.3-4.2 MB prompts (~330k-1M tokens) against a 262144
    window, reported as unexplained failures.

    Returns `(clamped_max_tokens, reason_if_unanswerable)`. A non-None reason means the
    prompt cannot fit at all and the request should not be sent.
    """
    if not max_model_len:
        max_model_len = get_run_knob("max_model_len")
    if not max_model_len:
        return max_tokens, None
    est_prompt = _estimate_prompt_tokens(messages)
    if est_prompt is None:
        return max_tokens, None
    headroom = max_model_len - est_prompt - 64        # 64 tokens of slack for the template
    if headroom < 256:
        return max_tokens, (
            f"prompt is ~{est_prompt:,} tokens, which does not fit the model's "
            f"{max_model_len:,}-token context window (no room left to answer)")
    if max_tokens is None or max_tokens > headroom:
        return headroom, None
    return max_tokens, None

_GRADE_VERDICT_RE = re.compile(r"GRADE\s*:\s*(NOT_ATTEMPTED|INCORRECT|CORRECT)", re.IGNORECASE)


def parse_grade_verdict(grade_text: str) -> bool:
    """Is the judge's verdict CORRECT? Reads the verdict, not the whole reply.

    The suites all asked for `Grade: CORRECT / INCORRECT` and then tested
    ``"CORRECT" in grade_str and "INCORRECT" not in grade_str`` over the entire response.
    Judges routinely ignore the one-word instruction and return several paragraphs of
    analysis, so a reply reasoning "the patch is not incorrect ... GRADE: CORRECT" scored
    as wrong: the word `INCORRECT` appeared somewhere in the prose.

    The last explicit `GRADE:` line wins (a judge that revises itself means the later
    one); only when the reply contains no explicit verdict at all do we fall back to the
    old substring behaviour.
    """
    text = str(grade_text or "")
    verdicts = _GRADE_VERDICT_RE.findall(text)
    if verdicts:
        return verdicts[-1].upper() == "CORRECT"
    upper = text.upper()
    return "CORRECT" in upper and "INCORRECT" not in upper
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else None

logger = logging.getLogger(__name__)
logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)
for noisy_logger in ["google_genai", "google", "grpc", "httpx", "httpcore", "urllib3"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# --- Judge-model resolution: loud, guarded, recordable ------------------------------
# GBENCH_JUDGE_MODEL silently redirects EVERY LLM-judge call (and tau2's user-sim / nl-judge,
# which default to `gemini/<DEFAULT_JUDGE_MODEL>`). Left unlogged and unrecorded, a whole sweep
# could be graded by the wrong model - even the model UNDER TEST judging its own outputs - and
# the only symptom was an exhausted API quota for that model. The pieces below make the override
# announce itself, flag self-judging, and let callers record the resolved model on the result.
if os.environ.get("GBENCH_JUDGE_MODEL"):
    logger.warning(
        "[judge] GBENCH_JUDGE_MODEL override is ACTIVE: all LLM-judging (and tau2 user-sim / "
        "nl-judge) will use '%s', NOT the default gemini-3.6-flash. Unset it to restore the "
        "default grader.", os.environ["GBENCH_JUDGE_MODEL"])


def _norm_model_id(m: Optional[str]) -> str:
    """Lowercased model id without provider prefix: `google/Gemma-4-26B-A4B-it` -> `gemma-4-26b-a4b-it`."""
    m = (m or "").strip().lower()
    return m.rsplit("/", 1)[-1] if "/" in m else m


def _same_model_family(judge: Optional[str], under_test: Optional[str]) -> bool:
    """True when the judge model is (a variant of) the model under test - i.e. self-judging."""
    j, u = _norm_model_id(judge), _norm_model_id(under_test)
    if not j or not u:
        return False
    return j == u or j.startswith(u) or u.startswith(j)


_JUDGE_RESOLVE_LOGGED: set = set()


def resolve_judge_model(under_test_model: Optional[str] = None) -> Tuple[str, str, bool]:
    """Resolve the LLM-judge model LOUDLY, with a self-judge guard.

    Returns `(model, source, is_self_judge)`; `source` is 'env:GBENCH_JUDGE_MODEL' or 'default'.
    Logs the resolved model once per (model, source, has-under-test). If the judge is the model
    under test, logs an ERROR - a model grading its own outputs is not a valid judgement and it
    routes grading to that model's own API quota. Never raises; grading proceeds, flagged.
    """
    env = os.environ.get("GBENCH_JUDGE_MODEL")
    model = env or "gemini-3.6-flash"
    source = "env:GBENCH_JUDGE_MODEL" if env else "default"
    is_self = bool(under_test_model and _same_model_family(model, under_test_model))
    key = (model, source, bool(under_test_model))
    if key not in _JUDGE_RESOLVE_LOGGED:
        _JUDGE_RESOLVE_LOGGED.add(key)
        logger.warning("[judge] grading with '%s' (source: %s)", model, source)
        if is_self:
            logger.error(
                "[judge] SELF-JUDGE DETECTED: judge '%s' is the model under test ('%s'). A model "
                "grading its own outputs is not a valid judgement and burns that model's API "
                "quota. Unset GBENCH_JUDGE_MODEL to use the default grader.",
                model, under_test_model)
    return model, source, is_self


def strip_thinking_tags(text: Optional[str]) -> str:
    """Strip <thought>...</thought> or <reasoning>...</reasoning> blocks if present."""
    if not text:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # If output was entirely wrapped in thought tags, return stripped raw text
    return cleaned.strip() if cleaned.strip() else text.strip()


def _sanitize_for_trace(obj: Any) -> Any:
    """Recursively truncate bulky payloads for human-readable JSON traces."""
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith("data:image/") and len(v) > 64:
                res[k] = v[:48] + "..."
            elif k == "tools" and isinstance(v, list) and len(v) > 4:
                # A tool-use suite may offer 100+ schemas, identical on every sample; kept
                # verbatim they dominate the result file.
                # Names are what a failure analysis needs, so keep only those.
                res[k] = [t.get("function", {}).get("name", t) if isinstance(t, dict) else t
                          for t in v]
            else:
                res[k] = _sanitize_for_trace(v)
        return res
    elif isinstance(obj, list):
        return [_sanitize_for_trace(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_trace(item) for item in obj)
    elif isinstance(obj, str) and obj.startswith("data:image/") and len(obj) > 64:
        return obj[:48] + "..."
    return obj


async def _send_single_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    extra_payload: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    max_output_tokens: int = 8192,
    temperature: float = 0.0,
    thinking: bool = False,
) -> "Reply":
    """Send a single chat completion request.

    Returns a `Reply`: rendered text, raw `tool_calls`, `finish_reason` and `reasoning`.

    The raw `tool_calls` array is kept because the text rendering is lossy (it stringifies
    every argument, hiding an INTEGER answered as "2010"). `finish_reason` and `reasoning`
    are kept because without them a TRUNCATED answer and an answer that lived entirely in
    the model's reasoning channel are both indistinguishable from an ordinary wrong answer
    (audit RC-3: 22 truncated, 32 empty in the 2026-08-15 sweep, none of it visible).
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    ALLOWED_API_KEYS = {
        "tools", "tool_choice", "response_format", "top_p", "top_k", "presence_penalty",
        "frequency_penalty", "stop", "seed", "stream", "n", "logit_bias",
        # CC3: let suites pass native reasoning toggle, vision soft-token control, and
        # per-sample generation overrides through to the server (previously silently dropped).
        "chat_template_kwargs", "mm_processor_kwargs", "extra_body", "max_tokens", "temperature",
    }
    if extra_payload:
        for k, v in extra_payload.items():
            if k in ALLOWED_API_KEYS:
                payload[k] = v
    # CC2: actually toggle the model's native reasoning channel to match --eval-thinking.
    # Only inject when a suite hasn't already set enable_thinking explicitly (e.g. via a
    # sample's extra_payload), so per-suite intent still wins.
    ctk = payload.get("chat_template_kwargs")
    if not (isinstance(ctk, dict) and "enable_thinking" in ctk):
        payload["chat_template_kwargs"] = {
            **(ctk if isinstance(ctk, dict) else {}),
            "enable_thinking": bool(thinking),
        }

    last_error: Optional[str] = None
    async with semaphore:
        for attempt in range(3):
            try:
                sent_at = time.monotonic()
                async with session.post(api_url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        elapsed = max(1e-3, time.monotonic() - sent_at)
                        choice = data["choices"][0]
                        msg = choice["message"]
                        content = msg.get("content") or ""
                        tool_calls = msg.get("tool_calls")
                        finish = choice.get("finish_reason")
                        stop_reason = choice.get("stop_reason")
                        usage_tokens = int((data.get("usage") or {}).get("completion_tokens") or 0)
                        # vLLM exposes the gemma-4 reasoning channel under either key.
                        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or None
                        if tool_calls:
                            call_strs = []
                            for tc in tool_calls:
                                fn = tc.get("function", {})
                                fname = fn.get("name", "")
                                fargs_str = fn.get("arguments", "{}")
                                py_kwargs_clean = ""
                                try:
                                    import json
                                    args_dict = json.loads(fargs_str) if isinstance(fargs_str, str) else fargs_str
                                    if isinstance(args_dict, dict):
                                        py_kwargs_clean = fname + "(" + ", ".join(f"{k}={v}" for k, v in args_dict.items()) + ")"
                                except Exception:
                                    py_kwargs_clean = ""
                                call_strs.append(f"{fname}({fargs_str}) {py_kwargs_clean}")
                        rate = round(usage_tokens / elapsed, 1) if usage_tokens else None
                        if tool_calls:
                            return Reply("\n".join(call_strs) + "\n" + str(content),
                                         tool_calls, finish, reasoning, None,
                                         usage_tokens, stop_reason, rate)
                        return Reply(str(content), None, finish, reasoning, None,
                                     usage_tokens, stop_reason, rate)
                    else:
                        err = await resp.text()
                        if resp.status == 400 and ("maximum context length" in err or "input tokens" in err):
                            match_input = re.search(r"contains at least (\d+) input tokens", err)
                            match_max = re.search(r"maximum context length is (\d+) tokens", err)
                            if match_input and match_max:
                                inp_tokens = int(match_input.group(1))
                                max_len = int(match_max.group(1))
                                clamped_tokens = max(64, max_len - inp_tokens - 16)
                                if clamped_tokens < payload.get("max_tokens", max_output_tokens):
                                    payload["max_tokens"] = clamped_tokens
                                    continue
                        logger.warning(f"HTTP {resp.status} on eval request: {err[:200]}")
            except asyncio.TimeoutError:
                # Distinct from any other failure: a timeout means the generation was still
                # running, so it is the LONG answers that get dropped - exactly the ones the
                # large budgets exist to capture. Recorded so it is never a mystery failure.
                last_error = f"timeout after {request_timeout_s(max_output_tokens)}s"
                if attempt == 2:
                    logger.warning("Request timed out after %ss (max_tokens=%s). Raise "
                                   "GBENCH_REQUEST_TIMEOUT_S / lower --batch-sizes, or the "
                                   "longest answers will be lost.",
                                   request_timeout_s(max_output_tokens), max_output_tokens)
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt == 2:
                    logger.warning(f"Request exception on attempt {attempt+1}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        return Reply(None, None, None, None, last_error)


def tally_batch_scores(sample_traces: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Dict[str, int]]]:
    """Numerator + per-category tally for batch-scored (async_eval_fn) suites.

    Samples the judge could not score (`judge_grade == "JUDGE_OUTAGE"`) or the suite flagged
    unmeasurable for this endpoint (`scoring_excluded`) are excluded from BOTH the numerator and
    every per-category total, so an unscored sample never surfaces as a category "0/1 failure"
    that contradicts a headline which already excluded it.

    Returns (correct_count, category_stats) where category_stats[cat] = {"correct", "total"}.
    """
    correct_count = 0
    category_stats: Dict[str, Dict[str, int]] = {}
    for trace in sample_traces:
        if str(trace.get("judge_grade") or "") == "JUDGE_OUTAGE":
            continue
        if trace.get("scoring_excluded"):
            continue
        is_corr = bool(trace.get("is_correct", False))
        if is_corr:
            correct_count += 1
        cat = trace.get("category")
        if cat:
            st = category_stats.setdefault(cat, {"correct": 0, "total": 0})
            st["total"] += 1
            if is_corr:
                st["correct"] += 1
    return correct_count, category_stats


def classify_scoring_mode(has_fallback: bool, declared: Optional[str],
                          has_async_scorer: bool) -> str:
    """Decide the honest top-level scoring_mode label for a run.

    Precedence:
      1. `judge_fallback` if any sample was scored by the no-judge substring fallback
         (a lower bound, not the canonical metric) - this overrides everything.
      2. the suite's `declared` mode when it told us how it actually scored - e.g.
         code-execution suites (bigcodebench/multipl_e/lcb/ojbench/spider2/cyberseceval)
         batch through async_eval_fn but run tests, not an LLM judge; coco_caption batches
         a deterministic corpus CIDEr metric.
      3. `judge` if a batch scorer is present and the suite did NOT declare otherwise (the
         legacy heuristic - most async scorers are judges).
      4. `deterministic` for a plain per-sample eval_fn.
    """
    if has_fallback:
        return "judge_fallback"
    if declared:
        return declared
    if has_async_scorer:
        return "judge"
    return "deterministic"


async def _run_suite_async(
    eval_name: str,
    model_name: str,
    base_url: str,
    concurrency: int,
    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
    eval_fn: Optional[Callable[[str, Any], bool]] = None,
    async_eval_fn: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    declared_scoring_mode: Optional[str] = None,
    thinking: bool = False,
    extra_payload: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.0,
    tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    max_tool_rounds: int = 8,
    attempt_count: int = 1,
    supports_attempts: bool = False,
) -> Dict[str, Any]:
    """Execute evaluation suite samples concurrently with tqdm progress bar."""
    api_url = f"{base_url.rstrip('/')}/chat/completions"

    # WS5 sharding for horizontal scale. GBENCH_SHARD="I/N" selects shard I of N over the
    # deterministically-ordered full sample set (round-robin), so a swarm node runs a reproducible,
    # non-overlapping subset that unions back to the full set. Applied BEFORE --eval-limit, so a
    # sharded smoke run caps within the shard. n_scored/n_total on the result reflect the shard.
    _shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    if _shard:
        _before = len(samples)
        samples = shard_select(samples, _shard)
        logger.info("[%s] shard %d/%d: %d of %d samples", eval_name, _shard[0], _shard[1],
                    len(samples), _before)

    # Enforce limit if provided. Stratified across the samples' own `category` metadata,
    # not a contiguous head: benchmark rows are stored grouped by category, so `[:limit]`
    # returned one category (audit RC-1 - 56 of 93 scored suites collapsed to a single
    # category at --eval-limit 20). Deterministic: seeded on the suite name.
    if limit and limit > 0 and len(samples) > limit:
        samples = stratified_sample(
            samples, limit,
            key_fn=lambda s: (s[2] or {}).get("category") if len(s) > 2 and isinstance(s[2], dict) else None,
            seed=eval_name)

    # --- repeated attempts (@k) -------------------------------------------------------
    # Most published numbers for small benchmarks are NOT one sample: AIME is avg@4-64,
    # GPQA-Diamond avg@10, ARC-AGI pass@2 by rule, tau-bench pass^k by definition. A
    # single greedy sample is the estimate every lab deliberately avoids - and on
    # 2026-08-17 we watched `astropy__astropy-14309` flip RESOLVED -> unresolved between
    # two identical temperature=0.0 runs, which is exactly the variance those protocols
    # average away.
    #
    # Implementation: replicate each sample k times and let the existing pipeline treat
    # every attempt as an ordinary sample. `accuracy` then IS avg@k for free, because it
    # is (correct attempts / total attempts). pass@k and pass^k are derived afterwards by
    # grouping on `source_sample_idx`.
    attempts_requested = max(1, int(attempt_count or 1))
    attempts = attempts_requested
    attempt_skip_reason = None
    if attempts > 1 and async_eval_fn is not None and not supports_attempts:
        # Batch scorers receive the whole trace list at once and several key it by an id
        # (`preds[instance_id]`, `results[task_id]`). Replicating samples would collapse
        # k attempts into one prediction and hand the same verdict back to all k traces -
        # a fabricated @k. Refuse rather than report a number we did not measure.
        attempt_skip_reason = (
            "this suite scores in a batch (async_eval_fn) and has not declared "
            "supports_attempts=True; running 1 attempt instead of "
            f"{attempts_requested} rather than reporting an @k it did not measure")
        logger.warning("[%s] --attempt-count=%d ignored: %s.",
                       eval_name, attempts_requested, attempt_skip_reason)
        attempts = 1
    #: expanded index -> (index of the sample it is an attempt of, attempt number)
    attempt_of: Dict[int, Tuple[int, int]] = {}
    if attempts > 1:
        expanded = []
        for src_idx, sample in enumerate(samples):
            for a in range(attempts):
                attempt_of[len(expanded)] = (src_idx, a)
                expanded.append(sample)
        logger.info("[%s] %d sample(s) x %d attempts = %d generations; accuracy will be "
                    "reported as avg@%d, with pass@%d and pass^%d alongside.",
                    eval_name, len(samples), attempts, len(expanded), attempts,
                    attempts, attempts)
        samples = expanded
    else:
        attempt_of = {i: (i, 0) for i in range(len(samples))}

    # Global fallback for suites with no floor entry. Raised from 8192/16384: the sweep
    # truncated 10 suites that ran at the old non-thinking default, and a thinking run
    # spends most of its budget in the reasoning channel before the answer starts (the
    # whole completion shares one max_tokens).
    effective_max_tokens = (
        max_output_tokens
        if max_output_tokens is not None
        else (32768 if thinking else 16384)
    )

    tool_rounds_used: Dict[int, int] = {}
    #: samples where the tool budget ran out and a final answer was forced
    forced_finals: Dict[int, bool] = {}
    sent_budgets: Dict[int, Any] = {}
    semaphore = asyncio.Semaphore(concurrency)
    if aiohttp is None:
        raise RuntimeError(
            "The 'aiohttp' package is required for native asynchronous evaluation runners.\n"
            "Install dependencies via: pip install aiohttp (or pip install -e .)"
        )
    connector = aiohttp.TCPConnector(limit=concurrency + 20)
    timeout_s = request_timeout_s(effective_max_tokens)
    if timeout_s > REQUEST_TIMEOUT_FLOOR_S:
        logger.info("[%s] per-request timeout %ds (max_tokens=%s at >=%.0f tok/s)",
                    eval_name, timeout_s, effective_max_tokens, MIN_DECODE_TOK_S)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    correct_count = 0
    total_count = len(samples)
    failed_requests = 0
    empty_responses = 0
    truncated_responses = 0
    judge_failures = 0
    judge_outages = 0
    scoring_excluded = 0
    health_counts: Dict[str, int] = {}
    category_stats = {}

    with tqdm(total=total_count, desc=f"Eval [{eval_name.upper()}]") as pbar:
        async def _fetch_sample(idx, messages, gold_answer, sample_extra_payload):
            # Merge suite-level extra_payload with sample-level category/extra payload
            request_payload = {}
            if extra_payload:
                request_payload.update(extra_payload)
            sample_cat = None
            if isinstance(sample_extra_payload, dict):
                request_payload.update({k: v for k, v in sample_extra_payload.items() if k != "category"})
                sample_cat = sample_extra_payload.get("category")
            elif isinstance(sample_extra_payload, str):
                sample_cat = sample_extra_payload

            # Fit the request to the context window before sending, rather than paying a
            # 400 + retry (or three silent failures when the prompt alone is over the limit).
            # A sample's own meta may carry `max_tokens` (culer 512, screenspot 128,
            # mmmu_pro 8192). That is legitimate - a per-sample override - but it lands in
            # the payload AFTER the resolved budget, so the suite-level number in the logs
            # was not what the server saw: culer logged `max_tokens=16384` while every
            # request sent 512, and its "8/20 truncated" was against 512, not 16384.
            # Resolve it here so one value is used, reported and clamped.
            declared = request_payload.get("max_tokens")
            budget = declared if isinstance(declared, int) and declared > 0 else effective_max_tokens
            sample_max_tokens, oversize = clamp_to_context(messages, budget)
            sent_budgets[idx] = sample_max_tokens
            if oversize:
                if pbar is not None:
                    pbar.update(1)
                return (idx, Reply(None, None, None, None), gold_answer, sample_cat,
                        messages, sample_extra_payload, oversize)

            # Agentic loop. A web-research suite cannot be answered in one shot: the model
            # asks for a search, reads the results, and asks again. Without it `gaia` and
            # `deepsearch_qa` reply "I do not have access to the internet" and score a
            # structural 0. Only runs when the suite supplies an executor; single-turn
            # suites take the same path they always did (one request, no extra state).
            convo = list(messages)
            rounds = 0
            while True:
                reply = await _send_single_request(
                    session=session,
                    api_url=api_url,
                    model_name=model_name,
                    messages=convo,
                    extra_payload=request_payload,
                    semaphore=semaphore,
                    pbar=pbar if rounds == 0 else None,
                    max_output_tokens=sample_max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                )
                if tool_executor is None or not reply.tool_calls:
                    # A turn that ran out of budget INSIDE the reasoning channel leaves no
                    # content and no tool call, so this break used to end the sample with
                    # nothing at all - the judge saw an empty response and recorded
                    # JUDGE_FAILED, which is indistinguishable from an outage. Measured
                    # 2026-08-18 on browsecomp: 2/3 samples reasoned 28,538 and 25,618
                    # characters into the 8192-token turn and emitted not one character of
                    # answer. Same remedy as running out of rounds, and for the same
                    # reason: ask once more, with thinking OFF so the model has to open the
                    # content channel, and grade whatever it commits to. A wrong answer is
                    # a real result; silence is not.
                    if (tool_executor is not None and reply.finish_reason == "length"
                            and not (reply.text or "").strip()):
                        convo.append({"role": "user", "content": (
                            "You ran out of room to think. Do not reason further. State "
                            "your best answer now from what you already have, in the "
                            "requested format.")})
                        final_max, oversize = clamp_to_context(convo, budget)
                        if not oversize:
                            reply = await _send_single_request(
                                session=session, api_url=api_url, model_name=model_name,
                                messages=convo, extra_payload={
                                    k: v for k, v in request_payload.items()
                                    if k not in ("tools", "tool_choice")},
                                semaphore=semaphore, pbar=None,
                                max_output_tokens=final_max, temperature=temperature,
                                thinking=False,
                            )
                            forced_finals[idx] = True
                            reply = await _retry_discarded_final(
                                reply=reply, convo=convo, session=session, api_url=api_url,
                                model_name=model_name,
                                payload={k: v for k, v in request_payload.items()
                                         if k not in ("tools", "tool_choice")},
                                semaphore=semaphore, budget=budget,
                                temperature=temperature)
                    break
                if rounds >= max_tool_rounds:
                    # Out of rounds while the model is still calling tools. Breaking here
                    # returns a reply whose only content is a TOOL CALL, which then gets
                    # flattened to text and graded as the answer - measured 2026-08-18 on
                    # browsecomp, where a 183-character `web_search({...})` was scored
                    # against the gold. Give it one final turn with the tools withdrawn so
                    # it commits to an answer; a wrong answer is a real result, a dangling
                    # tool call is not.
                    convo.append({"role": "assistant", "content": reply.text or "",
                                  "tool_calls": reply.tool_calls})
                    convo.append({"role": "user", "content": (
                        "No further tool calls are possible. Answer now, using only what "
                        "you already have, and state your final answer explicitly.")})
                    # Withdraw the tools so the model cannot simply call one again.
                    no_tools_payload = {k: v for k, v in request_payload.items()
                                        if k not in ("tools", "tool_choice")}
                    final_max, oversize = clamp_to_context(convo, budget)
                    if not oversize:
                        reply = await _send_single_request(
                            session=session, api_url=api_url, model_name=model_name,
                            messages=convo, extra_payload=no_tools_payload,
                            semaphore=semaphore, pbar=None,
                            max_output_tokens=final_max, temperature=temperature,
                            thinking=thinking,
                        )
                        forced_finals[idx] = True
                        # Withdrawing the tools does not stop gemma-4 emitting a tool call -
                        # it just leaves the call nowhere to go, so vLLM's parser strips it
                        # out of `content` and the turn returns NOTHING. Measured on the
                        # 2026-08-20 mcp_atlas run: 35 of 152 forced finals (23%) came back
                        # `output_discarded`, i.e. 7% of the suite scored zero on an answer
                        # the model had already paid to generate. The prose instruction
                        # above is evidently not enough on its own; say it in the imperative
                        # and give it one more turn.
                        reply = await _retry_discarded_final(
                            reply=reply, convo=convo, session=session, api_url=api_url,
                            model_name=model_name, payload=no_tools_payload,
                            semaphore=semaphore, budget=budget, temperature=temperature)
                    break
                rounds += 1
                convo.append({"role": "assistant", "content": reply.text or "",
                              "tool_calls": reply.tool_calls})
                for call in reply.tool_calls:
                    fn = (call.get("function") or {})
                    raw_args = fn.get("arguments")
                    try:
                        parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        parsed = {}
                    try:
                        # Stateful tool environments (a notebook kernel, a scratch
                        # filesystem) need to keep state PER SAMPLE - concurrent samples
                        # sharing one dict would interleave their cells. Executors that
                        # want it declare a third parameter; the two-argument ones are
                        # unaffected.
                        if _executor_wants_session(tool_executor):
                            out = await tool_executor(fn.get("name", ""), parsed, idx)
                        else:
                            out = await tool_executor(fn.get("name", ""), parsed)
                    except Exception as e:                     # a broken tool is reported,
                        out = json.dumps({"error": f"tool raised {type(e).__name__}: {e}"})
                    convo.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                  "name": fn.get("name", ""), "content": str(out)})
                # the follow-up must fit too, now that the transcript has grown
                sample_max_tokens, oversize = clamp_to_context(convo, budget)
                if oversize:
                    break
            if rounds:
                tool_rounds_used[idx] = rounds
            return (idx, reply, gold_answer, sample_cat, messages, sample_extra_payload, None)

        sample_traces = []
        completed = 0
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            coroutines = [
                _fetch_sample(idx, sample[0], sample[1], sample[2] if len(sample) > 2 else {})
                for idx, sample in enumerate(samples)
            ]
            for future in asyncio.as_completed(coroutines):
                (idx, reply, gold_answer, category, messages,
                 sample_extra_payload, skip_reason) = await future
                resp_text, tool_calls = reply.text, reply.tool_calls
                health = classify_reply(reply)
                health_counts[health] = health_counts.get(health, 0) + 1
                completed += 1
                pbar.update(1)

                if resp_text is None:
                    failed_requests += 1
                    is_correct = False
                    # Record WHY. A bare `request_failed` with `error: None` is what mrcr's
                    # six over-context rows looked like on the 2026-08-15 sweep.
                    if skip_reason:
                        status = "OVER_CONTEXT"
                    elif health == "request_timeout":
                        status = "TIMEOUT"
                        skip_reason = reply.error
                    else:
                        status = "FAILED"
                        skip_reason = reply.error
                elif health in ("truncated", "looping"):
                    # The generation hit the output-token budget mid-answer. Scored on
                    # what arrived (it may still be right), but counted so the run says so
                    # instead of reporting a quietly depressed accuracy.
                    truncated_responses += 1
                    cleaned_pred = strip_thinking_tags(resp_text)
                    if async_eval_fn is not None:
                        is_correct, status = False, "PENDING_JUDGE"
                    else:
                        is_correct = _apply_eval_fn(eval_fn, cleaned_pred, gold_answer, tool_calls)
                        if is_correct:
                            correct_count += 1
                        status = "TRUNCATED"
                elif not str(resp_text).strip() and not tool_calls:
                    # The server answered 200 with empty `content`. That is not a wrong
                    # answer, it is a request that produced nothing - most often the
                    # generation hitting the output-token budget mid-reasoning. Scored as
                    # incorrect either way, but counted so the run reports
                    # `completed_with_errors` instead of a quietly depressed accuracy.
                    empty_responses += 1
                    failed_requests += 1
                    is_correct = False
                    status = "EMPTY_RESPONSE"
                else:
                    cleaned_pred = strip_thinking_tags(resp_text)
                    if async_eval_fn is not None:
                        is_correct = False
                        status = "PENDING_JUDGE"
                    else:
                        is_correct = _apply_eval_fn(eval_fn, cleaned_pred, gold_answer, tool_calls)
                        if is_correct:
                            correct_count += 1
                        status = "OK"

                if category and async_eval_fn is None:
                    if category not in category_stats:
                        category_stats[category] = {"correct": 0, "total": 0}
                    category_stats[category]["total"] += 1
                    if is_correct:
                        category_stats[category]["correct"] += 1

                sample_traces.append({
                    "sample_idx": idx,
                    "source_sample_idx": attempt_of.get(idx, (idx, 0))[0],
                    "attempt_idx": attempt_of.get(idx, (idx, 0))[1],
                    "category": category,
                    "messages": _sanitize_for_trace(messages),
                    "extra_payload": _sanitize_for_trace(sample_extra_payload) if sample_extra_payload else None,
                    "gold_answer": gold_answer,
                    "response_text": resp_text,
                    "tool_calls": tool_calls,
                    # Diffable form of the same thing; see `normalize_function_calls`.
                    "emitted_function_calls": normalize_function_calls(tool_calls),
                    "finish_reason": reply.finish_reason,
                    "health": health,
                    "reasoning_chars": len(reply.reasoning or ""),
                    "reasoning": reply.reasoning,   # thinking content (None on no-think)
                    "repetition_run": repetition_run(resp_text),
                    "repetition_onset": repetition_onset(resp_text),
                    "max_tokens_sent": sent_budgets.get(idx),
                    "completion_tokens": reply.completion_tokens,
                    "decode_tok_s": reply.decode_tok_s,
                    "stop_reason": reply.stop_reason,
                    "error": skip_reason,
                    "is_correct": is_correct,
                    "status": status,
                })

                if async_eval_fn is None:
                    pbar.set_postfix(
                        correct=f"{correct_count}/{completed} ({correct_count / completed * 100.0:.1f}%)"
                    )

    # Phase 2: Post-generation batch/async judging if provided
    if async_eval_fn is not None:
        await async_eval_fn(sample_traces)
        correct_count = 0
        category_stats = {}
        # A grader that never returned a verdict is a harness failure, not a "no". Without
        # this the run still reported `success` and the unjudged samples simply counted as
        # incorrect, so a judge outage looked like a low score.
        # Only an explicit grader marker counts. `status == "FAILED"` must NOT be treated as
        # a judge failure: execution-based suites use it for an ordinary wrong answer (lcb
        # sets it whenever a solution does not pass its tests), so inferring from it counted
        # every incorrect sample as a harness error and downgraded clean runs to
        # `completed_with_errors`.
        # Only markers that mean THE JUDGE failed. An empty response is not one: the
        # judge is never called, and the sample is already counted under
        # `empty_responses` / `failed_requests` / `discarded_outputs`. Adding it here
        # would blame grading for a generation problem and count the same sample a
        # fourth time.
        JUDGE_FAILURE_GRADES = ("judge_error", "unparsed", "JUDGE_FAILED")
        judge_failures = sum(
            1 for t in sample_traces
            if str(t.get("judge_grade") or "") in JUDGE_FAILURE_GRADES)
        failed_requests += judge_failures
        # JUDGE_OUTAGE is a distinct outcome: the judge (Gemini) cascade was exhausted for
        # this sample - an infrastructure outage, NOT a model failure. It is excluded from
        # BOTH the accuracy numerator and denominator (below) and reported on its own, so an
        # outage can never masquerade as a wrong answer. Kept separate from
        # JUDGE_FAILURE_GRADES (parse/format errors that count as failed_requests).
        judge_outages = sum(
            1 for t in sample_traces
            if str(t.get("judge_grade") or "") == "JUDGE_OUTAGE")
        # A suite may declare a sample UNMEASURABLE for this endpoint (e.g. gdpval: a task
        # whose whole rubric grades a produced file a text model cannot emit). Like a judge
        # outage, it is excluded from the numerator, the denominator AND the per-category
        # accuracy - counting it as a category "0/1 failure" contradicts a headline that says
        # it was not scored at all.
        scoring_excluded = sum(1 for t in sample_traces if t.get("scoring_excluded"))
        correct_count, category_stats = tally_batch_scores(sample_traces)

    sample_traces.sort(key=lambda x: x["sample_idx"])

    # Judge outages AND suite-declared unmeasurable samples are excluded from the denominator:
    # accuracy is over the samples that were actually scored, not over ones an outage prevented
    # scoring or the suite flagged as impossible to measure for this endpoint.
    scored_count = total_count - judge_outages - scoring_excluded
    accuracy = (correct_count / scored_count * 100.0) if scored_count > 0 else 0.0

    # --- @k metrics -------------------------------------------------------------------
    # `accuracy` above is already avg@k (correct attempts / total attempts). The other two
    # shapes are what the remaining protocols ask for and cannot be derived from it:
    #   pass@k  - at least one attempt correct   (ARC-AGI's rule, OJBench, aider)
    #   pass^k  - EVERY attempt correct          (tau-bench's reliability metric)
    attempts_report = None
    if attempts > 1:
        by_source: Dict[int, List[bool]] = {}
        for t in sample_traces:
            by_source.setdefault(t.get("source_sample_idx", t["sample_idx"]), []).append(
                bool(t.get("is_correct")))
        n_src = len(by_source) or 1
        any_ok = sum(1 for v in by_source.values() if any(v))
        all_ok = sum(1 for v in by_source.values() if v and all(v))
        # How often the k attempts disagreed with each other. This is the run's own
        # measurement of its reproducibility, and the reason to use @k at all.
        unstable = sum(1 for v in by_source.values() if any(v) and not all(v))
        attempts_report = {
            "attempts_requested": attempts_requested,
            "attempts_per_sample": attempts,
            "samples": n_src,
            "generations": total_count,
            "avg_at_k": round(accuracy, 2),
            "pass_at_k": round(any_ok / n_src * 100.0, 2),
            "pass_hat_k": round(all_ok / n_src * 100.0, 2),
            "unstable_samples": unstable,
        }
        logger.info(
            "[%s] avg@%d %.2f%% | pass@%d %.2f%% | pass^%d %.2f%% over %d sample(s); "
            "%d sample(s) gave different verdicts across attempts.",
            eval_name, attempts, accuracy, attempts, any_ok / n_src * 100.0,
            attempts, all_ok / n_src * 100.0, n_src, unstable)
    elif attempts_requested > 1:
        attempts_report = {
            "attempts_requested": attempts_requested,
            "attempts_per_sample": 1,
            "not_applied": attempt_skip_reason,
        }

    category_accuracy = {}
    for cat, stats in category_stats.items():
        cat_acc = (stats["correct"] / stats["total"] * 100.0) if stats["total"] > 0 else 0.0
        category_accuracy[cat] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": round(cat_acc, 2),
        }

    # CC6: never emit a clean "success" 0% when nothing was actually scored. Zero samples
    # (loader produced nothing / dataset failed to materialize) is an error, not a 0% pass.
    if total_count == 0:
        status_str = "error"
    elif failed_requests == total_count or (judge_outages >= total_count and total_count > 0):
        # nothing was scored - every sample either failed generation or hit a judge outage
        status_str = "failed"
    elif failed_requests == 0 and judge_outages == 0 and truncated_responses * 10 <= total_count:
        status_str = "success"
    else:
        # More than 10% of answers cut off at the token budget is a measurement problem,
        # not a model result: those samples were never given the chance to be right.
        status_str = "completed_with_errors"
    # A suite that fell back to substring matching because the judge was unavailable must
    # not report its number as the canonical metric (audit RC-5).
    fallback = sum(1 for t in sample_traces if t.get("scoring_mode") == "judge_fallback")
    scoring_mode = classify_scoring_mode(
        has_fallback=bool(fallback),
        declared=declared_scoring_mode,
        has_async_scorer=async_eval_fn is not None,
    )
    if fallback:
        logger.warning(
            "[%s] %d/%d samples were graded by the no-judge fallback (substring match), "
            "not the canonical judge. This is a lower bound, not the benchmark's metric.",
            eval_name, fallback, total_count)

    over_context = sum(1 for t in sample_traces if t.get("status") == "OVER_CONTEXT")
    discarded = sum(1 for t in sample_traces if t.get("health") == "output_discarded")
    if discarded:
        logger.warning(
            "[%s] %d/%d responses were GENERATED BUT DISCARDED by the server: the model "
            "emitted a tool call and the tool-call parser removed it, but the request "
            "declared no `tools` so there was nowhere to put it. These are not empty "
            "answers. Declare the suite's tool schemas in the sample meta "
            "(`{\"tools\": [...]}`) to capture them.", eval_name, discarded, total_count)
    timed_out = sum(1 for t in sample_traces if t.get("status") == "TIMEOUT")
    if timed_out:
        logger.warning(
            "[%s] %d/%d requests timed out at %ds. Timeouts drop the LONGEST answers, so "
            "the accuracy is biased downward. Raise GBENCH_REQUEST_TIMEOUT_S or lower "
            "--batch-sizes before quoting this.",
            eval_name, timed_out, total_count, request_timeout_s(effective_max_tokens))
    if over_context:
        logger.warning(
            "[%s] %d/%d prompts exceed the model's context window and could not be sent. "
            "These are a dataset/serving mismatch, not wrong answers - they are excluded "
            "from nothing, so the accuracy denominator still counts them.",
            eval_name, over_context, total_count)

    looping = sum(1 for t in sample_traces if t.get("health") == "looping")
    if looping:
        onsets = [t["repetition_onset"] for t in sample_traces
                  if t.get("health") == "looping" and t.get("repetition_onset")]
        detail = ""
        if onsets:
            fracs = sorted(o["onset_frac"] for o in onsets)
            periods = sorted(o["period_words"] or 0 for o in onsets)
            detail = (f" Onset at {fracs[len(fracs)//2]:.0%} through the output (range "
                      f"{fracs[0]:.0%}-{fracs[-1]:.0%}), cycle period "
                      f"{periods[len(periods)//2]} words. Stuck on: "
                      f"{onsets[0]['cycle_preview'][:70]!r}.")
        logger.warning(
            "[%s] %d/%d responses hit the cap while REPEATING THEMSELVES (a 20-word window "
            "recurring >=%d times), not while writing a longer answer. Raising "
            "--max-output-tokens buys more of the same and costs wall-clock; this is a "
            "decoding/model behaviour, not a budget problem.%s",
            eval_name, looping, total_count, REPETITION_THRESHOLD, detail)

    # Every timeout in this harness is sized from an ASSUMED decode rate. Nothing checked
    # that assumption against the box actually running, so on slower hardware a suite could
    # only fail as a bare timeout with no hint that the sizing, not the model, was wrong.
    # The rate is measured for free from `completion_tokens / wall time`, so state it, and
    # say so when reality is under the assumption.
    rates = sorted(t["decode_tok_s"] for t in sample_traces if t.get("decode_tok_s"))
    observed_rate = rates[len(rates) // 2] if rates else None
    if observed_rate is not None and observed_rate < MIN_DECODE_TOK_S:
        logger.warning(
            "[%s] measured decode rate is %.1f tok/s, BELOW the %.1f tok/s the request "
            "timeout assumes. Long generations on this endpoint will time out and be "
            "dropped, which biases the score downward. Lower GBENCH_MIN_DECODE_TOK_S (it "
            "sizes the timeout) or reduce --batch-sizes.",
            eval_name, observed_rate, MIN_DECODE_TOK_S)

    # Looping responses also hit the cap, but "raise the budget" is exactly the wrong
    # advice for them - they are counted and explained above instead. Same for responses
    # that ran many times longer than this suite's healthy median without converging.
    non_convergent = reclassify_non_convergent(sample_traces)
    if non_convergent:
        logger.warning(
            "[%s] %d/%d responses hit the cap while running >=%.0fx this suite's healthy "
            "median length WITHOUT repeating verbatim - the model kept re-deriving instead "
            "of converging. Raising --max-output-tokens buys more of the same; these are "
            "reported as `non_convergent`, not `truncated`.",
            eval_name, non_convergent, total_count, NON_CONVERGENT_LENGTH_RATIO)
    genuinely_truncated = truncated_responses - looping - non_convergent
    if genuinely_truncated > 0:
        # Report the budget the SERVER saw, not the suite-level resolution: a sample's meta
        # can override it, and quoting the wrong number sent culer's investigation astray.
        hit = sorted({t.get("max_tokens_sent") for t in sample_traces
                      if t.get("status") == "TRUNCATED" and t.get("max_tokens_sent")})
        shown = ", ".join(str(h) for h in hit) or str(effective_max_tokens)
        if tool_rounds_used:
            # "Raise the budget" is the wrong advice once a suite runs a tool loop: the
            # budget bounds ONE TURN, and browsecomp at 65536/turn produced no answer at
            # all where 8192/turn produced a judged one. A truncated turn here means the
            # model chose to spend the turn thinking rather than acting.
            logger.warning(
                "[%s] %d/%d TURNS hit the per-turn budget (max_tokens=%s). In a tool loop "
                "this bounds one turn, not the run - raising it lets the model think for "
                "longer without acting, which is how this suite stopped answering at all. "
                "Adjust GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS if the turn is genuinely too "
                "small (tool calls per sample will be near zero).",
                eval_name, genuinely_truncated, total_count, shown)
        else:
            logger.warning(
                "[%s] %d/%d responses hit the output-token budget (max_tokens=%s) and were "
                "scored on a partial answer. Raise --max-output-tokens before quoting this.",
                eval_name, genuinely_truncated, total_count, shown)

    return {
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "thinking": thinking,
        "total_questions": total_count,
        "correct_answers": correct_count,
        "failed_requests": failed_requests,
        "empty_responses": empty_responses,
        "truncated_responses": truncated_responses,
        "non_convergent_responses": non_convergent,
        "genuinely_truncated": max(0, truncated_responses - looping),
        "max_tokens_sent": sorted({t.get("max_tokens_sent") for t in sample_traces
                                   if t.get("max_tokens_sent")}) or None,
        "over_context_prompts": over_context,
        "timed_out_requests": timed_out,
        "looping_responses": looping,
        "discarded_outputs": discarded,
        "request_timeout_s": request_timeout_s(effective_max_tokens),
        # What the harness assumed vs what this endpoint actually did. Every derived
        # timeout follows from `assumed`; publishing both is what makes a timeout on
        # unfamiliar hardware diagnosable instead of a bare failure.
        "decode_tok_s": {"observed_median": observed_rate,
                         "assumed_floor": MIN_DECODE_TOK_S} if observed_rate else None,
        "tool_rounds": {"samples_using_tools": len(tool_rounds_used),
                        "total_rounds": sum(tool_rounds_used.values()),
                        # samples that exhausted the round budget and were made to commit to
                        # an answer instead of being graded on a dangling tool call
                        "forced_final_answers": len(forced_finals),
                        # the budget each TURN was given, which in a loop is not the run's
                        "per_turn_max_tokens": effective_max_tokens} if tool_rounds_used else None,
        "scoring_mode": scoring_mode,
        "judge_failures": judge_failures,
        "judge_outages": judge_outages,
        "judge_outage_rate": round(judge_outages / total_count, 4) if total_count else 0.0,
        "scoring_excluded": scoring_excluded,
        "response_health": health_counts,
        "attempts": attempts_report,
        "accuracy": round(accuracy, 2),
        "category_accuracy": category_accuracy,
        "sample_traces": sample_traces,
        "status": status_str,
    }


def run_eval_suite(
    eval_name: str,
    model_name: str,
    base_url: str,
    concurrency: int,
    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
    eval_fn: Optional[Callable[[str, Any], bool]] = None,
    async_eval_fn: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    declared_scoring_mode: Optional[str] = None,
    thinking: bool = False,
    extra_payload: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    max_tool_rounds: int = 8,
    attempt_count: Optional[int] = None,
    supports_attempts: bool = False,
) -> Dict[str, Any]:
    """Synchronous entry point to run an evaluation suite.

    `temperature`, `max_output_tokens` and `attempt_count` fall back to the run-level
    knobs when the suite does not pass them, so `--temperature` / `--attempt-count` reach
    every suite rather than only the 15 that happened to forward them (audit RC-2).

    `supports_attempts` is opt-in and only consulted for batch-scored suites: a scorer
    that keys its predictions by instance id cannot be handed k copies of the same
    sample. See `_run_suite_async` for why that would fabricate an @k.
    """
    temperature, temperature_source = resolve_temperature(eval_name, temperature,
                                                          thinking=thinking)
    if temperature_source.startswith("env:"):
        logger.info("[%s] temperature %.2f from %s (overrides --temperature)",
                    eval_name, temperature, temperature_source.split(":", 1)[1])
    if attempt_count is None:
        attempt_count = get_run_knob("attempt_count", 1)
    penalties = decoding_penalties()
    if penalties:
        extra_payload = {**(extra_payload or {}), **penalties}
        logger.warning("Decoding penalties active: %s. Canonical protocol is greedy with no "
                       "penalty, so these numbers are NOT comparable with published results "
                       "or with runs that did not set them.", penalties)
    knob = get_run_knob("max_output_tokens")
    # The floor must know about thinking: reasoning and the answer share one budget, so the
    # non-thinking floor cuts the trace off before the answer starts.
    default_floor = THINKING_MIN_OUTPUT_TOKENS if thinking else DEFAULT_MIN_OUTPUT_TOKENS
    floor = max(SUITE_MIN_OUTPUT_TOKENS.get(eval_name, 0), default_floor)
    if knob:
        max_output_tokens = knob          # operator was explicit; never override it
    elif floor and (max_output_tokens is None or max_output_tokens < floor):
        if max_output_tokens is not None:
            logger.info("[%s] raising max_output_tokens %d -> %d (%s floor)",
                        eval_name, max_output_tokens, floor,
                        "thinking" if thinking else "long-answer suite")
        max_output_tokens = floor
    turn_ceiling = tool_loop_turn_ceiling(tool_executor is not None, max_output_tokens)
    if turn_ceiling:
        logger.warning(
            "[%s] capping the PER-TURN budget %d -> %d: this suite runs a tool loop, so "
            "`max_output_tokens` bounds one turn, not the run. Set "
            "GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS=0 to disable.",
            eval_name, max_output_tokens, turn_ceiling)
        max_output_tokens = turn_ceiling
    start_time = time.time()
    result = asyncio.run(
        _run_suite_async(
            eval_name=eval_name,
            model_name=model_name,
            base_url=base_url,
            concurrency=concurrency,
            samples=samples,
            eval_fn=eval_fn,
            async_eval_fn=async_eval_fn,
            declared_scoring_mode=declared_scoring_mode,
            thinking=thinking,
            extra_payload=extra_payload,
            limit=limit,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
            attempt_count=attempt_count,
            supports_attempts=supports_attempts,
        )
    )
    result["duration_s"] = round(time.time() - start_time, 2)
    result["decoding_penalties"] = penalties or None
    result["temperature"] = temperature
    result["temperature_source"] = temperature_source
    return result


def gemini_key_live_valid(key: str) -> Tuple[bool, str]:
    """Live ping of the Gemini OpenAI-compat endpoint. Returns (False, reason) ONLY on a definitive
    auth rejection; a network/other error is inconclusive -> (True, ...) so a valid key is never
    blocked by gate-time infra flakiness. Shared by every judge suite (gaia2/wildclawbench import it)
    so the presence+live-ping gate is uniform.

    `key` may be a comma/semicolon-separated LIST (a multi-key GEMINI_API_KEY): it is live when ANY
    member is live, so one dead/expired key in the list never blocks the run."""
    cands = [k.strip() for k in re.split(r"[;,]", str(key or "")) if k.strip()]
    if not cands:
        return False, "no GEMINI_API_KEY"
    if len(cands) > 1:
        reasons = []
        for k in cands:
            ok, why = _single_key_live(k)
            if ok:
                return True, ""
            reasons.append(why)
        return False, "; ".join(reasons)
    return _single_key_live(cands[0])


def _single_key_live(key: str) -> Tuple[bool, str]:
    """Live ping of ONE key (see gemini_key_live_valid for the (False, reason)-only-on-auth contract)."""
    base = os.environ.get("GEMINI_OPENAI_BASE_URL",
                          "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")
    req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return (200 <= getattr(r, "status", 200) < 300), ""
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"HTTP {e.code}"
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")[:300]
        except Exception:
            pass
        # Gemini returns 400 for a bad key, e.g. {"error":{"message":"Please pass a valid API key",
        # "status":"INVALID_ARGUMENT"}} or "API key not valid" / "API_KEY_INVALID". Any 400 that
        # mentions the API key is an auth rejection; other 400s are inconclusive (malformed request).
        bl = body.lower()
        if e.code == 400 and ("api key" in bl or "api_key" in bl):
            return False, f"HTTP 400 {body[:120]}"
        return True, f"inconclusive HTTP {e.code}"
    except Exception as e:
        return True, f"inconclusive {e}"


def gemini_required_skip(eval_name: str, model_name: str) -> None:
    """HARD-ERROR (infra_required) if GEMINI_API_KEY is absent OR live-rejected, else return None.

    An LLM-judge suite needs the key for canonical grading; per the no-skip policy a missing judge
    key is missing INFRA and must hard-error (never a silent skip, never a heuristic downgrade) so
    a judge-less run is loudly flagged as status:"error", not omitted as status:"skipped". The
    presence check is followed by a live auth ping (gemini_key_live_valid) so a present-but-invalid /
    expired / quota-exhausted key fails FAST at gate time instead of fail-slow after a full
    (up to 880-prompt) generation run turns every sample into a JUDGE_OUTAGE. A network blip at
    gate time is inconclusive and never blocks a valid key.

    Historical name kept to avoid churn across ~15 call sites, which use
    ``skip = gemini_required_skip(...); if skip: return skip`` - the call now RAISES on a missing or
    rejected key, so that guard is a no-op (this returns None when the key is present and valid).
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        from .swebench_common import infra_required
        raise infra_required(
            eval_name,
            "requires GEMINI_API_KEY for canonical LLM-judge grading",
            f"docs/evals/{eval_name}.md")
    ok, why = gemini_key_live_valid(key)
    if not ok:
        from .swebench_common import infra_required
        raise infra_required(
            eval_name,
            f"GEMINI_API_KEY was rejected by the judge endpoint ({why}); a valid key is required "
            "for canonical LLM-judge grading",
            f"docs/evals/{eval_name}.md")
    return None
