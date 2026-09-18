# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: mcp_atlas
# Description: MCP-Atlas (Scale AI Multi-Turn Agentic MCP Tool-Use Evaluation across 20+ MCP Servers)

"""gbench native built-in runner for mcp_atlas (Tool Use & Function Calling).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MCP_ATLAS_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import ast
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .search_tool import (backend_tally, reset_backend_tally,
                          search_available, search_backend_name)
from .base import run_eval_suite, DEFAULT_JUDGE_MODEL, judge_config, judge_generate_cascade, gemini_required_skip
from .sampling import stratified_sample

logger = logging.getLogger(__name__)

PILLAR = "Tool Use & Function Calling"


def _parse_list_field(raw: Any) -> List[str]:
    """MCP-Atlas stores `ENABLED_TOOLS` / `GTFA_CLAIMS` as a *string* holding a list.

    The literal uses Python quoting (mixed `'`/`"`), so `json.loads` alone is not enough.
    """
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        if isinstance(value, list):
            return [s for s in (_element_name(x) for x in value) if s]
    return [text]


def _element_name(x: Any) -> str:
    """One list element as a string, unwrapping the dict form `ENABLED_TOOLS` sometimes uses.

    5 of the 500 rows ship their tools as objects rather than bare names -
    `{'name': 'mongodb_list-collections', 'optionalParams': [], 'requiredParams': ['database']}`.
    `str(x)` turned the whole repr into the tool NAME, which is not a legal OpenAI function
    name (`^[a-zA-Z0-9_-]{1,64}$`), so those tools could not be called at all. The
    `requiredParams` those rows carry are still discarded - the stub schema below is
    deliberately uniform across all 500 rows rather than richer on 1% of them.
    """
    if isinstance(x, dict):
        return str(x.get("name") or "").strip()
    return str(x).strip()


def _load_mcp_atlas_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MCP-Atlas benchmark dataset directly from HF Hub (ScaleAI/MCP-Atlas)."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("ScaleAI/MCP-Atlas", split="train")
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for mcp_atlas: {e}")
        raise RuntimeError(f"Could not load dataset for mcp_atlas: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for mcp_atlas returned empty rows")

    # Stratified, not a contiguous head (audit RC-1).
    rows = stratified_sample(rows, limit, None, seed="mcp_atlas")

    samples = []
    for item in rows:
        prompt_text = str(item.get("PROMPT") or "").strip()
        enabled_tools = _parse_list_field(item.get("ENABLED_TOOLS"))
        gtfa_claims = _parse_list_field(item.get("GTFA_CLAIMS"))
        if not prompt_text or not gtfa_claims:
            continue

        # Both columns are *strings* holding a list literal. Slicing the raw value took the
        # first ten CHARACTERS and joined them, so the prompt advertised the tool list as
        # `[, ", f, e, t, c, h, _, f, e`; and the claims never split, so the whole literal
        # (brackets and quotes included) was one "claim".
        tools_str = ", ".join(enabled_tools) if enabled_tools else "none"
        server = enabled_tools[0].split("_")[0] if enabled_tools else "mcp"

        prompt = (
            f"MCP tools available on this task: {tools_str}\n\n"
            f"User Task: {prompt_text}\n\n"
            "Answer the task. State every fact your answer depends on explicitly, and "
            "finish with your conclusion on the last line as: Final Answer: <response>"
        )
        messages = [{"role": "user", "content": prompt}]
        # Declare the named tools on the request. The prompt already advertises them, so
        # the model tries to call one; gemma-4 then emits `<|tool_response>` (a stop token)
        # and vLLM's tool-call parser lifts the call out of `content`. With no `tools` on
        # the request the extraction is DISCARDED - measured live as completion_tokens=34,
        # content=null, tool_calls=[], stop_reason=50, i.e. 13/20 "empty responses" that
        # were really answers thrown away. MCP-Atlas ships names but no JSON schemas, so
        # these are permissive stubs: enough for the parser to bind the call.
        meta = {"category": server, "claims": gtfa_claims}
        if enabled_tools:
            meta["tools"] = [{"type": "function",
                              "function": {"name": t,
                                           "description": f"MCP tool {t}.",
                                           "parameters": {"type": "object",
                                                          "properties": {},
                                                          "additionalProperties": True}}}
                             for t in enabled_tools if isinstance(t, str) and t.strip()]
        samples.append((messages, json.dumps(gtfa_claims), meta))

    logger.info(f"Loaded {len(samples)} mcp_atlas samples.")
    return samples


#: The official per-claim rubric, copied verbatim from `services/scoring/score_claims.py`
#: in scaleapi/mcp-atlas@main (`CoverageEvaluator._get_single_claim_evaluation_prompt`).
#: It grades ONE claim at a time - that is not an implementation detail, it is what the
#: scorer does, and batching all claims into a single call changes the judgements.
_CLAIM_JUDGE_PROMPT = """You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.
NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, +/-1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match
CLAIM TO EVALUATE:
{claim}
MODEL RESPONSE TO ANALYZE:
{response}
INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
   - When numbers differ slightly, note if they're within acceptable range
5. Provide a confidence level (0.0-1.0) for your assessment
Be rigorous but fair in your assessment. Focus on whether the response conveys the same information as the claim, not on exact numerical precision unless precision is critical to the claim's meaning.

Reply with ONLY a JSON object: {{"coverage_outcome": "fulfilled"|"partially_fulfilled"|"not_fulfilled", "confidence_level": <0.0-1.0>}}"""

#: `coverage_to_score` from the official evaluator. The middle rung is the whole point:
#: a binary supported/not-supported judgement discards every partial answer.
_COVERAGE_TO_SCORE = {"fulfilled": 1.0, "partially_fulfilled": 0.5, "not_fulfilled": 0.0}

#: Official secondary metrics: `pass_rate_0.50` / `pass_rate_0.75` in `_coverage_stats`.
_PASS_THRESHOLDS = (0.50, 0.75)


def _eval_mcp_atlas(response_text: str, gold_target: str) -> bool:
    """Deterministic floor for MCP-Atlas: every ground-truth claim stated verbatim.

    NOT the scoring authority - `_async_judge_mcp_atlas` is. MCP-Atlas grades an answer
    against `GTFA_CLAIMS`, natural-language facts the answer must assert ("The domain
    registration year of the AssaultCube's official site is 2006."), which is a semantic
    judgement. The previous scorer accepted a claim when 75% of its 4+ character tokens
    appeared anywhere in the response, so echoing the task's own nouns largely satisfied it
    while the decisive value (the year) was never checked. With no judge available this
    falls back to exact containment of each claim, which under-credits paraphrase but
    cannot invent a pass.
    """
    claims = _parse_list_field(gold_target)
    if not response_text or not claims:
        return False
    resp_lower = response_text.lower()
    return all(c.lower() in resp_lower for c in claims)


async def _async_judge_mcp_atlas(
    sample_traces: List[Dict[str, Any]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_concurrency: int = 32,
) -> None:
    """Score claim coverage the way the official harness does.

    Matched against `services/scoring/score_claims.py` in scaleapi/mcp-atlas@main
    (fetched 2026-08-20), which the dataset card names as the released eval harness:

    * **one judge call per claim**, not one per task
    * a **ternary** outcome - fulfilled 1.0 / partially_fulfilled 0.5 / not_fulfilled 0.0
    * per-task ``coverage_score = round(total_score / len(claims), 3)``
    * a claim whose judge cascade is exhausted lifts the whole task to JUDGE_OUTAGE
      (an infra outage, excluded from the coverage mean and from base's accuracy) rather
      than scoring its claims `not_fulfilled`
    * empty and errored responses are **kept and scored 0.0** ("All rows are kept
      (including errors) for a holistic picture"), so they sit in the denominator

    Known remaining deviation: the official default judge is
    `gemini/gemini-3.1-pro-preview`; gbench uses its shared judge (`GBENCH_JUDGE_MODEL`,
    default gemini-3.6-flash). Set `GBENCH_MCP_ATLAS_JUDGE_MODEL` to pin the canonical one.
    """
    import asyncio
    from tqdm import tqdm

    judge_model = os.environ.get("GBENCH_MCP_ATLAS_JUDGE_MODEL") or judge_model
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Defensive: run_mcp_atlas's gemini_required_skip gate already hard-errors on an absent/
        # invalid key, so this is unreachable in normal flow. Never silently downgrade to verbatim
        # claim-containment (a heuristic number masquerading as scoring_mode="judge").
        from .swebench_common import infra_required
        raise infra_required(
            "mcp_atlas",
            "requires GEMINI_API_KEY for canonical LLM-judge grading",
            "docs/evals/mcp_atlas.md")

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _score_claim(claim: str, answer: str) -> Optional[float]:
        """One claim -> one score in {0.0, 0.5, 1.0}, or None on a judge-cascade outage.

        None is a JUDGE_OUTAGE signal (the Gemini judge cascade was exhausted - an infra
        outage) that the caller lifts to the whole trace, so the task is excluded rather than
        having its claims scored not_fulfilled by an outage. A parseable-but-unrecognised
        judge reply still returns 0.0, preserving the suite's existing "unparsed judge =>
        not_fulfilled" behaviour (that is NOT an outage).
        """
        prompt = _CLAIM_JUDGE_PROMPT.format(claim=claim, response=answer[:8000])
        async with semaphore:
            text, _judge_used = await judge_generate_cascade(prompt)
        if text is None:
            return None
        m = re.search(r'"coverage_outcome"\s*:\s*"([a-z_]+)"', text)
        if m:
            return _COVERAGE_TO_SCORE.get(m.group(1), 0.0)
        return 0.0

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        claims = _parse_list_field(trace.get("gold_answer"))
        answer = str(trace.get("response_text") or "").strip()
        if not claims:
            trace["is_correct"] = False
            trace["judge_grade"] = "no_claims"
            pbar.update(1)
            return
        if not answer:
            # Kept, not skipped - an empty answer covers no claims and belongs in the mean.
            trace["claim_recall"] = 0.0
            trace["is_correct"] = False
            trace["judge_grade"] = f"0.0/{len(claims)} claims (no response)"
            pbar.update(1)
            return

        scores = await asyncio.gather(*[_score_claim(c, answer) for c in claims])
        if any(s is None for s in scores):
            # At least one claim's judge cascade was exhausted (infra outage): exclude the
            # whole task from the coverage mean and from base's accuracy, rather than scoring
            # its claims not_fulfilled and depressing coverage with an outage.
            trace["judge_grade"] = "JUDGE_OUTAGE"
            pbar.update(1)
            return
        total = sum(scores)
        coverage = round(total / len(claims), 3)
        trace["claim_recall"] = coverage
        trace["claim_scores"] = scores
        trace["fully_covered_claims"] = sum(1 for s in scores if s >= 1.0)
        trace["partially_covered_claims"] = sum(1 for s in scores if 0.5 <= s < 1.0)
        # `correct_answers` is the count at or above the official 0.50 pass bar; the
        # headline `accuracy` is overridden to mean_coverage in `run_mcp_atlas`.
        trace["is_correct"] = coverage >= 0.50
        trace["judge_grade"] = f"{total:g}/{len(claims)} claims (coverage {coverage})"
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [MCP_ATLAS]") as pbar:
        await asyncio.gather(*[_judge_single(t, pbar) for t in sample_traces])


#: MCP tool names whose job is "go and look something up". MCP-Atlas references 143 tools
#: across its tasks; these are the ones a web-search backend can genuinely stand in for.
_SEARCH_TOOL_HINTS = ("search", "scraper", "fetch", "browse", "google", "brave", "ddg",
                      "duckduckgo", "arxiv", "wikipedia", "web", "whois", "crawl", "query")

#: Argument keys models actually use for the query string, in the order we try them.
_QUERY_KEYS = ("query", "queries", "q", "search_term", "searchTerm", "keywords", "term",
               "text", "url", "prompt", "question")


def _query_from_args(args: Dict[str, Any]) -> str:
    """Pull a search string out of whatever shape the model used."""
    for k in _QUERY_KEYS:
        v = (args or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v:
            joined = " ".join(str(x) for x in v if str(x).strip())
            if joined.strip():
                return joined.strip()
    # last resort: any non-empty string value
    for v in (args or {}).values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


#: Services scoped to a private account or to local state. Their names often contain
#: "search" (`notion_API-post-search`, `memory_search_nodes`), but answering them with a
#: PUBLIC web search would hand the model plausible, unrelated facts about someone else's
#: data - worse than saying the tool is unavailable, because it looks like evidence.
_PRIVATE_SERVICE_HINTS = ("notion", "slack", "jira", "gmail", "gdrive", "google-drive",
                          "filesystem", "memory", "sqlite", "postgres", "mysql", "redis",
                          "cli-mcp", "code-executor", "shell", "terminal", "docker",
                          "kubernetes", "aws", "gcp", "azure", "stripe", "sentry")


def is_search_like(tool_name: str) -> bool:
    """True only for tools a PUBLIC web search can honestly stand in for."""
    low = (tool_name or "").lower()
    if any(h in low for h in _PRIVATE_SERVICE_HINTS):
        return False
    return any(h in low for h in _SEARCH_TOOL_HINTS)


async def execute_mcp_tool(name: str, args: Dict[str, Any]) -> str:
    """Serve MCP-Atlas tool calls with the resources gbench actually has.

    The tasks are graded on the FACTS the answer contains, and the model reliably asks for
    the right thing (`brave-search_brave_web_search({"queries": [...]})`,
    `ddg-search_search(...)`, `arxiv_search_papers(...)`). Previously nothing served those
    calls, so the suite scored 0/20 with the answers going nowhere.

    Search-flavoured tools are backed by the same Gemini grounding `gaia`/`deepsearch_qa`
    use. Anything stateful (filesystem, memory graph, notion, code execution) cannot be
    faked, so it returns an explicit "unavailable" telling the model to answer from what it
    already has - which is a real signal, not a silent void.
    """
    from .search_tool import search_with_fallback
    import asyncio
    if is_search_like(name):
        query = _query_from_args(args)
        if not query:
            return json.dumps({"tool": name, "error": "no query argument supplied"})
        # Via `search_with_fallback`, NOT `gemini_search` directly. Calling the grounding
        # backend by hand meant this suite alone had no fallback, and - worse - it never
        # recorded which backend answered. On the 2026-08-20 sweep that hid a total outage:
        # `GBENCH_SEARCH_MODEL` defaulted to a model whose Google-Search grounding was
        # quota'd, so all 2161 tool rounds across 498 samples came back as errors while the
        # result still reported `search_backend: gemini-google-search-grounding`.
        results, backend = await asyncio.to_thread(search_with_fallback, query)
        return json.dumps({"tool": name, "query": query, "results": results,
                           "backend": backend}, ensure_ascii=False)
    return json.dumps({
        "tool": name,
        "error": ("this MCP server is not available in the gbench environment "
                  "(no filesystem, memory graph, code execution or third-party account "
                  "access). Answer from what you already know and from any search results "
                  "you have, and state the facts your answer depends on."),
    })


def run_mcp_atlas(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native MCP_ATLAS evaluation benchmark.

    **This is a lower bound, and structurally so.** The official harness
    (scaleapi/mcp-atlas) stands up 36 real MCP servers in Docker - `services/agent-environment`,
    with Notion / Slack / MongoDB / Airtable / filesystem / git fixtures shipped as data
    exports - and lets the agent drive them. gbench runs none of that. It declares the task's
    tool names, serves the lookup-flavoured ones from a web search, and returns an explicit
    "unavailable" for everything stateful.

    Measured over the 500 public tasks on 2026-08-20: of 261 distinct tools referenced, only
    65 (24.9%) are search-like. Per task a mean of **29.7%** of the declared tools are
    servable; 20 tasks have none and exactly 1 has all of them. `servable_tool_fraction` and
    `samples_with_no_servable_tool` are recorded on every result, and
    `leaderboard_comparable` is always False. Closing that gap means running the official
    agent-environment, not tuning anything here.
    """
    # Fail-fast: the LLM judge is load-bearing. A missing/invalid key must hard-error BEFORE the
    # generation run, never downgrade to verbatim claim-containment (which would publish a heuristic
    # substring number labelled scoring_mode="judge").
    gemini_required_skip("mcp_atlas", model_name)
    samples = _load_mcp_atlas_samples(limit=limit)
    # Per-RUN, not per-process: a sweep runs many search-backed suites in one process
    # and they would otherwise inherit each other's counts.
    reset_backend_tally()
    result = run_eval_suite(
        eval_name="mcp_atlas",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_async_judge_mcp_atlas,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        tool_executor=execute_mcp_tool if search_available() else None,
    )
    # Judge-outage traces carry no claim_recall and are excluded here (and from base's
    # pass/fail accuracy) so an infra outage never depresses mean_coverage.
    recalls = [t["claim_recall"] for t in result.get("sample_traces", [])
               if t.get("judge_grade") != "JUDGE_OUTAGE"
               and isinstance(t.get("claim_recall"), (int, float))]
    result["closed_book"] = not search_available()
    result["search_backend"] = search_backend_name()
    # What was CONFIGURED vs what actually ANSWERED. The 2026-08-20 run reported
    # `gemini-google-search-grounding` while all 2161 lookups 429'd; this is the field
    # that would have shown it.
    result["search_backend_calls"] = backend_tally()
    result["leaderboard_comparable"] = False
    # Headline = `mean_coverage`, matching `_coverage_stats` in the official
    # `services/scoring/score_claims.py`. gbench previously headlined all-claims-supported,
    # which the harness does not compute at all: on the 2026-08-20 run that read 0.20%
    # where coverage read 12.56% on identical responses. Same override pattern as gdpval /
    # healthbench / ruler / mrcr, which also headline a continuous rubric mean.
    result["metric"] = "mean_coverage (GTFA claim coverage, scaleapi/mcp-atlas score_claims.py)"
    if recalls:
        result["mean_coverage"] = round(sum(recalls) / len(recalls) * 100.0, 2)
        result["accuracy"] = result["mean_coverage"]
        for th in _PASS_THRESHOLDS:
            result[f"pass_rate_{th:.2f}"] = round(
                sum(1 for r in recalls if r >= th) / len(recalls) * 100.0, 2)
        result["tasks_fully_covered"] = sum(1 for r in recalls if r >= 1.0)
        result["claims_scored"] = sum(
            len(t.get("claim_scores") or []) for t in result.get("sample_traces", []))
        # `correct_answers` counts tasks at or above the official 0.50 bar, so it is
        # `pass_rate_0.50` - NOT accuracy x total. Named here so it does not read as a bug.
        result["correct_answers_metric"] = "tasks with coverage >= 0.50 (pass_rate_0.50)"
        # Kept for continuity with pre-2026-08-20 result files, which headlined this.
        result["strict_all_claims_covered_pct"] = round(
            result["tasks_fully_covered"] / len(recalls) * 100.0, 2)
    # How much of each task's declared tool surface gbench can actually serve. MCP-Atlas
    # spans 36 real MCP servers; gbench executes none of them and substitutes a web search
    # for the lookup-flavoured ones, so this is a hard ceiling on the score and belongs on
    # the result rather than in a docstring.
    fracs = []
    for tr in result.get("sample_traces", []):
        declared = [f.get("function", {}).get("name", "") if isinstance(f, dict) else str(f)
                    for f in (tr.get("extra_payload") or {}).get("tools") or []]
        if declared:
            fracs.append(sum(is_search_like(n) for n in declared) / len(declared))
    if fracs:
        result["servable_tool_fraction"] = round(sum(fracs) / len(fracs) * 100.0, 1)
        result["samples_with_no_servable_tool"] = sum(1 for f in fracs if f == 0)
    return result
