# bfcl_v4_agentic

The **BFCL v4 Agentic** track (Berkeley Function-Calling Leaderboard), run through
Berkeley's own `bfcl-eval` harness. This is a *stateful tool-execution* benchmark, not an
answer-matching dataset: the agent calls memory / web-search tools across many turns and is
graded on the resulting state, so gbench wraps the canonical harness rather than
re-implementing it (the same approach as [tau2](tau2.md) / [tau3](tau3.md)).

> **Not to be confused with [`bfcl_v3_live`](bfcl_v3_live.md).** That suite (formerly
> mis-named `bfcl_v4_agentic`) loads the BFCL **v3 LIVE single-turn** subsets plus
> irrelevance abstention, roughly the "Live" 10% slice of the v4 leaderboard. It measures
> something real, just not the agentic track.

## Categories
Verified from the installed harness (`TEST_COLLECTION_MAPPING["agentic"]`):

| Category | Needs a search backend |
|---|---|
| `memory_kv` | no |
| `memory_vector` | no (local faiss + sentence-transformers) |
| `memory_rec_sum` | no |
| `web_search_base` | **yes** |
| `web_search_no_snippet` | **yes** |

Leaderboard weighting for context: Agentic 40% · Multi-Turn 30% · Live 10% · Non-Live 10% ·
Hallucination 10%.

## Requirements
- **`pip install bfcl-eval`**: the canonical harness (*not* the unrelated `bfcl` package on
  PyPI). The v4 data ships **inside** the package (`BFCL_v4_memory.json`,
  `BFCL_v4_web_search.json`); nothing is downloaded from HF.
- **A reachable model endpoint.** gbench points the harness at it via `OPENAI_BASE_URL` /
  `REMOTE_OPENAI_BASE_URL` and passes `--skip-server-setup`, so no second GPU server is
  started.
- **A web-search backend** *(only for the two `web_search_*` categories)*; see below.
- If a prerequisite is missing the suite **hard-errors** (`infra_required`) with the reason;
  it never skips and never reports a fabricated score.

## Search backend (`BFCL_SEARCH_BACKEND`)
Canonical BFCL queries DuckDuckGo **through SerpAPI**. gbench supports two backends and
records which one ran in `bfcl_report.search_backend`:

| Value | Needs | Leaderboard-comparable |
|---|---|---|
| `serpapi` | `SERPAPI_API_KEY` (paid) | ✅ yes, the canonical path |
| `gemini` *(default when only a Gemini key exists)* | `GEMINI_API_KEY` | ❌ no |

The `gemini` backend **cascades across grounding models** (default
`gemini-3.7-flash → 3.6 → 3.5 → 3-flash-preview → 2.5-flash`, overridable with
`GBENCH_SEARCH_MODELS`, or a single model with `BFCL_SEARCH_GEMINI_MODEL`). Grounding quota
is per-model, so a per-model 429 under BFCL's concurrent web_search steps is recovered by the
next model rather than failing the query. The direct-DuckDuckGo backend was removed
(2026-08-21): it is captcha-blocked on this box, and a second grounding model is a strictly
better stand-in than a blind scraper.

**Why Gemini is not comparable:** results come from Google-Search grounding, so the
retrievable evidence differs from DuckDuckGo. Each grounding chunk is mapped to BFCL's
required `{title, href, body}` shape: `title` is the source domain, `href` is a
`vertexaisearch` redirect, and `body` is taken from the grounding *supports* (the answer
spans a chunk backs). The numbers are internally consistent and reproducible, but should not
be compared against published BFCL results. `memory_*` performs no search and stays
canonical regardless of backend.

**Degraded mode:** with no search backend at all, the suite runs the **three memory
categories only** and says so in `bfcl_report` (`search_backend: "none (memory-only)"`,
`web_search_categories_run: []`) rather than skipping everything or implying full coverage.

## Model registration
BFCL grades through per-model handlers and ships **no gemma-4 entry**, so gbench registers
one at runtime using **`OpenAICompletionsHandler` in function-calling mode**. The local
`GemmaHandler` is deliberately *not* used: it hand-builds a raw gemma-3 prompt and parses
prompt-style `[func(a=1)]` text back out, which fails on every row against a vLLM-served
gemma-4 with native tool calling ("Failed to decode the model response"). Because the
harness runs as a subprocess, the registration and the search-backend patch are re-applied
inside the child before the CLI dispatches.

## Thinking (`--thinking`)
The delegated `bfcl-eval` harness builds its own OpenAI requests, so gbench's usual
`chat_template_kwargs` injection (base.py) never runs here. `--thinking` is instead plumbed
into the harness subprocess via `GBENCH_BFCL_ENABLE_THINKING=1`, and the gbench handler
subclass adds `extra_body={"chat_template_kwargs": {"enable_thinking": true}}` to every call -
the same native-reasoning toggle every other suite uses. Without this the track always ran
no-think regardless of the flag. `--thinking` also selects the 1.0 temperature default (via
`resolve_temperature`), and the run's `thinking` field records which mode was measured.

## Env knobs
`BFCL_PROJECT_ROOT` (default `~/.cache/gbench/bfcl` or `./bfcl`; results/scores are written here,
never into site-packages) · `BFCL_SEARCH_BACKEND` · `BFCL_SEARCH_GEMINI_MODEL`
(default `gemini-3.5-flash`) · `BFCL_TIMEOUT_S` (default 10800) · `GBENCH_BFCL_ENABLE_THINKING`
(set to `1` by `--thinking`; injects the reasoning toggle described above). `--sandboxes`/concurrency
maps to the harness' `--num-threads`.

## Run
```bash
# memory-only (no search key needed)
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals bfcl_v4_agentic

# full agentic track with Gemini-grounded search
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals bfcl_v4_agentic --sandboxes 16

# canonical (leaderboard-comparable) search
export SERPAPI_API_KEY="..."   # BFCL_SEARCH_BACKEND=serpapi is then chosen automatically
```

`accuracy` is the sample-weighted mean over the categories that ran; `category_accuracy`
breaks it down per category; `bfcl_report` records the categories run, the search backend,
and whether the result is leaderboard-comparable.

> **Runtime:** this is genuinely slow: each item is a multi-turn agentic episode
> (~30s+/item on a 26B model), so a full category takes tens of minutes.
