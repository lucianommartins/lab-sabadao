# deepsearch_qa

Canonical DeepSearchQA (`xbench/DeepSearch-2510`): autonomous, multi-step **web-search
research** questions whose prompts/answers are XOR-protected with a per-row canary (decoded
canonically before use). This is a *deep-research agent* benchmark (the model must look things
up), so gbench offers a search tool and drives a multi-round tool loop, then grades with an LLM
judge.

## Scoring

- The model answers with a live **search tool** (`web_search`, backed by Gemini grounding); gbench
  runs the canonical multi-round tool loop and the model concludes with `Final Answer: <answer>`.
- Grading is **LLM-judge only** (gbench's shared Gemini judge, pinned at temperature 0): CORRECT
  iff the model's final answer matches the gold factual answer in meaning. There is **no**
  string-match fallback. The dataset's short factual answers can be phrased many ways, so a
  substring match is not canonical.
- A judge-cascade outage is recorded as `JUDGE_OUTAGE` and **excluded from the denominator**
  (never silently scored wrong).
- `leaderboard_comparable` is **False**: gbench's Gemini-grounded search is a faithful search
  backend but not the exact browsing agent a published DeepSearch number uses.

## Requirements

- A running OpenAI-compatible `/v1` endpoint for the model under test.
- **`GEMINI_API_KEY`**: REQUIRED. It powers *both* the search backend and the LLM judge. Without
  it the suite **hard-errors** (it never skips and never falls back to a non-canonical score):
  ```bash
  export GEMINI_API_KEY="…"
  ```
- Python deps `datasets`, `google-genai`, `aiohttp`, `tqdm` (already in the gbench env). The
  dataset is public/ungated, with no checkout to provision.

Optional: `GBENCH_DEEPSEARCH_QA_TEMPERATURE` overrides the sampling temperature for this suite
only (the judge stays at 0).

## Run

```bash
export GEMINI_API_KEY="…"
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals deepsearch_qa --eval-limit 20
```

The result records `search_backend` (the grounding cascade that served the lookups),
`search_backend_calls` (a per-backend tally of how many lookups each backend actually served this
run, so a run that made zero real searches or silently fell back is visible, not hidden) and
`leaderboard_comparable: false`.
