# gaia setup

Canonical GAIA (General AI Assistants, `gaia-benchmark/GAIA`, 2023 validation, 165
tasks) scored with the official `question_scorer` (number/list/string normalized
exact-match on the model's `FINAL ANSWER:` line). gbench runs GAIA with a **web_search
tool loop** (the model drives a Gemini-grounded search tool), not the full browsing
agent.

## Requirements
- The dataset is **gated**: log in to HF, accept the license at
  <https://huggingface.co/datasets/gaia-benchmark/GAIA>, and export `HF_TOKEN`
  (or `HUGGING_FACE_HUB_TOKEN`). The suite **hard-errors** (`infra_required`, never
  skips) without gated access.
- **`GEMINI_API_KEY`**: required. GAIA is a web-research benchmark; without a search
  backend every answer scores a structural 0, so the key is a hard prerequisite (the
  suite hard-errors up front without it, before downloading the dataset). It powers the
  Gemini-grounded `web_search` tool.
- No other deps (the scorer is stdlib).

## Run
```bash
export HF_TOKEN=hf_xxx
export GEMINI_API_KEY=...
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gaia
```

## Notes / caveats
- **Sampling:** `GBENCH_GAIA_TEMPERATURE` overrides the temperature for this suite (else the run
  default: 0.0 greedy / 1.0 with `--thinking`); it takes precedence over `--temperature`. The
  Gemini judge/grounding is pinned at 0.0.
- **`leaderboard_comparable` is always `false`**: this is a **search-only** measurement, not the
  full browsing agent the public GAIA leaderboard uses, and ~a quarter of GAIA tasks need a file
  attachment this harness does not pass. The result reports accuracy over the **file-free subset**
  and a per-`Level` breakdown alongside the headline, plus `search_backend` and
  `search_backend_calls` (a per-backend tally so a run that made zero real searches is visible).
