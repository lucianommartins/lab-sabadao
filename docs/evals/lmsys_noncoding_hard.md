# lmsys_noncoding_hard

Canonical LMSYS / WildBench hard non-coding subset (`WildEval/WildBench`, config `v2-hard`,
coding tags filtered out): complex, multi-turn human prompts that are not coding tasks.

**Scoring: canonical WildBench WB-Score.** An LLM judge (the gbench Gemini cascade) rates each
response 1-10 against the item's evaluation checklist; the per-item score is rescaled
`(score-1)/9` and the headline is the mean over 0-100 (matches the WildBench leaderboard scale).
`pass_rate_at_7` (fraction scoring ≥7) is reported as a secondary. A failed/empty response
counts as the worst WB-Score (0) rather than being dropped; a judge-side outage is excluded.
If `GEMINI_API_KEY` is unset the suite falls back to a checklist keyword-overlap heuristic
(`scoring_mode=judge_fallback`) so it still runs, but only the judged run is the canonical metric.

## Requirements
A running OpenAI-compatible `/v1` endpoint, plus `GEMINI_API_KEY` for the canonical WB-Score
judge (without it the run uses the non-canonical heuristic fallback).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals lmsys_noncoding_hard --eval-limit 20
```
