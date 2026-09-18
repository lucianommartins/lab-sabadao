# bfcl_v3_live

BFCL **v3 LIVE** single-turn function calling plus irrelevance abstention, loaded from
`gorilla-llm/Berkeley-Function-Calling-Leaderboard`.

Subsets: `live_simple`, `live_parallel`, `live_multiple`, `live_parallel_multiple`, and
`live_irrelevance` (abstention: the correct behaviour is to emit **no** tool call).
Roughly the "Live" 10% slice of the BFCL v4 leaderboard.

> **Renamed.** This suite was previously registered as `bfcl_v4_agentic`, which implied the
> v4 *agentic* track (web search / memory / format sensitivity). It never loaded v4 data.
> The HF dataset contains no v4 files at all. The real agentic track now lives in
> [`bfcl_v4_agentic`](bfcl_v4_agentic.md), which drives Berkeley's `bfcl-eval` harness.
> Results labelled `bfcl_v4_agentic` from before this rename are v3-live numbers.

## Scoring
Deterministic AST/JSON matching of the predicted call(s) against gold: function name plus
arguments. For `live_irrelevance` the sample is correct only if the response contains no
tool call.

## Requirements
`datasets` + network access to the HF hub. No API keys.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals bfcl_v3_live
```
