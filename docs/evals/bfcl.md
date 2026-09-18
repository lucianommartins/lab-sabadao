# bfcl

Canonical BFCL (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`, v3): function
calling across 11 execution/parsing tiers (simple, multiple, parallel, multi-turn,
irrelevance/abstention, etc.). **Scoring: the predicted function call is checked
against ground truth** via the BFCL AST/execution checker per category.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals bfcl --eval-limit 20
```
`--eval-categories` filters BFCL categories; `category_accuracy` reports per tier.
