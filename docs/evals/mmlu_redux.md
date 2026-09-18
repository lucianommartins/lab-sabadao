# mmlu_redux

Canonical MMLU-Redux (`edinburgh-dawg/mmlu-redux`): a manually re-annotated,
error-corrected subset of MMLU. **Scoring: 100% deterministic letter matching** of the
predicted option against gold.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mmlu_redux --eval-limit 20
```
