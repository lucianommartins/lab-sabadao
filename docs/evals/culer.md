# culer

CULER (Code-RULER, loaded from `zai-org/LongBench-v2`): long-context needle retrieval
across multi-file codebase repositories (64k-512k context, left-cropped to 128k).
**Scoring: validate the predicted answer against the gold needle/option.**

> **Scope:** LongBench-v2 is filtered to its **code** domain (CULER = Code-RULER);
> roughly 10% of the split qualifies. Set `CULER_ALL_DOMAINS=1` to score every domain
> instead, which makes the result a general LongBench-v2 number, not a code one.

## Requirements
- A serving endpoint with a **large context window** (run at the model's native
  context length; a short `max_model_len` truncates the codebase inputs).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals culer --eval-limit 20
```
