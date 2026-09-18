# cimemories

Canonical CIMemories (`facebook/CIMemories`): context-integrated continual memory,
tested as safe non-leakage of sensitive personal attributes across a conversation.
**Scoring: the response is checked for non-leakage** of the protected attribute
(a leak fails the item).

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals cimemories --eval-limit 20
```
