# seal_tools

Canonical SEAL-Tools (`casey-martin/Seal-Tools`): complex multi-step API parameter
mapping and validation. **Scoring: the predicted tool call(s) are matched against the
gold call specification** (name + parameters).

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals seal_tools --eval-limit 20
```
