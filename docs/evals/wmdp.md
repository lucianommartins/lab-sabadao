# wmdp

Canonical WMDP (`cais/wmdp`): the Weapons of Mass Destruction Proxy, multiple-choice
questions probing hazardous knowledge (biosecurity, cyber, chemical) as a proxy for
unlearning/safety measurement. **Scoring: match the predicted option letter against
gold.**

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals wmdp --eval-limit 20
```
