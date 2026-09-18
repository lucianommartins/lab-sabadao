# amc_aime

Canonical AMC/AIME combined (`AI-MO/aimo-validation-amc` + `AI-MO/aimo-validation-aime`):
modern high-school AMC 10/12 and AIME competition problems. **Scoring: extract the
final integer and match the gold AMC/AIME answer.**

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals amc_aime --eval-thinking --eval-limit 20
```
