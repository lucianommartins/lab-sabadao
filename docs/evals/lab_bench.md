# lab_bench

Canonical LAB-Bench (`futurehouse/lab-bench`): wet-lab biology and chemistry
experimental reasoning, posed as multiple choice. **Scoring: match the predicted
option letter against the gold answer.**

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals lab_bench --eval-thinking --eval-limit 20
```
