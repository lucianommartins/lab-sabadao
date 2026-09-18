# infographicvqa

Canonical InfographicVQA (`mm-eval/InfographicVQA`): visual question answering over
infographic posters (dense text + graphics). Images are sent losslessly (PNG).
**Scoring: the predicted answer must match any gold answer using normalized
matching** (ANLS-style).

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals infographicvqa --eval-limit 20
```
