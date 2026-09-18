# chartqa

Canonical ChartQA (`ahmed-masry/ChartQA`): visual reasoning and quantitative
extraction over plots and charts. Images are sent losslessly (PNG). **Scoring:
predicted text is validated against the gold answer with relaxed numerical tolerance**
(the canonical ChartQA relaxed-accuracy metric).

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals chartqa --eval-limit 20
```
