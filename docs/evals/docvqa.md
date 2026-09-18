# docvqa

Canonical DocVQA (`lmms-lab-encoder/DocVQA`): document visual question answering over
scanned pages and PDF forms. Images are sent losslessly (PNG). **Scoring: ANLS /
normalized matching** of the predicted text against the gold answers (the canonical
DocVQA metric).

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals docvqa --eval-limit 20
```
