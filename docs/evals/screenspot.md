# screenspot

Canonical ScreenSpot V2 (`HongxinLi/ScreenSpot_v2`): pixel-level GUI coordinate
grounding across web, mobile, and desktop UIs. The model outputs a click point.
**Scoring: the predicted (x, y) must fall inside the target bounding box** (evaluated
in a `[0, 1000]` normalized coordinate scale).

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals screenspot \
       --eval-max-soft-tokens 1120 --eval-limit 20
```
`--eval-max-soft-tokens` sets the image soft-token budget; grounding benefits from a
higher budget.
