# semantic_keypoint

Canonical semantic keypoint grounding (`HongxinLi/ScreenSpot_v2`): continuous 2D
spatial pointing / coordinate localization. The model outputs a target coordinate.
**Scoring: the predicted (x, y) must fall inside the gold target's bounding box**
(point-in-bbox), the same grounding metric ScreenSpot uses.

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals semantic_keypoint \
       --eval-max-soft-tokens 1120 --eval-limit 20
```
