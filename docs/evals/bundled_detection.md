# bundled_detection

Canonical bundled object detection & grounding (`detection-datasets/coco`, MS-COCO):
the model predicts bounding boxes for objects in an image. **Scoring: predicted boxes
are matched to gold objects with IoU ≥ 0.5.**

## Requirements
- A **vision-language model** endpoint (images are sent as data URIs).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals bundled_detection --eval-limit 20
```
