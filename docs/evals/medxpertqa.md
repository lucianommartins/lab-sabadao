# medxpertqa

Canonical MedXpertQA (`TsinghuaC3I/MedXpertQA`): expert clinical diagnostic case
studies as multiple choice; some items include medical images. **Scoring: extract the
predicted option letter and match the gold answer.**

## Requirements
- A **vision-language model** endpoint (some questions carry images sent as data URIs).
  Text-only items also run on a text model.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals medxpertqa --eval-thinking --eval-limit 20
```
