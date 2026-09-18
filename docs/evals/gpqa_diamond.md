# gpqa_diamond

Canonical GPQA Diamond (`Idavidrein/gpqa`, diamond split): the 198-question
expert-curated, highest-difficulty subset of GPQA. Zero-shot / CoT multiple choice.
**Scoring: extract the answer letter from the response and match the gold letter**
(`\boxed{}`, then an explicit anchor, then the last standalone letter). Answer
options are shuffled per item with a seed derived from the question, so the correct
answer's position does not track its alphabetical order.
For the full split, see [gpqa](gpqa.md).

## Requirements
- The `Idavidrein/gpqa` dataset is **gated** on HuggingFace; accept its terms and set
  `HF_TOKEN`.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gpqa_diamond --eval-thinking --eval-limit 20
```
