# gpqa

Canonical GPQA (`Idavidrein/gpqa`, main 448-question split): graduate-level,
Google-proof multiple-choice science (biology, physics, chemistry). Zero-shot / CoT.
**Scoring: extract the answer letter from the response and match the gold letter**
(`\boxed{}`, then an explicit anchor, then the last standalone letter). Answer
options are shuffled per item with a seed derived from the question, so the correct
answer's position does not track its alphabetical order.
For the harder curated subset, see [gpqa_diamond](gpqa_diamond.md).

## Requirements
- The `Idavidrein/gpqa` dataset is **gated** on HuggingFace; accept its terms and set
  `HF_TOKEN` so the loader can download it.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gpqa --eval-thinking --eval-limit 20
```
`--eval-thinking` recommended. `--max-output-tokens` ≥ 2048 with thinking.
