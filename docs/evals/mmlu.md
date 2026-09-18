# mmlu

Canonical MMLU (`cais/mmlu`): Massive Multitask Language Understanding across 57
subjects, 4 options (A-D). Zero-shot / CoT / few-shot. **Scoring: match the predicted
letter (A-D) against gold.** See also [mmlu_pro](mmlu_pro.md) (10-option) and
[mmlu_redux](mmlu_redux.md) (error-corrected split).

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mmlu --eval-n-shot 5 --eval-limit 20
```
`--eval-n-shot` sets the few-shot count; `--eval-thinking` for CoT.
