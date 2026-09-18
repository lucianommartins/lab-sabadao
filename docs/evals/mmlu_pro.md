# mmlu_pro

Canonical MMLU-Pro (`TIGER-Lab/MMLU-Pro`): a harder, reasoning-focused MMLU with 10
options (A-J) across 14 subjects. **Scoring: match the predicted letter (A-J) against
gold.** Zero-shot by default; `--eval-n-shot N` prepends N chain-of-thought exemplars
from the dataset's official `validation` split for the item's own category (max 5).

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mmlu_pro --eval-thinking --eval-limit 20
```
