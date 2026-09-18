# mmmu_pro

Canonical MMMU-Pro (`MMMU/MMMU_Pro`): multimodal, multidisciplinary reasoning with a
10-option leaderboard format. **Scoring: MMMU-Pro letter match** (canonical
`parse_answer` extraction, compared against gold).

## Requirements
- A **vision-language model** endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mmmu_pro --eval-thinking \
       --eval-max-soft-tokens 280 --eval-limit 20
```
`--eval-max-soft-tokens` sets the image soft-token budget (`70`/`140`/`280`/`560`/`1120`);
`--eval-thinking` for CoT.
