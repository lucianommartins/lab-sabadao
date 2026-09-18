# aime

American Invitational Mathematics Examination, answers are integers 0-999. **Scoring:
extract the final integer and match the gold**. `\boxed{}` first, then an explicit answer
anchor, then the last number in the response (exact match; random-guess baseline 0.1%).

## Two datasets, split by contamination window

| set | dataset | role | result field |
| --- | --- | --- | --- |
| **AIME 2025** (post-cutoff, 30) | `yentinglin/aime_2025` | **headline**, clean reasoning number | `accuracy` / `accuracy_post_cutoff` |
| AIME 2022-2024 (pre-cutoff, 90) | `AI-MO/aimo-validation-aime` | contaminated **reference** | `accuracy_pre_cutoff_reference` |

> **Why the split.** The 2022-2024 problems *and their full worked solutions* are on
> artofproblemsolving.com (the dataset's own `url` field) and pre-date the training cutoff,
> so a high score there can be recall, not reasoning; measured 2026-08-20, a 26B model
> scored ~82% no-think on the pre-cutoff set. AIME 2025 is post-cutoff, so it is the number
> to trust. Both are run and reported; `category_accuracy` breaks the score down per contest
> year, and `accuracy` is the post-cutoff headline. `accuracy_all_years` keeps the combined
> figure for continuity.

Knobs: `GBENCH_AIME_INCLUDE_PRECUTOFF=0` runs the clean 2025 set only;
`GBENCH_AIME_POSTCUTOFF_DATASET` / `GBENCH_AIME_PRECUTOFF_DATASET` override the sources.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals aime --eval-thinking --eval-limit 20
```
`--eval-thinking` strongly recommended; `--max-output-tokens` ≥ 8192 for full working.
