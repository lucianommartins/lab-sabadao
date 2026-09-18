# t_eval

Canonical T-Eval (Chen et al., ACL 2024; arXiv:2312.14033;
[open-compass/T-Eval](https://github.com/open-compass/T-Eval); HF `lovesnowbest/T-Eval`):
six separable tool-use sub-skills, scored deterministically (no LLM judge).

## What runs

All **8 English data files** (the `_zh` files are the separate Chinese leaderboard and are
excluded): `instruct_v2`, `plan_str_v2`, `plan_json_v2`, `reason_str_v2`, `retrieve_str_v2`,
`understand_str_v2`, `reason_retrieve_understand_json_v2`, `review_str_v2`. Each sample is a
pre-baked single-turn context (`origin_prompt`, mapped to chat: `function`→`user`, adjacent
same-role turns merged); the model emits one completion.

## Scoring (canonical `convert_results`)

The upstream evaluators are vendored verbatim in
`gbench/runners/eval_suites/_teval_vendor/` and driven per sample; the six dimensions combine
exactly as upstream:

- **Instruct** = mean of the json and string groups, each `(format + args_em)/2` (groups weighted equally)
- **Plan** = mean(plan_str F1, plan_json F1), bertscore graph match (Hungarian) + LIS
- **Reason** = mean(reason_str thought, rru_json thought), sentence-embedding cosine
- **Retrieve** = mean(retrieve_str name, rru_json name)
- **Understand** = mean(understand_str args, rru_json args)
- **Review** = review_str review_quality
- **Overall** = mean of the six → reported as `accuracy`; per-dimension scores in `dimension_scores`.

## Requirements

- An OpenAI-compatible `/v1` endpoint (single-turn generation).
- **`sentence-transformers`** + **`torch`** + **`networkx`**: required by the Plan/Reason/
  Retrieve/Understand scorers (BERTScore + the plan Hungarian matching). `torch` comes with your
  vLLM serving stack (`pip install vllm`, or your validated CUDA build); `sentence-transformers` and
  `networkx` are **not** in any gbench extra, so install them into the same environment gbench runs in:
  ```bash
  pip install sentence-transformers networkx        # torch: comes with `pip install vllm`, or your CUDA build
  ```
  The Plan/Reason scorers download **`all-mpnet-base-v2`** (~420MB, once, cached; GPU if available)
  on first use. If `sentence-transformers` is missing the suite **hard-errors** (`infra_required`).
  It never skips and never scores on a partial matrix.
- No API key, no LLM judge, no server/sandbox. Scoring is fully deterministic and offline.
- **Sampling:** `GBENCH_T_EVAL_TEMPERATURE` overrides the temperature for this suite; else the run
  default (0.0 greedy / 1.0 with `--thinking`). Scoring itself is deterministic regardless.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals t_eval --eval-limit 20
```
`--eval-limit N` samples N rows **per file** (stratified within each), so all six dimensions
are always represented.
