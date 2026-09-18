# humanitys_last_exam setup

Canonical Humanity's Last Exam (`cais/hle`): a frontier, multi-disciplinary academic
benchmark of expert-authored questions spanning many fields. Answers are open-form
and expert-level, so correctness is decided by an **LLM judge** against the gold
answer (with abstention handling), not string matching.

## Requirements
- **`GEMINI_API_KEY`** (the judge model). Without it the suite **hard-errors** (`infra_required`); it never skips and never fabricates a number.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals humanitys_last_exam --eval-thinking --eval-limit 20
```
`--eval-thinking` recommended (frontier reasoning). `--max-output-tokens` ≥ 2048.
