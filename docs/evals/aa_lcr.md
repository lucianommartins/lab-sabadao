# aa_lcr setup

Canonical AA-LCR (`ArtificialAnalysis/AA-LCR`): Artificial Analysis Long-Context
Reasoning. Questions require reasoning over long documents; open-form answers are
graded by an **LLM judge** against the gold answer.

## Requirements
- **`GEMINI_API_KEY`** (the judge model). Without it the suite **hard-errors** (`infra_required`); it never skips and never fabricates a number.
- A serving context window large enough for the long inputs (run against an endpoint
  configured at the model's native context length).

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals aa_lcr --eval-limit 20
```
`--max-output-tokens` ≥ 4096.
