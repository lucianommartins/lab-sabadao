# hmmt

Canonical HMMT (`MathArena/hmmt_feb_2025`): Harvard-MIT Mathematics Tournament
competition problems. **Scoring: 100% deterministic symbolic answer equivalence** (the
MathArena/leaderboard grader) of the extracted final answer against gold.

## Requirements
- A running OpenAI-compatible `/v1` endpoint.
- **`math_verify`** (the precise symbolic-equivalence backend, shared by all math suites via
  `math_equiv.require_backend`): install with `pip install 'gbench[evals]'` (pulls
  `math_verify` + `latex2sympy2_extended` + `antlr4-python3-runtime`). The suite **hard-errors**
  (`infra_required`, never skips, never downgrades to string matching) if it is absent; `sympy`
  alone cannot robustly parse competition LaTeX.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals hmmt --eval-thinking --eval-limit 20
```
`--eval-thinking` recommended; `--max-output-tokens` generous for working.
`GBENCH_HMMT_TEMPERATURE` overrides the temperature for this suite (else 0.0 greedy / 1.0 with
`--thinking`).
