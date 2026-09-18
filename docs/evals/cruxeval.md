# cruxeval

Canonical CRUXEval (`cruxeval-org/cruxeval`): code reasoning via execution tracing and
output prediction. Given a function and an input, predict the output. **Scoring: literal equality**. The stated answer (`\boxed{}`, an explicit anchor, or
the last non-empty line) must equal the gold Python literal, textually or after
`ast.literal_eval` on both sides. Comparison is case- and type-sensitive: `true` is
not `True`, and the string `'0'` is not the integer `0`.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint (predictions are compared
literally; no code is executed by gbench for this suite).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals cruxeval --eval-limit 20
```
