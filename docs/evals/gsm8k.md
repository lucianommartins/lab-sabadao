# gsm8k

Canonical GSM8K (`openai/gsm8k`): grade-school multi-step arithmetic word problems.
**Scoring: extract the final number from the response and match the gold numeric
answer.**

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gsm8k --eval-n-shot 5 --eval-limit 20
```
`--eval-n-shot` for few-shot; `--eval-thinking` for CoT.
