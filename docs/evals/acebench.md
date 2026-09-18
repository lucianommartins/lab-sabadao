# acebench

Canonical ACEBench (`chenchen0103/ACEBench`, via the MIT-licensed HF reformatting
`oliveirabruno01/acebench`): **function-calling / tool-use** evaluation. The model must emit
the correct tool call(s) for each task. **Scoring: the ACEBench checker (LLM-free AST match)
compares the predicted call(s) against the gold call specification.**

> **Scope (`leaderboard_comparable=false`):** the **English** config and only the **Normal +
> Special** categories run. The **`agent` category is excluded** - it needs a stateful simulated
> environment and a user-simulator LLM - and the Chinese half is excluded. So this is *not*
> "agentic coding"; it is the non-agent tool-use subset, and the number is a lower bound on the
> full ACEBench leaderboard.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint (the checker is LLM-free; no judge key).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals acebench --eval-thinking --eval-limit 20
```
