# gorilla_apibench

Canonical Gorilla APIBench (`gorilla-llm/APIBench`): real-world API invocation across
HuggingFace, TorchHub, and TensorHub. **Scoring: 100% deterministic AST matching** of
the predicted API call against gold.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gorilla_apibench --eval-limit 20
```
