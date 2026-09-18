# nexus_function_calling

Canonical NexusRaven single-call benchmark (`Nexusflow/NexusRaven_API_evaluation`):
each query is presented with the available Python function signatures and the model
must emit the correct single call. **Scoring: AST-equivalence match** of the predicted
call against gold: function name plus arguments, order-independent, falsy/default
values dropped.

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals nexus_function_calling --eval-limit 20
```
