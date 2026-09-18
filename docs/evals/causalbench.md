# causalbench

Canonical CausalBench (`causal-nlp/corr2cause`): causal-discovery and counterfactual
reasoning. Each item poses a causal question over a described correlation structure;
the model answers the causal verdict. **Scoring: exact match of the predicted
Yes/No causal verdict against gold.**

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals causalbench --eval-limit 20
```
