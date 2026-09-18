# frames setup

Canonical FRAMES (`google/frames-benchmark`): multi-hop factuality, retrieval, and
temporal reasoning. Each item requires synthesizing evidence across several
documents to reach a single answer. Because answers are open-form, correctness is
decided by an **LLM judge** comparing the model's response against the gold answer,
not by string matching.

> **Closed book:** gbench gives the model no search tool and no retrieval corpus, so it answers FRAMES questions from parametric memory alone. The score is a lower bound and is NOT comparable with a published number produced by a browsing/retrieval agent.

## Requirements
- **`GEMINI_API_KEY`**: the judge model. The suite **hard-errors** (`infra_required`) if it is unset.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals frames --eval-thinking --eval-limit 20
```
`--eval-thinking` is recommended (multi-hop). `--max-output-tokens` ≥ 2048.
