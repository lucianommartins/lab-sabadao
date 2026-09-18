# healthbench setup

Canonical HealthBench (`openai/healthbench`): realistic clinical conversations scored
against physician-authored **rubrics**. Each example carries a set of weighted rubric
criteria; an **LLM judge** decides which criteria the model's response satisfies, and
the score is the achieved rubric weight over the total possible weight (not a simple
right/wrong). The headline `accuracy` is the **mean rubric score**;
`pass_rate_at_0.5` reports the share of conversations above 0.5. The judge is required. There is no heuristic fallback.

## Requirements
- **`GEMINI_API_KEY`**: the rubric-grading judge. The suite **hard-errors** if it is unset.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals healthbench --eval-limit 20
```
`--max-output-tokens` ≥ 4096 (clinical responses are long-form). Scores are weighted
rubric fractions, so per-example values are continuous in [0, 1].
