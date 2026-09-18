# arc_agi

Canonical ARC-AGI-1 (François Chollet, `dataartist/arc-agi`): the public 400-task
evaluation corpus of 2D integer-grid abstraction puzzles. Each task's train
input→output examples are serialized into the prompt and the model must produce the
test output grid. **Scoring: exact grid match** (predicted 2D int grid == gold grid).

## Requirements
None beyond a running OpenAI-compatible `/v1` endpoint. No judge, no sandbox.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals arc_agi --eval-thinking --eval-limit 20
```
`--eval-thinking` helps (abstract reasoning). `--max-output-tokens` should fit a full grid.
