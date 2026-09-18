# bigcodebench setup

Canonical BigCodeBench (`bigcode/bigcodebench`, instruct split) scored by
**calibrated pass@1**: the model's raw output is sanitized + calibrated
(`code_prompt` prepended) and executed against each task's `unittest` TestCase
inside the official Docker image.

## Requirements
- **Docker** (image `bigcodebench/bigcodebench-evaluate:latest`, ~5.4 GB).
- The image is **auto-pulled on first run** (no advance pull needed); the suite
  hard-errors if Docker is unreachable or the pull fails.
- `datasets` (base install). No pip install of `bigcodebench` into the gbench venv
  (it pins transformers/accelerate and would clobber the vLLM env). The image
  carries the harness.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals bigcodebench \
       --sandboxes 8 --eval-limit 20
```
Use `--subset hard` (kwarg) for BigCodeBench-Hard (148 tasks). `--sandboxes` maps
to the harness `--parallel`. The instruct prompt is sent verbatim (canonical);
`--max-output-tokens` ≥ 2048.
