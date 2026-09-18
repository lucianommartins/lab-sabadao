# lcb

Canonical LiveCodeBench (`livecodebench/code_generation`, plus the `execution`
scenario): contamination-free algorithmic coding. **Scoring: execution-based.** After
generation, each solution's public+private test suites are run in parallel
(ProcessPoolExecutor, inside a bubblewrap jail: read-only root, no network) and a
problem passes iff all tests pass (pass@1). A problem whose tests could not be decoded
is reported incorrect, not auto-passed. **Code execution**
is graded by extracting the model's stated `Final Output:` (or the whole/last-line value)
and comparing it to the expected value, not a substring-containment test, which used to
over-credit responses that merely mentioned the value.

## `test_generation` is excluded by default
The `test_generation` scenario is **not loaded by default**. Faithful scoring requires
running the model's generated tests against a reference solution to check they discriminate
correct vs. buggy code (infrastructure this lightweight harness doesn't have) and the
cheap length proxy that was there fake-passed ~100% of cases, inflating the overall score.
Set **`LCB_INCLUDE_TEST_GEN=1`** to load it anyway (it is not credited by the scorer).

## Requirements
- Executes generated code in **local subprocesses** (no Docker), but the host must
  allow spawning Python processes. `--sandboxes` bounds concurrency.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals lcb --eval-thinking --sandboxes 8 --eval-limit 10
```
