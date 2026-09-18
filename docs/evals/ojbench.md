# ojbench setup

Canonical OJBench (`He-Ren/OJBench_testdata`, 232 NOI/ICPC competitive-programming
problems) scored by **online-judge Pass@1**: the model emits a full stdin/stdout
program, judged Accepted iff **every** testcase passes within the per-problem
time/memory limits. Judging uses the official `ojbench` library over the DMOJ
sandbox; this is the only correct scorer (a substring/heuristic check cannot
verify program behaviour).

The dataset prompt already embeds the required response format, so it is sent
**verbatim**; the response has its thinking tags stripped before judging.

## Why the judge runs in a container

gbench does the model rollout itself (HTTP against the served model), then hands the generated
programs to a local image for judging. The judge (`ojbench` + the DMOJ `dmoj` judge-server + PyPy3
+ g++) is containerized because DMOJ's `cptbox` sandbox does **not** build on Python 3.12 (its
bundled Cython C uses `PyLongObject.ob_digit`, removed in 3.12) and the gbench serving env is 3.12.
On the image's Python **3.11** it compiles cleanly. Nothing DMOJ-versioned touches the main env.

> **DMOJ version is load-bearing.** OJBench's `judger.py` calls `problem.cases()`, an API present
> only on the DMOJ judge-server commit its README pins (`f098cd3a49a60186d1fadde5132329ec5f4f2213`).
> The PyPI `dmoj==4.1.0` that `pip install -e OJBench` would pull has **no** `Problem.cases()`, so
> every executable submission raises `AttributeError` inside the judge worker and NOTHING gets
> graded - a silent all-0. The Dockerfile therefore installs DMOJ from that exact commit and asserts
> `hasattr(Problem, "cases")` at build time. The runner also treats a submission the judge returned
> **no verdict** for as a harness failure (`scoring_excluded`), and hard-errors if the judge graded
> nothing at all - a judge failure can never surface as a clean 0%.

## Requirements (else the suite hard-errors `infra_required`)
- **Docker** (a reachable daemon). The judge run needs `--cap-add SYS_PTRACE` and
  `--security-opt seccomp=unconfined` (gbench passes both) so the inner cptbox ptrace+seccomp
  sandbox works; Docker's default seccomp profile blocks installing the inner filter.
- **The local image**, built once (gbench never pulls it):
  ```bash
  docker build -t gbench-ojbench -f docker/ojbench.Dockerfile docker
  ```
  Override the name with `GBENCH_OJBENCH_IMAGE` if you retag it. The image clones He-Ren/OJBench and
  installs `ojbench` + `dmoj` + PyPy3 + g++ internally.
- **Testdata**, bind-mounted read-only: set `GBENCH_OJBENCH_TESTDATA` (legacy alias
  `OJBENCH_TESTDATA`) to a snapshot of `He-Ren/OJBench_testdata` containing `NOI/` and `ICPC/`
  (~7.85 GB; `hf download He-Ren/OJBench_testdata --repo-type dataset --local-dir <dir>`). It is
  NOT baked into the image.

## Run
```bash
export GBENCH_OJBENCH_TESTDATA=/path/to/OJBench_testdata
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals ojbench \
       --sandboxes 8 --eval-limit 20
```
`--sandboxes` maps to the judge `num_workers` inside the container; `category_accuracy` reports per
`{dataset}_{difficulty}` (e.g. `NOI_hard`). Use `--max-output-tokens` >= 8192 for full solutions.

## Crash resilience

Upstream `ojbench.judger.judge_jsonl_data` did `result_queue.get()` with no timeout, so if a judge
worker crashed **hard** while executing a submission (a C-level cptbox death, no Python traceback),
its result was never queued and the whole judge **hung forever**. The image bakes a patch
(`docker/ojbench_patch_judger.py`): a bounded `get()` plus a break once **every** worker has exited
(a slow-but-alive worker keeps waiting; only all-dead stops the loop). A crashed-worker submission is
left **not-passed**; all others still get real verdicts; the judge never deadlocks. `OJBENCH_GET_TIMEOUT_S`
(default 120) tunes the poll interval, and the host wraps the whole container in
`GBENCH_OJBENCH_JUDGE_TIMEOUT_S` as an outer backstop. NOTE: a small number of submissions can crash
the cptbox sandbox during execution (root cause not yet isolated - the sandbox self-tests pass for
pypy3/C/C++, so it is submission-specific); those score not-passed rather than a true verdict.
