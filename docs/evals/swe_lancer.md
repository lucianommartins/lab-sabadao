# swe_lancer setup

> **Status: deregistered (roadmap-only, not runnable).** `swe_lancer` is not in the `SUITES`
> registry, so `--evals swe_lancer` / `--evals all` will not run it. The setup below is retained
> for the roadmap; a graded run needs `/dev/kvm` (see docs/evals_roadmap.md).

> **Docker setup:** this suite runs one container/network per task and can exhaust the
> Docker address pool at high `--sandboxes`. See [docker-sandboxes.md](docker-sandboxes.md)
> for the recommended `default-address-pools` config and pre-run prune, strongly advised
> before running at concurrency.


SWE-Lancer scored by **execution**: the
model's patch is applied inside the task's Docker image and the hidden end-to-end
(Playwright/pytest) test suite is run; an IC-SWE task resolves iff those tests pass
(SWE-Manager tasks are scored by the correct proposal selection). Only the official
harness can verify this, so the earlier substring/filename heuristic was **removed**
(a substring match on a patch measures nothing).

> **Dataset provenance:** the prompts are loaded from the community mirror
> **`DCAgent2/swe-lancer`**, while the scoring harness is a clone of
> `openai/SWELancer-Benchmark`. If the two number their tasks differently every
> lookup misses; gbench detects zero task-id overlap and reports an explicit
> `task_id mismatch` error rather than a 0% resolved rate.

## Upstream harness state (read this first)

`openai/SWELancer-Benchmark`'s **`main` branch contains only `README.md`** - cloning it the
obvious way gives you nothing. The harness lives on other branches:

| branch | files |
| --- | ---: |
| `main` | 1 (README only) |
| `remove-pwds` | 3093 |
| `mw/lite` | 2045 |

```bash
git clone https://github.com/openai/SWELancer-Benchmark
cd SWELancer-Benchmark && git checkout remove-pwds
```

**Checking out a populated branch is necessary but not sufficient.** The entrypoint is
`run_swelancer.py`, a [nanoeval](https://github.com/openai/nanoeval) script that:

* runs **its own agent loop** (`SimpleAgentSolver`) rather than scoring predictions
  produced elsewhere;
* has the **model hardcoded in the script** - `SimpleAgentSolver(model="gpt-4o")` - not
  passed on the command line;
* accepts only `--issue_ids`, with concurrency and recording edited inside the file;
* needs a `.env` copied from `sample.env` (Pusher credentials and friends).

> **CRITICAL: running this as shipped benchmarks the wrong model.** Because the model is
> hardcoded to `gpt-4o` in the upstream agent, an unadapted run produces a real, plausible
> SWE-Lancer number for **GPT-4o**, not for the model under test, and nothing in the output
> says so. A wrong number here is worse than no number, which is why the suite skips rather
> than running. The one encouraging detail: `swelancer_agent.py` builds `AsyncOpenAI(...)`
> with **no `base_url`**, so the SDK honours `OPENAI_BASE_URL` from the environment; an
> adapter can redirect the agent at the gbench endpoint without patching that call.

gbench's contract is the opposite: it generates the patches itself and hands the harness a
predictions file to score. **gbench now SHIPS that adapter**, `run_swelancer_eval.py` +
`predictions_solver.py` (a nanoeval `PredictionsSolver` that applies gbench's patch instead of the
gpt-4o agent, then reuses SWELancer's own container grading), and installs them into
`$SWELANCER_HARNESS_DIR` automatically at run time. You do **not** write an adapter. The suite
**hard-errors** (`infra_required`, never skips, never a fabricated number) if any prerequisite below
is missing, and reports `status:"error"` if the harness produces no report or the task-ids mismatch.

## Requirements
- **Docker** + the Python **`docker` SDK** (`pip install gbench[evals]`). The harness is
  Docker-out-of-Docker: the adapter spawns one `swelancer` container per task (non-privileged; Xvfb
  runs *inside* the container, so no host GUI is needed), so it needs a reachable Docker daemon.
- **The `swelancer` image, built LOCALLY** (heavy: clones the Expensify/App monorepo + Playwright;
  ~10 GB, tens of minutes; never pulled). gbench ships the Dockerfile:
  ```bash
  docker build -t swelancer -f docker/swe_lancer.Dockerfile $SWELANCER_HARNESS_DIR
  ```
  (It vendors upstream's `Dockerfile_x86` with one fix: recent Miniconda needs channel ToS accepted
  before `conda create`. The build context is the SWELancer checkout, which supplies `requirements.txt`
  + `issues/` + `runtime_scripts/`.)
- **Harness checkout**: set `SWELANCER_HARNESS_DIR` to a checkout of `openai/SWELancer-Benchmark`
  on a populated branch (see the table above; it provides `swelancer.py` + the vendored
  `project/nanoeval` + `project/alcatraz`, run via `uv`). gbench installs its adapter into this dir.
- **`.env`**: copy `sample.env` to `.env` in the harness dir (`USE_WEB_PROXY`, `EXPENSIFY_URL`,
  `NEW_EXPENSIFY_URL`, `ALCATRAZ_TIMEOUT`); the defaults work (grading replays recorded traffic via
  mitmproxy). No `OPENAI_API_KEY` is needed; the gpt-4o agent path is not used.
- **Runtime network**: each task container does an `npm install` at start; keep egress to npm/PyPI
  available (or pre-bake `node_modules`). The pytest grading itself is hermetic (mitmproxy replay).
- **`uv` on `PATH`**: the default scorer command launches the harness with `uv run` on the host, so
  `uv` must be installed and resolvable. `pip install gbench[evals]` now declares it (the wheel is a
  self-contained binary with no Python deps), or use the standalone installer
  (`curl -LsSf https://astral.sh/uv/install.sh | sh` -> `~/.local/bin`). If uv installs into a venv
  `bin`, that dir must be on `PATH` for the scorer subprocess. The suite hard-errors with this hint if
  uv is missing (or override the command below to avoid uv).
- **Explicit opt-in**: set `SWELANCER_RUN=1` (a cost guard: real runs spawn very large per-task
  containers). Missing it hard-errors with this instruction.
- Override the invocation with **`SWELANCER_EVAL_CMD`** (placeholders `{harness} {predictions}
  {output_dir} {num_workers}`) only if you need a non-default command; the default is
  `uv run python run_swelancer_eval.py --predictions {predictions} --output_dir {output_dir}
  --num_workers {num_workers}` run in the harness dir.

## Run
```bash
export SWELANCER_HARNESS_DIR=/opt/SWELancer-Benchmark
export SWELANCER_RUN=1
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals swe_lancer \
       --sandboxes 4 --eval-limit 5
```
Predictions are keyed by `question_id` (ic_swe tasks; SWE-Manager tasks are not patch-shaped and are
excluded); `--sandboxes` maps to `--num_workers`; the result carries `swe_lancer_report` (total /
resolved / matched) and per-issue results under the output dir. The model returns a single unified
diff patch. `leaderboard_comparable` is True only for a full greedy run.
Start small: per-task containers dominate wall-clock.
