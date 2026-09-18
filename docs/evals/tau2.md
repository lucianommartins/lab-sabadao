# tau2 setup

> **Before a large sweep:** tau2/tau3 stage a ~700-file retrieval knowledge base into
> `/tmp/agentic_search_*` per simulation and can leak them (filling the tmpfs inode table,
> which crashes later evals with a misleading `No space left on device`). Raise the `/tmp`
> inode cap and clean stale dirs first. See [large-sweep-setup.md](large-sweep-setup.md).


tau2-bench (Sierra `sierra-research/tau2-bench`, domains: airline / retail / telecom)
is a multi-turn **environment** benchmark: an LLM user and the agent converse over
several turns, tools mutate a domain database, and each task is scored by the
simulator's oracle (final DB-state check) plus an nl_assertion LLM judge, yielding a
per-task reward.

## It cannot be scored single-turn; it hard-errors by default
Against a plain single-turn `/v1` endpoint, tau2 **cannot** be measured: many tasks are
assertion-only (no tool action to match), and the tasks that do require actions carry
gold arguments (e.g. `user_id`, `reservation_id`) that only exist in the multi-turn
environment state the model never sees. Scoring anyway yields a misleading ~0%, so by
default `tau2` **hard-errors** (`infra_required`) with that reason rather than emit a fake score.
(The single-turn strict-action matcher is retained as `_eval_tau2` for diagnostics/tests,
but is no longer wired as a default scorer.)

## Full environment (canonical): `TAU2_ENV_RUN=1`
The only way to get a real tau2 score is the actual simulator. gbench drives it through
the tau2-bench **Python API** (`run_domain(TextRunConfig(...))`), scoring each task by
its reward, matching the canonical reference setup. Three LLM roles are used:

- **agent** = your model, routed to the gbench endpoint via LiteLLM `openai/<model>`
  with `api_base`/`api_key` passed **per-call** (gbench does *not* set a global
  `OPENAI_API_BASE`, so the user simulator and judge keep their own providers);
- **user simulator** = `gemini/gemini-3.5-flash` by default (`TAU2_USER_LLM`);
- **nl-assertion judge** = `gemini/gemini-3.5-flash` by default (`TAU2_EVAL_LLM`).

### Requirements
- **Install the simulator** so `import tau2` works. tau2-bench is **not on PyPI**;
  clone it and install editable:
  ```bash
  git clone https://github.com/sierra-research/tau2-bench
  pip install -e ./tau2-bench            # installs the `tau2` package + its deps (litellm, …)
  ```
  Alternatively, if you already have a checkout and its deps, point **`TAU2_BENCH_SRC`**
  at its `src/` directory (gbench adds it to `sys.path`; you still need `litellm` etc.).
  - **Private-index note:** if `pip install -e ./tau2-bench` fails resolving a pinned dep
    (e.g. a 403 on `litellm`), install the small missing deps directly and skip tau2's
    dep graph: `pip install addict deepdiff && pip install -e ./tau2-bench --no-deps`
    (tau2 runs fine on a newer already-installed `litellm`).
  - **Python 3.13 note:** tau2's voice module imports the stdlib `audioop`, removed in
    3.13; install the backport once: `pip install audioop-lts`. (gbench never uses
    voice; this is only needed so `import tau2` succeeds.)
- **`GEMINI_API_KEY`**: required for the gemini user/judge defaults (or set
  `TAU2_USER_LLM` / `TAU2_EVAL_LLM` to another LiteLLM provider you have keys for).
- **`TAU2_ENV_RUN=1`** to opt in.

If the simulator isn't importable, the key is missing, or no domain produces results,
the suite **hard-errors** (`infra_required`); it never emits a misleading score.

### Tuning knobs (env)
`TAU2_TEMPERATURE` (1.0) · `TAU2_TOP_K` (64) · `TAU2_TOP_P` (0.95) ·
**`TAU2_NUM_TRIALS` (1**; set `=3` for 3-trial parity) · `TAU2_MAX_STEPS` (200) ·
`TAU2_MAX_ERRORS` (10) · `TAU2_SEED` (300) · `TAU2_USER_TEMPERATURE` (0.0) ·
`TAU2_PROGRESS_SECS` (30, upper bound on how often the progress bar's postfix refreshes;
the bar itself advances per completed task) ·
`TAU2_SAVE_TRACES` (unset; set to a directory to persist tau2's full SimulationResults
per domain, `tau2_<domain>_traces.json`: per-task messages + reward breakdown for auditing;
default keeps only the aggregate).
`--sandboxes` maps to the simulator's `max_concurrency`; `--eval-limit` maps to
`num_tasks`; `--eval-thinking` enables the model's thinking channel for the agent.

**Log noise & progress:** by default gbench quiets tau2's own output so the log stays
readable: (1) two cosmetic ERROR logs (litellm cost-mapping miss, `git rev-parse HEAD`
failure), (2) tau2's rich per-task "Simulation Overview" panels + live "Status: X/N
complete" progress reprints (which spam a file-redirected log since rich can't update in
place), (3) litellm's per-call Gemini-3+ sampling `DeprecationWarning` (emitted twice per
user-sim/judge call), and (4) tau2's import-time loguru noise: the `Registry info: {...}`
DEBUG dump (domains/agents/users/task_sets) and the benign `No .env file found` WARNING
(tau2 looks for an optional `.env` to load keys; gbench passes keys via the environment,
so a missing `.env` is irrelevant; genuine tau2 warnings such as task retries still
show). gbench builds its own summary from the results, so these are redundant. In their place gbench keeps the log **evolving like every other eval**:
- **per-domain lines** through its own logger (a "starting domain 'airline'" line then a
  per-domain mean-reward line when it finishes);
- a **tqdm progress bar**, one per domain, driven by tau2's own `StatusMonitor`, the same
  `tqdm(total=…, desc="Eval [X]")` mechanism every other eval uses. It advances per
  completed task and carries a live postfix (`reward`, `running`, `oldest`), e.g.
  `Eval [tau3 telecom]: 60%|██████ | 68/114 [12:03<08:15, reward=0.58, running=32, oldest=146s]`.
  Like the other evals' bars it updates **in place via carriage returns**, a single
  updating line on a TTY or under `tail -f`; a file redirect stores each `\r` frame (which
  a terminal collapses on display). The postfix refresh cadence is capped by
  `TAU2_PROGRESS_SECS` so the bar stays live between completions.

Set **`TAU2_VERBOSE=1`** to restore all tau2 panels + litellm logs + the registry dump +
tau2's original console heartbeat. Real errors still surface via loguru either way.

## Run
```bash
# default (no env): hard-errors (infra_required) with an explanation, not a misleading 0%
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals tau2
# full environment (the only scored path)
export GEMINI_API_KEY="..."
# after: git clone https://github.com/sierra-research/tau2-bench && pip install -e ./tau2-bench
# (or, if not installed as a package, point at the checkout instead:)
# export TAU2_BENCH_SRC=/path/to/tau2-bench/src
TAU2_ENV_RUN=1 gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals tau2 --sandboxes 8
```
`accuracy` is the canonical tau2 **mean reward**;
`correct_answers` is the perfect-task (reward ≥ 1) count; `category_accuracy` reports
per domain; a `tau2_report` field carries mean reward / perfect tasks / infra errors.
Pass a single domain with the `domain` kwarg.
