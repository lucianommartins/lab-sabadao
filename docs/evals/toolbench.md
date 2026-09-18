# toolbench

Canonical ToolBench scored by **StableToolBench** (THUNLP-MT/StableToolBench, arXiv:2403.07714):
the model runs a **multi-turn DFSDT agentic loop**, calling RapidAPI tools against a **cached** API
server (deterministic; an LLM simulator covers cache misses) and finishing with a `Finish` call. The
answer trees are scored by **ToolEval**:

- **SoPR** (Solvable Pass Rate): each attempt judged solved / unsolved / unsure (1 / 0.5 / 0), mean
  over queries, mean over `evaluate_times`. **Headline `accuracy`.**
- **SoWR** (Solvable Win Rate): pairwise preference vs the GPT-3.5-CoT reference
  (`virtual_chatgpt_cot`).

**Self-contained by design.** gbench bundles StableToolBench (pinned) + the 236 MB response cache +
the cached `/virtual` server into a **local Docker image built from `docker/toolbench.Dockerfile`**
(built locally, never pulled). The runner `docker run`s it: the container starts the server, runs the
DFSDT loop against your served `/v1`, and converts the answer trees; gbench then scores them with its
**Gemini cascade judge** (`base.judge_generate_cascade`), the same judge every gbench judged suite
uses, NOT the canonical gpt-4-turbo. ToolEval's exact SoPR/SoWR prompts + protocol are ported
verbatim; only the judge model changes. The cache-miss simulator is also pointed at Gemini's
OpenAI-compatible endpoint, so **the only key you need is `GEMINI_API_KEY`**, no OpenAI anywhere.
Because the judge differs from the published StableToolBench leaderboard (gpt-4-turbo), a run is
**never `leaderboard_comparable=true`**.

It **hard-errors** (`infra_required`, never skips, never a fabricated number) if Docker, the image,
or `GEMINI_API_KEY` is missing, and records `status:"error"` (not a 0) if the container/judge
produces nothing parseable.

## Build (one-time, local, never pulled)
```bash
docker build -t gbench-toolbench -f docker/toolbench.Dockerfile docker
```
This clones pinned StableToolBench, installs its inference + ToolEval env, downloads the 236 MB cache
(HF `stabletoolbench/Cache`) into the server, and bundles the cached `/virtual` server. Override the
tag with `GBENCH_TOOLBENCH_IMAGE`.

## Run
```bash
export GEMINI_API_KEY=...
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals toolbench
```
The runner `docker run --network host`s the image (so the container reaches your served endpoint +
Gemini), reads back the converted answers, and computes SoPR/SoWR with the Gemini judge.

## Knobs

| Env var | Default | Effect |
| --- | --- | --- |
| `GEMINI_API_KEY` | *(required)* | the gbench Gemini cascade judge **and** the cache-miss simulator |
| `GBENCH_TOOLBENCH_IMAGE` | `gbench-toolbench` | the locally-built image tag |
| `GBENCH_TOOLBENCH_GROUPS` | all 6 | comma/space test groups (subset ⇒ not comparable) |
| `GBENCH_TOOLBENCH_METHOD` | `DFS_woFilter_w2` | inference method (canonical DFSDT) |
| `GBENCH_TOOLBENCH_REFERENCE` | `virtual_chatgpt_cot` | SoWR reference model |
| `GBENCH_TOOLBENCH_REFERENCE_DIR` | *(image's shipped data)* | host dir of converted reference answers for SoWR (else SoWR is skipped, SoPR still reported) |
| `GBENCH_TOOLBENCH_EVAL_TIMES` | `1` | judge repetitions (gbench's judge is deterministic at 0.0, so repeats add nothing) |
| `GBENCH_TOOLBENCH_MAX_QUERY_COUNT` | `200` | per-task DFS **search budget** (canonical StableToolBench default). This is NOT a task count - `--eval-limit` caps the number of TASKS per group (the entrypoint slices the query file to the first N); `--max_query_count` bounds the search within each task. |
| `GBENCH_TOOLBENCH_SIMULATOR_MODEL` | `gemini-2.5-flash` | cache-miss simulator model (Gemini OpenAI-compat) |
| `GBENCH_TOOLBENCH_SIMULATOR_BASE` | Gemini OpenAI-compat endpoint | simulator API base |
| `GBENCH_TOOLBENCH_SIMULATOR_KEY` | `$GEMINI_API_KEY` | simulator key (if different) |
| `GBENCH_TOOLBENCH_TEMPERATURE` | *(run default)* | model-under-test sampling override (0.0 greedy / 1.0 `--thinking`) |

`leaderboard_comparable` is **always `false`**: judged by gbench's Gemini cascade, not the canonical
gpt-4-turbo ToolEval judge the published StableToolBench leaderboard uses. The number is a faithful
gbench-internal SoPR/SoWR (ToolEval prompts + protocol ported verbatim), consistent with every other
Gemini-judged gbench suite.
