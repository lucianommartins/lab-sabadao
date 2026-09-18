# skillsbench

Canonical **SkillsBench** ([benchflow-ai/skillsbench](https://github.com/benchflow-ai/skillsbench)):
a **deterministic, execution-based** agentic benchmark, **87 tasks**, each a self-contained
per-task Docker environment scored by a **pytest verifier** that writes a reward in `[0,1]` to
`/logs/verifier/reward.txt`. **There is no LLM judge anywhere** (the previous gbench runner both
quarantined the suite *and* scored an LLM-judge-over-text; neither is canonical; both are gone).

## How gbench runs it

The agentic run is **delegated to the upstream BenchFlow runner** (`bench eval run`) inside a LOCAL
orchestrator container (`gbench/docker/skillsbench.Dockerfile`), driven **docker-out-of-docker**:
`bench eval run --sandbox docker` builds and runs each **task** container as a sibling on the host
daemon. Inside each task container the agent produces the solution, then the task's deterministic
verifier produces the reward. gbench authors no agent loop and no judge; it drives `bench` directly
(the repo's bundled open-model runner script targets an older BenchFlow and is unused).

> **docker-out-of-docker note.** BenchFlow bind-mounts its working dirs into the sibling task
> containers, and a `docker run -v` from inside a container refers to the **host** filesystem. The
> harness therefore mounts its workdir as an **identity mount** (same path on host and in the
> orchestrator) and points `TMPDIR` there, so the sibling sandbox sees the real files and the reward
> is collected. This was verified: the model-free `oracle` agent scores `reward=1.0` end-to-end.

- **Headline `accuracy`** = mean reward over the tasks in the primary condition (default
  `without-skills` = the model's raw capability), ×100.
- **Conditions**: `without-skills` and `with-skills`; when both run, the **with-skills-vs-without
  lift** (upstream's signature metric) is reported.

### Agent + model wiring (BenchFlow 0.6.3)

The pinned BenchFlow (**0.6.3**, from the repo's own `uv.lock`) does **not** ship the leaderboard's
`opencode` agent; its open-model agents (`deepagents`, `codex`, …) read
`BENCHFLOW_PROVIDER_BASE_URL` / `BENCHFLOW_PROVIDER_API_KEY`. gbench defaults to **`deepagents`** and
injects the gbench `/v1` endpoint via `--agent-env BENCHFLOW_PROVIDER_BASE_URL=… --agent-env
BENCHFLOW_PROVIDER_API_KEY=…` (plus `OPENAI_BASE_URL`/`OPENAI_API_KEY`). Because the agent differs
from the leaderboard's `opencode` (and gbench runs a single trial vs upstream's 3),
`leaderboard_comparable` is False unless you supply the `opencode` agent and run a full, both-
condition, greedy pass, but the **deterministic verifier scoring is canonical either way**.

## Requirements

- **Docker** with a **reachable daemon whose socket is mounted** (the harness passes
  `-v /var/run/docker.sock` + the identity-mounted workdir; docker-out-of-docker was verified).
- **The `gbench-skillsbench` image, built LOCALLY** (never pulled): clones benchflow-ai/skillsbench
  at a pinned commit, `uv sync`s BenchFlow 0.6.3, and installs the Docker CLI **and the
  `docker compose` v2 plugin** (BenchFlow's docker sandbox drives `docker compose`):
  ```bash
  docker build -t gbench-skillsbench -f gbench/docker/skillsbench.Dockerfile gbench/docker
  ```
- **Network egress**: ~28 of the 87 task images fetch public toolchains/data at build time (uv,
  nodesource, pytorch cpu wheels, apt, a few github/arxiv/HF pulls; all public, no auth), and the
  verifier `pip install`s pytest.
- **A served model at `/v1` reachable FROM the sibling task containers.** The task containers are on
  the docker bridge, so the endpoint must be bridge-reachable; gbench rewrites a
  `127.0.0.1`/`localhost` endpoint to the bridge gateway (`GBENCH_SKILLSBENCH_TASK_HOST`, default
  `172.17.0.1`), or override the whole URL with `GBENCH_SKILLSBENCH_TASK_ENDPOINT`.
- **No GEMINI key, no paid API**: scoring is deterministic. The 14 credentialed/GPU tasks live in
  upstream's `tasks-extra/` and are excluded by default (only `mhc-layer-impl` is excluded from
  `tasks/`).
- The model-free **`oracle` agent** (`GBENCH_SKILLSBENCH_AGENT=oracle`) runs each task's reference
  solution + verifier, a pipeline smoke needing only Docker + egress (no served model).

## Environment knobs

| var | effect |
| --- | --- |
| `GBENCH_SKILLSBENCH_TEMPERATURE` | per-suite temperature override |
| `GBENCH_SKILLSBENCH_IMAGE` | override the image name (default `gbench-skillsbench`) |
| `GBENCH_SKILLSBENCH_CONDITIONS` | comma list from `{without-skills,with-skills}` (default `without-skills`) |
| `GBENCH_SKILLSBENCH_AGENT` | agent (default `deepagents`; `oracle` for a model-free smoke) |
| `GBENCH_SKILLSBENCH_TASK_HOST` | bridge host the task containers use to reach `/v1` (default `172.17.0.1`) |
| `GBENCH_SKILLSBENCH_TASK_ENDPOINT` | full override of the task-container `/v1` URL |
| `GBENCH_SKILLSBENCH_TASKS` | comma list of task ids to run |
| `GBENCH_SKILLSBENCH_TIMEOUT_S` | container timeout (default 24h; full runs are long) |

`--eval-limit` caps to the first N task dirs (sorted, deterministic). Run both conditions for the
canonical signature; expect a long wall-clock (per-task image build + agent + verifier).

## Run

```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals skillsbench --eval-limit 5
```

The result payload carries `mean_reward` (0-1), `condition_means`, `with_skills_lift`, `n_tasks`,
the raw summary, and the temperature provenance.
