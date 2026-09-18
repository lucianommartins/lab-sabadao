# wildclawbench setup

Canonical **WildClawBench** ([github.com/internlm/WildClawBench](https://github.com/internlm/WildClawBench),
[HF dataset](https://huggingface.co/datasets/internlm/WildClawBench)): 60 hand-built, real-world,
long-horizon agent tasks across 6 categories (EN + ZH). Each task runs in its **own Docker
container** from the prebaked image `wildclawbench-ubuntu:v1.3`, with the **OpenClaw** agent inside
chaining 10-60+ tool calls (browser, bash, filesystem, email, calendar) over 10-20 minutes; after
the agent finishes, the task's own `grade()` (a mix of **programmatic checks + an LLM judge**) is
`docker exec`'d for a per-metric 0-1 score. gbench delegates the whole run to the pinned upstream
harness (`eval/run_batch.py`) inside a LOCAL orchestrator image, driven **docker-out-of-docker**.

Pinned harness commit: `316334ccc4a87b9b5635ad73da99b4dfc0b3887e`.

## What gbench changes (everything else is upstream, verbatim)
- **Model endpoint**: the OpenClaw agent is pointed at the gbench `/v1` endpoint via an injected
  `my_api.json` custom provider (upstream's supported "custom endpoint" path), instead of OpenRouter.
- **Judge**: all 43 judged tasks call the judge through the stock OpenAI SDK
  (`OpenAI(base_url=OPENROUTER_BASE_URL).chat.completions.create(model=JUDGE_MODEL)`) and declare
  `OPENROUTER_API_KEY/OPENROUTER_BASE_URL/JUDGE_MODEL` in their `## Env`. gbench points those at an
  in-orchestrator **Gemini cascade proxy** (`wildclawbench_cascade_judge.py`): same model list /
  rounds / backoff as `base.judge_generate_cascade`, reached over Gemini's OpenAI-compatible
  endpoint. The judge **prompts are untouched**. Because the judge is Gemini (not the canonical
  `openai/gpt-5.4`), a run here is **never `leaderboard_comparable`**.
- **Task-container network**: the task containers run `--network host` by default
  (`GBENCH_WILDCLAWBENCH_TASK_NETWORK`) so they can reach the local model + judge proxy at
  `127.0.0.1`. Set it to `bridge` for the upstream default (then endpoints are rewritten to the
  docker bridge gateway).

## Metric
Headline **`accuracy`** = the harness's own **equal-weight global mean** of per-task `overall_score`
(0-1) over **all selected tasks**; a missing/errored task counts as 0. This is exactly `run_batch`'s
`global_avg` (`total_score / total_tasks`). **It is NOT the leaderboard's "Overall Score"**, which
follows a weighted multimodal/pure-text breakdown; that is approximated separately as
`weighted_overall` (0.5·MM + 0.5·pure-text). Also reported: `multimodal_mean` / `pure_text_mean`
(each task declares `modality`), `avg_time_min`, `avg_cost_usd` (≈0 with a self-hosted endpoint,
OpenRouter cost is unavailable), and three honesty counters: `judge_fallback_tasks` (a judge outage
triggered the task's own deterministic fallback: `regex`/`rule`/`keyword`/`heuristic`), `n_missing`
(selected tasks that produced no score.json, counted as 0), and `n_grading_errors`.

## Sampling & concurrency
- **Sampling**: gbench does **not** pin a temperature for wildclawbench; the OpenClaw agent samples
  at its harness default (the leaderboard also runs each model under its own harness defaults).
- **Concurrency**: with the default `--network host` task containers, the OpenClaw gateway's fixed
  port forces **serial** execution (`parallel` is clamped to 1). Set
  `GBENCH_WILDCLAWBENCH_TASK_NETWORK=bridge` (only on a host where the docker bridge can reach the
  model, not a `FORWARD DROP` host) to run tasks in parallel; each bridge container gets its own
  netns so the gateway port no longer collides.

## Judge fidelity & known limitations (all → `leaderboard_comparable=False`)
- The judge is the gbench **Gemini cascade** (not the canonical `openai/gpt-5.4`), delivered through
  an in-orchestrator OpenAI-compatible proxy. The proxy forwards each task's judge prompt verbatim,
  overriding only the model (cascade) and temperature (0.0), and strips fields Gemini's OpenAI-compat
  endpoint rejects (`thinking`/`reasoning` from three Safety tasks; non-`function` tools such as the
  OpenRouter server-side `web_search` used by one Search task).
- **04_Search_Retrieval_task_1** (Google-Scholar) has a *web-search verification* judge step; the
  Gemini flash judge has no web tool, so that step isn't fully reproducible and may bias the task
  toward its screening-only score, surfaced via `judge_fallback_tasks`/`judge_error`.
- **05_Creative_Synthesis_task_11** (video dub) grades an audio dimension by sending `input_audio` to
  the judge; if the Gemini cascade endpoint rejects audio content, that dimension degrades to 0,
  again a judge-infra limitation, not model quality.
- Three tasks (01_task_3, 01_task_8, 06_task_7) give the **agent** an OpenRouter multimodal helper
  via `## Env`; under gbench that helper is also routed to the Gemini proxy (temperature pinned 0,
  model ignored), so those tasks' agent-side multimodal calls run on Gemini.

## Prerequisites (all required; the suite HARD-ERRORS via `infra_required`, never skips)

### 1. Build the LOCAL orchestrator image (gbench never pulls)
```bash
docker build -t gbench-wildclawbench -f gbench/docker/wildclawbench.Dockerfile gbench/docker
```
Bakes the pinned harness's Python runtime (`python-dotenv`, `pyyaml`) + the Docker CLI + the gbench
adapter (launcher, cascade-judge proxy, entrypoint). Lean by design: the large workspace + the
13.5 GB task image are NOT baked (see below).

### 2. Provision the WildClawBench checkout + workspace  (`GBENCH_WILDCLAWBENCH_HOST_DIR`)
The task workspace includes 3 YouTube videos that cannot be redistributed, so it is provisioned
once on the host and IDENTITY-mounted into the orchestrator (docker-out-of-docker requires the
task bind-mount source to resolve on the host):
```bash
git clone https://github.com/internlm/WildClawBench.git /srv/WildClawBench
cd /srv/WildClawBench && git checkout 316334ccc4a87b9b5635ad73da99b4dfc0b3887e
pip install -U "huggingface_hub[cli]"
# NOTE: use --include "workspace/**" (a directory PREFIX). A bare positional `workspace`
# is resolved by `hf download` as a single FILE and 404s; --include also skips the repo's
# Images/ tarballs (~57 GB of prebaked task images you load separately via `docker load`).
hf download internlm/WildClawBench --repo-type dataset --local-dir . --include "workspace/**"   # ~1.1 GB
pip install -r requirements.txt          # yt-dlp, ffmpeg (system), modelscope, gdown for prepare
bash script/prepare.sh                    # downloads 3 YouTube videos + SAM3 weights, extracts dot_git
export GBENCH_WILDCLAWBENCH_HOST_DIR=/srv/WildClawBench
```
`script/prepare.sh` needs `yt-dlp`, `ffmpeg`, `gdown`, `modelscope` and network egress. YouTube may
require cookies (see the dataset README). Tasks whose input data is missing will score 0 (they are
NOT skipped).

### 3. Load the task image on the host daemon (13.5 GB, `docker load`, never `docker pull`)
```bash
hf download internlm/WildClawBench Images/wildclawbench-ubuntu_v1.3.tar --repo-type dataset --local-dir /srv/wcb-images
docker load -i /srv/wcb-images/Images/wildclawbench-ubuntu_v1.3.tar   # -> wildclawbench-ubuntu:v1.3
```

### 4. Keys + model
- **`GEMINI_API_KEY`**: the judge cascade (43 of 60 tasks grade with an LLM judge). Required.
- **`BRAVE_API_KEY`**: **required for ALL tasks**, not just Search & Retrieval: the OpenClaw
  gateway baked into `wildclawbench-ubuntu:v1.3` treats `tools.web.search.apiKey` as a required
  secret with no fallback and **refuses to start without it** (verified: gateway aborts with
  `WEB_SEARCH_KEY_UNRESOLVED_NO_FALLBACK`, so every task fails). Get a free key at
  [brave.com/search/api](https://brave.com/search/api/).
- A **served model at `/v1`** reachable from the task containers (with `--network host`, that is
  `127.0.0.1`). Pass it via `--remote-endpoint` / the sweep's model endpoint.

## Run
```bash
export GBENCH_WILDCLAWBENCH_HOST_DIR=/srv/WildClawBench
export GEMINI_API_KEY="..."
export BRAVE_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals wildclawbench
```

## Environment knobs
| Env var | Default | Meaning |
|---|---|---|
| `GBENCH_WILDCLAWBENCH_HOST_DIR` | (required) | provisioned WildClawBench checkout (harness + workspace), identity-mounted |
| `GBENCH_WILDCLAWBENCH_IMAGE` | `gbench-wildclawbench` | orchestrator image tag |
| `GBENCH_WILDCLAWBENCH_TASK_IMAGE` | `wildclawbench-ubuntu:v1.3` | prebaked task image (host-loaded) |
| `GBENCH_WILDCLAWBENCH_TASK_NETWORK` | `host` | task-container docker network (`host`, `bridge`, or a network name) |
| `GBENCH_WILDCLAWBENCH_TASK_ENDPOINT` | (derived) | override the model URL the task containers use |
| `GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT` | (derived) | override the judge URL the tasks use |
| `GBENCH_WILDCLAWBENCH_JUDGE_PORT` | `18790` | port the in-orchestrator cascade proxy listens on |
| `GBENCH_WILDCLAWBENCH_CATEGORIES` | (all six) | comma list of categories to run |
| `GBENCH_WILDCLAWBENCH_TASK` | (unset) | path to a single task `.md` (smoke) |
| `GBENCH_WILDCLAWBENCH_PARALLEL` | `--concurrency` | parallel task containers |
| `GBENCH_WILDCLAWBENCH_THINKING` | `high` (when `--thinking`) | OpenClaw thinking level (`agents.defaults.thinkingDefault`) |
| `GBENCH_WILDCLAWBENCH_TEMPERATURE` / `--temperature` | (not applied) | **no-op for this suite**: OpenClaw's harness owns sampling; gbench does not pin a temperature |
| `WILDCLAW_JUDGE_REQUEST_TIMEOUT_S` | 40 | per-judge-call cap on the cascade proxy (kept < the upstream 120s grade() exec cap) |
| `GBENCH_WILDCLAWBENCH_TIMEOUT_S` | 86400 | orchestrator wall-clock cap (timeout → infra_required + containers reaped) |
| `GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION` | (unset) | set to `1` to skip the live GEMINI/BRAVE key-validation ping at gate time (air-gapped) |
| `GEMINI_API_KEY`, `BRAVE_API_KEY` | (env) | judge cascade / search tasks |
| `GBENCH_JUDGE_MODELS`, `GBENCH_JUDGE_MODEL`, `GBENCH_JUDGE_CASCADE_ROUNDS`, `GBENCH_JUDGE_BACKOFF`, `GEMINI_OPENAI_BASE_URL` | gbench defaults | judge cascade tuning (shared with all judged suites) |

## What hard-errors (`infra_required`, never a skip / never a fake number)
- Docker CLI / daemon not reachable.
- Orchestrator image `gbench-wildclawbench` not built.
- `GBENCH_WILDCLAWBENCH_HOST_DIR` unset, or not a checkout, or missing `workspace/`.
- Task image `wildclawbench-ubuntu:v1.3` not loaded on the host.
- `GEMINI_API_KEY` or `BRAVE_API_KEY` unset.
- `GEMINI_API_KEY` / `BRAVE_API_KEY` **rejected by a live validation ping** (a definitive auth
  rejection: 401/403, or Brave 422, or Gemini `API_KEY_INVALID`). Network flakiness at gate time is
  inconclusive and does NOT block; set `GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION=1` for air-gapped /
  offline provisioning.
- The orchestrator produced no summary (a harness failure, reported as `status:error`, not 0%).

## Leaderboard comparability
Always **`leaderboard_comparable=False`**: the judge is the gbench Gemini cascade (not the canonical
`openai/gpt-5.4`), the reported `weighted_overall` is a 0.5/0.5 MM/pure-text approximation of the
paper's category weighting, and gbench runs a single trial. `judge_fallback_tasks > 0` (a judge
outage that regex-fell-back) is surfaced and added to the reason.
