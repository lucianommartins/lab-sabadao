# gaia2 setup

Canonical **GAIA2**, Meta's **Agents Research Environments (ARE)**
([github.com/facebookresearch/meta-agents-research-environments](https://github.com/facebookresearch/meta-agents-research-environments),
PyPI `meta-agents-research-environments`, HF dataset
[`meta-agents-research-environments/gaia2`](https://huggingface.co/datasets/meta-agents-research-environments/gaia2)).
A **stateful, multi-turn, time-sensitive** agentic benchmark: the agent acts inside an in-process
simulator (contacts / calendar / email / files with async, time-sensitive events), issuing tool
calls over many turns. Scoring is **hybrid**: deterministic "hard validation" (scripted/exact
checks) **plus a load-bearing LLM judge** for soft/semantic validation.

Pinned harness: `meta-agents-research-environments==1.2.0`.

## What gbench changes (everything else, incl. judge PROMPTS, is upstream verbatim)
- **No docker-out-of-docker.** ARE is a pure in-process Python simulator, so the orchestrator runs
  `are-benchmark` directly (no `/var/run/docker.sock`, no sibling task containers). It uses
  `--network host` to reach the served model.
- **Model wiring**, LiteLLM `local` provider: `--model openai/<served> --provider local --endpoint
  <gbench /v1>` (verified against `litellm_engine.py`: provider `local` → `custom_llm_provider=None`
  + `api_base=endpoint`; the `openai/` prefix routes LiteLLM to the OpenAI-compatible endpoint).
- **Judge**: ARE's default judge is `meta-llama/Meta-Llama-3.3-70B-Instruct`; gbench grades with its
  standard **Gemini cascade** via an in-container proxy (`--judge_provider local --judge_endpoint
  <proxy> --judge_model openai/gbench-cascade`), by convention across all judged suites. ARE's judge
  prompts are untouched. This is a gbench grader choice, not a defect.
- **`leaderboard_comparable` is always False** because gbench defaults to a single-run `run` (the
  canonical leaderboard uses the 3-phase `gaia2-run` x3 with Pass@k); a run here is also a
  gbench-internal, Gemini-graded number rather than a like-for-like leaderboard entry.

## Metric
Headline **`accuracy`** = the per-capability equal-weight **macro success rate** from ARE's own
`benchmark_stats.json` (`statistics.global.macro_success_rate`), the GAIA2 Overall. Also reported:
`micro_success_rate` (per-scenario), `pass_at_k_percent`/`pass_k_percent`, `total_scenarios`,
`no_validation_runs`, and the `per_capability` breakdown. gbench never fabricates a scalar: a run
that produces no `benchmark_stats.json` hard-errors.

## Modes
- **`run`** (default, `GBENCH_GAIA2_MODE=run`): `are-benchmark run` with **no `--hf-config`** runs
  all 5 capability configs (adaptability / ambiguity / execution / search / time) and writes one
  aggregated report. `GBENCH_GAIA2_CONFIG=mini` (or `demo`) narrows it for a smoke. gbench passes
  **`--agent`** (`GAIA2_AGENT`, default `default` - ARE's only registered agent) explicitly:
  `are-benchmark run` defaults `--agent` to `None`, which runs scenarios **agent-less** (the
  model-under-test is never invoked) yet still returns a real macro=0.0 - a fabricated score. gbench
  hard-errors (`infra_required`) if a run validated scenarios but no per-run trace shows any agent
  LLM usage, so an agent-less no-op can never be published as a 0%.
- **`gaia2-run`** (`GBENCH_GAIA2_MODE=gaia2-run`), the full canonical leaderboard submission: 3
  phases (Standard / Agent2Agent-mini / Noise-mini) × `--num_runs` (default 3), with Pass@k / Pass^k.
  Very heavy; opt-in.

## Prerequisites (the suite HARD-ERRORS via `infra_required`, never skips)

### 1. Build the LOCAL orchestrator image (gbench never pulls)
```bash
docker build -t gbench-gaia2 -f gbench/docker/gaia2.Dockerfile gbench/docker
```
Bakes the pinned ARE simulator + the **public** cc-by-4.0 GAIA2 dataset (no HF token) + the gbench
adapter (launcher, reused Gemini cascade-judge proxy, entrypoint). ARE hard-pins ~22 deps with `==`
(numpy/litellm/mcp/pydantic) that would break the vLLM stack, which is exactly why it lives in this
isolated container, never in the gbench serving environment.

### 2. Keys + model
- **`GEMINI_API_KEY`** (required): GAIA2's soft/semantic validation is done by an LLM judge, so
  without a judge model most scenarios score `no_validation`. gbench swaps in its Gemini cascade.
  The key is **live-validated** at gate time (a definitive auth rejection hard-errors; set
  `GBENCH_GAIA2_SKIP_KEY_VALIDATION=1` for air-gapped setups).
- A **served model at `/v1`** reachable from the container (with `--network host`, that is
  `127.0.0.1`).

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gaia2
```

## Environment knobs
| Env var | Default | Meaning |
|---|---|---|
| `GBENCH_GAIA2_IMAGE` | `gbench-gaia2` | orchestrator image tag |
| `GBENCH_GAIA2_MODE` | `run` | `run` (all 5 capabilities, single run) or `gaia2-run` (3-phase ×N) |
| `GBENCH_GAIA2_CONFIG` | (all 5) | single capability config, or `mini`/`demo` (smoke) |
| `GBENCH_GAIA2_LIMIT` | (none) | cap scenarios per config (smoke) |
| `GBENCH_GAIA2_NUM_RUNS` | 3 | `gaia2-run` repetitions (Pass@k/Pass^k) |
| `GBENCH_GAIA2_MODEL_ENDPOINT` | (derived) | override the model `/v1` URL the container uses |
| `GBENCH_GAIA2_JUDGE_ENDPOINT` / `GBENCH_GAIA2_JUDGE_MODEL` | cascade proxy | override the judge endpoint/model (e.g. self-judge for a keyless smoke) |
| `GBENCH_GAIA2_JUDGE_PORT` | 18790 | in-container cascade proxy port |
| `GBENCH_GAIA2_SCENARIO_TIMEOUT` / `GBENCH_GAIA2_MAX_CONCURRENT` | ARE defaults | passthrough tuning |
| `GBENCH_GAIA2_TIMEOUT_S` | 86400 | orchestrator wall-clock cap (timeout → infra_required + container reaped) |
| `GBENCH_GAIA2_SKIP_KEY_VALIDATION` | (unset) | `1` to skip the live GEMINI key ping (air-gapped) |
| `GBENCH_JUDGE_MODELS` / `GBENCH_JUDGE_MODEL` / `GBENCH_JUDGE_CASCADE_ROUNDS` / `GBENCH_JUDGE_BACKOFF` / `GEMINI_OPENAI_BASE_URL` | gbench defaults | judge cascade tuning (shared with all judged suites) |
| `GBENCH_GAIA2_TEMPERATURE` / `--temperature` | (not applied) | **no-op for this suite**; sampling follows the ARE/model default |

## What hard-errors (`infra_required`, never a skip / never a fake number)
- Docker CLI / daemon not reachable, or the orchestrator image `gbench-gaia2` not built.
- `GEMINI_API_KEY` unset, or rejected by a live validation ping (bypass with
  `GBENCH_GAIA2_SKIP_KEY_VALIDATION=1`).
- The orchestrator produced no `benchmark_stats.json` (a harness failure, `status:error`, not 0%).

## Leaderboard comparability
Always **`leaderboard_comparable=False`**: the judge is the gbench Gemini cascade (not the canonical
Llama-3.3-70B / GPT-OSS-120B), and gbench defaults to the single-run `run` (the canonical leaderboard
uses the 3-phase `gaia2-run` ×3 with Pass@k, and entries are maintainer-audited). Any `no_validation`
runs are surfaced in the reason.
