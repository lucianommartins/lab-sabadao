# mcp_bench

Canonical **MCP-Bench** ([Accenture/mcp-bench](https://github.com/Accenture/mcp-bench),
arXiv:2508.20453): a live agentic benchmark over **28 real Model Context Protocol servers**
(≈250 tools). For each task the model discovers the available tools, plans a multi-step
trajectory (sequential + parallel calls), invokes the tools against the live servers, and
grounds a final answer. gbench runs the **real** agent loop; it does not substitute a
single-turn proxy.

## What is measured

The 0-1 **Overall Score** (headline `accuracy`, reported ×100) is the mean of four dimensions:

| dimension | source |
| --- | --- |
| **schema understanding** | rule-based: `input_schema_compliance`, `valid_tool_name_rate` (0-1) |
| **task completion** | judge: mean(task_fulfillment, grounding) |
| **tool usage** | judge: mean(tool_appropriateness, parameter_accuracy) |
| **planning effectiveness** | judge: mean(dependency_awareness, parallelism_and_efficiency) |

The three judged dimensions come from six sub-dimensions scored 1-10, each judged **5× with
randomized dimension/criterion order and averaged** (upstream's "judge stability" protocol).
gbench keeps that protocol and the prompts **verbatim**; the 1-10 judge dimensions are divided
by 10 to sit on the 0-1 axis with the rule-based dimension.

### Judge: gbench Gemini cascade (not o4-mini)

Upstream judges with Azure **o4-mini**. gbench injects its **established Gemini cascade** as the
judge (the same model list / rounds / backoff as `base.judge_generate_cascade`, reached through
Gemini's OpenAI-compatible endpoint), leaving upstream's 6-dimension prompts and 5× stability
loop untouched; only the model underneath changes, for consistency with every other gbench
judged suite. The MCP-Bench README requires o4-mini to reproduce leaderboard numbers, so a
gbench run is **never `leaderboard_comparable=True`**.

## How gbench runs it

The agent loop + all 28 servers are **delegated to the upstream runner inside a LOCAL container**
(`gbench/docker/mcp_bench.Dockerfile`). gbench's launcher (baked into the image):

* points the model-under-test at the gbench `/v1` endpoint (upstream's `openai_compatible`
  provider; the agent uses plain `chat/completions`, so **no native tool-calling is required**);
* injects the Gemini cascade judge (neutralizing the hard-coded o4-mini default);
* runs only the tasks whose servers are **provisioned** and writes a manifest of what ran vs was
  dropped ("canonical-when-provisioned").

## Requirements

- **Docker** + the **`gbench-mcp-bench` image, built LOCALLY** (never pulled). Heavy: builds ~9
  TypeScript + ~19 Python servers from source (tens of minutes; BioMCP/alphagenome, the
  hand-built metmuseum server, and numpy/scipy servers are the slow/fragile ones; the upstream
  `install.sh` tolerates a per-server failure, so a flaky server narrows the runnable subset
  rather than breaking the image):
  ```bash
  docker build -t gbench-mcp-bench -f gbench/docker/mcp_bench.Dockerfile gbench/docker
  ```
  (Clones Accenture/mcp-bench at a pinned commit.)
- **`GEMINI_API_KEY`**: the Gemini cascade judge. **Hard-errors** (`infra_required`, never
  skips) if unset.
- **A served model at `/v1`**: the gbench endpoint (`--remote-endpoint`). The container reaches
  it via `--network host`.
- **Network egress + optional server keys** (all OPTIONAL: they only widen the provisioned
  subset; nothing is fabricated for an unprovisioned server):
  - **23 of 28 servers run keyless**: 6 fully offline (Bibliomantic, Math, Medical Calculator,
    Scientific Computing, Time, Unit Converter), 17 need only outbound internet.
  - **4 free-registration keys**: `NASA_API_KEY`, `NPS_API_KEY`, `HF_TOKEN`, `NCI_API_KEY`
    (BioMCP; NCI is optional, most BioMCP tools work without it).
  - **1 billing-gated key**: `GOOGLE_MAPS_API_KEY` (Google Cloud billing account).
  Set any of these in the environment and gbench passes them through; tasks whose servers are
  unavailable are dropped from the run and listed in the manifest.

With full provisioning (internet + all 5 keys) the run covers all **104 tasks** (56 single-server
+ 30 two-server + 18 three-server). Otherwise it runs the provisioned subset and says so; either
way `leaderboard_comparable` is False.

## Environment knobs

| var | effect |
| --- | --- |
| `GBENCH_MCP_BENCH_TEMPERATURE` | per-suite temperature override |
| `GBENCH_MCP_BENCH_IMAGE` | override the image name (default `gbench-mcp-bench`) |
| `GBENCH_MCP_BENCH_DISTRACTION` | `--distraction-count` (0 disables distraction servers) |
| `GBENCH_MCP_BENCH_DISABLE_STABILITY` | run the judge once instead of the 5× stability average |
| `GBENCH_MCP_BENCH_NO_SUBSET` | run every task regardless of provisioning (tasks with missing servers will fail) |
| `GBENCH_MCP_BENCH_ASSUME_NETWORK` | skip the egress probe and assume network is up |
| `GBENCH_MCP_BENCH_TIMEOUT_S` | container timeout (default 24h; full runs are long) |
| `GBENCH_JUDGE_MODELS` / `GBENCH_JUDGE_MODEL` / `GBENCH_JUDGE_CASCADE_ROUNDS` / `GBENCH_JUDGE_BACKOFF` | passed through to the in-container cascade judge (same knobs gbench uses) |

## Run

```bash
export GEMINI_API_KEY=...
# optional, to widen the server subset:
export NASA_API_KEY=... NPS_API_KEY=... HF_TOKEN=...
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mcp_bench --eval-limit 10
```

`--eval-limit` caps the number of tasks (applied after subset filtering). The result payload
carries `overall_score` (0-1), the four `dimensions`, `per_file`, `tasks_run`, the full
`raw_metrics`, the `subset_manifest` (available/unavailable servers, kept/dropped counts), and
the temperature provenance. Start small: the multi-server agent loop + 5× judge stability makes
a full run long.
