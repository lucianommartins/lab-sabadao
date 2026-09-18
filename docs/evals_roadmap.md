# gbench Evaluation Roadmap

gbench's evaluation suite is a **living system**, not a frozen list. New benchmarks appear, existing
ones publish fixes and new splits, and "state of the art" for what a *faithful, reproducible* eval looks
like keeps moving. This document is the standing plan for that continuous improvement: the standards
every eval must meet, the engineering playbook for adding or hardening one, and the prioritized
roadmap of evals to build next.

It is a technical contract, not marketing. If an eval is listed here as *planned*, it means the
canonical harness is **not yet implemented**, so there is **no runnable suite** for it: it is either
**deregistered** (absent from the registry and `--evals all`) or, if a placeholder exists, it
**hard-errors** (`infra_required`) rather than emit an approximate or fabricated number.

---

## 1. Non-negotiable standards (definition of done)

Every gbench eval, existing or new, must satisfy all of the following. These are hard rules;
"partially done" is not done.

1. **Canonical or nothing.** A suite is either a faithful measurement of the benchmark
   ("canonical-when-provisioned") or it **hard-errors** via `swebench_common.infra_required(...)`.
   There is **no skip, no quarantine, no partial, and no fabricated/proxy number**. A keyword-overlap
   or "first-action match" score published under a canonical suite name is *worse than no number* (it
   silently poisons the leaderboard) and is therefore forbidden.
2. **Delegate to the pinned upstream harness** when one exists, at a **pinned commit** (a benchmark
   with no release tags → pin a 40-char SHA). Do not re-implement scoring that upstream already
   defines; reproduce the exact split, decoding (greedy unless the benchmark says otherwise), metric,
   and answer extraction.
3. **Local by design.** Heavy or graph-conflicting dependencies go in a **LOCAL**
   `gbench/docker/<suite>.Dockerfile` built on the host; **never `docker pull`** an eval image into
   gbench's own build, and never a side venv. Additive pure-Python deps may go in the canonical env
   only after a `pip install --dry-run` shows no version changes to the pinned stack.
4. **Scoring: deterministic verifier preferred.** If the benchmark ships an execution-based / rule
   verifier, use it (no LLM judge). If it requires an LLM judge, reuse gbench's **Gemini cascade**
   (`base.judge_generate_cascade`) with the upstream judge **prompts verbatim**; swap only the model
   call, never the rubric.
5. **`leaderboard_comparable` is earned, not assumed.** It is `True` only for a full, greedy,
   canonical-protocol run scored by the canonical judge/agent. Any deviation (Gemini judge instead of
   the canonical one, a subset, a single trial where the leaderboard averages N, a self-hosted agent)
   sets it `False` **with a specific machine-readable reason**.
6. **Honest denominators.** The headline metric divides by the number of *selected* tasks; a
   missing/errored/timed-out task counts as **0**, never dropped from the denominator (dropping
   inflates the score). Surface `n_scored` / `n_missing` / judge-fallback counts.
7. **Complete docs + tests.** Ship `docs/evals/<suite>.md` with the full provisioning recipe
   (checkouts + commit, `hf download`s, images to load, keys, toolchains, the exact run command, and
   precisely what hard-errors), plus deterministic unit tests and a smoke/E2E to the maximum extent
   the environment allows.

## 2. Engineering playbook (how to add or harden an eval)

**Files & registration.** A suite `foo` is: `gbench/runners/eval_suites/foo.py` (the runner),
optionally `gbench/docker/foo.Dockerfile` + `foo_run.py` (launcher) + `foo_entrypoint.sh` (adapter),
`docs/evals/foo.md`, `tests/test_foo.py`. Register in `gbench/runners/eval_suites/__init__.py`
(import + `SUITES` map + `__all__`) and add the name to its pillar in `gbench/runners/evals.py`.

**Container-delegation pattern** (reference: `skillsbench`, `wildclawbench`, `mcp_bench`,
`ui_control_osworld`). The runner builds a `docker run` of the local orchestrator image; the
orchestrator drives the upstream harness. Two sub-shapes exist:
- **docker-out-of-docker (DooD):** the harness spawns *sibling* task containers on the host daemon
  (mount `/var/run/docker.sock`). **Identity-mount pitfall:** any path the harness bind-mounts into a
  sibling (`-v <path>`) is resolved on the **host**, so that path must be identical inside the
  orchestrator and on the host: identity-mount the workdir/checkout (`-v H:H`) and set `TMPDIR`
  there, or siblings get empty mounts. This bit skillsbench, wildclawbench, and OSWorld's qcow2.
- **In-process simulator** (reference: `gaia2`): no sockets, no siblings: the harness is a pure
  Python simulator; the orchestrator just needs `--network host`.

**Networking.** The model-under-test is served at gbench `/v1` (OpenAI-compatible). The orchestrator
runs `--network host` and reaches it at `127.0.0.1`. On a hardened host (`iptables FORWARD DROP`)
the docker bridge cannot reach host ports, so **sibling task containers must also use `--network
host`** (then the fixed agent ports may collide across parallel siblings → serialize, or use bridge
mode `172.17.0.1` on an unfirewalled host).

**Model wiring.** Point the upstream agent at gbench `/v1` via whatever it supports: an OpenAI base
URL (`OPENAI_BASE_URL`), a custom-provider config (OpenClaw `my_api.json`), or LiteLLM's `local`
provider (`--model openai/<name> --endpoint`). If the harness routes by model-name prefix or
hardcodes a name, and vLLM strictly rejects unknown names, pass a routing alias and **monkeypatch the
outgoing request `model` field to the served name** (reference: OSWorld's `gpt-4o` alias + payload
rewrite).

**Judge.** Reuse `gbench/docker/wildclawbench_cascade_judge.py`, a stdlib, OpenAI-compatible HTTP
proxy that runs the Gemini cascade (same model list / rounds / backoff as `base.judge_generate_cascade`)
and forwards the upstream prompt verbatim, overriding only model + temperature (0.0) and stripping
provider-only fields (`thinking`/`reasoning`, non-`function` tools). It has been reused unchanged
across wildclawbench and gaia2.

**Verify-don't-assume.** Pin the commit and read the **real** upstream (repo/wheel), not the gbench
docstring: the docstring's *hypothesis* has been wrong more than once (wildclawbench was assumed to
be a BenchFlow benchmark; it was InternLM's own harness). Verify CLI flags from `cli.py` /
`shared_params.py`, not from a README. Live-validate keys at gate time (auth-rejection → hard-error;
network flakiness → inconclusive → pass). Reap sibling containers by run-scoped label on
timeout/cleanup.

## 3. Roadmap (prioritized)

### #1: `lmarena_web_agent` → **WebArena**   ·   status: **planned**   ·   effort: **XL**

**What it should measure.** Canonical **WebArena** (`web-arena-x/webarena`): **812 tasks** in which
an agent drives a real Chromium browser (Playwright, `observation_type="accessibility_tree"`, the CoT
id+actree agent `p_cot_id_actree_2s`) across self-hosted websites and is scored by **programmatic**
per-task evaluators. The suite is *misnamed*: it is WebArena, not the LMArena human-preference arena.

**Current state.** Removed a **fabricated** number (keyword overlap between a text "plan" and a
checklist, over the wrong dataset `WPRM/annotated_webarena_checklist`, never opening a browser). The
suite is now **deregistered (roadmap-only)**: it is not in the registry, so `--evals all` never runs
it and `--evals lmarena_web_agent` is an unknown suite. WebArena will be (re)introduced as a runnable
suite (name TBD, see Open decisions) when the canonical harness lands.

**Sites & ports** (all self-hosted): OneStopShop shopping (Magento, `:7770`), shopping-admin CMS
(`:7780/admin`), Postmill reddit (`:9999`), GitLab (`:8023`), OpenStreetMap map + tile/routing
backend (`:3000`), Wikipedia (kiwix `.zim`, `:8888`), a Flask homepage (`:4399`).

**Scoring** (`evaluation_harness/evaluators.py`): **deterministic-dominant**, `URLEvaluator`
(`url_match`), `HTMLContentEvaluator` (`program_html`), `StringEvaluator` (`exact_match` /
`must_include`, a substring check, *not* an LLM call). A **minority** use an LLM:
`StringEvaluator.fuzzy_match → llm_fuzzy_match` and unachievable-task `ua_match → llm_ua_match`
(upstream `gpt-4-1106-preview` @ temp 0).

**Buildable path** (mirror wildclawbench). `gbench/docker/lmarena_web_agent.Dockerfile` bakes a pinned
webarena checkout + Playwright (`playwright install --with-deps chromium`) in the *orchestrator*. The
runner `docker run --network host -v /var/run/docker.sock:...`; the orchestrator exports the
`SHOPPING/REDDIT/GITLAB/...` env → the operator-provisioned sibling site containers, runs
`browser_env/auto_login.py` to mint `.auth` cookies, invokes `run.py --provider openai --model
<served> --temperature ...` over `--test_start_idx/--test_end_idx`, and reads the per-task
programmatic scores → success rate. Model-under-test wired by pointing the legacy `openai` module at
gbench `/v1` (`openai.api_base`/`OPENAI_API_BASE`). **Judge separation is mandatory:** the agent and
the fuzzy/ua judge **share the legacy `openai` module global**, so the fuzzy judge must be
monkeypatched onto the gbench Gemini cascade (upstream `gpt-4-1106-preview` prompts verbatim); else
the model-under-test grades itself.

**External prerequisites (`infra_required`).** The site **Docker images are un-bakeable and huge**:
`shopping_final`, `shopping_admin_final`, `postmill-populated`, `gitlab-populated` (each multi-GB),
the Wikipedia `.zim` (tens of GB), and especially the **map backend (~180 GB)**; upstream's own
recommended host is a **1 TB AMI**. gbench must not pull or bake these; the operator loads them,
exposes the ports, and sets the site URL env vars. `GEMINI_API_KEY` is required only when the selected
task set includes fuzzy/ua tasks.

**In-sandbox feasibility.** A *partial deterministic* smoke is possible: `docker load` the two small
sites (shopping + reddit), run a handful of `url_match`/`program_html` tasks (which need **no** judge)
with Playwright in the orchestrator against the served model; this exercises the full spine (site
container + accessibility-tree agent loop + programmatic evaluator + model wiring). The **full 812**
(GitLab + Wikipedia + the ~180 GB map + the fuzzy-judge subset) is not reproducible without the ~1 TB
host, and WebArena is **stateful**: sites must be reset between task batches or scores silently
corrupt.

**Open decisions.** (a) Rename the suite to `webarena` vs keep `lmarena_web_agent` for continuity.
(b) Full-812 (requires the 1 TB host + stateful reset) vs an honest deterministic site-subset
denominator. (c) Judge: reuse the Gemini cascade for the fuzzy/ua subset (leaderboard_comparable
False) vs a deterministic-only task subset (no judge, no GEMINI). (d) Agent scaffold: canonical
`run.py` CoT accessibility-tree agent (leaderboard-shaped) vs BrowserGym/AgentLab.

**Risks.** Un-bakeable/huge stateful sites; statefulness/reset correctness (a "ran but wrong number"
trap); legacy `openai<1.0` agent/judge self-grading contamination; hardcoded URLs + `.auth` cookies
fragile to host/port drift; map-backend build time (60-90 min).

### #2: `swe_lancer` → unblock the task image (upstream Expensify drift)   ·   status: **gbench-side done; image blocked**   ·   effort: **M-L**

**What it measures.** SWELancer (`openai/SWELancer-Benchmark`): real freelance Expensify tasks; the
model produces a patch, scored by the task's own `test.py` inside a per-task Expensify-app container
(alcatraz sandbox). IC-SWE (patch) + SWE-Manager (proposal-choice) variants.

**gbench side is FULLY FIXED** (2026-09-12, verified, full suite green). Six harness/adapter layers
were resolved: (1) nanoeval's hard-coded `setrlimit(RLIMIT_NOFILE, 131072)` crashes unprivileged →
`_patch_nanoeval_nofile` clamps to the host hard cap; (2) predictions written as JSONL but the adapter
`json.loads`-es one dict → writer emits `{question_id: diff}`; (3) missing `.env` → provisioned from
`sample.env`; (4) `USE_WEB_PROXY` KeyError → via `.env`; (5) opaque crash → full harness stdout/stderr
captured to a file; (6) **the real bug - task-id source mismatch**: gbench sourced prompts from the
`DCAgent2/swe-lancer` HF mirror (ids like `46053_566`, `28030-manager-0`) that do **not** exist in the
`swelancer:latest` image, so `run.sh`'s `check_flows /app/tests/issues/$ISSUE_ID/` FileNotFound-crashed
every task before any report. Fixed by sourcing tasks + prompts from the image's own
`/app/tests/issues/*/issue_data.json` (`_extract_issue_manifest`, cached) → ids always match the harness.

**The remaining blocker (image build, shared with SWELancer upstream).** `docker/swe_lancer.Dockerfile`
clones `Expensify/App --single-branch` at **HEAD** - this line is *identical to SWELancer's own
`Dockerfile:106`*. Expensify/App has drifted since SWELancer (early 2025): HEAD **removed `config/webpack/`**
(so `setup_mitmproxy.yml`'s `mkcert` step, which `chdir`s there *before* the per-issue checkout, dies with
`No such file or directory`) and renamed `master`→`main` (so `setup_expensify.yml`'s
`git reset --hard origin/master` is stale). The per-issue base commits (e.g. issue 179 → `a0ac979`) still
exist and *do* carry `config/webpack`, so the suite worked in 2025 and rots against a fast-moving app repo.

**Buildable path.** Pin the Expensify clone in the Dockerfile to an era-appropriate 2025 commit (with
`config/webpack`), fetch **full history** (drop `--single-branch`) so every issue's base commit is
checkout-able, and align the reset to `origin/main`. Then rebuild the ~14 GB image and iterate through any
further 2025-vs-current toolchain drift (npm/bun/ruby versions). Once a correct image exists the suite
runs unchanged (gbench code is done); remove it from `EVALS_ALL_EXCLUDE` at that point. Removed from
`--evals all` for now (still runnable when named, hard-errors honestly against a broken image).

### #3: `ui_control_osworld` → run on a nested-virt host   ·   status: **implemented; needs KVM host**   ·   effort: **S (infra only)**

**What it measures.** Canonical OSWorld (`github.com/xlang-ai/OSWorld`, pinned `fc31a90`): 369
execution-scored desktop computer-use tasks; a CU agent drives an Ubuntu-desktop VM (screenshots +
`pyautogui`) and each task is graded by a deterministic per-task evaluator. No LLM judge.

**gbench side is DONE.** The suite delegates to the pinned OSWorld harness inside a LOCAL orchestrator
image via the docker provider (sibling QEMU-VM container); model wiring is verified against the served
`/v1` (`pong`). It is deregistered from the default eval set only because of a **host hardware**
requirement, not a code gap.

**The blocker (host hardware).** OSWorld's Ubuntu VM only boots within its 300s timeout with
**hardware virtualization (`/dev/kvm`)**. GCP accelerator instances (A2/A3/G2) cannot enable nested
virtualization, and no container can conjure `/dev/kvm` the host kernel does not expose - so the model
serving box (an H100 A3) can never run it in place.

**Buildable path (no code change).** Run the orchestrator on a **nested-virt CPU VM** (GCP N2/N2D/C3
with `--enable-nested-virtualization`) and point `--remote-endpoint` at the model still served on the
GPU host - OSWorld needs no GPU of its own. The full split-topology recipe (VM creation, image build,
qcow2 + task image provisioning, firewall) is in `docs/evals/ui_control_osworld.md`. Re-register it
(remove from the deregistered set) once it is being run on such a host.

### Backlog: candidate evals (unordered)

New candidates are added here as they are identified. Inclusion criteria: (i) a widely-cited,
actively-maintained benchmark with a **reproducible canonical harness**; (ii) fills a capability gap
in the current pillars (Reasoning, Coding, Knowledge/QA, Tool-Use & Agentic, Long-Context,
Multimodal Vision & Grounding, Safety); (iii) has a **deterministic verifier or a well-specified
judge rubric** (avoid benchmarks scorable only by human preference). Each promoted item gets a full
technical spec in this section (like WebArena above) before implementation begins.

## 4. Recently landed (process in action)

The current Group-D harness-build program produced these canonical-when-provisioned suites, which
double as reference implementations of the patterns above: `gaia2` (in-process ARE simulator, hybrid
deterministic + Gemini-cascade judge), `ui_control_osworld` (DooD QEMU-VM siblings, deterministic
verifier), `wildclawbench` (DooD InternLM harness, in-orchestrator Gemini-cascade judge proxy),
`skillsbench` (DooD BenchFlow, deterministic verifier), `mcp_bench`, `toolbench`, `swe_lancer`,
`multipl_e`, `complexfuncbench` (self-contained). See each suite's `docs/evals/<suite>.md`.

## 5. Metric fidelity refinements (tracked)

These suites already produce a canonical or conservative number for their primary metric, but the
official benchmark defines an additional dimension or a stricter variant that is not yet implemented.
Each entry names the canonical target, the current behavior, and the concrete prerequisite. None is a
skip: the suite reports its primary metric today; these refinements widen fidelity, not enable a
score. A suite listed here still carries its `canonical_sync` stamp (its `method` note points back to
this section). A change here moves the headline number, so each one requires a full dataset re-baseline
at temperature 0.0 (see the launch smoke test).

- **`gorilla_apibench` domain-match functional accuracy.** Canonical (`ShishirPatil/gorilla`
  `eval/eval-scripts/ast_eval_hf.py`, verified 2026-09-10) matches the predicted call against the
  *whole* HuggingFace API database via tree-sitter AST subtree matching to get a `database_index`,
  then counts it correct iff that reference API's `domain` equals the gold `domain`; an unmatched call
  is a hallucination. Current: a stricter exact-gold-call AST match (name plus the checkpoint
  argument) via `fc_common`. This is conservative (it can only under-credit, never inflate).
  Prerequisite: load the HF API database and build the tree-sitter AST index.
- **`acebench` normal-category variable branch.** The special error_param subcategory (identify the
  incorrect VALUE) is confirmed canonical (verified against `chenchen0103/ACEBench`
  `model_eval/checker.py`, 2026-09-10). The residual gap is the normal-category `is_variable` branch:
  when a ground-truth possible-answer's type differs from the parameter's expected type, the official
  `type_checker` sets `is_variable=True` and skips literal value matching. The current `_value_checker`
  ports only the non-variable branch. Prerequisite: port the `is_variable` skip precisely (an imperfect
  port over-accepts, so it needs the re-baseline before it ships).
- **`charxiv` descriptive track.** Current scores the reasoning track (Gemini-cascade judge). The
  descriptive track needs the CharXiv descriptive-question template bank (shipped in the CharXiv repo,
  not the HF subset). Prerequisite: the template bank plus its scorer.
- **`nestful` Win Rate.** Current reports Full Sequence Match plus F1-Func / F1-Param (canonical).
  Win Rate additionally executes the nested call chain and compares the output to the reference.
  Prerequisite: a sandboxed reference-function execution runtime.
- **`beam_128k` event-ordering.** Current scores rubric-nugget coverage. The event-ordering subtask
  is scored by Kendall tau over predicted versus gold event order. Prerequisite: the event-ordering
  split with gold orderings, plus the tau metric.
- **`cimemories` Violation-at-n and Completeness.** Current removes the privacy-injection prompt and
  reports the base metric. The per-attribute Completeness and privacy Violation-at-n metrics need
  per-attribute privacy labels that are not in the loaded subset. Prerequisite: the labeled subset.
- **`livebench` per-category scorers.** Current scores the math category only (deterministic; the LLM
  judge is removed because canonical LiveBench forbids it). Canonical spans math, reasoning,
  data-analysis, language, coding, and instruction-following, each with its own deterministic scorer.
  Prerequisite: port each category's scorer and load its subset.
- **`api_bank` Level-3 and ROUGE-L.** Current scores Level-1 plus Level-2. Level-3 is the
  plan-plus-retrieve-plus-call flow, and response quality uses ROUGE-L. Prerequisite: the L3 flow and
  the `rouge-score` dependency.
- **`t_eval` six-dimension scoring.** The import blocker is fixed. Canonical T-Eval scores six
  separable sub-skills (instruct, plan, reason, retrieve, understand, review) with per-dimension graded
  formulas. Prerequisite: port the six graded scorers.
- **`wmdp` loglikelihood MCQA.** Current scores by generation plus letter extraction (a valid variant;
  the wmdp-cyber subset is included). Canonical (lm-eval-harness) scores by loglikelihood MCQA: compare
  the logprob of each answer-choice continuation. Prerequisite: logprobs plumbed through the generation
  harness.

## 6. Model portability refinements (tracked)

The model registry is fully dynamic (no built-in model table, no model-identity or weights paths in
shipping code, no "model engine"). The remaining non-gemma gaps are multimodal token-budgeting
constants, not identity: each has a safe conservative default plus an override, so a non-gemma
vision-language model still runs, but its image-token accounting can be off. Making these fully
model-derived means reading the served model's own processor/config for its per-image soft-token count.

- **`vision_tokens_per_image` default (`gbench/core/models.py`).** Defaults to 280 (a gemma-4
  per-image soft-token count) and is only overridden from gemma-style config keys. A non-gemma VLM
  whose config lacks those keys keeps 280, which mis-sizes multimodal prompt/context accounting.
  Prerequisite: read the image-token count from the served model's processor/config (or expose it via
  the registry) instead of the gemma fallback.
- **`IMAGE_SOFT_TOKENS` estimate (`gbench/runners/eval_suites/base.py`).** Defaults to 1120 (gemma-4's
  max per-image soft tokens), env-overridable via `GBENCH_IMAGE_SOFT_TOKENS`, but with no per-model
  derivation in the eval engine, so an image-heavy prompt for a non-gemma VLM can be mis-estimated
  (mitigated today by a conservative-high default plus a reactive 400/over-context retry).
  Prerequisite: plumb the served model's per-image token count into `_estimate_prompt_tokens`.
- **`~280 tok/img` summary annotation (`gbench/cli.py`).** The multimodal serving/throughput summary
  prints a literal "~280 tok/img"; the image count is dynamic but the per-image figure is a gemma
  constant. Prerequisite: derive the printed figure from the row's model or the recorded actual
  image-token count.

## 7. Packaging / operational (parked)

Lower-priority operational polish, tracked but not scheduled.

- **Local container image size.** The per-suite images under `docker/` (SWE-bench family,
  `toolbench`, `putnam_formal` with its multi-GB Mathlib cache, etc.) are large. Parked idea:
  shrink them with multi-stage builds, slimmer bases, and shared layers so a swarm of workers pulls
  less. Not a correctness issue; deferred until it becomes a throughput bottleneck.
