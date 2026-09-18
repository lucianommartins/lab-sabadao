# Optional toolchains

Most suites need nothing beyond `pip install gbench`. The ones below execute code,
drive containers, or wrap a third-party harness, and **hard-error** when their
prerequisite is missing: the error names the exact missing piece and points here.
Nothing is scored on a partial environment.

Install only what you intend to run. `gbench --list evals` names every suite;
`gbench --dry-run --evals <name>` shows what a run would execute without generating.

## Python extras

| Install | Needed by |
| --- | --- |
| `pip install gbench[evals]` | pulls `docker`, `swebench` (pinned `>=4.1,<5`; v5 broke the TestSpec build), `datasets`, `pandas`, `rapidfuzz`, `jsonschema`; covers most of this table |
| `pip install bfcl-eval` | `bfcl_v4_agentic` (the canonical Berkeley harness; **not** the unrelated `bfcl` package) |
| `pip install multi-swe-bench` | `multi_swe_bench` |
| `pip install harbor` (or `uv tool install harbor`) | `terminal_bench` |
| `pip install rank-bm25` | `tau3` banking_knowledge RAG domain |
| `pip install audioop-lts` | `tau2` / `tau3` on Python 3.13 (stdlib `audioop` was removed) |
| `pip install pycocoevalcap` | `coco_caption` (CIDEr/SPICE; without it the suite hard-errors (infra_required) rather than substituting BLEU) |
| `pip install sacrebleu` | `i18n_translate` chrF |
| `pip install rapidfuzz` | `omnidocbench` / `mrcr` edit distance (falls back to stdlib `difflib` if absent, less precise) |
| `pip install apted lxml Distance pylatexenc rapidfuzz` | `omnidocbench` **per-modality composite**: TEDS for tables (`apted`+`lxml`+`Distance`), Formula Edit Distance (`pylatexenc`). Without them the suite reports the single global text edit distance. CDM (formula-image metric) is a separate heavy add; see [omnidocbench.md](omnidocbench.md) |
| `pip install playwright && playwright install chromium` | `lmarena_web_agent` |
| `pip install mcp jsonschema` | `mcp_bench` |
| `pip install semgrep` + `CYBERSECEVAL_ICD_RULES` (absolute path to PurpleLlama's `CodeShield/insecure_code_detector/rules`) | `cyberseceval`: 62/351 rows score without it, 317/351 with it; see [cyberseceval.md](cyberseceval.md) |
| `pip install google-genai` | every LLM-judged suite (also needs `GEMINI_API_KEY`) |
| `pip install -e ./tau2-bench --no-deps` | `tau2`, `tau3` (or set `TAU2_BENCH_SRC`) |
| **separate venv** + `pip install -e SWE-bench-Live` | `swe_bench_live`. **Not** a plain install: it is `swebench` at an older version and would break `swe_bench_multilingual`. See [swe_bench_live.md](swe_bench_live.md) |
| `git clone He-Ren/OJBench && pip install -e .` + DMOJ `judge-server@f098cd3` | `ojbench` |

## System toolchains

| Install (Debian/Ubuntu) | Needed by |
| --- | --- |
| Docker CLI + running daemon | `swe_bench_*`, `multi_swe_bench`, `swe_lancer`, `bigcodebench`, `terminal_bench`, `putnam_formal` |
| `apt install cmake build-essential` (`g++`) | `multipl_e`, `aider_polyglot`, `ojbench` |
| `apt install golang-go` (`go`) | `multipl_e`, `aider_polyglot` |
| `apt install rustc cargo` | `multipl_e`, `aider_polyglot` |
| `apt install nodejs npm` (`node` + `npm`) | `aider_polyglot` (JavaScript tasks) |
| `apt install openjdk-21-jdk` (a **JDK ≤ 21**, not a JRE: you need `javac`) | `aider_polyglot` (Java tasks; tests run via the repo's own `./gradlew`, so **no system `gradle` is needed or checked**) |
| `apt install git` | `aider_polyglot` |
| `apt install git-lfs` **or** `pip install git-lfs`, then `git lfs install` | `ruler`: the NVIDIA/RULER checkout LFS-tracks its `*.json` data (`git clone` then `git lfs pull`); see [ruler.md](ruler.md) |
| `pypy3` | `ojbench` |
| `apt install ripgrep socat bubblewrap python3-pysrt` (`rg`, `socat`, `bwrap`, `srt`) | `tau3` (and any tau2 `banking_knowledge` run): tau2's agentic-shell sandbox hard-errors if any of `srt`/`rg`/`bwrap`/`socat` is missing. See [tau3.md](tau3.md) |
| `bubblewrap` (`bwrap`) | recommended for every code-executing suite; see below |

The `aider_polyglot` / `multipl_e` language toolchains in one line:
```bash
sudo apt-get install -y golang-go openjdk-21-jdk nodejs npm rustc cargo cmake build-essential git
```
Java must be a JDK **≤ 21** (`javac`, not a JRE-only `java`): aider's exercises run through a
pinned gradle-8.7 wrapper that rejects Java 22+. If a suitable JDK is installed but not first
on `PATH`, point `JAVA_HOME`/`PATH` at it (e.g. `/usr/lib/jvm/java-21-openjdk-amd64`).

## Sandboxing

Suites that execute model-written code (`codeforces`, `lcb`, `multipl_e`, `scicode`) run it
through `bwrap` (`apt install bubblewrap`). Isolation is mandatory by default: there is no
"degrade and run unsandboxed" policy. `GBENCH_SANDBOX` selects the policy:

- `required` (default): require `bwrap`; a suite **skips** if it is unavailable or blocked
- `bwrap`: require it and **fail loudly** (raise) if unavailable, a strict CI gate
- `none`: run on the host without isolation (explicit opt-out)

Model code never runs unsandboxed unless you set `GBENCH_SANDBOX=none`. On a shared or
untrusted host keep the default, or use `bwrap` to turn a missing sandbox into a hard
failure rather than a skip.

### Enabling bubblewrap

1. Install it:
   ```bash
   sudo apt-get install bubblewrap        # Debian/Ubuntu
   ```
2. **Ubuntu 23.10+/24.04 only.** These releases ship
   `kernel.apparmor_restrict_unprivileged_userns=1`, which blocks bubblewrap from creating
   the user namespace it needs; the suite skips with *"bubblewrap IS installed but is
   BLOCKED"*. Allow unprivileged user namespaces:
   ```bash
   sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
   # persist across reboot:
   echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/60-apparmor-userns.conf
   ```
   (A locked-down host can instead ship a per-binary AppArmor profile granting `bwrap` the
   `userns` permission: more surgical, but more setup.)
3. Verify the exact jail gbench probes. This must print `1`:
   ```bash
   bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --die-with-parent --new-session \
         --unshare-net python3 -c "print(1)"
   ```

If you cannot (or prefer not to) enable unprivileged user namespaces, run the code-executing
suites unsandboxed **on a host you trust** with `GBENCH_SANDBOX=none`.

## Harnesses and gates

| Variable | Purpose |
| --- | --- |
| `SWE_BENCH_PRO_HARNESS_DIR` | clone of `scaleapi/SWE-bench_Pro-os` (`swe_bench_pro_eval.py`, `run_scripts/`). The raw-sample CSV is generated for you; see [swe_bench_pro.md](swe_bench_pro.md) |
| `SWE_BENCH_PRO_RUN=1` | opt-in; a real run pulls tens-hundreds of GB of images |
| `SCICODE_TEST_DATA` | path to SciCode's ~1 GB `test_data.h5` reference outputs; see [scicode.md](scicode.md). Without it the suite hard-errors (infra_required) rather than reporting a structural 0% |
| `SWELANCER_HARNESS_DIR` | clone of `openai/SWELancer-Benchmark` (+ `SWELANCER_RUN=1`). Its `main` branch is EMPTY and the populated branches ship a nanoeval agent, not a predictions scorer - the suite hard-errors (infra_required) without an adapter. See [swe_lancer.md](swe_lancer.md) |
| `TAU2_ENV_RUN=1` | opt-in for the tau2/tau3 environment suites |
| `BFCL_PROJECT_ROOT` | where `bfcl-eval` keeps generations and scores |
| `GEMINI_API_KEY` | required by every judged suite; they hard-error (infra_required) without it rather than falling back to substring matching |
| `SERPAPI_API_KEY` | canonical web-search backend for `bfcl_v4_agentic`; otherwise Gemini grounding is used and the result says so |

## Checking what is available

There is no separate probe command: run the suite. Anything whose prerequisite is missing
returns `status: error` with the exact missing piece and a docs pointer, costs no
generation, and is reported as an **error**, never as 0%. Read the run summary and
install from the tables above.
