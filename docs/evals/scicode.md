# scicode

Canonical SciCode (`SciCode1/SciCode`): research-level scientific computing problems
(physics, chemistry, math) requiring multi-step code. **Scoring: execution-based**.
The generated Python is run against the problem's unit tests in a subprocess; it passes
iff the tests pass.

## Protocol (sequential, per sub-step)

gbench follows SciCode's canonical protocol rather than a single whole-problem shot. A
problem has N sub-steps; for step *k* the prompt contains every prior step's description
and **the model's own generated code for that step** (separated by `------`), then step
*k*'s description, function header and return line, plus the shared dependency block. The
model writes only step *k*'s function, which is fed forward as context for step *k+1*,
so a problem costs N model calls. Generation runs in parallel **across** problems and
strictly sequentially **within** a problem.

Two prompt modes match upstream: the default *without background* (the model must emit its
own `# Background:` comment) and, with `GBENCH_SCICODE_WITH_BACKGROUND=1`, *with background*
(the human-written step background is injected, canonical `--with-background`).

Three sub-steps (13.6, 62.1, 76.3) are **supplied** by the benchmark (they depend on prior
human scaffolding): gbench ships them as package data, injects them as context for the later
steps of those problems, and excludes them from scoring, exactly as upstream does.

## Metrics

SciCode reports two numbers and so does gbench: **sub-step accuracy** (correct sub-steps /
total scored sub-steps) is the headline `accuracy`; **problem accuracy** (a problem is
correct iff *every* scored sub-step passes) plus the raw counts are attached as extra keys
(`substep_accuracy`, `problem_accuracy`, `correct_steps`/`total_steps`,
`correct_problems`/`total_problems`). They land in the result JSON but not in
`eval_summary.csv`, which surfaces one headline per suite.

## Test data (fetched automatically)

SciCode scores generated code against **per-problem reference outputs**, bound as `target`
from the benchmark's `test_data.h5` (`process_hdf5_to_tuple`). Without that file every test
raises `NameError`, so even a perfect solution scores 0 - a structural zero, not a model
result. gbench therefore **hard-errors** (`infra_required`, never skips) until `SCICODE_TEST_DATA` points at the file.

The file is ~1 GB and is *not* in the HF dataset (`SciCode1/SciCode` ships only
`problems_dev.jsonl` / `problems_test.jsonl`).

**You do not normally need to do anything**: gbench fetches it at eval time like any other
dataset, from the mirror below, and huggingface_hub caches it. The suite records the file's
`sha256` and where it came from under `test_data` on the result, and logs a warning that
the mirror is unverified against the canonical copy. Set `SCICODE_TEST_DATA` to use your
own file, or `SCICODE_TEST_DATA_REPO` to point at a different mirror. If neither the local
path nor the fetch works, the suite **hard-errors** (`infra_required`, never skips) rather than reporting a structural 0%.

**Canonical source** - the SciCode repo README links a Google Drive folder, to be saved as
`./eval/data/test_data.h5`:
<https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR>
Google Drive folders are not scriptable without `gdown`/auth, so this is a manual download.

**Scriptable mirror** - a third-party copy on the Hub (1000.7 MiB, ~8.9k downloads).
Verify it against the canonical file before quoting a headline number:

```bash
hf download Srimadh/Scicode-test-data-h5 test_data.h5 \
    --repo-type dataset --local-dir ./scicode-data
```

Then point the suite at it:

```bash
export SCICODE_TEST_DATA=./scicode-data/test_data.h5
```

Doing this manually is only worth it if you want to pin a verified copy - the automatic
fetch pulls the same file.

### Verifying the mirror before publishing a number

The reference outputs in this file decide every pass/fail, so a headline SciCode score is
only as trustworthy as the h5 it was graded against. For a **published** number, verify the
mirror is byte-identical to the canonical Google Drive copy (for a smoke/validation run you
can skip this):

1. Every run records the h5's digest and source on the result JSON, under
   `test_data.sha256` and `test_data.source`. That is the exact file your score used.
2. Download the canonical file from the Google Drive link above, then hash both and compare:
   ```bash
   sha256sum ./scicode-data/test_data.h5        # the mirror (== result["test_data"]["sha256"])
   sha256sum /path/to/canonical/test_data.h5     # canonical, from Google Drive
   ```
3. If the two digests match, the mirror is identical to canonical - pin your verified copy so
   future runs use it and the provenance is unambiguous:
   ```bash
   export SCICODE_TEST_DATA=/path/to/canonical/test_data.h5
   ```
   If they differ, do **not** quote the number: regrade against the canonical file.

## Requirements
- **`h5py` + `scipy`** (both in the `evals` extra: `pip install -e .[evals]`). Scoring binds
  each test's `target` from `test_data.h5` via the canonical `process_hdf5_to_tuple`, which
  gbench **vendors** verbatim (Apache-2.0, attribution retained in `scicode.py`). The upstream
  `scicode` *package* pulls an invasively broad, conflicting dependency tree (numpy 2.x,
  litellm, inspect-ai, …) so we copy the one loader we need instead of installing it. Without
  `h5py`/`scipy` the suite **hard-errors** (`infra_required`, never skips; it will not report a structural 0%). Each sub-step's
  tests are keyed by `step_number` (e.g. `77.1`); a problem is correct iff **every** sub-step's
  tests pass.
- **`bubblewrap` is required.** This suite executes model-written code, so by default
  (`GBENCH_SANDBOX=required`) it runs each step inside a bubblewrap jail (read-only root,
  private `/tmp`, no network). Install it once (`sudo apt-get install bubblewrap`) and
  ensure unprivileged user namespaces are permitted (Ubuntu 23.10+/24.04 needs one extra
  `sysctl`, see [toolchains.md](toolchains.md#enabling-bubblewrap)). If bubblewrap is
  unavailable the suite **skips** rather than executing code unsandboxed. To run without isolation on a trusted
  host, set `GBENCH_SANDBOX=none`; use `GBENCH_SANDBOX=bwrap` to make a missing sandbox a
  hard failure instead of a skip. `--sandboxes` bounds how many steps execute at once.
- **Validation note:** SciCode withholds gold solutions (`ground_truth_code`/
  `general_solution` are empty in the HF dataset), so there is no gold self-check.
  Validate the wiring with a capable model (scores should be non-zero and track
  difficulty; an all-zero run means the h5/package/keying is wrong, not the model).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals scicode --eval-thinking --sandboxes 8 --eval-limit 10
```
