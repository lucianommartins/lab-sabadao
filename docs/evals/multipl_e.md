# multipl_e setup

Canonical MultiPL-E (`nuprl/MultiPL-E`, HumanEval family): execution-based **pass@1** across all
**24** target languages. For each problem the program is `prompt + completion + "\n" + tests`,
compiled/run per language; a completion passes iff its process exits 0 (the container's canonical
status rule: `OK` / `SyntaxError` / `Exception` / `Timeout`). Execution is delegated to the
**official MultiPL-E evaluator image**, which ships every language toolchain.

> **Program assembly:** a chat model usually answers with the *whole* function, not a raw
> continuation. gbench detects a re-emitted signature and keeps only the prompt's preamble before
> it, so the assembled code does not contain two definitions of the same function (which used to
> compile-fail and score correct solutions wrong). The container appends the test suite.

## Requirements

- `datasets` (from `pip install 'gbench[evals]'`) + network/HF-Hub access to `nuprl/MultiPL-E` on
  first run (the `humaneval-<lang>` configs; cached thereafter).
- **Docker** + the **`gbench-multipl-e`** evaluator image, built **LOCALLY** from gbench's own
  Dockerfile (never pulled). Build it once:
  ```bash
  docker build -t gbench-multipl-e -f docker/multipl_e.Dockerfile docker
  ```
  `docker/multipl_e.Dockerfile` is vendored from nuprl/MultiPL-E's `evaluation/Dockerfile`
  at a **pinned commit** (ubuntu 22.04 + all 24 toolchains) and clones the `evaluation/src` harness
  itself, so the image is reproducible + self-contained (no separate upstream clone needed).
  Override the tag with `GBENCH_MULTIPL_E_IMAGE`. (Ada/`adb` ships in a *separate* upstream
  `Adb.dockerfile` and is excluded from the default image + language set; add it via
  `GBENCH_MULTIPL_E_LANGS` only if your image includes it.)

The suite **hard-errors** (`infra_required`, never skips, never a fabricated 0%) if `datasets`,
Docker, or the image is missing. If the container runs but scores nothing, the run is recorded as
`status:"error"` (a harness failure, not "0% solved"); any individual problem the container failed
to score is counted in `multipl_e_unscored`, not silently read as wrong.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals multipl_e \
       --sandboxes 16 --eval-limit 40
```
gbench generates completions against the served model, writes one input file per completion into a
mounted work dir, runs the evaluator once (`docker run --network none <image> --dir /out
--output-dir /out --recursive`), reads back the `*.results.json.gz`, and reports pass@1
(`category_accuracy` = per-language). `--sandboxes` bounds concurrent model generation.

## Notes / caveats
- **Sampling & comparability:** the published MultiPL-E leaderboard is **n-sample pass@1 at
  temperature 0.2** (n = 20-200) over the full language set. Set `GBENCH_MULTIPL_E_SAMPLES`
  (completions/problem, default `1`) ≥ 20 **and** temperature `0.2` (`GBENCH_MULTIPL_E_TEMPERATURE=0.2`
  or `--temperature 0.2`) for a `leaderboard_comparable` run. The default single greedy completion is
  a valid pass@1 but higher-variance, so it reports `leaderboard_comparable=false` with the reason
  recorded. When `attempt_count > 1`, the framework's avg@k is the unbiased pass@1.
- `GBENCH_MULTIPL_E_LANGS` (comma/space list) restricts the languages; a subset is not
  leaderboard-comparable.
- `GBENCH_MULTIPL_E_TEMPERATURE` overrides the temperature for this suite (else 0.0 greedy / 1.0
  with `--thinking`).
