# swe_bench_live setup

> **Docker setup:** this suite runs one container/network per task and can exhaust the
> Docker address pool at high `--sandboxes`. See [docker-sandboxes.md](docker-sandboxes.md)
> for the recommended `default-address-pools` config and pre-run prune, strongly advised
> before running at concurrency.

Canonical SWE-bench-Live (`SWE-bench-Live/SWE-bench-Live`): the model emits a
unified-diff patch per GitHub issue and correctness is the execution-based
**resolved rate** (apply patch, run `FAIL_TO_PASS`/`PASS_TO_PASS`).

## Why this suite uses its own image

SWE-bench-Live ships the harness as a **fork of the `swebench` package**: the same package name at a
version incompatible with the upstream `swebench` the main gbench env uses for
`swe_bench_multilingual` and `copilot_bench_swe`. Only one `swebench` can be installed per
environment. Measured by building a `TestSpec` for one row of each dataset:

| `swebench` version | `copilot_bench_swe` | `swe_bench_live` | `swe_bench_multilingual` |
| --- | --- | --- | --- |
| upstream (PyPI) | OK | FAIL | OK |
| SWE-bench-Live fork | OK | OK | FAIL |

So gbench **isolates the fork in a local Docker image** rather than a separate host virtualenv. The
model rollout (patch generation) runs in the main gbench process against the served endpoint and
needs only `datasets`; only the scoring step (`swebench.harness.run_evaluation`) runs inside the fork
image, docker-out-of-docker (it spawns the per-instance `starryzhang/*` task containers on the host
daemon). Nothing fork-versioned touches the main env, so all four SWE suites run from one gbench
install.

## Requirements

- **Docker** (a reachable daemon; the per-instance task images are pulled from DockerHub namespace
  `starryzhang`).
- `datasets` (in the base install).
- The **local fork image**, built once:

  ```bash
  docker build -t gbench-swe-bench-live -f gbench/docker/swe_bench_live.Dockerfile gbench/docker
  ```

  Override the image name with `GBENCH_SWE_BENCH_LIVE_IMAGE` if you retag it. gbench never pulls this
  image; you build it locally.

If Docker or the image is missing the suite **hard-errors** (`infra_required`) with the build command,
never a skip and never a fabricated 0% resolved rate.

## Run

```bash
gbench --evals-only --evals swe_bench_live \
    --remote-endpoint http://127.0.0.1:8000/v1 \
    --tokenizer google/gemma-4-26B-A4B-it \
    --max-output-tokens 8192 --eval-limit 20 --sandboxes 8
```

No separate virtualenv and no separate `--results-dir` are needed: `swe_bench_live` now runs from the
same gbench install as `swe_bench_multilingual`, because the fork lives only inside the image.

## Notes

- `leaderboard_comparable` is `True` only for a full split at greedy decoding with a fully evaluated
  harness; a `--eval-limit` subset, a `--thinking` (non-greedy) run, or a harness that evaluated zero
  instances is reported honestly as not comparable (or as an error, never a 0%).
- `--shard I/N` is honored (the SWE family routes through the common path, so the instance set is
  partitioned before `--eval-limit`).
