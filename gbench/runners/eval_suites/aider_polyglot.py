# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: aider_polyglot
# Description: Aider Polyglot (native-edit-format code editing across 6 languages, pass@2)

"""gbench native built-in runner for aider_polyglot (Coding & Software Engineering).

Canonical Aider Polyglot (Aider-AI/aider `benchmark/benchmark.py`) drives aider's own Coder over
225 Exercism exercises in 6 languages (C++, Go, Java, JavaScript, Python, Rust). The model edits
the stub in its NATIVE edit format (diff / SEARCH-REPLACE for capable models, whole-file
otherwise); aider applies the edit, runs the exercise's hidden unit tests, and on failure feeds
the errors back for a second try. The headline is **pass@2**.

gbench delegates to aider's OWN benchmark harness inside aider's OWN benchmark container
(`aider-benchmark`, built from the checkout's `benchmark/Dockerfile`), for two reasons:

* aider + litellm would churn the torch/vLLM serving env (installing aider downgrades `openai`
  and moves `numpy`/`huggingface_hub`/`pydantic`/`pillow` - it breaks the pinned graph), so it is
  NOT installed into the serving venv; it runs in the isolated image.
* the image also carries the 6 language toolchains (go/rust/openjdk-21/node/gcc/python) needed to
  execute the exercises' unit tests.

gbench generates nothing itself here: it `docker run`s the image (`--network host`, so the
container reaches the gbench-served `/v1` endpoint), points aider at that endpoint via litellm
(`OPENAI_API_BASE` + `--model openai/<id>`), lets aider run its native-edit-format pass@2 loop,
and reads back the per-exercise `.aider.results.json`. This suite hard-errors (never skips) if
Docker, the image, or the exercises are missing. Because it shells out, it does NOT use
run_eval_suite.

Sampling: aider's benchmark controls the model's sampling itself (temperature 0 for
reproducibility), so gbench's `--temperature` / `GBENCH_AIDER_POLYGLOT_TEMPERATURE` knobs are
accepted for interface parity but are NOT injected here - aider owns the sampling knob. There is
no LLM judge (grading is the exercises' unit tests, executed in-container), so the usual
pinned-0.0 judge note does not apply.
"""

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/aider_polyglot.md"
PILLAR = "Coding & Software Engineering"

_IMAGE_DEFAULT = "aider-benchmark"
#: The polyglot benchmark's 6 languages.
_LANGUAGES = ["cpp", "go", "java", "javascript", "python", "rust"]
_EXPECTED_FULL = 225  # exercises in the canonical polyglot set across all 6 languages


def _image() -> str:
    return os.environ.get("GBENCH_AIDER_IMAGE", _IMAGE_DEFAULT)


def _bench_dir() -> str:
    """Host directory that holds `polyglot-benchmark/` and receives the run's results
    (mounted to /benchmarks; = aider's AIDER_BENCHMARK_DIR)."""
    d = prereqs_path("aider-bench", os.environ.get("GBENCH_AIDER_BENCHMARK_DIR"))
    return os.path.abspath(os.path.expanduser(d)) if d else ""


def _languages() -> List[str]:
    raw = (os.environ.get("GBENCH_AIDER_LANGUAGES") or "").replace(",", " ").split()
    return raw or list(_LANGUAGES)


def _served_model_id(base_url: str, fallback: str) -> str:
    """The model id the endpoint actually serves; litellm must send that exact id to vLLM (gbench
    passes a stripped model name, but the endpoint serves e.g. `google/gemma-4-...`)."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            return json.load(r)["data"][0]["id"]
    except Exception as e:                                              # noqa: BLE001
        logger.warning("aider_polyglot: could not read served model id (%s); using %r", e, fallback)
        return fallback


def _check_prereqs(image: str, bench_dir: str) -> None:
    build = (
        "Build the aider-benchmark image (gbench's Dockerfile = aider's own + a setuptools-scm "
        "version pin) and provide the exercises:\n"
        f"  docker build -t {image} -f docker/aider_polyglot.Dockerfile \\\n"
        "      $GBENCH_PREREQS_DIR/aider   # context = the aider checkout\n"
        "  git clone https://github.com/Aider-AI/polyglot-benchmark \\\n"
        "      $GBENCH_PREREQS_DIR/aider-bench/polyglot-benchmark\n"
        "  export GBENCH_AIDER_BENCHMARK_DIR=$GBENCH_PREREQS_DIR/aider-bench"
    )
    if not shutil.which("docker"):
        raise infra_required("aider_polyglot", "docker CLI not found. " + build, DOCS_URL)
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise infra_required("aider_polyglot", "docker daemon not reachable. " + build, DOCS_URL)
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        raise infra_required("aider_polyglot", f"image {image!r} not found. " + build, DOCS_URL)
    if not bench_dir or not os.path.isdir(os.path.join(bench_dir, "polyglot-benchmark")):
        raise infra_required(
            "aider_polyglot",
            "the polyglot exercises were not found. Set GBENCH_AIDER_BENCHMARK_DIR to a directory "
            "containing `polyglot-benchmark/`. " + build, DOCS_URL)


def _build_command(image: str, bench_dir: str, run_name: str, model: str, base_url: str,
                   threads: int, langs: List[str], edit_format: str, num_tests: Optional[int],
                   extra: str) -> List[str]:
    """The `docker run` that runs aider's own benchmark.py inside the image."""
    # benchmark.py reads the aider repo's git HEAD (`repo.head.object.hexsha`) for its run label.
    # The checkout's `.git` may be unreadable by the container's older git (it was written with the
    # `refstorage=reftable` extension), so if HEAD can't be read we replace `/aider/.git` with a
    # fresh single-commit repo. `safe.directory '*'` also clears the "dubious ownership" git raises
    # on the bind-mounted exercises. (This all happens on the container's ephemeral copy, never the
    # host checkout.)
    git_ensure = (
        "git config --global --add safe.directory '*' "
        "&& git config --global user.email gbench@localhost "
        "&& git config --global user.name gbench "
        "&& cd /aider "
        "&& { git rev-parse --verify HEAD >/dev/null 2>&1 "
        "|| { rm -rf /aider/.git && git init -q -b main && git add -A "
        "&& git commit -qm gbench-snapshot; }; } && "
    )
    args = (
        git_ensure +
        f"./benchmark/benchmark.py {run_name} --model {model} "
        f"--exercises-dir polyglot-benchmark --new --tries 2 "
        f"--threads {threads} --languages {','.join(langs)} "
    )
    if edit_format:
        args += f"--edit-format {edit_format} "
    if num_tests:
        args += f"--num-tests {int(num_tests)} "
    if extra:
        args += extra + " "
    return [
        "docker", "run", "--rm", "--network", "host",
        "-v", f"{bench_dir}:/benchmarks",
        "-e", "AIDER_DOCKER=1",
        "-e", "AIDER_BENCHMARK_DIR=/benchmarks",
        "-e", f"OPENAI_API_BASE={base_url}",
        "-e", "OPENAI_API_KEY=EMPTY",
        # aider/litellm cache + config land in a writable HOME
        "-e", "HOME=/benchmarks",
        image, "bash", "-lc", args.strip(),
    ]


def _parse_results(run_dir: str) -> Dict[str, Any]:
    """Aggregate the per-exercise `.aider.results.json` under a benchmark run directory.

    Each file has `tests_outcomes` (a bool per try) and `language`. pass@1 = passed on the first
    try; pass@2 = passed within two tries (aider stops early once a try passes)."""
    files = glob.glob(os.path.join(run_dir, "**", ".aider.results.json"), recursive=True)
    per_lang: Dict[str, Dict[str, int]] = {}
    n = p1 = p2 = 0
    for f in files:
        try:
            data = json.loads(open(f, encoding="utf-8").read())
        except Exception:
            continue
        if not isinstance(data, dict) or "tests_outcomes" not in data:
            continue                        # skip exception/placeholder result files
        outcomes = data.get("tests_outcomes") or []
        lang = str(data.get("language") or _lang_from_path(f) or "unknown")
        passed1 = bool(outcomes and outcomes[0])
        passed2 = bool(True in outcomes[:2])
        n += 1
        p1 += 1 if passed1 else 0
        p2 += 1 if passed2 else 0
        d = per_lang.setdefault(lang, {"total": 0, "pass1": 0, "pass2": 0})
        d["total"] += 1
        d["pass1"] += 1 if passed1 else 0
        d["pass2"] += 1 if passed2 else 0
    return {"n": n, "pass1": p1, "pass2": p2, "per_lang": per_lang, "n_files": len(files)}


def _lang_from_path(path: str) -> Optional[str]:
    m = re.search(r"/(cpp|go|java|javascript|python|rust)/exercises/", path)
    return m.group(1) if m else None


def run_aider_polyglot(model_name: str, base_url: str, concurrency: int = 1,
                       enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run canonical Aider Polyglot (native edit format, pass@2) inside aider's benchmark image."""
    image = _image()
    bench_dir = _bench_dir()
    _check_prereqs(image, bench_dir)

    # litellm routes `openai/<id>` to OPENAI_API_BASE; <id> must be the endpoint's served id.
    model = os.environ.get("GBENCH_AIDER_MODEL") or f"openai/{_served_model_id(base_url, model_name)}"
    langs = _languages()
    edit_format = os.environ.get("GBENCH_AIDER_EDIT_FORMAT", "")   # "" => aider's native default
    limit = kwargs.get("limit")
    extra = os.environ.get("GBENCH_AIDER_EXTRA_ARGS", "").strip()
    threads = max(1, int(concurrency or 1))
    run_name = "gbench"

    # Snapshot existing run dirs so we can identify the one this run creates (--new is timestamped).
    before = set(glob.glob(os.path.join(bench_dir, f"*--{run_name}")))
    cmd = _build_command(image, bench_dir, run_name, model, base_url, threads, langs,
                         edit_format, limit, extra)
    logger.info("aider_polyglot: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    after = sorted(set(glob.glob(os.path.join(bench_dir, f"*--{run_name}"))) - before,
                   key=lambda d: os.path.getmtime(d) if os.path.isdir(d) else 0)
    if not after:
        raise RuntimeError(
            "aider_polyglot: aider's benchmark produced no results directory (inference/build "
            "failed) - a harness/infra failure, not a 0%. Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])
    run_dir = after[-1]
    agg = _parse_results(run_dir)
    if not agg["n"]:
        raise RuntimeError(
            "aider_polyglot: no exercise results were scored under "
            f"{run_dir!r} (found {agg['n_files']} result files). Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])

    n = agg["n"]
    default_langs = set(langs) == set(_LANGUAGES)

    # No-partial gate. A canonical full run (all 6 languages, no --eval-limit) must complete the
    # entire 225-exercise set; anything short is a crashed/interrupted container, not a low score,
    # so refuse to report it as a result. A --eval-limit smoke run scores a subset by design, but
    # a non-zero container exit still means the run aborted mid-flight -> hard-error either way.
    is_full = default_langs and not limit
    if is_full and n != _EXPECTED_FULL:
        raise RuntimeError(
            f"aider_polyglot: partial run - scored {n}/{_EXPECTED_FULL} exercises "
            f"(container exit={proc.returncode}). A canonical full run must complete all "
            f"{_EXPECTED_FULL}; refusing to report a partial score. Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])
    if not is_full and proc.returncode != 0:
        raise RuntimeError(
            f"aider_polyglot: benchmark container exited {proc.returncode} after scoring {n} "
            "exercise(s) - the run aborted mid-flight (harness failure, not a 0%). Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])

    result = {
        "benchmark_type": "eval",
        "eval_name": "aider_polyglot",
        "model_name": model_name,
        "status": "success",
        "total_questions": n,
        "correct_answers": agg["pass2"],
        "accuracy": round(100.0 * agg["pass2"] / n, 2),          # canonical headline = pass@2
        "pass_rate_1": round(100.0 * agg["pass1"] / n, 2),
        "pass_rate_2": round(100.0 * agg["pass2"] / n, 2),
        "tries": 2,
        "languages_evaluated": sorted(agg["per_lang"].keys()),
        "category_accuracy": {
            lang: {"total": d["total"], "correct": d["pass2"],
                   "accuracy": round(100.0 * d["pass2"] / d["total"], 2)}
            for lang, d in sorted(agg["per_lang"].items()) if d["total"]},
        "edit_format": edit_format or "native (aider default per model)",
        "metric": ("execution pass@2 via aider's own benchmark.py in the aider-benchmark "
                   "container (native edit format); pass_rate_1 also reported"),
        # Comparable to the published Aider Polyglot leaderboard when it is the FULL 225-exercise
        # set (no --eval-limit) across all 6 languages with aider's native edit format.
        "leaderboard_comparable": bool(default_langs and not limit and not edit_format),
    }
    return result
