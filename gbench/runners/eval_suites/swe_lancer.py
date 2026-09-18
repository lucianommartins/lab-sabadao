# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: swe_lancer
# Description: SWE-Lancer (OpenAI) - real-world freelance SWE tasks, resolved via the official Docker harness

"""gbench native built-in runner for swe_lancer (Coding & Software Engineering).

Canonical SWE-Lancer (openai/SWELancer-Benchmark) scored by execution: the model's
patch is applied inside the task's Docker image and the hidden end-to-end
(Playwright/pytest) test suite is run - an IC-SWE task resolves iff those tests pass
(SWE-Manager tasks are scored by the correct proposal selection). Only the official
harness can score this; a substring/filename check cannot verify program behaviour,
so the previous heuristic scorer was removed. SANDBOX_EVAL. A real run pulls very
large per-task images, so it is gated behind an explicit opt-in
(GBENCH_SWE_LANCER_RUN=1; the bare SWELANCER_RUN still works as a deprecated alias)
and skips cleanly otherwise.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SWE_LANCER_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, strip_thinking_tags, suite_env
from .sampling import stratified_sample
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_lancer.md"
_DATASET = "DCAgent2/swe-lancer"
_IMAGE = "swelancer"   # the per-task grading image (built LOCALLY from docker/swe_lancer.Dockerfile)
#: gbench SHIPS the predictions-scorer adapter (upstream has none) and installs it into the harness
#: dir at run time; it runs in the harness's uv env (which has the vendored nanoeval/alcatraz).
_DEFAULT_EVAL_CMD = ("uv run python run_swelancer_eval.py "
                     "--predictions {predictions} --output_dir {output_dir} --num_workers {num_workers}")
_ADAPTER_FILES = ("predictions_solver.py", "run_swelancer_eval.py")


def _harness_dir() -> Optional[str]:
    return prereqs_path("SWELancer-Benchmark",
                        suite_env("GBENCH_SWE_LANCER_HARNESS_DIR", "SWELANCER_HARNESS_DIR"))


def _eval_cmd_template() -> str:
    return suite_env("GBENCH_SWE_LANCER_EVAL_CMD", "SWELANCER_EVAL_CMD") or _DEFAULT_EVAL_CMD


def _install_adapter(harness_dir: str) -> None:
    """Copy gbench's predictions-scorer adapter into the harness dir (upstream ships none)."""
    src_dir = os.path.join(os.path.dirname(__file__), "_swelancer_adapter")
    for fn in _ADAPTER_FILES:
        shutil.copyfile(os.path.join(src_dir, fn), os.path.join(harness_dir, fn))


def _ensure_dotenv(harness_dir: str) -> None:
    """SWELancer's get_tasks() reads dotenv_values('.env') for USE_WEB_PROXY / EXPENSIFY_URL /
    NEW_EXPENSIFY_URL / ALCATRAZ_TIMEOUT (README step 3: copy sample.env -> .env). Without it the
    harness KeyErrors on 'USE_WEB_PROXY' before loading any task. Provision it from the shipped
    sample.env (USE_WEB_PROXY=false = offline) if the operator has not created one."""
    dotenv = os.path.join(harness_dir, ".env")
    sample = os.path.join(harness_dir, "sample.env")
    if not os.path.isfile(dotenv) and os.path.isfile(sample):
        shutil.copyfile(sample, dotenv)
        logger.info("swe_lancer: created .env from sample.env (USE_WEB_PROXY=false, offline).")


#: The exact upstream line nanoeval uses to raise the fd limit, and gbench's clamped
#: replacement. nanoeval hard-sets NOFILE to 131072 on entry; on a host whose hard cap is
#: lower (e.g. 100000) an UNPRIVILEGED process cannot raise the hard limit, so the whole
#: harness dies with `ValueError: not allowed to raise maximum limit` before scoring a single
#: instance. 100k fds is far more than a gbench-sized run needs, so clamp the request to the
#: current hard cap (never lowering below what nanoeval wanted, never trying to raise it).
_NANOEVAL_NOFILE_ORIG = "    resource.setrlimit(resource.RLIMIT_NOFILE, (131_072, 131_072))"
_NANOEVAL_NOFILE_MARK = "# gbench: clamp NOFILE to the current hard cap (unprivileged-safe)"
_NANOEVAL_NOFILE_PATCH = (
    "    " + _NANOEVAL_NOFILE_MARK + "\n"
    "    _gb_soft, _gb_hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
    "    _gb_target = 131_072\n"
    "    if _gb_hard == resource.RLIM_INFINITY or _gb_hard >= _gb_target:\n"
    "        resource.setrlimit(resource.RLIMIT_NOFILE, (_gb_target, _gb_hard))\n"
    "    else:\n"
    "        resource.setrlimit(resource.RLIMIT_NOFILE, (_gb_hard, _gb_hard))"
)


def _patch_nanoeval_nofile(harness_dir: str) -> None:
    """Idempotently clamp nanoeval's hard-coded RLIMIT_NOFILE raise so an unprivileged host
    cannot crash the harness on entry. `uv run` rebuilds nanoeval from this local source, so a
    source edit takes effect on the next run. Warn (do not fail) if upstream changed the line."""
    setup = os.path.join(harness_dir, "project", "nanoeval", "nanoeval", "setup.py")
    if not os.path.isfile(setup):
        return
    try:
        with open(setup, encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        logger.warning("swe_lancer: could not read nanoeval setup.py to patch NOFILE (%s)", e)
        return
    if _NANOEVAL_NOFILE_MARK in src:
        return   # already patched
    if _NANOEVAL_NOFILE_ORIG not in src:
        logger.warning("swe_lancer: nanoeval NOFILE line not found as expected; upstream may have "
                       "changed it. Skipping the unprivileged-fd-limit patch (harness may fail to "
                       "start if the host hard cap is < 131072). See %s", DOCS_URL)
        return
    src = src.replace(_NANOEVAL_NOFILE_ORIG, _NANOEVAL_NOFILE_PATCH, 1)
    try:
        with open(setup, "w", encoding="utf-8") as f:
            f.write(src)
        logger.info("swe_lancer: patched nanoeval to clamp RLIMIT_NOFILE to the host hard cap.")
    except Exception as e:
        logger.warning("swe_lancer: could not write nanoeval NOFILE patch (%s)", e)


def check_swe_lancer_prerequisites() -> Tuple[bool, str]:
    """Docker + docker SDK + the SWELancer harness checkout + explicit opt-in (expensive)."""
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    try:
        import docker  # noqa: F401
    except ImportError:
        return False, "Python 'docker' SDK is not installed (pip install gbench[evals])."
    # The per-task grading image, built LOCALLY (gbench ships the Dockerfile; never pulled).
    if subprocess.run(["docker", "image", "inspect", _IMAGE], capture_output=True).returncode != 0:
        return False, (f"the {_IMAGE!r} image is not built. Build it LOCALLY (heavy - Expensify "
                       "monorepo + Playwright):\n  docker build -t swelancer -f "
                       "docker/swe_lancer.Dockerfile $GBENCH_SWE_LANCER_HARNESS_DIR\nSee " + DOCS_URL)
    hd = _harness_dir()
    if not hd or not os.path.isdir(hd):
        return False, ("SWE-Lancer harness not found: set GBENCH_SWE_LANCER_HARNESS_DIR (legacy alias "
                       "SWELANCER_HARNESS_DIR still works) to a checkout of "
                       "openai/SWELancer-Benchmark (the harness lives on a non-main branch; it "
                       "provides swelancer.py + the vendored project/nanoeval + project/alcatraz). "
                       "gbench SHIPS the predictions-scorer adapter (run_swelancer_eval.py + "
                       "predictions_solver.py) and installs it into that dir automatically.")
    if suite_env("GBENCH_SWE_LANCER_RUN", "SWELANCER_RUN") != "1":
        return False, ("SWE-Lancer is gated as a COST guard: a real run spawns very large per-task "
                       "Docker containers (Expensify E2E). Set GBENCH_SWE_LANCER_RUN=1 (legacy alias "
                       "SWELANCER_RUN=1 still works) to enable.")
    # The default scorer command shells out to `uv run ...` on the host to launch the harness.
    # Fail fast with an install hint instead of running the whole rollout and then dying on
    # "uv: not found" (a discarded run that reads like a 0%).
    if "uv " in _eval_cmd_template() and not shutil.which("uv"):
        return False, ("the SWE-Lancer scorer command uses `uv`, which is not on PATH. Install it "
                       "(pip install gbench[evals] declares it, or `pip install uv`, or the standalone "
                       "installer to ~/.local/bin) and ensure its bin dir is on PATH, or override "
                       "GBENCH_SWE_LANCER_EVAL_CMD with a command that does not use uv.")
    return True, ""


def _extract_patch(text: str) -> str:
    """Delegate to the shared extractor.

    This file used to carry its own copy, which only inspected the FIRST fenced block and
    otherwise took everything from `diff --git` to end-of-response - sweeping up the
    closing fence and the prose after it, so the harness rejected the patch as malformed
    ("patch: **** malformed patch at line 20: ```"). Three suites had the same copy.
    """
    from .swebench_common import extract_patch
    return extract_patch(strip_thinking_tags(text or ""))

_MANIFEST_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "gbench", "swe_lancer",
                               "issues_manifest.json")
_MANIFEST_BEGIN = "GBENCH_SWELANCER_MANIFEST_BEGIN"
_MANIFEST_END = "GBENCH_SWELANCER_MANIFEST_END"
# Read the task set + prompts from the HARNESS IMAGE ITSELF (/app/tests/issues/<id>/issue_data.json).
# This is the ONLY source whose ids match what the harness resolves: run.sh's setup does
# `check_flows /app/tests/issues/$ISSUE_ID/`, so an id absent from the image FileNotFound-crashes the
# task container before any report (measured 2026-09-12 - the old DCAgent2/swe-lancer mirror used ids
# like `46053_566` / `28030-manager-0` that do NOT exist in this image, so every task died and the
# suite reported "harness report not produced"). Sourcing from the image guarantees id parity.
_EXTRACT_SCRIPT = """
import os, json
base = "/app/tests/issues"
out = []
for i in sorted(os.listdir(base)):
    p = os.path.join(base, i, "issue_data.json")
    if not os.path.isfile(p):
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    out.append({"id": i, "title": d.get("title") or "",
                "issue_repo_steps": d.get("issue_repo_steps") or "",
                "price": d.get("price"), "issue_id": d.get("_issue_id")})
print("GBENCH_SWELANCER_MANIFEST_BEGIN")
print(json.dumps(out))
print("GBENCH_SWELANCER_MANIFEST_END")
"""


def _extract_issue_manifest() -> List[Dict[str, Any]]:
    """Canonical task list, read once from the harness image (cached). Raises on failure."""
    if os.path.isfile(_MANIFEST_CACHE):
        try:
            data = json.load(open(_MANIFEST_CACHE, encoding="utf-8"))
            if data:
                return data
        except Exception:
            pass
    # Image entrypoint is `bash -l -c`; pass "python3 -" as the command and feed the script on stdin.
    proc = subprocess.run(["docker", "run", "-i", "--rm", _IMAGE, "python3 -"],
                          input=_EXTRACT_SCRIPT, capture_output=True, text=True, timeout=300)
    out = proc.stdout or ""
    if _MANIFEST_BEGIN not in out or _MANIFEST_END not in out:
        raise RuntimeError(
            f"swe_lancer: could not extract the issue manifest from image {_IMAGE!r} "
            f"(rc={proc.returncode}; stderr tail: {(proc.stderr or '')[-300:]})")
    blob = out.split(_MANIFEST_BEGIN, 1)[1].split(_MANIFEST_END, 1)[0].strip()
    manifest = json.loads(blob)
    try:
        os.makedirs(os.path.dirname(_MANIFEST_CACHE), exist_ok=True)
        json.dump(manifest, open(_MANIFEST_CACHE, "w", encoding="utf-8"))
    except Exception:
        pass
    return manifest


def _load_swe_lancer_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Build SWE-Lancer prompts from the harness image's own issue set so prediction ids match the
    ids the harness resolves. The harness owns the tests + repo state; gbench only builds the prompt
    and keys each prediction by the image's issue id."""
    try:
        manifest = _extract_issue_manifest()
    except Exception as e:
        logger.error(f"Failed to load issue manifest for swe_lancer: {e}")
        raise RuntimeError(f"Could not load issue manifest for swe_lancer: {e}") from e
    if not manifest:
        raise RuntimeError("swe_lancer: image issue manifest is empty")

    # Stratified, not a contiguous head (audit RC-1).
    manifest = stratified_sample(manifest, limit, None, seed="swe_lancer")
    samples: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = []
    for it in manifest:
        task_id = str(it.get("id") or "").strip()
        title = (it.get("title") or "").strip()
        steps = (it.get("issue_repo_steps") or "").strip()
        if not task_id or not (title or steps):
            raise RuntimeError("swe_lancer: empty task_id/issue text; refusing to fabricate sample data")
        prompt = (
            f"[SWE-Lancer freelance task {task_id}]\n\n"
            f"{title}\n\n{steps}\n\n"
            "Resolve this issue in the repository. Output a single unified git diff "
            "(`diff --git a/... b/...`) inside a ```diff code block."
        )
        samples.append(([{"role": "user", "content": prompt}], task_id,
                        {"category": "swe_lancer", "task_id": task_id}))

    if not samples:
        raise RuntimeError("swe_lancer: no samples built from the image manifest")
    logger.info("Loaded %d swe_lancer samples from image %s (issue ids matched to harness).",
                len(samples), _IMAGE)
    return samples


def _parse_results(out_dir: str) -> Dict[str, bool]:
    """Tolerant parse of the harness report: {task_id: resolved} or {resolved_ids: [...]}."""
    for root, _, fs in os.walk(out_dir):
        for fn in fs:
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(root, fn), encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                if isinstance(data, dict) and "resolved_ids" in data:
                    return {str(t): True for t in data.get("resolved_ids", [])}
                if isinstance(data, dict) and all(isinstance(v, bool) for v in data.values()) and data:
                    return {str(k): bool(v) for k, v in data.items()}
    return {}


def _make_scorer(num_workers: int, metrics: Dict[str, Any]):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        hd = _harness_dir()
        _install_adapter(hd)   # gbench ships the predictions-scorer; upstream has none
        _patch_nanoeval_nofile(hd)   # unprivileged-safe fd limit (else the harness dies on entry)
        _ensure_dotenv(hd)     # SWELancer get_tasks() needs .env (USE_WEB_PROXY etc.)
        workdir = tempfile.mkdtemp(prefix="gbench_swelancer_")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)
        # The adapter (run_swelancer_eval.py / predictions_solver.py) reads --predictions with a
        # single json.loads() and expects a DICT {question_id: unified_diff}. Emitting JSONL
        # {task_id, patch} per line made it die with "JSONDecodeError: Extra data: line 2". Write the
        # dict the adapter's documented contract requires.
        preds_path = os.path.join(workdir, "predictions.json")
        preds = {}
        for tr in sample_traces:
            tid = (tr.get("extra_payload") or {}).get("task_id")
            if tid:
                preds[str(tid)] = _extract_patch(tr.get("response_text") or "")
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)

        cmd = _eval_cmd_template().format(
            harness=hd, predictions=preds_path, output_dir=out_dir, num_workers=max(1, num_workers))

        # The harness resolves deps with `uv`, which git-fetches transitive deps at runtime
        # (swelancer -> nanoeval -> chz @ https://github.com/openai/chz). A host ~/.gitconfig
        # `insteadOf` rule that rewrites https://github.com -> ssh://git@github.com turns that PUBLIC
        # HTTPS fetch into an SSH fetch needing a passphrase-protected key: it prompts on /dev/tty
        # (can hang a nohup'd run) and fails `Permission denied (publickey)`. Neutralize the host git
        # rewrites for this subprocess (public deps need no auth) and forbid interactive prompts so a
        # credential issue fails fast instead of blocking.
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = os.environ.get("GBENCH_SWE_LANCER_GIT_CONFIG_GLOBAL", os.devnull)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault("GIT_SSH_COMMAND",
                       "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10")

        def _run():
            return subprocess.run(cmd, shell=True, cwd=hd, capture_output=True, text=True, env=env)
        proc = await asyncio.to_thread(_run)

        results = _parse_results(out_dir)
        if results:
            # Safety net: prompts are now built from the image's own issue ids (_extract_issue_manifest),
            # so they should always match the harness. If a future change reintroduces a mismatched
            # source, every lookup below would miss and report a clean 0% that looks like a model
            # failure - fail loudly on zero overlap instead.
            sample_ids = {str((tr.get("extra_payload") or {}).get("task_id"))
                          for tr in sample_traces}
            overlap = sample_ids & {str(k) for k in results}
            if not overlap:
                logger.error(
                    "swe_lancer: the harness report shares NO task ids with the loaded "
                    "prompts (%d predictions vs %d scored ids). The prompt ids do not match "
                    "the harness image %r; the run cannot be scored.",
                    len(sample_ids), len(results), _IMAGE)
                metrics["swe_lancer_report"] = {
                    "error": "task_id mismatch between the prompt mirror and the harness",
                    "prompt_ids": len(sample_ids), "report_ids": len(results)}
                for tr in sample_traces:
                    tr["is_correct"] = False
                    tr["status"] = "FAILED"
                return
            metrics["swe_lancer_report"] = {"total_instances": len(sample_traces),
                                            "resolved_instances": sum(1 for v in results.values() if v),
                                            "matched_instances": len(overlap)}
        else:
            # Dump the FULL harness stderr to a file (the 800-char tail hides the real task-level
            # exception nanoeval wraps); log its path so a crash deep in the executor is diagnosable.
            full_err = (proc.stdout or "") + "\n----- STDERR -----\n" + (proc.stderr or "")
            err_path = os.path.join(workdir, "harness_full_output.txt")
            try:
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(full_err)
            except Exception:
                err_path = "(could not write)"
            logger.error("swe_lancer: no harness report parsed. Full output -> %s\nstderr tail: %s",
                         err_path, (proc.stderr or "")[-1500:])
            metrics["swe_lancer_report"] = {"error": "harness report not produced",
                                            "harness_output_file": err_path}

        for tr in sample_traces:
            tid = (tr.get("extra_payload") or {}).get("task_id")
            tr["is_correct"] = bool(results.get(str(tid), False))
            tr["status"] = "OK"
    return _score


def run_swe_lancer(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-Lancer resolved-rate via the SWELancer container harness + gbench predictions adapter."""
    # No-skip: missing Docker / image / harness / opt-in HARD-ERRORS (never a fabricated skip row).
    ok, reason = check_swe_lancer_prerequisites()
    if not ok:
        raise infra_required("swe_lancer", reason, DOCS_URL)
    limit = kwargs.get("limit")
    samples = _load_swe_lancer_samples(limit=limit)
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="swe_lancer",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(concurrency, metrics),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature"),
    )
    result.update(metrics)
    # CC6: a crashed/mismatched harness (no report, or a task_id mismatch) is a harness failure,
    # not a 0% resolved rate - surface it as status:error rather than a fabricated score.
    rep = metrics.get("swe_lancer_report") or {}
    if isinstance(rep, dict) and rep.get("error"):
        result["status"] = "error"
        result["error"] = rep["error"]
    # SWE-Lancer's published resolved rate is the FULL task set at greedy decoding; a --eval-limit
    # subset or a non-greedy (--thinking) run is not that number.
    noncanon: List[str] = []
    if limit:
        noncanon.append(f"subset run (--eval-limit {limit})")
    if result.get("temperature") not in (0.0, 0):
        noncanon.append(f"non-greedy temperature={result.get('temperature')}")
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    return result
