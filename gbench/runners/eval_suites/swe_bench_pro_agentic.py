# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Shared AGENTIC runner for SWE-bench Pro (the built-in `swe_bench_pro` eval and its
agentic plugin variants).

SWE-bench Pro is an AGENTIC task: an agent solves the issue *inside* the repo container (checked
out at base_commit), not a single-shot diff from the prompt. gbench's default single-turn path
(model emits a diff with no repo access) is the wrong mode - the patches can't apply, so it
scores a structural ~0. This runner reproduces the agentic setup with the canonical
**mini-swe-agent** (shipped in the SWE-bench Pro harness), pointed at the local vLLM endpoint,
then grades the extracted patches with the **existing** `swe_bench_pro` execution scorer.

Both evals share this one code path; they differ only in which instances they run:
  * `swe_bench_pro` (built-in) -> the full public ScaleAI/SWE-bench_Pro set (only_instance_ids=None)
  * a plugin variant          -> the subset its export covers (only_instance_ids=<subset ids>)

Reused as-is (no reinvention): the harness' `generate_sweagent_instances.generate_instances()`
(canonical instance dicts: image_name=jefzda/sweap-images:..., problem_statement, base_commit;
repo at /app), `minisweagent.run.extra.swebench.process_instance()`, and `swe_bench_pro`'s
`swe_bench_pro_eval.py` scorer. Setup/prereqs + gotchas: docs/evals/swe_bench_pro.md.

NOTE: the docker/gated-image path can't be unit-tested; the scaffolding (instance build, config,
scorer bridge, log routing) is. Not an eval entry point itself (deliberately no `run_*` name).
"""
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .sampling import stratified_sample

logger = logging.getLogger(__name__)

#: SWE-bench Pro images check the repo out here (generate_sweagent_instances: repo_name='app').
_REPO_CWD = "/app"
#: The published Pro images live under this dockerhub user (matches _score_patches' hardcoded
#: --dockerhub_username). A plain constant, not an env knob - the run is driven by gbench flags.
_DOCKERHUB_USER = "jefzda"
#: Canonical mini-swe-agent SWE-bench budgets, and the per-turn output default when the run sets
#: no --max-output-tokens. Constants, NOT env knobs (temperature/tokens come from gbench flags).
_STEP_LIMIT = 250
_COST_LIMIT = 3.0
_DEFAULT_TURN_MAX_TOKENS = 8192

# instance_id is `instance_<org>__<repo>-<40-hex-commit>[-v<hex>|-vnan]`; strip the commit tail
# to recover the <org>__<repo> category (for stratified --limit; audit RC-1). A repo name may
# contain '-' (element-web), but the commit is a distinct 40-hex block, so match on that.
_COMMIT_TAIL = re.compile(r"-[0-9a-f]{40}.*$")

# Set at MODULE IMPORT, before anything imports mini-swe-agent: several of its config fields take
# their DEFAULT from the env at class-definition (import) time, so setting them later has no
# effect. OPENAI_API_KEY: LiteLLM's openai/ provider needs a key even for a keyless local vLLM.
# MSWEA_COST_TRACKING: LiteLLM can't price a local model and mini-swe-agent RAISES on that by
# default, killing every turn. (build_agent_config also sets cost_tracking on the model config as
# the import-timing-proof belt-and-suspenders.)
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")


def _harness_dir() -> str:
    hd = os.environ.get("SWE_BENCH_PRO_HARNESS_DIR")
    if not hd or not os.path.isdir(hd):
        raise RuntimeError(
            "SWE_BENCH_PRO_HARNESS_DIR is not set to a valid SWE-bench Pro harness clone; "
            "see docs/evals/swe_bench_pro.md")
    return hd


def _add_harness_to_path() -> str:
    hd = _harness_dir()
    for p in (os.path.join(hd, "helper_code"), hd):
        if p not in sys.path:
            sys.path.insert(0, p)
    return hd


def _repo_of(instance: Dict[str, Any]) -> str:
    """Stratification category: the source `<org>__<repo>` recovered from the instance_id."""
    iid = str(instance.get("instance_id", ""))
    iid = iid[len("instance_"):] if iid.startswith("instance_") else iid
    return _COMMIT_TAIL.sub("", iid) or iid


def build_instances(only_instance_ids: Optional[Set[str]] = None,
                    limit: Optional[int] = None,
                    seed: str = "swe_bench_pro") -> List[Dict[str, Any]]:
    """Canonical mini-swe-agent instance dicts for SWE-bench Pro (optionally filtered).

    Reuses the harness' own builder so image URIs / problem statements / the /app repo path are
    exactly what the Pro agentic pipeline expects. `only_instance_ids=None` -> the full set.
    Under `limit`, sample stratified across repos (the pool is stored grouped by repo, so a
    contiguous head would be one repo - audit RC-1), deterministically seeded on the suite name.
    """
    _add_harness_to_path()
    import generate_sweagent_instances as gen  # from helper_code/
    instances = gen.generate_instances(_DOCKERHUB_USER)
    if only_instance_ids is not None:
        instances = [i for i in instances if i.get("instance_id") in only_instance_ids]
    if limit is not None and limit > 0:
        instances = stratified_sample(instances, limit, key_fn=_repo_of, seed=seed)
    return instances


def build_agent_config(model_name: str, base_url: str, step_limit: int, cost_limit: float,
                       temperature: float, max_tokens: int) -> Dict[str, Any]:
    """mini-swe-agent config: point the model at the local vLLM, work in /app, bound the run.

    `temperature` / `max_tokens` are the resolved run-level values (gbench --temperature /
    --max-output-tokens), passed in - not read from a private env knob.
    """
    import yaml
    from minisweagent.config import builtin_config_dir
    cfg = yaml.safe_load((Path(builtin_config_dir) / "extra" / "swebench.yaml").read_text())

    # Model -> the model under test via LiteLLM's OpenAI-compatible path against the local vLLM.
    model = cfg.setdefault("model", {})
    model["model_name"] = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
    mk = model.setdefault("model_kwargs", {})
    mk.setdefault("api_base", base_url)
    mk.setdefault("api_key", os.getenv("OPENAI_API_KEY", "dummy"))
    mk["temperature"] = temperature      # honor gbench --temperature (default 1.0; 0.0 loops)
    # Cap per-turn OUTPUT tokens (gbench --max-output-tokens). mini-swe-agent leaves this unbounded,
    # so a single turn on a slow local model can generate tens of thousands of tokens over many
    # minutes and blow the whole agent budget on one turn.
    mk["max_tokens"] = max_tokens
    # Import-timing-proof cost switch: a local model has no LiteLLM price -> ignore cost errors
    # rather than raise (which killed every turn).
    model["cost_tracking"] = "ignore_errors"

    # Only real AgentConfig fields here (step_limit / cost_limit); cwd is NOT one of them.
    agent = cfg.setdefault("agent", {})
    agent["step_limit"] = step_limit
    agent["cost_limit"] = cost_limit

    # Repo path: the builtin swebench.yaml assumes /testbed, but SWE-bench PRO images check the
    # repo out at /app. Set the container working dir there AND rewrite /testbed -> /app in the
    # agent's prompt templates.
    repo_cwd = _REPO_CWD
    if repo_cwd != "/testbed":
        for k, v in list(agent.items()):
            if isinstance(v, str) and "/testbed" in v:
                agent[k] = v.replace("/testbed", repo_cwd)
    env = cfg.setdefault("environment", {})
    env["cwd"] = repo_cwd                      # DockerEnvironmentConfig.cwd -> `docker exec -w`
    # Pro images set ENTRYPOINT=["/bin/bash"]; mini-swe-agent's `<image> sleep 2h` keep-alive then
    # becomes `/bin/bash sleep 2h` -> bash can't find a script "sleep" -> exits -> `--rm` deletes
    # the container -> every `docker exec` fails "No such container". Clear the entrypoint so
    # `sleep 2h` runs as the command (exec commands still get the image env/cwd).
    env["run_args"] = ["--rm", "--entrypoint", ""]
    return cfg


def _score_patches(preds_json: Path, num_workers: int, eval_name: str) -> Dict[str, Any]:
    """Run the existing SWE-bench Pro execution harness on mini-swe-agent's preds.json.

    mini-swe-agent writes {iid: {model_name_or_path, model_patch}}; the harness wants a list of
    {instance_id, patch, prefix}. Convert, then invoke swe_bench_pro_eval.py exactly as
    swe_bench_pro._make_scorer does, and read eval_results.json.
    """
    import subprocess
    import tempfile
    from gbench.runners.eval_suites.swe_bench_pro import _harness_dir as sbp_hd, raw_sample_path
    from gbench.runners.eval_suites import swe_thread_cap

    raw = json.loads(preds_json.read_text()) if preds_json.exists() else {}
    prefix = f"gbench__{eval_name}_agentic"
    preds = [{"instance_id": iid, "patch": (v or {}).get("model_patch") or "", "prefix": prefix}
             for iid, v in raw.items()]
    empty = sum(1 for p in preds if not p["patch"].strip())

    hd = sbp_hd()
    workdir = Path(tempfile.mkdtemp(prefix="gbench_swepro_agentic_"))
    out_dir = workdir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = workdir / "preds.json"
    preds_path.write_text(json.dumps(preds))

    cmd = [
        # sys.executable, NOT bare "python": the scorer imports pandas + docker, which live in
        # the gbench venv. Bare "python" resolves via PATH and, if gbench is launched from a shell
        # without that venv first on PATH, silently ModuleNotFdErrors -> no eval_results.json ->
        # every instance reads False (a false "all tests failed"). Pin it to this interpreter.
        sys.executable, os.path.join(hd, "swe_bench_pro_eval.py"),
        f"--raw_sample_path={raw_sample_path()}",
        f"--patch_path={preds_path}", f"--output_dir={out_dir}",
        f"--scripts_dir={os.path.join(hd, 'run_scripts')}",
        f"--num_workers={max(1, num_workers)}",
        "--dockerhub_username=jefzda", "--use_local_docker",
    ]
    cmd, _threads = swe_thread_cap.apply(cmd, num_workers, "swe_bench_pro")
    proc = subprocess.run(cmd, cwd=hd, capture_output=True, text=True)

    report_path = out_dir / "eval_results.json"
    results: Dict[str, bool] = {}
    if report_path.is_file():
        results = {k: bool(v) for k, v in json.loads(report_path.read_text()).items()}
    else:
        logger.error("%s agentic: eval_results.json missing. stderr tail: %s",
                     eval_name, (proc.stderr or "")[-800:])
    return {
        "results": results,
        "total_instances": len(results),
        "resolved_instances": sum(results.values()),
        "empty_patch_instances": empty,
        "submitted_with_patch": len(preds) - empty,
    }


@contextmanager
def quiet_agent_logs(eval_name: str, fallback_dir: Optional[str] = None):
    """Route the verbose sub-logs (mini-swe-agent's RichHandler, litellm's per-call INFO, the http
    clients) to a DEDICATED FILE and off gbench's stdout handler, so the main log stays clean (just
    the progress bar). litellm re-adds its own handler each call, so a handler swap alone doesn't
    hold - also raise it to WARNING (drops INFO at the source) + suppress_debug_info. Everything is
    restored on exit, so a sweep's later evals keep their normal logging. Yields the log path.

    Shared by both agentic runners (mini-swe-agent and the alternate agent-SDK harness).
    """
    agent_log = Path(os.getenv("GBENCH_RESULTS_DIR", fallback_dir or ".")) / f"{eval_name}_agent.log"
    fh = logging.FileHandler(agent_log)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    noisy = ("minisweagent", "LiteLLM", "litellm", "LiteLLM Router",
             "LiteLLM Proxy", "httpx", "httpcore", "openai")
    #: These carry the agent's own per-turn detail - keep them verbose in the file; the rest
    #: (http/litellm chatter) drop to WARNING so the file stays readable.
    verbose = ("minisweagent",)
    saved = {}
    for n in noisy:
        lg = logging.getLogger(n)
        saved[n] = (lg.handlers[:], lg.propagate, lg.level)
        lg.handlers = [fh]
        lg.propagate = False
        if n in verbose:
            lg.setLevel(logging.INFO)
        else:
            lg.setLevel(logging.WARNING)
    try:
        import litellm as _ll
        saved_suppress = getattr(_ll, "suppress_debug_info", False)
        _ll.suppress_debug_info = True
    except Exception:
        _ll, saved_suppress = None, None
    try:
        yield agent_log
    finally:
        for n, (handlers, prop, lvl) in saved.items():
            lg = logging.getLogger(n)
            lg.handlers = handlers
            lg.propagate = prop
            lg.setLevel(lvl)
        if _ll is not None and saved_suppress is not None:
            _ll.suppress_debug_info = saved_suppress
        fh.close()


class _QuietProgress:
    """Minimal progress_manager for process_instance (mini-swe-agent's rich Live UI is skipped)."""

    def on_instance_start(self, instance_id):
        logger.info("[swe-agent] start %s", instance_id)

    def update_instance_status(self, instance_id, status):
        logger.debug("[swe-agent] %s: %s", instance_id, status)

    def on_instance_end(self, instance_id, exit_status):
        logger.info("[swe-agent] end %s: %s", instance_id, exit_status)

    def on_uncaught_exception(self, instance_id, exc):
        logger.error("[swe-agent] uncaught on %s: %s", instance_id, exc)


def execute_agentic(model_name: str, base_url: str, concurrency: int = 1,
                    limit: Optional[int] = None, only_instance_ids: Optional[Set[str]] = None,
                    eval_name: str = "swe_bench_pro",
                    temperature: Optional[float] = None,
                    max_output_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Generate patches agentically (mini-swe-agent, per-instance docker) then score them.

    `only_instance_ids=None` runs the full public set (built-in swe_bench_pro); pass a set to
    restrict to a subset (a plugin variant). `eval_name` labels the result / progress bar / agent log.

    Driven by gbench's own flags - no private env knobs: `concurrency` <- --sandboxes (parallel
    containers, one per instance), `temperature` <- --temperature (default DEFAULT_TEMPERATURE=1.0),
    `max_output_tokens` <- --max-output-tokens (per-turn cap; default 8192), `limit` <- --limit.
    step_limit 250 / cost_limit 3.0 are the canonical mini-swe-agent SWE-bench constants.
    """
    import concurrent.futures
    import tempfile

    from .base import DEFAULT_TEMPERATURE
    temp = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    turn_max_tokens = int(max_output_tokens) if max_output_tokens else _DEFAULT_TURN_MAX_TOKENS

    instances = build_instances(only_instance_ids=only_instance_ids, limit=limit, seed=eval_name)
    base = {"benchmark_type": "eval", "eval_name": eval_name,
            "model_name": model_name, "mode": "agentic_mini_swe_agent"}
    if not instances:
        return {**base, "status": "error", "total_questions": 0,
                "correct_answers": 0, "accuracy": 0.0,
                "error": "no instances to run (empty instance set or build failure)"}

    # mini-swe-agent drives the per-instance docker rollout. It is an optional dep
    # (pip install mini-swe-agent); import it AFTER the no-instances guard so that
    # path never needs it, and surface a clean SKIP (not a ModuleNotFoundError
    # crash) when it is absent — matching the codebase's optional-dep convention.
    try:
        from minisweagent.run.extra.swebench import process_instance
    except ImportError:
        return {**base, "status": "skipped",
                "total_questions": len(instances), "correct_answers": 0, "accuracy": 0.0,
                "error": "mini-swe-agent not installed (pip install mini-swe-agent)"}

    cfg = build_agent_config(model_name, base_url, step_limit=_STEP_LIMIT, cost_limit=_COST_LIMIT,
                             temperature=temp, max_tokens=turn_max_tokens)
    out_dir = Path(tempfile.mkdtemp(prefix="gbench_swepro_agent_"))
    workers = max(1, concurrency)          # == --sandboxes; parallel containers, one per instance
    pm = _QuietProgress()
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - tqdm is a gbench dep, but degrade gracefully
        def tqdm(*_a, **_k):  # type: ignore
            class _N:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def update(self, *_): pass
            return _N()

    with quiet_agent_logs(eval_name, str(out_dir)) as agent_log:
        logger.info("%s agentic: %d instance(s), %d worker(s); verbose agent logs -> %s",
                    eval_name, len(instances), workers, agent_log)
        with tqdm(total=len(instances), desc=f"Eval [{eval_name.upper()}]") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(process_instance, inst, out_dir, cfg, pm): inst["instance_id"]
                           for inst in instances}
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:  # noqa: BLE001 - one instance dying must not sink the batch
                        logger.error("instance %s failed: %s", futures[fut], e)
                    pbar.update(1)

    logger.info("%s agentic: generation done; scoring %d patch(es) via the SWE-bench Pro "
                "harness (docker apply + tests) ...", eval_name, len(instances))
    rep = _score_patches(out_dir / "preds.json", num_workers=workers, eval_name=eval_name)
    total = rep["total_instances"] or len(instances)
    resolved = rep["resolved_instances"]
    acc = round(100.0 * resolved / total, 2) if total else 0.0
    return {
        **base,
        "status": "success" if total else "error",
        "total_questions": total,
        "correct_answers": resolved,
        "accuracy": acc,
        "swe_bench_pro_report": {k: rep[k] for k in (
            "total_instances", "resolved_instances", "empty_patch_instances",
            "submitted_with_patch")},
        "agent_output_dir": str(out_dir),
    }
