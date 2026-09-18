# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""Shared gated full-environment harness for tau2 / tau3 (Sierra tau2-bench).

The canonical tau-bench is a multi-turn *environment* benchmark: an LLM user and the
agent converse over several turns, tools mutate a domain database, and each task is
scored by the simulator's oracle (final DB-state check) plus an nl_assertion LLM
judge, yielding a per-task reward. It cannot be measured from a single response, so
gbench's default tau2/tau3 runners skip; this module runs the REAL simulator when the
operator opts in (GBENCH_TAU2_ENV_RUN=1) and the `tau2` package is importable.

Design (mirrors the reference tau-bench setup):
  * Three distinct LLM roles:
      - agent  = the model under test, routed to the gbench endpoint via LiteLLM's
                 `openai/<model>` with `api_base`/`api_key` passed PER-CALL;
      - user   = user simulator (GBENCH_TAU2_USER_LLM, defaults to GBENCH_JUDGE_MODEL);
      - judge  = nl_assertion evaluator (GBENCH_TAU2_EVAL_LLM, same default).
    Neither is the model under test: the agent is the local endpoint. A 503 from
    these two is upstream capacity, not an agent failure - and since 2026-08-21 a gemini/
    user/judge call CASCADES through the same model chain grounding uses (see
    `_build_gemini_cascade`), so one overloaded model (the 500 throttling::OVERLOADED that
    depressed tau3's reward) falls over to the next instead of failing the task. The agent
    (openai/<local>) is never cascaded.
  * We deliberately DO NOT set OPENAI_API_BASE globally: tau2's evaluator/user also
    issue LLM calls, and redirecting all OpenAI traffic to the endpoint would corrupt
    grading. Only the agent is routed, per-call.
  * Best-effort robustness patches (empty/malformed-response retries, markdown-fence
    stripping) matching the reference, applied defensively so tau2 version drift can
    never crash the run.

Sampling/run knobs default to the reference values and are overridable via env
(GBENCH_TAU2_TEMPERATURE / GBENCH_TAU2_TOP_K / GBENCH_TAU2_TOP_P / GBENCH_TAU2_NUM_TRIALS /
GBENCH_TAU2_MAX_STEPS / GBENCH_TAU2_MAX_ERRORS / GBENCH_TAU2_SEED / GBENCH_TAU2_USER_LLM /
GBENCH_TAU2_EVAL_LLM / GBENCH_TAU2_USER_TEMPERATURE). Every knob here uses the canonical
`GBENCH_TAU2_<KNOB>` name; the bare `TAU2_<KNOB>` names still work as deprecated aliases
(canonical wins; a one-time deprecation warning is logged when a legacy name is used).

Per-task simulation traces (messages + reward breakdown) are saved BY DEFAULT into
`<run results dir>/tau_traces/` - tau is a wrapped harness, so without this its per-task
data would live only in memory and be lost at run end, unlike every other eval's saved
sample_traces. `GBENCH_TAU2_SAVE_TRACES=<path>` overrides the location;
`GBENCH_TAU2_SAVE_TRACES=` (empty) disables it. See `_resolve_tau_trace_dir`.
"""

import importlib
import logging
import os
import random
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import suite_env
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

_PATCHED = False   # robustness patches are process-global; apply once
_QUIETED = False   # cosmetic-log suppression; apply once
_TAU_PROGRESS_DESC = "tau"   # tqdm bar label; run_tau_env sets it per domain


def _quiet_tau2_noise() -> None:
    """Quiet tau2's redundant output so gbench logs stay readable. Two parts:

    1. Silence two cosmetic ERROR logs *at the source* (a loguru handler-filter does not
       survive: tau2's batch runner calls `logger.remove(); logger.add(...)` when it runs,
       wiping any filter). We no-op the emitting functions - neither value affects scoring:
         - get_response_cost: litellm can't price a custom model name -> 0.0;
         - get_commit_hash: `git rev-parse HEAD` fails off a repo -> "unknown".
    2. Mute tau2's rich console (`ConsoleDisplay.console`) - the per-task "Simulation
       Overview" panels and the live "Status: X/N complete" progress reprints, which are
       redundant because gbench builds its own summary from results.simulations, and which
       spam the log when stdout is redirected to a file (rich can't update in place). Set
       GBENCH_TAU2_VERBOSE=1 (bare TAU2_VERBOSE still works) to keep the full tau2 panels.
       Real errors still surface via loguru.

    Best-effort across tau2 versions; patches the by-value import in runner.helpers too.
    """
    global _QUIETED
    if _QUIETED:
        return
    _QUIETED = True
    try:
        import tau2.utils.llm_utils as _llm            # called module-locally (llm_utils:420)
        _llm.get_response_cost = lambda *a, **k: 0.0
    except Exception as e:
        logger.debug("tau2: could not patch get_response_cost: %s", e)
    try:
        import tau2.utils.utils as _u
        _u.get_commit_hash = lambda *a, **k: "unknown"
        import tau2.runner.helpers as _h               # `from ...utils import get_commit_hash`
        if hasattr(_h, "get_commit_hash"):
            _h.get_commit_hash = lambda *a, **k: "unknown"
    except Exception as e:
        logger.debug("tau2: could not patch get_commit_hash: %s", e)
    if suite_env("GBENCH_TAU2_VERBOSE", "TAU2_VERBOSE") != "1":
        try:
            from rich.console import Console
            import tau2.utils.display as _disp
            _disp.ConsoleDisplay.console = Console(quiet=True)   # all panels route through this
        except Exception as e:
            logger.debug("tau2: could not mute rich console: %s", e)
        # litellm logs a Gemini-3+ sampling DeprecationWarning on every user-sim/judge call
        # (twice per call, via two loggers) - raise its loggers to ERROR so they don't flood.
        try:
            os.environ.setdefault("LITELLM_LOG", "ERROR")
            for _n in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy"):
                logging.getLogger(_n).setLevel(logging.ERROR)
            import litellm as _ll
            _ll.suppress_debug_info = True
        except Exception as e:
            logger.debug("tau2: could not quiet litellm logging: %s", e)
        # The openai/httpx/httpcore HTTP clients log routine "Retrying request .../
        # HTTP Request: POST ..." at INFO for every agent call - pure noise here (gbench
        # builds its own summary). Raise them to WARNING; genuine failures still surface.
        try:
            for _n in ("openai", "openai._base_client", "httpx", "httpcore"):
                logging.getLogger(_n).setLevel(logging.WARNING)
        except Exception as e:
            logger.debug("tau2: could not quiet http client logging: %s", e)
        # Muting the console also killed tau2's useful 30s "Status: X/N complete" heartbeat
        # (StatusMonitor._monitor prints it through the same console). Redirect that heartbeat
        # to gbench's logger so the run shows live per-task progress like every other eval,
        # WITHOUT the per-task panels. Reuses tau2's own counters/reward math.
        _redirect_tau2_progress()


def _tau_progress_postfix(self, _time) -> Dict[str, Any]:
    """Build the tqdm postfix (avg reward, in-flight count, oldest elapsed) from a monitor."""
    post: Dict[str, Any] = {}
    try:
        results = self._simulation_results
        if results is not None:
            rewards = [s.reward_info.reward for s in results.simulations
                       if getattr(s, "reward_info", None) is not None]
            if rewards:
                post["reward"] = f"{sum(rewards) / len(rewards):.3f}"
        with self._lock:
            running = list(self.running_tasks.values())
        post["running"] = len(running)
        if running:
            now = _time.time()
            post["oldest"] = f"{max(now - i['start_time'] for i in running):.0f}s"
    except Exception:
        pass
    return post


def _redirect_tau2_progress() -> None:
    """Drive a tqdm progress bar from tau2's StatusMonitor - like every other gbench eval.

    tau2's native progress ("Status: X/N complete ...") prints through the console we mute.
    Instead we attach a tqdm bar to StatusMonitor (the same `tqdm(total=...)` mechanism
    `run_eval_suite` uses for every other suite): it advances per completed task and its
    postfix (avg reward, in-flight count, oldest elapsed) refreshes on a fixed cadence
    (min(GBENCH_TAU2_PROGRESS_SECS, 5)s) so it stays live between completions. On a TTY / `tail -f`
    this renders as one updating line, exactly like the other evals' `Eval [X]` bars.

    Falls back to periodic gbench-logger lines if tqdm is unavailable. Never fatal.
    """
    try:
        import time as _time
        from tau2.runner.progress import StatusMonitor
    except Exception as e:
        logger.debug("tau2: could not import StatusMonitor for progress: %s", e)
        return

    interval = max(1.0, float(suite_env("GBENCH_TAU2_PROGRESS_SECS", "TAU2_PROGRESS_SECS", default="30")))

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if tqdm is None:
        # --- fallback: one gbench-logger line per interval (no tqdm available) ---
        def _logger_monitor(self) -> None:
            while not self._stop_event.wait(timeout=interval):
                try:
                    with self._lock:
                        completed, total = self.completed_count, self.total_count
                    post = _tau_progress_postfix(self, _time)
                    logger.info("%s progress: %d/%d done | reward %s | %s running (oldest %s)",
                                _TAU_PROGRESS_DESC, completed, total,
                                post.get("reward", "n/a"), post.get("running", 0),
                                post.get("oldest", "-"))
                except Exception:
                    pass
        StatusMonitor._monitor = _logger_monitor
        return

    # --- tqdm bar, updated per-task; postfix refreshed by the monitor thread ---
    bar_interval = min(interval, 5.0)
    _orig_start = StatusMonitor.start
    _orig_finished = StatusMonitor.task_finished
    _orig_stop = StatusMonitor.stop

    def start(self) -> None:
        try:
            self._pbar = tqdm(total=self.total_count, initial=self.completed_count,
                              desc=f"Eval [{_TAU_PROGRESS_DESC}]", unit="task")
        except Exception:
            self._pbar = None
        _orig_start(self)                       # starts the (patched) _monitor thread

    def task_finished(self, task_key) -> None:
        _orig_finished(self, task_key)          # increments completed_count, pops running
        pb = getattr(self, "_pbar", None)
        if pb is not None:
            try:
                pb.update(1)                    # tqdm.update is thread-safe across workers
            except Exception:
                pass

    def stop(self) -> None:
        _orig_stop(self)
        pb = getattr(self, "_pbar", None)
        if pb is not None:
            try:
                pb.set_postfix(_tau_progress_postfix(self, _time), refresh=True)
                pb.close()
            except Exception:
                pass
            self._pbar = None

    def _monitor(self) -> None:                 # keep postfix + elapsed live between completions
        while not self._stop_event.wait(timeout=bar_interval):
            pb = getattr(self, "_pbar", None)
            if pb is None:
                continue
            try:
                pb.set_postfix(_tau_progress_postfix(self, _time), refresh=True)
            except Exception:
                pass

    StatusMonitor.start = start
    StatusMonitor.task_finished = task_finished
    StatusMonitor.stop = stop
    StatusMonitor._monitor = _monitor


def env_requested() -> bool:
    """True iff the operator explicitly opted into the full tau2 simulator."""
    return suite_env("GBENCH_TAU2_ENV_RUN", "TAU2_ENV_RUN") == "1"


def _user_llm() -> str:
    from .base import DEFAULT_JUDGE_MODEL
    return suite_env("GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM", default=f"gemini/{DEFAULT_JUDGE_MODEL}")


def _eval_llm() -> str:
    from .base import DEFAULT_JUDGE_MODEL
    return suite_env("GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM", default=f"gemini/{DEFAULT_JUDGE_MODEL}")


def _import_tau2() -> Tuple[bool, str]:
    """Attempt a real `import tau2`, adding GBENCH_TAU2_BENCH_SRC to sys.path first if set.

    We import (not just find_spec) because tau2 can be *findable* yet not *importable* -
    e.g. on Python 3.13 its voice module imports the removed stdlib `audioop`. Callers
    need the real failure reason, so we return (ok, error_message).
    """
    src = suite_env("GBENCH_TAU2_BENCH_SRC", "TAU2_BENCH_SRC")
    if src and os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    # tau2 emits import-time loguru noise before it ever does useful work: a DEBUG
    # "Registry info: {...}" block (domains/agents/users/task_sets) and a WARNING
    # "No .env file found" (it looks for an optional .env to load keys; we pass keys via
    # the environment, so a missing .env is irrelevant). Configure loguru *before*
    # importing: raise the threshold to WARNING (kills the DEBUG dump) and add a filter
    # that drops the benign .env line while keeping real warnings (e.g. task retries).
    # GBENCH_TAU2_VERBOSE=1 keeps everything.
    if suite_env("GBENCH_TAU2_VERBOSE", "TAU2_VERBOSE") != "1":
        try:
            from loguru import logger as _loguru

            def _drop_import_noise(record):
                return "No .env file found" not in record["message"]

            _loguru.remove()
            _loguru.add(sys.stderr, level="WARNING", filter=_drop_import_noise)
        except Exception:
            pass
    try:
        import tau2  # noqa: F401
        return True, ""
    except Exception as e:  # ImportError + anything tau2.__init__ raises
        return False, f"{type(e).__name__}: {e}"


def check_tau_env_prerequisites() -> Tuple[bool, str]:
    """tau2-bench actually importable + a judge/user key for the gemini defaults."""
    ok, err = _import_tau2()
    if not ok:
        if "No module named 'tau2'" in err:
            return False, ("tau2-bench is not importable. It is not on PyPI - clone it and "
                           "install: `git clone https://github.com/sierra-research/tau2-bench && "
                           "pip install -e ./tau2-bench --no-deps`, or set GBENCH_TAU2_BENCH_SRC to a "
                           "tau2-bench/src checkout.")
        hint = ""
        if "audioop" in err:
            hint = (" tau2's voice module imports the stdlib `audioop`, removed in Python 3.13 - "
                    "install the backport: `pip install audioop-lts`.")
        return False, f"tau2-bench is installed but failed to import ({err}).{hint}"
    raw_key = os.getenv("GEMINI_API_KEY", "")
    if raw_key and "," in raw_key:
        os.environ["GEMINI_API_KEY"] = raw_key.split(",")[0].strip()

    if (_user_llm().startswith("gemini/") or _eval_llm().startswith("gemini/")) \
            and not os.getenv("GEMINI_API_KEY"):
        return False, ("tau2 environment needs GEMINI_API_KEY for the user simulator and the "
                       "nl-assertion judge (or set GBENCH_TAU2_USER_LLM / GBENCH_TAU2_EVAL_LLM to another "
                       "LiteLLM provider you have credentials for).")
    if (_user_llm().startswith("gemini/") or _eval_llm().startswith("gemini/")) \
            and os.getenv("GEMINI_API_KEY"):
        from .base import gemini_key_live_valid
        _ok, _why = gemini_key_live_valid(os.environ["GEMINI_API_KEY"])
        if not _ok:
            return False, (f"GEMINI_API_KEY was rejected by the judge endpoint ({_why}); a valid key "
                           "is required for the tau2 user simulator / nl-assertion judge (fail-fast).")
    return True, ""


def check_banking_prerequisites() -> Tuple[bool, str]:
    """Extra prerequisites for the banking_knowledge (tau3) RAG domain.

    banking_knowledge is a knowledge-retrieval domain, so on top of the shared tau2 env
    prereqs it needs a BM25 backend (`rank-bm25`) for the default/alltools retrieval configs,
    and - for the default Gemini-embedding dense retrieval - an embeddings key (the same
    GEMINI_API_KEY works via Google's OpenAI-compatible endpoint). Mirrors the house-style
    prereq skips (audioop, GEMINI key) with a docs pointer.
    """
    try:
        import rank_bm25  # noqa: F401
    except Exception:
        return False, ("banking_knowledge (tau3) needs the BM25 retrieval backend - install it: "
                       "`pip install rank-bm25`. See 'docs/evals/tau3.md' for setup.")
    # The default retrieval config (no GBENCH_TAU2_RETRIEVAL_CONFIG override) uses Gemini
    # embeddings for dense retrieval; that needs a key. A custom config may use other creds, so
    # only enforce this for the default path.
    if not suite_env("GBENCH_TAU2_RETRIEVAL_CONFIG", "TAU2_RETRIEVAL_CONFIG") and not (
        suite_env("GBENCH_TAU2_EMBED_API_KEY", "TAU2_EMBED_API_KEY")
        or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    ):
        return False, ("banking_knowledge (tau3) default dense retrieval uses Gemini embeddings and "
                       "needs GEMINI_API_KEY (or GBENCH_TAU2_EMBED_API_KEY / OPENAI_API_KEY). See "
                       "'docs/evals/tau3.md'.")
    return True, ""


# Google's OpenAI-compatible base URL - lets the OpenAI-SDK embedder hit Gemini embeddings.
_GEMINI_EMBED_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _setup_banking_retrieval() -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Resolve the banking_knowledge retrieval config, wiring Gemini embeddings by default.

    Returns (retrieval_config_name, retrieval_config_kwargs) to set on TextRunConfig.

    Default (no GBENCH_TAU2_RETRIEVAL_CONFIG): a canonical `alltools` variant whose dense half
    uses a Gemini embedding model (GBENCH_TAU2_EMBED_MODEL, default `gemini-embedding-001`) via
    Google's OpenAI-compatible endpoint - so no OpenAI/OpenRouter key is required. We register a
    dedicated `alltools-gemini` variant (embedder_type "openai" -> OpenAI SDK -> Gemini) and
    point the OpenAI SDK env at Gemini. Set GBENCH_TAU2_RETRIEVAL_CONFIG to any stock tau2-bench
    variant (e.g. `bm25`, `grep_only`, or `alltools` with your own OpenAI/OpenRouter creds)
    to bypass all of this.
    """
    override = suite_env("GBENCH_TAU2_RETRIEVAL_CONFIG", "TAU2_RETRIEVAL_CONFIG")
    if override:
        return override, None

    # `gemini-embedding-001` is the published embeddings model id. The previous default,
    # `gemini-embedding-2`, is not a served id: the KB warm-up 404s, retrieval never
    # builds, and tau3 banking_knowledge skips - which reads as "harness unavailable"
    # rather than "one env var is wrong".
    embed_model = suite_env("GBENCH_TAU2_EMBED_MODEL", "TAU2_EMBED_MODEL", default="gemini-embedding-001")
    embed_key = suite_env("GBENCH_TAU2_EMBED_API_KEY", "TAU2_EMBED_API_KEY") or os.getenv("GEMINI_API_KEY")
    embed_base = suite_env("GBENCH_TAU2_EMBED_BASE_URL", "TAU2_EMBED_BASE_URL", default=_GEMINI_EMBED_BASE)
    # Route the OpenAI-SDK embedder to Gemini unless real OpenAI creds are already present.
    if embed_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = embed_key
        os.environ.setdefault("OPENAI_BASE_URL", embed_base)
    # Gemini's OpenAI-compat embeddings endpoint caps a batch at 100 inputs; tau2-bench's
    # OpenAIEmbedder sends all ~700 docs in one request (400 on Gemini). Chunk embed().
    _patch_openai_embedder_batch(int(suite_env("GBENCH_TAU2_EMBED_BATCH", "TAU2_EMBED_BATCH", default="100")))

    variant_name = "alltools-gemini"
    try:
        from tau2.domains.banking_knowledge import retrieval as _r
        if variant_name not in _r.RETRIEVAL_VARIANTS:
            _r.RETRIEVAL_VARIANTS[variant_name] = _r.all_tools_variant(
                name=variant_name, embedder_type="openai", embedder_model=embed_model)
        # The KB-cache warmer resolves embedder configs from a hardcoded table keyed by
        # config name (embeddings_cache.get_unique_embedder_configs_for_retrieval_configs),
        # which doesn't know our custom variant. Wrap it so the doc embeddings are pre-warmed
        # with our Gemini model instead of being skipped (which would leave dense retrieval
        # to embed lazily or, worse, fall back to a wrong model).
        _patch_embedder_config_resolver(variant_name, embed_model)
        logger.info("tau3: banking_knowledge retrieval = %s (embedder=openai:%s via %s)",
                    variant_name, embed_model,
                    os.getenv("OPENAI_BASE_URL", "OpenAI default"))
        return variant_name, None
    except Exception as e:
        logger.warning("tau3: could not register gemini alltools variant (%s); using stock 'alltools'", e)
        return "alltools", None


def _patch_openai_embedder_batch(max_batch: int = 100) -> None:
    """Chunk OpenAIEmbedder.embed() into <=max_batch inputs per request (idempotent).

    tau2-bench sends every document in one embeddings.create() call, which is fine for
    OpenAI (batch limit ~2048) but 400s on Gemini's OpenAI-compatible endpoint (max 100
    per BatchEmbedContents). We wrap embed() to split large inputs and concatenate.
    """
    if max_batch < 1:
        return
    try:
        import numpy as _np
        from tau2.knowledge.embedders.openai_embedder import OpenAIEmbedder
        if getattr(OpenAIEmbedder.embed, "_gbench_chunked", False):
            return
        _orig = OpenAIEmbedder.embed

        def _chunked(self, texts):
            if not texts or len(texts) <= max_batch:
                return _orig(self, texts)
            parts = [_orig(self, texts[i:i + max_batch])
                     for i in range(0, len(texts), max_batch)]
            return _np.concatenate(parts, axis=0)

        _chunked._gbench_chunked = True
        OpenAIEmbedder.embed = _chunked
    except Exception as e:
        logger.debug("tau3: could not patch OpenAIEmbedder batch size: %s", e)


def _patch_embedder_config_resolver(variant_name: str, embed_model: str) -> None:
    """Teach the KB-cache warmer about our Gemini variant (idempotent, best-effort)."""
    try:
        from tau2.knowledge import embeddings_cache as _ec
        if getattr(_ec.get_unique_embedder_configs_for_retrieval_configs, "_gbench_wrapped", False):
            return
        _orig = _ec.get_unique_embedder_configs_for_retrieval_configs

        def _wrapped(names, retrieval_config_kwargs=None):
            out = list(_orig(names, retrieval_config_kwargs) or [])
            if variant_name in (names or []):
                cfg = ("openai", {"model": embed_model})
                if cfg not in out:
                    out.append(cfg)
            return out

        _wrapped._gbench_wrapped = True
        _ec.get_unique_embedder_configs_for_retrieval_configs = _wrapped
    except Exception as e:
        logger.debug("tau3: could not patch embedder-config resolver: %s", e)


def _strip_md_fences(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    c = text.strip()
    if c.startswith("```"):
        lines = c.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


#: Transport-level failures seen during a tau run, by class. tau2 turns a failed
#: generation into a reward-0 simulation with a populated `reward_info`, so the existing
#: `reward_info is None` check never sees them and they are indistinguishable from the
#: agent simply answering badly. Measured on tau3 2026-08-20: 4 ContextWindowExceededError
#: (prompt reached 262,145 tokens against the endpoint's 262,144 limit) plus 1 timeout,
#: while the result reported `infra_errors: 0` and an accuracy of 9.28%.
_LLM_FAILURES: Dict[str, int] = {}


def _note_llm_failure(exc: BaseException) -> None:
    name = type(exc).__name__
    text = f"{name}: {exc}"
    if "ContextWindowExceeded" in text:
        key = "context_window_exceeded"
    elif "Timeout" in text or "timed out" in text.lower():
        key = "timeout"
    elif "RateLimit" in text or "429" in text:
        key = "rate_limited"
    else:
        key = "other_llm_error"
    _LLM_FAILURES[key] = _LLM_FAILURES.get(key, 0) + 1


def llm_failures() -> Dict[str, int]:
    return dict(_LLM_FAILURES)


def reset_llm_failures() -> None:
    _LLM_FAILURES.clear()


def _resolve_tau_trace_dir() -> Optional[str]:
    """Where to persist tau2's per-task simulation traces. ON BY DEFAULT.

    Precedence:
      * `GBENCH_TAU2_SAVE_TRACES` set to a path  -> use it (explicit override);
      * `GBENCH_TAU2_SAVE_TRACES` set but empty  -> DISABLE (opt-out);
      * `GBENCH_TAU2_SAVE_TRACES` unset          -> default to `<run results dir>/tau_traces`
        (`GBENCH_RESULTS_DIR`, exported by LogManager), so tau matches every other eval and
        its traces live beside the eval JSONs. If no run dir is known (e.g. a bare unit
        test), returns None rather than dumping traces somewhere surprising.
    """
    explicit = suite_env("GBENCH_TAU2_SAVE_TRACES", "TAU2_SAVE_TRACES")
    if explicit is not None:
        return explicit.strip() or None
    run_dir = os.getenv("GBENCH_RESULTS_DIR")
    return os.path.join(run_dir, "tau_traces") if run_dir else None


def _build_gemini_cascade(inner_completion, backoff, cascade_fn=None):
    """Wrap a litellm `completion` so a gemini/<model> call falls through the grounding cascade.

    tau2's user simulator and nl-assertion judge default to gemini/<DEFAULT_JUDGE_MODEL>
    (gemini-3.6-flash); a single overloaded model used to fail the whole task (2026-08-21
    tau3, 500 throttling::OVERLOADED). This tries the requested model first (so an explicit
    GBENCH_TAU2_USER_LLM / GBENCH_TAU2_EVAL_LLM is honoured), then the rest of the SAME cascade grounding
    uses, with exponential backoff + jitter between models. Non-gemini models - notably the
    agent under test, routed as openai/<local> - pass straight through, uncascaded.

    Factored out (rather than defined inside the patch) so it is unit-testable without
    mutating the global litellm.completion.
    """
    if cascade_fn is None:
        from .search_tool import _search_cascade as cascade_fn

    def _completion_cascade(*args, **kwargs):
        model = kwargs.get("model")
        if not (isinstance(model, str) and model.startswith("gemini/")):
            return inner_completion(*args, **kwargs)
        chain = ["gemini/" + m for m in cascade_fn()]
        ordered = [model] + [m for m in chain if m != model]
        last = None
        for i, m in enumerate(ordered):
            try:
                return inner_completion(*args, **{**kwargs, "model": m})
            except BaseException as e:      # noqa: BLE001 - fall over to the next model
                last = e
                if i < len(ordered) - 1:
                    logger.debug("tau2: judge/user model %s failed (%s); falling to %s",
                                 m, type(e).__name__, ordered[i + 1])
                    time.sleep(backoff * (2 ** i) + random.uniform(0, backoff))
        raise last

    return _completion_cascade


def _install_robustness_patches(retries: int = 3) -> None:
    """Retry empty/malformed LLM responses and strip stray markdown fences.

    All patches are best-effort: any import/attribute drift in a given tau2 version is
    caught and logged, never fatal (the run proceeds with whatever could be patched).
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # --- Non-interactive nohup / background safe console input ---
    try:
        from rich.console import Console
        import tau2.utils.display as _disp
        if hasattr(_disp, "ConsoleDisplay") and hasattr(_disp.ConsoleDisplay, "console"):
            _disp.ConsoleDisplay.console.input = lambda *a, **k: "y"
    except Exception as e:
        logger.debug("tau2: could not patch console input: %s", e)

    # --- litellm.completion: retry when the first response is empty/malformed ---
    try:
        import json as _json
        import litellm
        _orig_completion = litellm.completion

        def _completion_retry(*args, **kwargs):
            try:
                resp = _orig_completion(*args, **kwargs)
            except BaseException as e:      # noqa: BLE001 - record, then re-raise unchanged
                _note_llm_failure(e)
                raise
            for attempt in range(retries - 1):
                try:
                    msg = resp.choices[0].message
                    tool_calls = getattr(msg, "tool_calls", None)
                    if not msg.content and not tool_calls:
                        raise ValueError("empty response")
                    if tool_calls:
                        for tc in tool_calls:
                            a = tc.function.arguments
                            if not a:
                                raise ValueError("empty tool arguments")
                            _json.loads(a)  # unparseable -> retry
                    return resp
                except (IndexError, AttributeError):
                    return resp  # unexpected shape; hand back untouched
                except (ValueError, TypeError):
                    logger.debug("tau2: empty/malformed response, retry %d/%d", attempt + 2, retries)
                    resp = _orig_completion(*args, **kwargs)
            return resp

        litellm.completion = _completion_retry
        try:
            import litellm.main
            litellm.main.completion = _completion_retry
        except Exception:
            pass
        try:
            import tau2.utils.llm_utils as _llm
            _llm.completion = _completion_retry
        except Exception:
            pass
    except Exception as e:
        logger.warning("tau2: could not install litellm retry patch: %s", e)

    # --- gemini judge/user-sim CASCADE: fall through the SAME model chain grounding uses ---
    # tau2's user simulator and nl-assertion judge default to gemini/<DEFAULT_JUDGE_MODEL>
    # (gemini-3.6-flash). On the 2026-08-21 sweep, tau3 tasks failed after their same-model
    # retries when that single alias returned 500 throttling::OVERLOADED, but the model
    # we requested was the cascade's own gemini-3.6-flash. Grounding survives an overloaded
    # model by cascading to the next one (each has its own capacity); the judge/user path did
    # not - it retried the one overloaded model 4x and gave up, depressing tau3's reward.
    # Wrap the completion chokepoint (which tau2.generate() calls) so a gemini/ call falls
    # through the exact same cascade as search_tool, with backoff+jitter between models. The
    # AGENT (model under test) is routed as openai/<local> and is deliberately NOT cascaded.
    try:
        _cascade_backoff = float(suite_env("GBENCH_TAU2_LLM_BACKOFF", "TAU2_LLM_BACKOFF",
                                           default=os.getenv("GBENCH_SEARCH_BACKOFF", "1.0")))
        _completion_cascade = _build_gemini_cascade(litellm.completion, _cascade_backoff)
        litellm.completion = _completion_cascade
        try:
            import litellm.main
            litellm.main.completion = _completion_cascade
        except Exception:
            pass
        try:
            import tau2.utils.llm_utils as _llm
            _llm.completion = _completion_cascade
        except Exception:
            pass
    except Exception as e:
        logger.warning("tau2: could not install judge/user cascade: %s", e)

    # --- tau2 generate(): retry empty content + strip markdown fences ---
    try:
        import tau2.utils.llm_utils as _llm
        _orig_generate = _llm.generate

        def _generate_retry(*args, **kwargs):
            last = None
            for attempt in range(retries):
                try:
                    res = _orig_generate(*args, **kwargs)
                except BaseException as e:  # noqa: BLE001 - record, then re-raise unchanged
                    _note_llm_failure(e)
                    raise
                last = res
                content = getattr(res, "content", None)
                tool_calls = getattr(res, "tool_calls", None)
                if content is not None:
                    stripped = _strip_md_fences(content)
                    if stripped != content:
                        try:
                            res.content = stripped
                        except Exception:
                            pass
                if (content or tool_calls) and (content is None or str(content).strip() or tool_calls):
                    return res
                logger.debug("tau2: empty generate(), retry %d/%d", attempt + 2, retries)
            return last

        for modname in ("tau2.utils.llm_utils", "tau2.agent.llm_agent",
                        "tau2.user.user_simulator", "tau2.environment.utils.interface_agent",
                        "tau2.evaluator.evaluator_nl_assertions",
                        "tau2.evaluator.hallucination_reviewer"):
            try:
                mod = importlib.import_module(modname)
                if hasattr(mod, "generate"):
                    mod.generate = _generate_retry
            except Exception:
                pass
    except Exception as e:
        logger.warning("tau2: could not install generate() retry patch: %s", e)


def _configure_evaluator_llm(eval_model: str) -> None:
    """Point tau2's nl-assertion evaluator + env-interface LLMs at `eval_model`.

    `from X import Y` copies values at import time, so we patch both the config module
    and the already-imported evaluator module. Best-effort across versions.
    """
    eval_args = {"temperature": 0.0}
    try:
        import tau2.config as cfg
        for attr, val in (("DEFAULT_LLM_NL_ASSERTIONS", eval_model),
                          ("DEFAULT_LLM_NL_ASSERTIONS_ARGS", eval_args),
                          ("DEFAULT_LLM_ENV_INTERFACE", eval_model),
                          ("DEFAULT_LLM_ENV_INTERFACE_ARGS", eval_args)):
            if hasattr(cfg, attr):
                setattr(cfg, attr, val)
    except Exception as e:
        logger.warning("tau2: could not set evaluator config: %s", e)
    try:
        import tau2.evaluator.evaluator_nl_assertions as nl
        if hasattr(nl, "DEFAULT_LLM_NL_ASSERTIONS"):
            nl.DEFAULT_LLM_NL_ASSERTIONS = eval_model
            nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = eval_args
    except Exception:
        pass


#: What `_build_agent_args` last resolved, so the result can record it (13).
_LAST_AGENT_TEMPERATURE: Dict[str, Any] = {}


def _build_agent_args(base_url: str, enable_thinking: bool,
                      eval_name: str = "tau2") -> Dict[str, Any]:
    """LiteLLM args routing ONLY the agent to the gbench endpoint, per-call."""
    # The agent is the model under test, so it must honour the same precedence as every
    # other suite, owned by the standard resolver: GBENCH_<EVAL>_TEMPERATURE (per-suite env
    # override) > --temperature > the suite's own 1.0. The bare TAU2_TEMPERATURE is now a
    # deprecated ALIAS of the canonical GBENCH_TAU2_TEMPERATURE (it no longer outranks the
    # canonical name or the CLI): if it is set and the canonical is not, we copy it onto the
    # canonical name and let the resolver decide. The evaluator/judge LLM stays at 0.0 and is
    # deliberately NOT routed through this - grading should not drift with the run's sampling
    # knob. tau2's own DEFAULT_LLM_TEMPERATURE_AGENT is 0.0; 1.0/0.95/64 is the reference tau2
    # configuration, which is what this suite reproduces. Resolving under `eval_name` keeps
    # GBENCH_TAU3_TEMPERATURE working when tau3 delegates here (it was hardcoded to "tau2").
    from .base import resolve_temperature, temperature_env_var
    _canon_temp = temperature_env_var(eval_name or "tau2")   # GBENCH_TAU2_TEMPERATURE / GBENCH_TAU3_TEMPERATURE
    _legacy_temp = os.getenv("TAU2_TEMPERATURE")
    if _legacy_temp and not os.getenv(_canon_temp):
        os.environ[_canon_temp] = _legacy_temp   # legacy alias -> canonical, then let the resolver own precedence
        logger.warning("env var TAU2_TEMPERATURE is a deprecated alias; use %s instead.", _canon_temp)
    agent_temp, temp_source = resolve_temperature(eval_name or "tau2", 1.0)
    _LAST_AGENT_TEMPERATURE["value"] = agent_temp
    _LAST_AGENT_TEMPERATURE["source"] = temp_source
    args: Dict[str, Any] = {
        "temperature": agent_temp,
        "api_base": base_url,
        "api_key": os.getenv("OPENAI_API_KEY", "EMPTY"),
        "top_p": float(suite_env("GBENCH_TAU2_TOP_P", "TAU2_TOP_P", default="0.95")),
    }
    extra_body: Dict[str, Any] = {"top_k": int(suite_env("GBENCH_TAU2_TOP_K", "TAU2_TOP_K", default="64"))}
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    args["extra_body"] = extra_body
    return args


def _run_one_domain(domain, model_name, base_url, concurrency, limit, enable_thinking,
                    eval_name="tau2"):
    """Run a single tau2 domain via the Python API; return (total, reward_sum, perfect, infra)."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    # Persist tau2's full SimulationResults (per-task messages + reward breakdown) so a run
    # can be audited afterwards - tau is a wrapped harness, so without this its per-task data
    # lives only in memory and is lost when the run ends (unlike every other eval, whose
    # sample_traces are saved). ON BY DEFAULT into the run's results dir; see
    # `_resolve_tau_trace_dir`.
    save_to = None
    trace_dir = _resolve_tau_trace_dir()
    if trace_dir:
        try:
            os.makedirs(trace_dir, exist_ok=True)
            save_to = os.path.join(trace_dir, f"tau2_{domain}_traces.json")
        except Exception as e:
            logger.warning("tau2: could not prepare trace dir %r: %s", trace_dir, e)

    # banking_knowledge (tau3) is a RAG domain: select/wire its retrieval backend. Other
    # domains (airline/retail/telecom) ignore retrieval_config.
    retrieval_config = None
    retrieval_config_kwargs = None
    if domain == "banking_knowledge":
        retrieval_config, retrieval_config_kwargs = _setup_banking_retrieval()

    config = TextRunConfig(
        domain=domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=f"openai/{model_name}",
        llm_args_agent=_build_agent_args(base_url, enable_thinking, eval_name),
        llm_user=_user_llm(),
        llm_args_user={"temperature": float(suite_env("GBENCH_TAU2_USER_TEMPERATURE", "TAU2_USER_TEMPERATURE", default="0.0"))},
        num_trials=int(suite_env("GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS", default="1")),
        max_steps=int(suite_env("GBENCH_TAU2_MAX_STEPS", "TAU2_MAX_STEPS", default="200")),
        max_errors=int(suite_env("GBENCH_TAU2_MAX_ERRORS", "TAU2_MAX_ERRORS", default="10")),
        max_concurrency=max(1, concurrency),
        seed=int(suite_env("GBENCH_TAU2_SEED", "TAU2_SEED", default="300")),
        log_level="ERROR",
        task_ids=None,
        num_tasks=(limit if (limit and limit > 0) else None),
        retrieval_config=retrieval_config,
        retrieval_config_kwargs=retrieval_config_kwargs,
        save_to=save_to,
        verbose_logs=False,
        auto_resume=True,
        hallucination_retries=0,
    )
    results = run_domain(config)
    sims = list(results.simulations)
    reward_sum = 0.0
    perfect = 0
    infra = 0
    for s in sims:
        ri = getattr(s, "reward_info", None)
        if ri is None:                       # infra error == reward 0 (not excluded)
            infra += 1
            continue
        r = float(ri.reward or 0.0)
        reward_sum += r
        if r >= 1.0:
            perfect += 1
    return len(sims), reward_sum, perfect, infra


#: Age (hours) above which a leftover tau retrieval sandbox is pruned at the start of a tau
#: run. tau2's SandboxManager stages a ~700-file knowledge base into
#: /tmp/agentic_search_<id>/ per simulation and cleans up only on graceful __exit__; a sim
#: killed or timed out (e.g. a context-window overflow) leaks its dir. On a RAM-backed tmpfs
#: these accumulate across runs until the INODE table fills and unrelated evals crash with a
#: misleading "No space left on device" (measured 2026-08-20: 1,141 leaked dirs, ~800K
#: inodes, killed aider_polyglot). gbench runs tau suites sequentially, so at the start of a
#: tau run nothing tau owns any such dir; pruning those older than this threshold is safe and
#: bounds accumulation to a single run. 0 disables. See docs/evals/large-sweep-setup.md.
_TAU_SANDBOX_MAX_AGE_H = float(os.environ.get("GBENCH_TAU_SANDBOX_MAX_AGE_H", "2"))


def _prune_stale_search_sandboxes() -> None:
    """Remove leftover /tmp/agentic_search_* dirs older than the age threshold. Never raises."""
    if _TAU_SANDBOX_MAX_AGE_H <= 0:
        return
    import glob
    import shutil
    import time
    try:
        cutoff = time.time() - _TAU_SANDBOX_MAX_AGE_H * 3600.0
    except Exception:            # time.time patched out in some sandboxes
        return
    base = os.path.join(tempfile.gettempdir(), "agentic_search_*")
    removed = 0
    for path in glob.glob(base):
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("tau: pruned %d stale retrieval sandbox(es) from %s (older than %.1fh); "
                    "see docs/evals/large-sweep-setup.md", removed,
                    tempfile.gettempdir(), _TAU_SANDBOX_MAX_AGE_H)


def run_tau_env(
    eval_name: str,
    domains: List[str],
    model_name: str,
    base_url: str,
    concurrency: int,
    limit: Optional[int],
    docs_url: str,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Run the real tau2 simulator per domain and aggregate into a standard result dict.

    Accuracy is the canonical tau2 mean reward; correct_answers
    is the perfect-task (reward>=1) count. Skips cleanly if prerequisites are missing or
    no domain produced results.
    """
    _prune_stale_search_sandboxes()
    ok, reason = check_tau_env_prerequisites()
    if not ok:
        raise infra_required(eval_name, reason, docs_url)
    if "banking_knowledge" in domains:
        ok_b, reason_b = check_banking_prerequisites()
        if not ok_b:
            raise infra_required(eval_name, reason_b, docs_url)

    _quiet_tau2_noise()
    reset_llm_failures()
    _install_robustness_patches()
    _configure_evaluator_llm(_eval_llm())

    category_accuracy: Dict[str, Any] = {}
    total = 0
    reward_sum_all = 0.0
    perfect_all = 0
    infra_all = 0
    ran_any = False

    # tau2's own per-task console is muted by default (it spams a file-redirected log), so
    # progress is shown via a tqdm bar driven by StatusMonitor - one per domain, exactly like
    # every other eval's `Eval [X]` bar. GBENCH_TAU2_VERBOSE=1 restores tau2's live per-task panels.
    # --eval-limit is a TOTAL budget across the domains being run, not per-domain: distribute it
    # (ceil) so the total stays ~limit while each domain keeps >=1 task. A single-domain run
    # (tau3, or an explicit `domain` kwarg) is unchanged.
    per_domain_limit = limit
    if limit and limit > 0 and len(domains) > 1:
        per_domain_limit = max(1, (int(limit) + len(domains) - 1) // len(domains))

    global _TAU_PROGRESS_DESC
    scope = f"{per_domain_limit} tasks/domain" if (per_domain_limit and per_domain_limit > 0) else "all tasks"
    trials = int(suite_env("GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS", default="1"))
    logger.info("%s: running %d domain(s) %s via tau2 simulator (%s, concurrency=%d, trials=%d); "
                "each task is a multi-turn agent<->user-simulator conversation scored by an LLM "
                "judge, so a domain runs for several minutes - a tqdm progress bar follows "
                "(set GBENCH_TAU2_VERBOSE=1 for per-task detail)",
                eval_name, len(domains), domains, scope, max(1, concurrency), trials)

    for i, domain in enumerate(domains, 1):
        _TAU_PROGRESS_DESC = f"{eval_name} {domain}" if len(domains) > 1 else eval_name
        logger.info("%s: [%d/%d] starting domain '%s' ...", eval_name, i, len(domains), domain)
        try:
            d_total, d_reward, d_perfect, d_infra = _run_one_domain(
                domain, model_name, base_url, concurrency, per_domain_limit, enable_thinking,
                eval_name)
        except Exception as e:
            logger.error("%s env: domain %s failed: %s", eval_name, domain, e, exc_info=True)
            continue
        if d_total == 0:
            logger.error("%s env: domain %s produced no simulations", eval_name, domain)
            continue
        ran_any = True
        total += d_total
        reward_sum_all += d_reward
        perfect_all += d_perfect
        infra_all += d_infra
        category_accuracy[domain] = {
            "correct": d_perfect,
            "total": d_total,
            "accuracy": round(d_reward / d_total * 100.0, 2),  # mean reward for the domain
        }
        logger.info("%s: [%d/%d] domain '%s' done - mean reward %.3f (%d/%d perfect, %d infra errors)",
                    eval_name, i, len(domains), domain,
                    d_reward / d_total, d_perfect, d_total, d_infra)

    if not ran_any:
        raise infra_required(
            eval_name,
            "tau2 simulator produced no results for any domain (check tau2 install, "
            "GEMINI_API_KEY, and the model endpoint)",
            docs_url)

    accuracy = (reward_sum_all / total * 100.0) if total > 0 else 0.0
    return {
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "mode": "full_environment",
        "total_questions": total,
        "correct_answers": perfect_all,
        "accuracy": round(accuracy, 2),          # canonical tau2 mean reward
        "category_accuracy": category_accuracy,
        # tau2 grades with its canonical harness (DB-state check + nl-assertion judge), run
        # here on gbench's standard Gemini cascade user-sim/judge. The remaining reason the
        # number is not directly comparable to the published leaderboard is the trial count:
        # the default is a single trial, while the leaderboard reports pass^k.
        "leaderboard_comparable": False,
        "leaderboard_comparable_reason": (
            "single-trial by default (the tau2 leaderboard reports pass^k over multiple "
            "trials; set TAU2_NUM_TRIALS=k to match)"
        ),
        "tau2_report": {
            "mean_reward": round(reward_sum_all / total, 4) if total else 0.0,
            "perfect_tasks": perfect_all,
            "total_simulations": total,
            "infra_errors": infra_all,
            "num_trials": int(suite_env("GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS", default="1")),
            "agent_llm": f"openai/{model_name}",
            "user_llm": _user_llm(),
            "eval_llm": _eval_llm(),
        },
        # Transport failures, counted where they happen rather than inferred from the
        # simulation record. tau2 scores a failed generation as reward 0 with a populated
        # `reward_info`, so `infra_errors` above stays 0 and an outage is indistinguishable
        # from a bad agent. tau3 on 2026-08-20 reported infra_errors 0 / accuracy 9.28%
        # while hitting 4 context-window overflows and a timeout.
        "llm_failures": llm_failures(),
        "llm_failures_total": sum(llm_failures().values()),
        # The model under test's sampling knob, so a run can be audited from the artifact
        # like every other suite. The user simulator and judge stay at 0.0 and are separate.
        "temperature": _LAST_AGENT_TEMPERATURE.get("value"),
        "temperature_source": _LAST_AGENT_TEMPERATURE.get("source"),
        "status": "success",
    }
