# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
"""Hardware calibration (WS4).

`gbench --calibrate` inspects the local hardware (GPUs and VRAM, CPU cores, RAM, Docker) and prints
recommended `--num-gpus`, `--sandboxes`, and `--batch-sizes` so a user does not have to guess. The
same detection also backs a pre-run guardrail that aborts gracefully, with an actionable message,
when a requested `--sandboxes` count would over-subscribe the machine, instead of letting it OOM or
exhaust the Docker address pool mid-sweep.

The recommendation is deliberately conservative and heuristic; it is a starting point, not a
guarantee. The two knobs it sizes:

* `--sandboxes` (concurrency for containerized evals): each concurrent sandbox runs a heavy Docker
  task container (compilers, test suites, model rollouts), budgeted here at about
  `_CORES_PER_SANDBOX` cores and `_RAM_GB_PER_SANDBOX` GB.
* `--batch-sizes` (the eval client's HTTP concurrency to the served model): this is bound by the
  served model's throughput, not local CPU, so the suggestion is a moderate starting concurrency.
"""

import os
import shutil
import subprocess
from typing import Any, Dict, Tuple

# A containerized-eval sandbox runs a heavy Docker task; budget cores + RAM per concurrent sandbox.
_CORES_PER_SANDBOX = 2
_RAM_GB_PER_SANDBOX = 8
#: env override to bypass the pre-run over-subscription guard (explicit opt-out).
GUARD_BYPASS_ENV = "GBENCH_SKIP_CALIBRATION_GUARD"


def _cpu_cores() -> int:
    """Usable CPU cores, affinity-aware (respects cgroup/taskset limits), min 1."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _ram_gb() -> Tuple[float, float]:
    """(total_gb, available_gb). Reads Linux /proc/meminfo; falls back to sysconf for total and
    assumes ~80% available when MemAvailable is missing."""
    total = avail = 0.0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    if total <= 0:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
        except (ValueError, OSError):
            total = 0.0
    if avail <= 0:
        avail = total * 0.8
    return round(total, 1), round(avail, 1)


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def detect_hardware() -> Dict[str, Any]:
    """Inspect local hardware. GPU/VRAM values are 0 when nvidia-smi is absent (fail open)."""
    from ..core.config import get_available_gpus, _gpu_total_vram_gb, _gpu_min_free_vram_gb
    total_ram, avail_ram = _ram_gb()
    return {
        "gpus": get_available_gpus(),
        "gpu_total_vram_gb": round(_gpu_total_vram_gb(), 1),
        "gpu_free_vram_gb": round(_gpu_min_free_vram_gb(), 1),
        "cpu_cores": _cpu_cores(),
        "ram_total_gb": total_ram,
        "ram_available_gb": avail_ram,
        "docker": _docker_ok(),
    }


def max_safe_sandboxes(hw: Dict[str, Any]) -> int:
    """The most concurrent container-eval sandboxes this host should run (0 if Docker is absent)."""
    if not hw.get("docker"):
        return 0
    cores, ram = hw["cpu_cores"], hw["ram_available_gb"]
    return max(1, min(cores // _CORES_PER_SANDBOX, int(ram // _RAM_GB_PER_SANDBOX)))


def _largest_pow2(n: int) -> int:
    """Largest power of two <= n (>=1 when n>=1, else 0). Tensor-parallel size must be a
    power of two, and validate_gpu_config hard-rejects other counts, so calibrate must not
    recommend e.g. 6 GPUs on a 6-GPU host - it recommends 4."""
    if n < 1:
        return 0
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def recommend(hw: Dict[str, Any]) -> Dict[str, Any]:
    """Recommended knob values for this hardware."""
    cores = hw["cpu_cores"]
    return {
        "num_gpus": _largest_pow2(hw["gpus"]) if hw["gpus"] else None,
        "sandboxes": max_safe_sandboxes(hw),
        # HTTP concurrency to the served model; endpoint-throughput-bound, so a moderate default.
        "eval_concurrency": max(1, min(64, cores * 4)),
    }


def check_oversubscription(requested_sandboxes: Any, hw: Dict[str, Any]) -> Tuple[bool, str]:
    """(ok, message). Not ok when a requested sandbox count clearly exceeds CPU/RAM capacity.

    A bypass via GUARD_BYPASS_ENV always returns ok (explicit opt-out). The hard cap allows headroom
    over the conservative recommendation so a mild over-shoot is not blocked, only a clear one."""
    if os.environ.get(GUARD_BYPASS_ENV):
        return True, ""
    try:
        n = int(requested_sandboxes)
    except (TypeError, ValueError):
        return True, ""
    if n <= 0:
        return True, ""
    if not hw.get("docker"):
        # No Docker: container evals hard-error on their own prereq check; nothing to guard here.
        return True, ""
    rec = max_safe_sandboxes(hw)
    hard_cap = max(hw["cpu_cores"], rec * 3)   # headroom: block only a clear over-subscription
    if n > hard_cap:
        return False, (
            f"--sandboxes {n} over-subscribes this host ({hw['cpu_cores']} CPU cores, "
            f"{hw['ram_available_gb']:.0f} GB RAM available): a containerized-eval sandbox needs "
            f"about {_CORES_PER_SANDBOX} cores and {_RAM_GB_PER_SANDBOX} GB each. Recommended "
            f"--sandboxes {rec}. Lower it, run `gbench --calibrate`, or set {GUARD_BYPASS_ENV}=1 "
            f"to override.")
    return True, ""


def format_report(hw: Dict[str, Any], rec: Dict[str, Any]) -> str:
    gpu_line = (f"{hw['gpus']} GPU(s), {hw['gpu_total_vram_gb']:.0f} GB VRAM each "
                f"({hw['gpu_free_vram_gb']:.0f} GB free)" if hw["gpus"]
                else "none detected (nvidia-smi absent or no visible GPU)")
    docker_line = "available" if hw["docker"] else "NOT available (containerized evals will hard-error)"
    lines = [
        "gbench hardware calibration",
        "",
        "Detected:",
        f"  GPU:    {gpu_line}",
        f"  CPU:    {hw['cpu_cores']} usable cores",
        f"  RAM:    {hw['ram_total_gb']:.0f} GB total, {hw['ram_available_gb']:.0f} GB available",
        f"  Docker: {docker_line}",
        "",
        "Recommended flags:",
        (f"  --num-gpus {rec['num_gpus']}" if rec["num_gpus"]
         else "  --num-gpus: n/a (no local GPU; use --remote-endpoint against a served model)"),
        (f"  --sandboxes {rec['sandboxes']}   (concurrency for containerized evals)"
         if rec["sandboxes"] else
         "  --sandboxes: n/a (Docker not available; containerized evals hard-error)"),
        f"  --batch-sizes {rec['eval_concurrency']}   (eval client concurrency; endpoint-bound, tune to the server)",
        "",
        "Notes:",
        "  - --num-gpus / tensor-parallel also depends on the model size (a >80B model needs 8).",
        "  - --sandboxes is a conservative starting point; raise it if the host stays underloaded.",
        f"  - A requested --sandboxes above about {max(hw['cpu_cores'], rec['sandboxes'] * 3)} is refused "
        f"pre-run unless {GUARD_BYPASS_ENV}=1.",
    ]
    return "\n".join(lines)


def calibration_report() -> str:
    """Convenience: detect + recommend + format, for `gbench --calibrate`."""
    hw = detect_hardware()
    return format_report(hw, recommend(hw))
