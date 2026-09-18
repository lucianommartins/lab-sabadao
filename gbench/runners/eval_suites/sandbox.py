# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process isolation for eval suites that execute model-written code (audit CC5).

`--sandboxes` only bounds how many of these run at once; it does not isolate anything.
Until now every code-executing suite ran `sys.executable -c <model output>` directly on the
host, with the user's full filesystem and network.

This wraps those calls in `bubblewrap`: read-only root, private `/tmp`, no network, and an
explicit writable bind for the scratch directory the suite needs.

    GBENCH_SANDBOX=required (default) require bwrap. If it is unavailable or blocked,
                                     code-executing suites SKIP (via sandbox_skip_reason)
                                     rather than run unsandboxed.
    GBENCH_SANDBOX=bwrap             require bwrap and FAIL LOUDLY (raise) if it is
                                     unavailable or blocked - a strict gate for CI.
    GBENCH_SANDBOX=none              run directly on the host (explicit unsandboxed opt-out)

Isolation is mandatory by default: executing model-written code without a sandbox is a
security risk, so there is no "degrade and run unsandboxed" policy - the only way to run
without isolation is to ask for it explicitly with `none`. Every mode self-probes once (a
bwrap that is installed but blocked - no user namespaces, a restrictive container - is
treated as unavailable) rather than assuming, so a broken sandbox surfaces as a clean skip
(or, under `bwrap`, a loud failure), not as a model that "cannot write code".
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["sandbox_mode", "sandbox_available", "sandbox_skip_reason",
           "wrap_argv", "run_sandboxed"]

_PROBE_TIMEOUT = 10


def sandbox_mode() -> str:
    mode = (os.environ.get("GBENCH_SANDBOX") or "required").strip().lower()
    return mode if mode in ("required", "bwrap", "none") else "required"


@functools.lru_cache(maxsize=1)
def _probe_bwrap() -> bool:
    """Does bwrap actually run here? Cached: this is asked once per code-executing sample."""
    if not shutil.which("bwrap"):
        return False
    try:
        proc = subprocess.run(
            wrap_argv([sys.executable, "-c", "print(1)"], force=True),
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT)
        return proc.returncode == 0 and proc.stdout.strip() == "1"
    except Exception as e:  # pragma: no cover - environment dependent
        logger.debug("bwrap probe failed: %s", e)
        return False


@functools.lru_cache(maxsize=1)
def sandbox_available() -> bool:
    """True when execution will actually be isolated (bubblewrap works here).

    Under the default `required` mode a False return means code-executing suites must
    SKIP (see `sandbox_skip_reason`) rather than run unsandboxed - so this stays silent
    there and lets the skip carry the message.
    """
    mode = sandbox_mode()
    if mode == "none":
        logger.warning("GBENCH_SANDBOX=none: model-written code runs directly on this host "
                       "with full filesystem and network access.")
        return False
    if _probe_bwrap():
        return True
    if mode == "bwrap":
        raise RuntimeError(
            "GBENCH_SANDBOX=bwrap was requested but bubblewrap is unavailable or blocked "
            "here. Install `bubblewrap` and ensure unprivileged user namespaces are "
            "permitted, set GBENCH_SANDBOX=none to run unsandboxed, or GBENCH_SANDBOX="
            "required to skip code-executing suites instead of failing.")
    # required: not isolated -> callers skip via sandbox_skip_reason(); never degrade.
    return False


def sandbox_skip_reason() -> Optional[str]:
    """Reason a code-executing suite must SKIP, or None if it may proceed.

    Default (`required`): model-written code must not run without isolation, so if
    bubblewrap is unavailable or blocked, return a message to skip with. `none` (explicit
    unsandboxed opt-out) and `bwrap` (fails loudly at the exec call instead of skipping)
    return None, so the suite proceeds.
    """
    mode = sandbox_mode()
    if mode in ("none", "bwrap"):
        return None
    if _probe_bwrap():
        return None
    if shutil.which("bwrap"):
        # Installed but the probe failed - almost always blocked unprivileged user
        # namespaces (Ubuntu 23.10+/24.04 ships apparmor_restrict_unprivileged_userns=1).
        return ("this suite executes model-written code and gbench requires bubblewrap "
                "isolation (GBENCH_SANDBOX=required, the default). bubblewrap IS installed "
                "but is BLOCKED from creating an unprivileged user namespace here. On "
                "Ubuntu 23.10+/24.04 this is the AppArmor restriction - allow it with "
                "`sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0` (persist under "
                "/etc/sysctl.d/). Or set GBENCH_SANDBOX=none to run UNSANDBOXED (model code "
                "executes directly on this host)")
    return ("this suite executes model-written code and gbench requires bubblewrap for "
            "process isolation (GBENCH_SANDBOX=required, the default), but bubblewrap is not "
            "installed. Install it (`sudo apt-get install bubblewrap`) and allow unprivileged "
            "user namespaces, or set GBENCH_SANDBOX=none to run UNSANDBOXED (model code "
            "executes directly on this host)")


def wrap_argv(argv: Sequence[str], writable: Sequence[str] = (), network: bool = False,
              force: bool = False) -> List[str]:
    """Prefix `argv` with a bubblewrap jail, or return it unchanged when disabled.

    `writable` lists directories the command legitimately needs to write (a compiler's
    scratch dir, the temp file holding the program). Everything else is read-only.
    """
    if not force and not sandbox_available():
        if sandbox_mode() in ("required", "bwrap"):
            # Defense in depth: a code-executing suite reached here without gating on
            # sandbox_skip_reason(). Fail loudly rather than silently run unsandboxed.
            raise RuntimeError(
                "Refusing to execute model-written code unsandboxed: bubblewrap is required "
                "(GBENCH_SANDBOX=required, the default) but is unavailable or blocked. Install "
                "`bubblewrap`, or set GBENCH_SANDBOX=none to run unsandboxed on purpose.")
        return list(argv)

    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
           "--tmpfs", "/tmp", "--die-with-parent", "--new-session"]
    if not network:
        cmd += ["--unshare-net"]
    for path in writable:
        if path and os.path.isdir(path):
            cmd += ["--bind", path, path]
    return cmd + list(argv)


def run_sandboxed(argv: Sequence[str], writable: Sequence[str] = (), network: bool = False,
                  **kwargs: Any) -> "subprocess.CompletedProcess":
    """`subprocess.run` for model-written code, isolated when bubblewrap is available."""
    return subprocess.run(wrap_argv(argv, writable=writable, network=network), **kwargs)
