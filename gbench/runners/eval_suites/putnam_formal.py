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

"""Native Putnam Mathematical Competition Formal Track (Lean 4) via Docker Sandboxes.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_PUTNAM_FORMAL_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, suite_env
from .sampling import limit_dataset
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/putnam_formal.md"
# Pin GBENCH_PUTNAM_FORMAL_LEAN_IMAGE (legacy alias PUTNAM_LEAN_IMAGE) to the exact
# Lean/Mathlib the PutnamBench statements target (a floating :latest can rename/deprecate
# lemmas -> non-reproducible). The default stays overridable; the exact-digest pin + a
# Lean-compile-vs-gold check is tracked for the reproducibility wave.
LEAN_IMAGE = suite_env("GBENCH_PUTNAM_FORMAL_LEAN_IMAGE", "PUTNAM_LEAN_IMAGE",
                       default="leanprovercommunity/lean4:latest")
# `import Mathlib` elaboration alone can approach 30s; a non-trivial proof adds more,
# so a 30s cap scored correct-but-slow proofs as failures. Generous + configurable.
COMPILE_TIMEOUT_S = int(suite_env("GBENCH_PUTNAM_FORMAL_COMPILE_TIMEOUT_S",
                                  "PUTNAM_COMPILE_TIMEOUT_S", default="300"))


def _lean_image_provenance() -> Dict[str, Any]:
    """Resolve the Lean image to an immutable digest for the result, and flag mutable tags.

    A floating tag like ':latest' means two runs can compile against different Lean/Mathlib
    (lemmas renamed/deprecated) - so the score is not reproducible. Record the RepoDigest
    actually used (docker inspect) and mark reproducible=False when the operator pinned a
    mutable tag rather than an @sha256 digest.
    """
    ref = LEAN_IMAGE
    pinned = "@sha256:" in ref
    digest = None
    try:
        insp = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", ref],
            capture_output=True, text=True, timeout=10)
        if insp.returncode == 0:
            digest = (insp.stdout or "").strip() or None
    except Exception as e:  # noqa: BLE001
        logger.debug("putnam_formal: could not resolve Lean image digest: %s", e)
    if not pinned:
        logger.warning(
            "putnam_formal: GBENCH_PUTNAM_FORMAL_LEAN_IMAGE=%r is a MUTABLE tag - the Lean/Mathlib "
            "behind it changes over time, so this score is not reproducible. Pin a digest for a "
            "publishable number, e.g. GBENCH_PUTNAM_FORMAL_LEAN_IMAGE=%s",
            ref, digest or f"{ref.split(':')[0]}@sha256:...")
    return {"ref": ref, "digest": digest, "reproducible": bool(pinned)}


def check_putnam_formal_prerequisites() -> Tuple[bool, str]:
    """Check if Docker daemon is running and automatically pull Lean 4 image if missing."""
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH. Install Docker to run formal Lean 4 verification."

    try:
        res = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        if res.returncode != 0:
            return False, f"Docker daemon is not running or accessible: {res.stderr.strip()}"
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"

    # Auto-pull Lean 4 Docker image if not present locally (matching Harbor / Terminal-Bench behavior)
    try:
        insp = subprocess.run(["docker", "image", "inspect", LEAN_IMAGE], capture_output=True, timeout=5)
        if insp.returncode != 0:
            logger.info(f"Auto-pulling Lean 4 Docker image '{LEAN_IMAGE}' for formal verification...")
            pull = subprocess.run(["docker", "pull", LEAN_IMAGE], capture_output=True, text=True, timeout=300)
            if pull.returncode != 0:
                return False, f"Failed to pull Lean 4 Docker image '{LEAN_IMAGE}': {pull.stderr.strip()}"
    except Exception as e:
        return False, f"Error ensuring Lean 4 Docker image '{LEAN_IMAGE}': {e}"

    # Every PutnamBench statement uses Mathlib (Finset.Icc, Tendsto, Polynomial, 𝓝 ...).
    # The default lean4 image ships the toolchain only, so without Mathlib EVERY compile
    # fails and the suite reports a structural 0% that says nothing about the model.
    try:
        probe = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "sh", LEAN_IMAGE, "-c",
             "printf 'import Mathlib\\n' > /tmp/p.lean && lean /tmp/p.lean"],
            capture_output=True, text=True, timeout=180)
        if probe.returncode != 0:
            return False, (
                f"Lean image '{LEAN_IMAGE}' has no Mathlib, but every PutnamBench statement "
                f"depends on it, so all verifications would fail regardless of the model. "
                f"Point GBENCH_PUTNAM_FORMAL_LEAN_IMAGE (legacy alias PUTNAM_LEAN_IMAGE) at a "
                f"Mathlib-provisioned image (or a prebuilt lake project with a Mathlib cache).")
    except Exception as e:
        return False, f"Could not verify Mathlib availability in '{LEAN_IMAGE}': {e}"

    return True, ""


def _load_putnam_formal_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load formal Lean 4 theorem statements directly from canonical HF Hub dataset ('amitayusht/PutnamBench')."""
    from datasets import load_dataset

    ds = load_dataset("amitayusht/PutnamBench", split="train")
    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, 'tags', seed="putnam_formal")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} Putnam formal samples from HF Hub ('amitayusht/PutnamBench').")

    samples = []
    for item in raw_samples:
        lean_stmt = item.get("lean4_statement", "")
        if not lean_stmt:
            continue

        problem_name = item.get("name", "putnam_theorem")
        tags = item.get("tags", ["formal_proof"])
        if isinstance(tags, str) and tags.startswith("["):
            import ast
            try:
                tags = ast.literal_eval(tags)
            except Exception:
                pass
        category = str(tags[0]).strip("'\"[] ") if tags else "formal_proof"

        prompt = (
            "Complete the Lean 4 formal proof below.\n"
            "CRITICAL REQUIREMENT: Output ONLY valid Lean 4 code enclosed in a ```lean ... ``` block. "
            "Do NOT include conversational explanation, markdown commentary, or English prose before or after the code block.\n\n"
            f"```lean\n{lean_stmt}\n```"
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, lean_stmt, {"category": category, "theorem_name": problem_name}))

    return samples


def _eval_putnam_formal(response_text: str, lean_stmt: str) -> bool:
    """Compile generated Lean 4 proof in sandboxed Docker container."""
    if not response_text or not lean_stmt:
        return False

    # Extract Lean code from response block
    code = response_text.strip()
    match = re.search(r"```lean\s*(.*?)\s*```", code, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()

    # If model provided full replacement or tactics
    if "theorem" in code:
        full_code = code
    else:
        full_code = lean_stmt.replace(":= sorry", f":= by\n{code}")

    # PutnamBench statements ship with NO preamble yet use unqualified Mathlib names
    # (Ici, Tendsto, atTop, 𝓝, ...), so without the canonical header every proof
    # fails to elaborate (structural ~0%). Prepend the canonical PutnamBench Lean4
    # header. NOTE: per-problem `open`s vary slightly upstream - validate with a Lean
    # compile against gold before release. Ref: trishullab/PutnamBench lean4/src/*.lean.
    if "import Mathlib" not in full_code:
        full_code = "import Mathlib\nopen Set Function Filter Topology\n\n" + full_code

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chmod(tmpdir, 0o777)
        src_path = Path(tmpdir) / "Proof.lean"
        src_path.write_text(full_code, encoding="utf-8")
        os.chmod(src_path, 0o666)

        cmd = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "",
            "--network",
            "none",
            "--memory",
            "4g",
            "--cpus",
            "2",
            "-v",
            f"{tmpdir}:/workspace:ro",
            LEAN_IMAGE,
            "lean",
            "/workspace/Proof.lean",
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
            output = f"{res.stdout}\n{res.stderr}".lower()
            # Valid proof: exit code 0 AND no 'sorry' or 'error' in compiler output (stdout/stderr)
            if res.returncode == 0 and "sorry" not in output and "error" not in output:
                return True
        except Exception:
            pass

    return False


def run_putnam_formal(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Putnam Mathematical Competition formal verification suite (Lean 4) via Docker sandbox."""
    ok, reason = check_putnam_formal_prerequisites()
    if not ok:
        raise infra_required("putnam_formal", reason, DOCS_URL)

    samples = _load_putnam_formal_samples(limit=kwargs.get("limit"))

    result = run_eval_suite(
        eval_name="putnam_formal",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_putnam_formal,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    # CC6 reproducibility: the Lean/Mathlib version decides every compile, so record the exact
    # image digest used and flag a mutable tag rather than let it silently drift.
    result["lean_image"] = _lean_image_provenance()
    return result
