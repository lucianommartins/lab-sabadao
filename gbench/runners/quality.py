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

"""Quality benchmark runner using gemmaclaw QA suites."""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import fcntl
from pathlib import Path
from typing import Optional

from ..core.config import BenchmarkConfig
from ..core.models import ModelConfig, ModelFormat
# The same constant the sha resolver in scaffold.py resolves against. A
# sha resolved against one remote and a checkout taken from another would
# pin a scorer this run never used.
from ..core.scaffold import GEMMACLAW_REMOTE
from .serving import ServingBenchmarkRunner

logger = logging.getLogger(__name__)

#: Minimum Node.js (major, minor) required by gemmaclaw. One source of truth for
#: both the nvm-upgrade resolver and the preflight gate.
_MIN_NODE = (22, 19)


def _node_version(node_bin: Optional[str]) -> tuple:
    """Return the (major, minor) of a node binary, or (0, 0) if it is missing or
    unrunnable. ``None`` (e.g. shutil.which found nothing) -> (0, 0)."""
    if not node_bin:
        return (0, 0)
    try:
        out = subprocess.run([node_bin, "-v"], capture_output=True, text=True,
                             check=True).stdout.strip().lstrip("v")
        major, minor, *_ = (out.split(".") + ["0", "0"])
        return (int(major), int(minor))
    except (subprocess.SubprocessError, OSError, ValueError):
        return (0, 0)


class FileLock:
    """Simple Unix file locking context manager."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.lock_file = None

    def __enter__(self):
        self.lock_file = open(self.lock_path, "w")
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_file:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()


class QualityBenchmarkRunner(ServingBenchmarkRunner):
    """Runner for quality benchmarks using gemmaclaw QA suite."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the quality benchmark runner.

        Args:
            config: Benchmark configuration
        """
        super().__init__(config)
        self.gemmaclaw_repo_path = (
            Path(self.config.gemmaclaw_path)
            if self.config.gemmaclaw_path
            else Path("~/.cache/gbench/gemmaclaw").expanduser()
        )
    def _get_plugin_id_and_validity(self, plugin_dir: Path) -> tuple[Optional[str], bool]:
        """Resolve plugin ID and validity from its manifest or package.json.

        Returns:
            A tuple of (plugin_id, is_valid_plugin).
        """
        # Check openclaw.plugin.json first
        manifest_path = plugin_dir / "openclaw.plugin.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "id" in data:
                        return data["id"], True
            except Exception:
                pass
            return plugin_dir.name, True

        # Check package.json
        pkg_path = plugin_dir / "package.json"
        if pkg_path.exists():
            try:
                with open(pkg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "openclaw" in data:
                        pkg_name = data.get("name", "")
                        if pkg_name.startswith("@openclaw/"):
                            return pkg_name.split("/", 1)[1], True
                        return pkg_name or plugin_dir.name, True
            except Exception:
                pass

        return None, False

    def _preflight_check(self) -> bool:
        """Perform environment preflight check for Node.js, pnpm, and Git.

        Returns:
            True if all checks pass, False otherwise.
        """
        logger.info("Performing quality runner preflight checks...")

        # Resolve a Node >= 22.19 from NVM. The system node may be present but too
        # OLD (e.g. Ubuntu ships v18): shutil.which then finds it and the version
        # gate below fails, so we must STILL look in nvm for a satisfying newer
        # version and put it first on PATH. Handles both "no node" and "old node".
        if _node_version(shutil.which("node")) < _MIN_NODE:
            nvm_dir = Path("~/.nvm/versions/node").expanduser()
            if nvm_dir.exists():
                def _ver_key(d):
                    return [int(c) if c.isdigit() else 0
                            for c in d.name.lstrip("v").split(".")]
                # Newest nvm version whose node actually satisfies the floor.
                satisfying = sorted(
                    (d for d in nvm_dir.iterdir()
                     if d.is_dir() and (d / "bin" / "node").exists()
                     and _node_version(str(d / "bin" / "node")) >= _MIN_NODE),
                    key=_ver_key,
                )
                if satisfying:
                    node_bin = satisfying[-1] / "bin"
                    logger.info(
                        f"System node < v{_MIN_NODE[0]}.{_MIN_NODE[1]}; using nvm "
                        f"node at {node_bin}")
                    os.environ["PATH"] = str(node_bin) + os.pathsep + os.environ.get("PATH", "")

        # 1. Check Node.js version
        try:
            node_ver_res = subprocess.run(
                ["node", "-v"],
                capture_output=True, text=True, check=True
            )
            node_ver = node_ver_res.stdout.strip().lstrip("v")
            major, minor, _ = (node_ver.split(".") + ["0", "0"])[:3]
            major_num = int(major)
            minor_num = int(minor)
            
            if major_num < 22 or (major_num == 22 and minor_num < 19):
                logger.error(
                    f"Outdated Node.js version: v{node_ver}. "
                    "Node.js v22.19+ is required by gemmaclaw."
                )
                return False
            logger.info(f"Node.js check passed: v{node_ver}")
        except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
            logger.error(f"Node.js preflight check failed: {e}")
            return False

        # 2. Check pnpm / npx pnpm availability
        try:
            # We can check npx availability which is bundled with Node
            npx_check = subprocess.run(
                ["npx", "--version"],
                capture_output=True, text=True, check=True
            )
            logger.info(f"npx check passed: v{npx_check.stdout.strip()}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"npx preflight check failed: {e}")
            return False

        # 3. Check Git availability
        try:
            git_check = subprocess.run(
                ["git", "--version"],
                capture_output=True, text=True, check=True
            )
            logger.info(f"git check passed: {git_check.stdout.strip()}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"git preflight check failed: {e}")
            return False

        return True

    def _git_rewrite_safe_env(self) -> dict:
        """Env that neutralizes a host ~/.gitconfig ``insteadOf`` rule rewriting
        ``https://github.com`` -> ``ssh://git@github.com``. gemmaclaw is a PUBLIC
        repo and pnpm resolves git-based transitive deps at install time; the
        rewrite turns those public HTTPS fetches into SSH fetches needing a key,
        which hang on a prompt or fail ``Permission denied (publickey)``. Public
        deps need no auth, so we blank the global git config for these
        subprocesses and forbid interactive prompts (fail fast). Mirrors the
        swe_lancer eval fix; overridable via GBENCH_QUALITY_GIT_CONFIG_GLOBAL for
        a host that legitimately needs its rewrites (e.g. private deps)."""
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = os.environ.get(
            "GBENCH_QUALITY_GIT_CONFIG_GLOBAL", os.devnull)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault(
            "GIT_SSH_COMMAND",
            "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10")
        return env

    def _setup_gemmaclaw_repo(self) -> Optional[str]:
        """Clone or update the gemmaclaw repository to the target commit.

        Returns:
            The commit hash of the checked out gemmaclaw repository, or None on failure.
        """
        commit = self.config.gemmaclaw_commit
        logger.info(f"Setting up gemmaclaw repo at {self.gemmaclaw_repo_path} (ref: {commit})")

        # Neutralize host github HTTPS->SSH rewrites for every git/pnpm subprocess
        # below (clone, fetch, and pnpm's git-based deps).
        git_env = self._git_rewrite_safe_env()

        lock_path = self.gemmaclaw_repo_path.parent / "gemmaclaw.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        is_prod = os.environ.get("GBENCH_ENV") == "prod"

        try:
            with FileLock(lock_path):
                if is_prod and self.gemmaclaw_repo_path.exists():
                    # In production, trust the baked-in repo. Avoid network and rebuilds.
                    commit_hash_res = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=str(self.gemmaclaw_repo_path),
                        capture_output=True, text=True, check=True
                    )
                    commit_hash = commit_hash_res.stdout.strip()
                    logger.info(f"[PROD] Using baked-in gemmaclaw repo at commit: {commit_hash}")
                    
                    # Verify build stamp
                    build_stamp_path = self.gemmaclaw_repo_path / "dist" / ".buildstamp"
                    if build_stamp_path.exists():
                        try:
                            with open(build_stamp_path) as sf:
                                stamp_data = json.load(sf)
                            if stamp_data.get("head") == commit_hash:
                                logger.info("[PROD] gemmaclaw build is up to date.")
                                return commit_hash
                        except Exception as e:
                            logger.warning(f"[PROD] Failed to read build stamp: {e}")
                    
                    logger.warning("[PROD] Build stamp missing or mismatch. Will attempt build fallback.")
                
                else:
                    # Local mode or repo doesn't exist (clone/fetch logic)
                    if self.config.gemmaclaw_path:
                        # If path was explicitly provided by user, do not clone or fetch, just checkout the commit.
                        if not self.gemmaclaw_repo_path.exists():
                            logger.error(f"Provided gemmaclaw path does not exist: {self.gemmaclaw_repo_path}")
                            return None
                    else:
                        # Clone or update from remote
                        if not self.gemmaclaw_repo_path.exists():
                            self.gemmaclaw_repo_path.parent.mkdir(parents=True, exist_ok=True)
                            logger.info("Cloning gemmaclaw repo from GitHub...")
                            subprocess.run(
                                [
                                    "git", "clone",
                                    GEMMACLAW_REMOTE,
                                    str(self.gemmaclaw_repo_path)
                                ],
                                check=True, env=git_env
                            )
                        else:
                            logger.info("gemmaclaw repo exists, fetching latest updates...")
                            subprocess.run(
                                ["git", "fetch", "origin"],
                                cwd=str(self.gemmaclaw_repo_path),
                                check=True, env=git_env
                            )

                    # Checkout target commit/branch
                    logger.info(f"Checking out '{commit}'...")
                    subprocess.run(
                        ["git", "checkout", commit],
                        cwd=str(self.gemmaclaw_repo_path),
                        check=True, env=git_env
                    )

                # Get exact commit hash (re-read if we did checkout)
                commit_hash_res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(self.gemmaclaw_repo_path),
                    capture_output=True, text=True, check=True
                )
                commit_hash = commit_hash_res.stdout.strip()
                logger.info(f"gemmaclaw is at commit: {commit_hash}")

                # Check if build stamp exists and matches HEAD to skip build (for local mode fallback)
                build_stamp_path = self.gemmaclaw_repo_path / "dist" / ".buildstamp"
                if build_stamp_path.exists():
                    try:
                        with open(build_stamp_path) as sf:
                            stamp_data = json.load(sf)
                        if stamp_data.get("head") == commit_hash:
                            logger.info("gemmaclaw build is up to date, skipping installation and build.")
                            return commit_hash
                    except Exception:
                        pass

                # Install dependencies (pnpm resolves git-based transitive deps,
                # so it needs the github-rewrite-safe env too).
                logger.info("Installing GemmaClaw dependencies via pnpm (this may take ~1 min on first run)...")
                subprocess.run(
                    ["npx", "-y", "pnpm", "install"],
                    cwd=str(self.gemmaclaw_repo_path),
                    check=True, env=git_env
                )

                # Build packages
                logger.info("Building GemmaClaw/OpenClaw TypeScript workspace (109 packages)... Initial compilation takes ~1-2 minutes, please wait.")
                subprocess.run(
                    ["npx", "-y", "pnpm", "build"],
                    cwd=str(self.gemmaclaw_repo_path),
                    check=True, env=git_env
                )

                # Write build stamp to skip future compiles
                from datetime import datetime
                build_stamp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(build_stamp_path, "w") as sf:
                    json.dump({"head": commit_hash, "builtAt": datetime.utcnow().isoformat() + "Z"}, sf, indent=2)

                return commit_hash
        except Exception as e:
            logger.error(f"Failed to set up gemmaclaw repo: {e}")
            return None


    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
    ) -> dict:
        """Run gemmaclaw quality benchmark suite against local vLLM.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)

        Returns:
            Dictionary with quality benchmark results

        Raises:
            RuntimeError: If benchmark execution fails
        """
        lm = self.config.log_manager
        output_file = lm.get_quality_result_path(model.short_name, format.value)
        log_file = lm.get_quality_log_path(model.short_name, format.value)

        # Preflight checks
        if not self.config.dry_run:
            if not self._preflight_check():
                raise RuntimeError("Quality runner preflight environment checks failed")

            commit_hash = self._setup_gemmaclaw_repo()
            if not commit_hash:
                raise RuntimeError("Failed to set up gemmaclaw repository")

        # Start vLLM server
        if not self.config.dry_run and not self.config.remote_endpoint:
            if not self._start_server(model, format):
                raise RuntimeError("Failed to start vLLM server")

        temp_dir = None
        config_path = None
        try:
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would run quality benchmarks for {model.short_name}")
                return {"dry_run": True}

            # Create a temporary directory for config
            temp_dir = tempfile.mkdtemp(prefix="gbench_quality_")
            qa_output_dirname = "gbench_qa_out"
            qa_output_dir = os.path.join(str(self.gemmaclaw_repo_path), qa_output_dirname)
            if os.path.exists(qa_output_dir):
                shutil.rmtree(qa_output_dir, ignore_errors=True)
            os.mkdir(qa_output_dir)

            if self.config.remote_endpoint:
                if "run.app" in self.config.remote_endpoint:
                    model_path = model.short_name
                else:
                    model_path = model.hf_model_id
            else:
                model_path = model.get_model_path(format)

            # Determine endpoint URL and API key/token dynamically
            api_key = os.environ.get("VLLM_API_KEY")
            # GCP identity-token auto-auth (metadata server + gcloud fallback) is OPT-IN: a public
            # user pointing at any *.googleapis.com / *.run.app endpoint should not trigger a
            # surprise metadata-server call or a `gcloud` subprocess. Enable with GBENCH_GCP_AUTH=1.
            _gcp_auth = os.environ.get("GBENCH_GCP_AUTH", "").strip().lower() in ("1", "true", "yes")
            if self.config.remote_endpoint:
                endpoint_url = self.config.remote_endpoint
                _looks_gcp = ("run.app" in endpoint_url or "googleapis.com" in endpoint_url)
                if not api_key and _looks_gcp and not _gcp_auth:
                    logger.info("Endpoint looks like a Google Cloud service but GBENCH_GCP_AUTH is not "
                                "set; not contacting the GCP metadata server or gcloud. Set VLLM_API_KEY "
                                "for auth, or GBENCH_GCP_AUTH=1 to fetch a GCP identity token automatically.")
                if not api_key and _looks_gcp and _gcp_auth:
                    # Try metadata server first (for Cloud Run)
                    try:
                        logger.info("Retrieving GCP identity token from Metadata Server...")
                        import urllib.request
                        # Use the service URL (without path) as audience
                        audience = endpoint_url.split("/v1")[0]
                        req = urllib.request.Request(
                            f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={audience}",
                            headers={"Metadata-Flavor": "Google"}
                        )
                        with urllib.request.urlopen(req, timeout=2) as resp:
                            api_key = resp.read().decode("utf-8").strip()
                        logger.info("Successfully retrieved token from Metadata Server.")
                    except Exception as e:
                        logger.debug(f"Metadata server token retrieval failed: {e}")

                    # Fallback to gcloud (for local dev)
                    if not api_key:
                        try:
                            logger.info("Retrieving GCP identity token via gcloud...")
                            api_key = subprocess.check_output(
                                ["gcloud", "auth", "print-identity-token"],
                                text=True
                            ).strip()
                        except Exception as e:
                            logger.warning(f"Could not retrieve GCP identity token: {e}. Defaulting to unauthenticated.")
            else:
                endpoint_url = f"http://127.0.0.1:{self.server_port}/v1"

            if not api_key:
                api_key = "local-vllm"

            # Generate custom openclaw.json config pointing to the server
            enabled_plugins = {"google", "qa-lab", "active-memory"}
            plugin_entries = {}

            # Scan extensions directory to disable others
            extensions_dir = self.gemmaclaw_repo_path / "extensions"
            if extensions_dir.exists():
                for item in extensions_dir.iterdir():
                    if item.is_dir():
                        p_id, is_valid = self._get_plugin_id_and_validity(item)
                        if is_valid and p_id not in enabled_plugins:
                            plugin_entries[p_id] = {"enabled": False}

            # Explicitly enable the ones we need
            for p in enabled_plugins:
                plugin_entries[p] = {"enabled": True}

            config_data = {
                "models": {
                    "providers": {
                        "vllm": {
                            "baseUrl": endpoint_url,
                            "apiKey": api_key,
                            "api": "openai-completions",
                            "models": [
                                {
                                    "id": model_path,
                                    "name": f"Local {model.short_name}",
                                    "contextWindow": model.max_context_length,
                                    "maxTokens": 4096
                                }
                            ]
                        }
                    }
                },
                "plugins": {
                    "entries": plugin_entries
                }
            }

            config_path = os.path.join(temp_dir, "openclaw.json")
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            logger.info(
                f"\n{'='*60}\n"
                f"  QUALITY BENCHMARK CONFIGURATION\n"
                f"{'='*60}\n"
                f"  Model:          {model.short_name} ({format.value})\n"
                f"  vLLM Port:      {self.server_port}\n"
                f"  Gemmaclaw Commit: {commit_hash[:10]}\n"
                f"{'='*60}"
            )

            # Run gemmaclaw suite subcommand
            # Env overrides for gemmaclaw configuration. Base it on the
            # github-rewrite-safe env so any runtime npx/pnpm/git fetch by
            # openclaw also bypasses the host HTTPS->SSH rewrite. The Gemini
            # judge key (GEMINI_API_KEY) is inherited from os.environ here.
            env = self._git_rewrite_safe_env()
            env["OPENCLAW_CONFIG_PATH"] = config_path
            env["VLLM_API_KEY"] = api_key
            env["OPENCLAW_ENABLE_PRIVATE_QA_CLI"] = "1"

            cmd = [
                "npx", "-y", "pnpm", "openclaw", "qa", "suite",
                "--provider-mode", "live-frontier",
                "--model", f"vllm/{model_path}",
                "--output-dir", qa_output_dirname
            ]

            if getattr(self.config, "selected_scenarios", None):
                for scenario in self.config.selected_scenarios:
                    cmd.extend(["--scenario", scenario])

            logger.info(f"Running gemmaclaw suite: {' '.join(cmd)}")
            
            import time
            with open(log_file, "w") as log_fh:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.gemmaclaw_repo_path),
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                try:
                    # Read from log_file and write to sys.stdout in a deadlock-free way
                    with open(log_file, "r") as log_read:
                        while process.poll() is None:
                            line = log_read.readline()
                            if line:
                                sys.stdout.write(line)
                                sys.stdout.flush()
                            else:
                                time.sleep(0.1)
                        
                        # Read remaining lines
                        for line in log_read:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                            
                    process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    logger.error("gemmaclaw suite timed out after 3600 seconds. Killing process...")
                    process.kill()
                    process.wait()
                    raise RuntimeError("gemmaclaw suite timed out")

            if process.returncode != 0:
                logger.error(f"gemmaclaw suite exited with code {process.returncode}")
                # We can check log file contents for traceback if needed

            # Read result summary
            summary_path = os.path.join(qa_output_dir, "qa-suite-summary.json")
            if not os.path.exists(summary_path):
                raise RuntimeError(
                    f"gemmaclaw failed to write suite summary. "
                    f"Check logs at: {log_file}"
                )

            with open(summary_path) as f:
                qa_results = json.load(f)

            counts = qa_results.get("counts", {})
            total_scenarios = counts.get("total", 0)
            passed_scenarios = counts.get("passed", 0)
            failed_scenarios = counts.get("failed", 0)
            blocked_scenarios = counts.get("blocked", 0)
            
            pass_rate = (passed_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0.0

            result = {
                "benchmark_type": "quality",
                "model_name": model.name,
                "model_short": model.short_name,
                "format": format.value,
                "gemmaclaw_commit": commit_hash,
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "failed_scenarios": failed_scenarios,
                "blocked_scenarios": blocked_scenarios,
                "pass_rate": pass_rate,
                "raw_results": qa_results
            }

            # Save result to gbench logs
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(
                f"Quality benchmark complete: Pass rate={pass_rate:.1f}% "
                f"({passed_scenarios}/{total_scenarios} scenarios passed)"
            )

            return result

        except Exception as e:
            logger.error(f"Quality benchmark failed: {e}")
            raise RuntimeError(f"Quality benchmark failed: {e}")

        finally:
            # Clean up temp folder
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            # Clean up relative output folder inside gemmaclaw repo
            qa_output_dir = os.path.join(str(self.gemmaclaw_repo_path), "gbench_qa_out")
            if os.path.exists(qa_output_dir):
                shutil.rmtree(qa_output_dir, ignore_errors=True)
            # Cleanup server
            if not self.config.dry_run:
                self._cleanup_server()
