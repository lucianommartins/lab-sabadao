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

"""Serving benchmark runner using vLLM Python API directly."""

import asyncio
import atexit
import json
import logging
import os
import requests
import shutil
import signal
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import Optional, Any

from ..core.config import BenchmarkConfig, get_max_model_len, get_server_timeout
from ..core.models import ModelConfig, ModelFormat
from ..analysis.statistics import (
    aggregate_benchmark_results,
    compute_statistics,
    format_statistics_summary,
    summarize_latency_metric,
    validate_repeatability,
)

# vLLM benchmark imports will be loaded dynamically in active methods

logger = logging.getLogger(__name__)


class ServingBenchmarkRunner:
    """Runner for vLLM serving benchmarks using Python API."""

    # Multimodal request shape - kept in lockstep with StressTestRunner so all
    # three MM pillars measure the same campaign-sized workload: N images + text
    # padded to the campaign input length, output = campaign output length with
    # ignore_eos. Needs the server started with --limit-mm-per-prompt image=N.
    IMAGES_PER_REQUEST = 4
    PER_IMAGE_SOFT_TOKENS = 280   # gemma-4 vision_soft_tokens_per_image
    MIN_MM_TEXT_TOKENS = 16
    # No-think system prompt (default). A reasoning model (e.g. gemma-4 on Ollama)
    # otherwise streams its chain-of-thought into the `reasoning` field and delays/
    # replaces `content` - inflating TTFT and, when the whole budget goes to
    # reasoning, producing 0-content "empty" replies. gbench sends no system
    # message, so the backend injects the model's default ("...with reasoning...")
    # system; sending OUR own system prompt is ONE no-think lever, but it is NOT
    # sufficient on every backend (Ollama's gemma reasons regardless over /v1), so the
    # perf path ALSO sends the portable switches below. --thinking skips both.
    NOTHINK_SYSTEM = ("Answer directly with your final response only. Do not show "
                      "any reasoning, analysis, or thinking. Follow the user's "
                      "length and formatting instructions.")

    def no_think_fields(self) -> dict:
        """Portable no-think request fields for the PERF path. The serving/stress
        workload is a controlled, fixed-length generation, so a reasoning prefix is
        uncontrolled noise that must not enter the measurement. NOTHINK_SYSTEM alone
        does not stop some backends, so also send the switches vLLM/SGLang honor;
        backends that don't support them ignore them (HTTP 200, no error). Returns {}
        under --thinking (then the model reasons on purpose)."""
        if getattr(self.config, "thinking", False):
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}

    def __init__(self, config: BenchmarkConfig):
        """Initialize the serving benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.server_process: Optional[subprocess.Popen] = None
        self._server_log_file = None  # File handle for server logs
        self.server_port = int(os.environ.get("GBENCH_SERVER_PORT", "8000"))
        self._mm_image_dir = None  # Temp dir for generated images
        self._mm_image_paths = []  # Generated image file paths
        self._mm_tokenizer = None  # Lazily loaded; sizes MM padded-text prompt

    def _images_per_request(self) -> int:
        """Images per MM request: config override (--personal sets 1) else default 4."""
        return int(getattr(self.config, "mm_images_per_request", None) or self.IMAGES_PER_REQUEST)

    def _build_mm_text(self, n_tokens: int, tokenizer) -> str:
        """Text prompt of ~n_tokens tokens used to pad a multimodal request."""
        base = "Analyze the attached images and the following context in detail. "
        if not tokenizer or n_tokens <= 0:
            return base * max(1, n_tokens // 12)  # rough char-based fallback
        filler = base * (max(1, n_tokens // 8) + 8)
        ids = tokenizer(filler).input_ids[:n_tokens]
        return tokenizer.decode(ids)
        # Register cleanup on exit
        atexit.register(self._cleanup_server)

    def _build_text_prompt(self, in_len: int, out_len: int, tokenizer,
                           seed: int = 0) -> str:
        """Coherent chat prompt of ~in_len tokens that reliably elicits a long reply.

        Used for REMOTE endpoints that ignore ignore_eos (e.g. Ollama), where the
        gibberish RandomDataset makes the model emit empty / immediate-EOS replies.
        A real passage padded to EXACTLY in_len tokens + an explicit long-output
        directive keeps the controlled input length (so TTFT/TPOT stay comparable
        to the local RandomDataset shape) while giving the model a real task.
        Varied by ``seed`` so identical prompts don't skew a prefix-caching backend.
        """
        passage = (
            "Consider how a large city keeps itself supplied with fresh water. "
            "Reservoirs collect rainfall across a wide watershed, aqueducts carry "
            "it toward the population, and treatment plants remove sediment and "
            "pathogens before distribution. Engineers balance demand against "
            "capacity, plan for droughts, and maintain aging pipes that leak more "
            "with every passing year. The same logic of collection, transport, "
            "treatment, and delivery reappears in power grids, food systems, and "
            "communication networks, each a quiet feat of coordination that most "
            "residents never pause to notice. "
        )
        words = passage.split()
        rot = (seed * 13) % len(words)
        filler = " ".join(words[rot:] + words[:rot])
        target_words = max(150, int(out_len * 0.5))
        directive = (
            f"\n\nUsing the passage above as context, write a detailed, "
            f"well-structured explanation of at least {target_words} words about "
            f"how complex systems stay reliable. Develop several distinct points."
        )
        # Unique, coherent per-request lead-in so no two of the N requests are
        # byte-identical. Rotation alone repeats every len(words) (~83), so an
        # N>len(words) run would emit duplicate prompts a prefix-caching backend
        # dedups (skewing TTFT); the seed nonce guarantees distinctness at any N.
        nonce = f"Document {seed}. "
        if tokenizer is None:
            reps = max(1, in_len // max(1, len(words)) + 1)
            return nonce + (filler + " ") * reps + directive
        dir_ids = tokenizer(directive, add_special_tokens=False).input_ids
        nonce_ids = tokenizer(nonce, add_special_tokens=False).input_ids
        if seed == 0 and int(in_len) < len(dir_ids) + len(nonce_ids) + 1:
            logger.warning(
                f"--input-lengths {in_len} is below the coherent-prompt floor "
                f"(~{len(dir_ids) + len(nonce_ids)} tok for directive + lead-in); "
                f"actual input length will exceed it - use in_len >= "
                f"{len(dir_ids) + len(nonce_ids) + 1}.")
        body_budget = max(1, int(in_len) - len(dir_ids) - len(nonce_ids))
        body_ids = tokenizer(filler, add_special_tokens=False).input_ids
        if not body_ids:
            # A corrupt tokenizer emitted no tokens for real English text - avoid
            # a non-terminating tile loop; fall back to the directive alone.
            return directive.strip()
        body_ids = (body_ids * (body_budget // len(body_ids) + 1))[:body_budget]
        return tokenizer.decode(nonce_ids + body_ids) + directive

    def _resolve_model_id(self, model: ModelConfig, format: ModelFormat) -> str:
        """Resolve the model ID to use for API requests."""
        if not self.config.remote_endpoint:
            return model.get_model_path(format)
            
        base_url = self.config.remote_endpoint.rstrip("/")
        try:
            import requests
            resp = requests.get(f"{base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    remote_ids = [m["id"] for m in data["data"]]
                    for candidate in [model.hf_model_id, model.name, model.short_name]:
                        if candidate in remote_ids:
                            return candidate
                    return remote_ids[0]
        except Exception:
            pass
            
        return model.hf_model_id


    def _wait_for_gpu_memory(self, min_free_gb: Optional[float] = None, max_wait_seconds: int = 120):
        """Wait for GPU memory to be reclaimed by CUDA driver.

        Uses nvidia-smi to actively poll until sufficient memory is free.
        Respects CUDA_VISIBLE_DEVICES to only check the assigned GPUs.

        Args:
            min_free_gb: Minimum free memory required in GB. None => 80% of the
                GPU's total VRAM (scales to any card; the old fixed 70GB made this
                wait the full timeout on every <70GB GPU).
            max_wait_seconds: Maximum time to wait before giving up
        """
        if self.config.remote_endpoint:
            return

        if min_free_gb is None:
            from ..core.config import _gpu_total_vram_gb
            _total = _gpu_total_vram_gb()
            min_free_gb = 0.8 * _total if _total > 0 else 8.0

        logger.info(f"Waiting for GPU memory to be reclaimed (need {min_free_gb:.0f}GB free)...")
        
        # Build nvidia-smi command targeting only our assigned GPUs
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        smi_cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        if gpu_ids:
            smi_cmd.extend(["--id=" + gpu_ids])
        
        for elapsed in range(0, max_wait_seconds, 5):
            try:
                result = subprocess.run(
                    smi_cmd,
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Check ALL assigned GPUs have enough free memory
                    free_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
                    if free_values:
                        min_free_mb = min(free_values)
                        free_gb = min_free_mb / 1024
                        if free_gb >= min_free_gb:
                            logger.info(f"GPU memory ready: {free_gb:.1f}GB free (min across {len(free_values)} GPU(s))")
                            return
                        logger.info(f"  GPU memory: {free_gb:.1f}GB free (min), waiting... ({elapsed}s)")
            except Exception:
                pass  # nvidia-smi failed, fall back to fixed wait
            time.sleep(5)
        
        logger.warning(f"GPU memory wait timed out after {max_wait_seconds}s, proceeding anyway")

    def _setup_mm_images(self, num_images: int = 50) -> str:
        """Generate synthetic images for multimodal serving benchmark.

        Creates a temp directory with random JPEG images.

        Args:
            num_images: Number of images to generate

        Returns:
            Path to the temp directory containing images
        """
        import numpy as np
        from PIL import Image

        img_dir = tempfile.mkdtemp(prefix="serving_mm_images_")
        self._mm_image_paths = []

        for i in range(num_images):
            # 256x256 random images - small enough for fast I/O,
            # large enough to exercise the vision encoder
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(img_dir, f"serve_img_{i:03d}.jpg")
            img.save(img_path, format="JPEG", quality=85)
            self._mm_image_paths.append(img_path)

        logger.info(f"Generated {num_images} synthetic images in {img_dir}")
        self._mm_image_dir = img_dir
        return img_dir

    def _cleanup_mm_images(self):
        """Clean up temp image directory."""
        if self._mm_image_dir and os.path.exists(self._mm_image_dir):
            shutil.rmtree(self._mm_image_dir, ignore_errors=True)
            logger.info(f"Cleaned up MM image directory: {self._mm_image_dir}")
            self._mm_image_dir = None
            self._mm_image_paths = []

    def _generate_sample_requests(
        self,
        model: ModelConfig,
        format: ModelFormat,
        num_prompts: int,
        input_len: int = 512,
        output_len: int = 128,
    ) -> tuple[list[Any], Any]:
        """Generate sample requests for benchmarking using vLLM's RandomDataset.

        Args:
            model: Model configuration
            format: Model format
            num_prompts: Number of prompts to generate
            input_len: Input token length
            output_len: Output token length

        Returns:
            Tuple of (list of SampleRequest, tokenizer)
        """
        from gbench.utils import check_vllm_available, require_vllm_engine

        if not check_vllm_available():
            if not self.config.remote_endpoint:
                require_vllm_engine("Serving benchmark datasets")

            in_len = self.config.input_lengths[0] if self.config.input_lengths else input_len
            out_len = self.config.output_lengths[0] if self.config.output_lengths else output_len

            from dataclasses import dataclass
            @dataclass
            class FallbackSampleRequest:
                prompt: str
                prompt_len: int
                expected_output_len: int
                multi_modal_data: Optional[dict] = None

            sample_text = "The quick brown fox jumps over the lazy dog. " * (in_len // 8 + 1)
            requests = [
                FallbackSampleRequest(
                    prompt=sample_text,
                    prompt_len=in_len,
                    expected_output_len=out_len,
                )
                for _ in range(num_prompts)
            ]
            return requests, None

        if self.config.dataset == "sharegpt":
            from vllm.benchmarks.datasets import ShareGPTDataset
            from gbench.utils import safe_get_tokenizer
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
            
            # Use ShareGPT default path convention or download
            sharegpt_path = getattr(self.config, "dataset_path", None) or os.environ.get("SHAREGPT_PATH", os.path.expanduser("~/.cache/gbench/ShareGPT_V3_unfiltered_cleaned_split.json"))
            if not os.path.exists(sharegpt_path):
                # Download just-in-time if missing
                os.makedirs(os.path.dirname(sharegpt_path), exist_ok=True)
                import urllib.request
                logger.info(f"Downloading ShareGPT dataset to {sharegpt_path}...")
                urllib.request.urlretrieve("https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json", sharegpt_path)

            dataset = ShareGPTDataset(dataset_path=sharegpt_path)
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
            )
            return requests, tokenizer
        elif self.config.dataset == "custom":
            from vllm.benchmarks.datasets import CustomDataset
            from gbench.utils import safe_get_tokenizer
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
            
            dataset_path = getattr(self.config, "dataset_path", None)
            if not dataset_path:
                raise ValueError("--dataset-path is required when using custom dataset")
                
            dataset = CustomDataset(dataset_path=dataset_path)
            out_len = self.config.output_lengths[0] if self.config.output_lengths else 128
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
                output_len=out_len,
            )
            return requests, tokenizer
        elif self.config.dataset == "hf":
            from vllm.benchmarks.datasets import get_samples
            from gbench.utils import safe_get_tokenizer
            import argparse
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
            
            dataset_path = getattr(self.config, "dataset_path", None)
            if not dataset_path:
                raise ValueError("--dataset-path is required when using hf dataset")
                
            args = argparse.Namespace(
                dataset_name="hf",
                dataset_path=dataset_path,
                hf_name=dataset_path,
                hf_split="train",
                hf_subset=None,
                disable_shuffle=False,
                seed=83,
                num_prompts=num_prompts,
                custom_output_len=self.config.output_lengths[0] if self.config.output_lengths else 128,
                skip_chat_template=False,
                request_id_prefix="",
                no_oversample=False,
            )
            requests = get_samples(args, tokenizer)
            return requests, tokenizer
        else:
            from vllm.benchmarks.datasets import RandomDataset
            from gbench.utils import safe_get_tokenizer
            
            tokenizer_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            tokenizer = safe_get_tokenizer(tokenizer_path, custom_tokenizer=self.config.tokenizer)
            if tokenizer is None:
                raise RuntimeError(
                    f"Could not load a tokenizer for '{tokenizer_path}' - text "
                    f"serving needs one to synthesize prompts. Pass --tokenizer "
                    f"<hf-id-or-path>"
                    + (" (a remote run against a gated model needs a locally-"
                       "loadable tokenizer or HF auth)."
                       if self.config.remote_endpoint else "."))

            # Use config's input/output lengths
            in_len = self.config.input_lengths[0] if self.config.input_lengths else input_len
            out_len = self.config.output_lengths[0] if self.config.output_lengths else output_len

            # REMOTE endpoints may ignore ignore_eos (e.g. Ollama), so gibberish
            # RandomDataset prompts draw empty / immediate-EOS replies. Use a
            # COHERENT prompt (real passage padded to in_len + an explicit long-
            # output directive), varied per request, so replies are sustained.
            # LOCAL vLLM honors ignore_eos, so it KEEPS the validated RandomDataset
            # controlled-shape workload below - the local/large-machine path is
            # untouched.
            if self.config.remote_endpoint:
                from dataclasses import dataclass

                @dataclass
                class _TextRequest:
                    prompt: str
                    prompt_len: int
                    expected_output_len: int
                    multi_modal_data: Optional[dict] = None

                reqs = [
                    _TextRequest(
                        prompt=self._build_text_prompt(in_len, out_len, tokenizer, seed=i),
                        prompt_len=in_len,
                        expected_output_len=out_len,
                    )
                    for i in range(num_prompts)
                ]
                return reqs, tokenizer

            # range_ratio=0.0 => every request is EXACTLY (in_len, out_len).
            # For a latency measurement this is the right design: it isolates the
            # shape so the reported percentiles reflect system/queueing variance,
            # not input-length variance (which would smear the tails and make two
            # models incomparable). It also means all requests share one output
            # length, so pooled ITL/TPOT percentiles are clean. Deterministic
            # shape also removes the 1.5x tail, so max_model_len only needs to
            # clear (in_len+out_len) - campaign_ctx() sizes it with slack.
            dataset = RandomDataset(random_seed=83)
            requests = dataset.sample(
                tokenizer=tokenizer,
                num_requests=num_prompts,
                input_len=in_len,
                output_len=out_len,
                range_ratio=0.0,
            )
            return requests, tokenizer

    async def _run_mm_serving_benchmark(
        self,
        model: ModelConfig,
        format: ModelFormat,
        max_concurrency: int,
        num_prompts: int,
    ) -> dict:
        """Run multimodal serving benchmark via aiohttp + SSE.

        Sends chat completion requests with images, measures TTFT, TPOT, ITL
        from SSE streaming response tokens.

        Args:
            model: Model configuration
            format: Model format
            max_concurrency: Maximum concurrent requests (from batch_size)
            num_prompts: Number of requests to send

        Returns:
            Dictionary with serving metrics matching text serving output format
        """
        import aiohttp

        model_path = self._resolve_model_id(model, format)
        base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self.server_port}"
        url = base_url.rstrip("/")
        api_url = f"{url}/chat/completions" if url.endswith("/v1") else f"{url}/v1/chat/completions"

        # Honor the campaign shape (mirror the MM stress path): N images + text
        # padded to the campaign input length, output = campaign output length
        # with ignore_eos. Without this every campaign sent the same
        # 1-image/128-token stub, so all MM serving rows measured one workload
        # mislabeled per campaign.
        in_len = self.config.input_lengths[0] if self.config.input_lengths else 1024
        out_len = self.config.output_lengths[0] if self.config.output_lengths else 128
        max_tokens = max(int(out_len), 16)
        n_img = self._images_per_request()
        img_tokens = n_img * (getattr(model, "vision_tokens_per_image", None) or self.PER_IMAGE_SOFT_TOKENS)
        max_len = get_max_model_len(getattr(self.config, "max_model_len", None))
        # Full campaign text with images ON TOP (not replacing text): MM = same
        # text as the text row + 4 images, so it is a strict superset and directly
        # comparable. Clamp only if it would overflow the context window.
        target_text = min(int(in_len), max_len - img_tokens - max_tokens - 128)
        target_text = max(self.MIN_MM_TEXT_TOKENS, target_text)
        if self._mm_tokenizer is None:
            from gbench.utils import safe_get_tokenizer
            tok_path = model.hf_model_id if (self.config.remote_endpoint or format == ModelFormat.GGUF) else model.get_model_path(format)
            self._mm_tokenizer = safe_get_tokenizer(tok_path, custom_tokenizer=self.config.tokenizer)
        prompt_text = self._build_mm_text(target_text, self._mm_tokenizer)

        def _img_url(path):
            if self.config.remote_endpoint:
                import base64
                with open(path, "rb") as f:
                    return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
            return f"file://{path}"

        semaphore = asyncio.Semaphore(max_concurrency)
        all_ttfts = []       # Time to first token (ms), one per request
        all_tpots = []       # Mean inter-token latency (ms), one per request
        all_itls = []        # Inter-token latencies (ms), flattened (back-compat)
        all_itl_blocks = []  # Inter-token latencies (ms) grouped per request
        all_e2els = []       # End-to-end latencies (ms), one per request
        all_out_lens = []    # Output tokens, one per request
        all_prompt_lens = [] # Server-measured input (prompt) tokens, one per request
        completed = 0
        empty = 0            # HTTP 200 but the model produced 0 content tokens
        failed = 0           # transport / HTTP errors
        error_reasons = []   # reason string per failed request
        total_output_tokens = 0

        async def send_mm_request(session, request_id, pbar=None):
            nonlocal completed, empty, failed, total_output_tokens
            # Rotate a window of n_img distinct images per request.
            paths = self._mm_image_paths
            image_urls = [
                _img_url(paths[(request_id * n_img + k) % len(paths)])
                for k in range(n_img)
            ]
            content = [{"type": "text", "text": prompt_text}]
            content += [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
            # No-think by default (see NOTHINK_SYSTEM); --thinking lets it reason.
            _messages = []
            if not getattr(self.config, "thinking", False):
                _messages.append({"role": "system", "content": self.NOTHINK_SYSTEM})
            _messages.append({"role": "user", "content": content})
            payload = {
                "model": model_path,
                "messages": _messages,
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "stream": True,
                # Ask the server for token-accurate usage in the final SSE chunk so
                # output length is real tokens (matching the text path's tokenizer
                # accounting), not a count of non-empty content deltas.
                "stream_options": {"include_usage": True},
            }
            payload.update(self.no_think_fields())   # portable no-think for the perf workload
            async with semaphore:
                start = time.perf_counter()
                content_times = []  # arrival time of each REAL content token
                try:
                    async with session.post(api_url, json=payload) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.debug(
                                f"MM request {request_id} HTTP {resp.status}: "
                                f"{error_text[:200]}"
                            )
                            failed += 1
                            error_reasons.append(f"http_{resp.status}")
                            return
                        # Count only chunks carrying real generated text, not the
                        # role-delta / finish control chunks (mirrors the text
                        # path at _run_lightweight_remote_benchmark), so TPOT and
                        # output throughput aren't biased by 1-2 empty SSE lines.
                        usage_tokens = None
                        usage_prompt = None
                        n_data_chunks = 0
                        finish_reason = None
                        choice_keys = set()
                        delta_keys = set()
                        async for raw_line in resp.content:
                            line = raw_line.decode("utf-8", errors="ignore").strip()
                            if not line.startswith("data:") or "[DONE]" in line:
                                continue
                            n_data_chunks += 1
                            try:
                                data = json.loads(line[5:].strip())
                                choices = data.get("choices") or []
                                if choices:
                                    choice_keys.update(choices[0].keys())
                                    _d = choices[0].get("delta") or {}
                                    delta_keys.update(_d.keys())
                                    if choices[0].get("finish_reason"):
                                        finish_reason = choices[0].get("finish_reason")
                                    # Count content OR reasoning/thinking tokens
                                    # (see text path): an all-reasoning reply is
                                    # real decode, not an empty completion.
                                    delta = (_d.get("content") or _d.get("reasoning")
                                             or _d.get("reasoning_content") or "")
                                else:
                                    delta = ""
                                # Final chunk (include_usage) carries token-accurate
                                # usage with an empty choices list.
                                usage = data.get("usage")
                                if usage and usage.get("completion_tokens") is not None:
                                    usage_tokens = int(usage["completion_tokens"])
                                if usage and usage.get("prompt_tokens") is not None:
                                    usage_prompt = int(usage["prompt_tokens"])
                            except Exception:
                                delta = ""
                            if delta:
                                content_times.append(time.perf_counter())

                    end = time.perf_counter()

                    if not content_times:
                        # HTTP 200 with 0 content chunks: an EMPTY completion.
                        # Diagnose WHY: completion_tokens>0 => tokens generated but
                        # missed (parse bug; keys show where); ==0 => genuinely empty.
                        logger.warning(
                            "EMPTY MM reply diag (req %s): usage.completion_tokens=%s "
                            "finish_reason=%s data_chunks=%s choice_keys=%s delta_keys=%s",
                            request_id, usage_tokens, finish_reason, n_data_chunks,
                            sorted(choice_keys), sorted(delta_keys))
                        empty += 1
                        return

                    ttft_ms = (content_times[0] - start) * 1000
                    e2el_ms = (end - start) * 1000
                    # Output length = server-reported tokens when available (accurate),
                    # else the count of non-empty content chunks (close proxy: vLLM
                    # streams ~1 token/chunk). TTFT/ITL timing still uses content_times.
                    num_tokens = usage_tokens if usage_tokens is not None else len(content_times)
                    total_output_tokens += num_tokens

                    all_ttfts.append(ttft_ms)
                    all_e2els.append(e2el_ms)
                    all_out_lens.append(num_tokens)
                    if usage_prompt is not None:
                        all_prompt_lens.append(usage_prompt)

                    # Per-request inter-token latencies (ms). Kept grouped for
                    # block-bootstrap CIs (ITL is autocorrelated within a request)
                    # and flattened for the pooled percentiles. TPOT = mean ITL
                    # for the request == (last-first)/(num_tokens-1).
                    req_itls_ms = [
                        (content_times[j] - content_times[j - 1]) * 1000
                        for j in range(1, len(content_times))
                    ]
                    all_itl_blocks.append(req_itls_ms)
                    if req_itls_ms:
                        all_itls.extend(req_itls_ms)
                        all_tpots.append(sum(req_itls_ms) / len(req_itls_ms))

                    completed += 1
                except Exception as e:
                    logger.debug(f"MM request {request_id} error: {e}")
                    failed += 1
                    error_reasons.append(
                        "timeout" if isinstance(e, asyncio.TimeoutError)
                        else type(e).__name__)
                finally:
                    if pbar:
                        pbar.update(1)

        connector = aiohttp.TCPConnector(limit=max_concurrency + 20)
        # aiohttp total = a TOTAL wall-clock cap on the whole request. Tunable via
        # --timeout (config.timeout); see config for the text-vs-MM semantics.
        timeout = aiohttp.ClientTimeout(total=getattr(self.config, "timeout", 600) or 600)
        from tqdm import tqdm
        with tqdm(total=num_prompts, desc="MM Serving") as pbar:
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout
            ) as session:
                overall_start = time.perf_counter()
                tasks = [
                    send_mm_request(session, i, pbar)
                    for i in range(num_prompts)
                ]
                await asyncio.gather(*tasks)
                overall_elapsed = time.perf_counter() - overall_start

        if not all_ttfts:
            raise RuntimeError(
                f"All {num_prompts} MM serving requests produced no tokens "
                f"(empty={empty}, error={failed})"
            )

        # Compute percentile metrics matching text serving output format
        import numpy as np
        ttft_arr = np.array(all_ttfts)
        e2el_arr = np.array(all_e2els)
        request_throughput = completed / overall_elapsed
        output_throughput = total_output_tokens / overall_elapsed

        reason_counts = {}
        for _rsn in error_reasons:
            reason_counts[_rsn] = reason_counts.get(_rsn, 0) + 1

        result = {
            "request_throughput": request_throughput,
            "output_throughput": output_throughput,
            "total_output_tokens": int(total_output_tokens),
            "completed": completed,
            "empty": empty,
            "failed": failed,
            "error_reasons": reason_counts,
            "total": num_prompts,
            "duration": overall_elapsed,
            # Actual MM prefill = clamped text + image placeholder tokens (what
            # drives TTFT), plus the nominal campaign input for reference. Small
            # campaigns (in=128) collapse text to the floor, so the real prefill is
            # dominated by the 4 images - report both so the row isn't mislabeled.
            "input_length": int(target_text + img_tokens),
            "input_length_nominal": int(in_len),
            # Server-measured ACTUAL prefill (usage.prompt_tokens) - the real
            # image cost on backends whose projector != the nominal 280/img
            # budget (e.g. Ollama GGUF ~56/img). None when the server omits usage.
            "input_length_actual": (int(np.median(all_prompt_lens))
                                    if all_prompt_lens else None),
            "mm_images": int(n_img),
            "output_length": int(out_len),
            # TTFT metrics
            "mean_ttft_ms": float(np.mean(ttft_arr)),
            "median_ttft_ms": float(np.median(ttft_arr)),
            "p99_ttft_ms": float(np.percentile(ttft_arr, 99)),
            # E2EL metrics
            "mean_e2el_ms": float(np.mean(e2el_arr)),
            "median_e2el_ms": float(np.median(e2el_arr)),
            "p99_e2el_ms": float(np.percentile(e2el_arr, 99)),
        }

        # TPOT metrics (need >1 token per request)
        if all_tpots:
            tpot_arr = np.array(all_tpots)
            result["mean_tpot_ms"] = float(np.mean(tpot_arr))
            result["median_tpot_ms"] = float(np.median(tpot_arr))
            result["p99_tpot_ms"] = float(np.percentile(tpot_arr, 99))

        # ITL metrics
        if all_itls:
            itl_arr = np.array(all_itls)
            result["mean_itl_ms"] = float(np.mean(itl_arr))
            result["median_itl_ms"] = float(np.median(itl_arr))
            result["p99_itl_ms"] = float(np.percentile(itl_arr, 99))

        # Raw per-request samples (ms) so the pooled aggregator can recompute
        # percentiles + bootstrap CIs across iterations exactly as for the text
        # path (never averaging per-iteration percentiles).
        result["raw_ttfts_ms"] = all_ttfts
        result["raw_itl_blocks_ms"] = all_itl_blocks
        result["raw_e2els_ms"] = all_e2els
        result["output_lens"] = all_out_lens

        return result

    async def _run_benchmark_async(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: int,
        input_requests: list[Any],
        tokenizer,
        request_rate: Optional[float] = None,
    ) -> dict:
        """Run serving benchmark using vLLM's benchmark() API directly.

        Args:
            model: Model configuration
            format: Model format
            batch_size: Max concurrency (closed-loop). Ignored when
                ``request_rate`` is given.
            num_prompts: Number of prompts
            input_requests: Pre-generated sample requests
            tokenizer: Tokenizer
            request_rate: If set, drive the benchmark OPEN-LOOP at this
                arrival rate (QPS) with Poisson arrivals and NO concurrency
                cap (max_concurrency=None) - the SOTA capacity-under-SLO
                setup. If None, keep the closed-loop behaviour: request rate
                from config, max_concurrency=batch_size.

        Returns:
            Dictionary with benchmark results
        """
        from gbench.utils import check_vllm_available, require_vllm_engine

        model_path = self._resolve_model_id(model, format)
        base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self.server_port}"

        if self.config.remote_endpoint or not check_vllm_available():
            if not self.config.remote_endpoint:
                require_vllm_engine("Serving benchmark execution")
            return await self._run_lightweight_remote_benchmark(
                base_url=base_url,
                model_id=model_path,
                input_requests=input_requests,
                batch_size=batch_size,
            )

        from vllm.benchmarks.serve import benchmark, TaskType
        api_url = f"{base_url}/v1/completions"

        # Open-loop (rate-driven, no concurrency cap) vs closed-loop (fixed
        # concurrency at the config/infinite rate).
        if request_rate is not None:
            rate = float(request_rate)
            max_conc = None
        else:
            rate = float("inf") if self.config.request_rate == "inf" else float(self.config.request_rate)
            max_conc = batch_size

        result = await benchmark(
            task_type=TaskType.GENERATION,
            endpoint_type="openai",  # Maps to async_request_openai_completions
            api_url=api_url,
            base_url=base_url,
            model_id=model_path,
            model_name=model_path,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=None,
            request_rate=rate,
            burstiness=1.0,
            max_concurrency=max_conc,
            disable_tqdm=False,
            num_warmups=0,  # We handle warmups ourselves
            profile=False,
            selected_percentile_metrics=["ttft", "tpot", "itl", "e2el"],
            # Per-iteration record only; the authoritative percentiles are
            # recomputed from POOLED raw samples in _aggregate_serving_results
            # (averaging per-iteration percentiles is statistically invalid).
            selected_percentiles=[50.0, 95.0, 99.0, 99.9],
            # RandomDataset prompts are gibberish token IDs; with EOS honored an
            # instruct model stops after a handful of tokens, so decode-heavy
            # (out=2048) / long-decode (out=8192) would measure TPOT over almost
            # no decode and output_throughput would collapse. vLLM's own random
            # benchmark CLI forces ignore_eos for exactly this reason; match it so
            # every request emits its requested output length and the decode /
            # goodput numbers are real and comparable across the 4 models.
            ignore_eos=True,
            goodput_config_dict={},
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
        )

        return result

    async def _run_lightweight_remote_benchmark(
        self,
        base_url: str,
        model_id: str,
        input_requests: list,
        batch_size: int,
    ) -> Any:
        """Pure-Python async streaming benchmark runner for remote HTTP endpoints."""
        import time
        import urllib.request
        import json
        import asyncio
        from dataclasses import dataclass

        @dataclass
        class MetricResult:
            metrics: dict

        url = base_url.rstrip("/")
        api_url = f"{url}/chat/completions" if url.endswith("/v1") else f"{url}/v1/chat/completions"
        if not api_url.startswith("http://") and not api_url.startswith("https://"):
            api_url = f"http://{api_url}"

        semaphore = asyncio.Semaphore(batch_size)
        all_ttfts = []       # ms, one per COMPLETED request
        all_itl_blocks = []  # ms, per-request inter-token latency lists
        all_e2els = []       # ms, one per COMPLETED request
        all_out_lens = []    # output tokens, one per COMPLETED request
        all_prompt_lens = [] # server-measured input (prompt) tokens, one per COMPLETED request
        completed = 0
        empty = 0            # HTTP 200 but the model produced 0 content tokens
        failed = 0           # transport / HTTP errors (exceptions / non-200)
        error_reasons = []   # reason string per failed request
        total_output_tokens = 0
        start_time = time.perf_counter()

        def _send_single_request(req):
            """Stream one request; return per-request samples or None on empty.

            Mirrors the MM path (_run_mm_serving_benchmark): TPOT/ITL come from
            inter-arrival gaps between REAL content chunks (tail-EXCLUDED - the
            numerator never includes the post-last-token usage chunk / [DONE] /
            connection close), and a request that produced <=1 content chunk
            yields NO TPOT/ITL sample (it has no inter-token interval) instead of
            a full-e2e-latency value masquerading as a per-token time. Output
            length is the server's token-accurate usage when present, else the
            content-chunk count.
            """
            # No-think by default: our system prompt overrides the backend's
            # default (reasoning-enabling) one. --thinking skips it (model reasons).
            _messages = []
            if not getattr(self.config, "thinking", False):
                _messages.append({"role": "system", "content": self.NOTHINK_SYSTEM})
            _messages.append({"role": "user", "content": req.prompt})
            _pl = {
                "model": model_id,
                "messages": _messages,
                "max_tokens": getattr(req, "expected_output_len", 128),
                "stream": True,
                # Force full-length decode where honored (a local vLLM engine);
                # a remote OpenAI-compat endpoint may drop it (Ollama does).
                "ignore_eos": True,
                "stream_options": {"include_usage": True},
            }
            _pl.update(self.no_think_fields())       # portable no-think for the perf workload
            payload = json.dumps(_pl).encode("utf-8")

            request = urllib.request.Request(
                api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            req_start = time.perf_counter()
            content_times = []   # arrival time of each REAL content token
            usage_tokens = None
            usage_prompt = None
            # Empty-reply diagnostics: why does a 200 yield 0 content chunks?
            n_data_chunks = 0
            finish_reason = None
            choice_keys = set()
            delta_keys = set()

            # urllib's timeout is per-socket-read: a STALL cap (fires if the
            # server sends no bytes for this long, e.g. a hung prefill), not a cap
            # on total decode. Tunable via --timeout (config.timeout); see config.
            req_timeout = getattr(self.config, "timeout", 600) or 600
            with urllib.request.urlopen(request, timeout=req_timeout) as resp:
                for line in resp:
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                        continue
                    n_data_chunks += 1
                    try:
                        data = json.loads(line_str[6:])
                        choices = data.get("choices") or []
                        if choices:
                            choice_keys.update(choices[0].keys())
                            _d = choices[0].get("delta") or {}
                            delta_keys.update(_d.keys())
                            if choices[0].get("finish_reason"):
                                finish_reason = choices[0].get("finish_reason")
                            # Count a generated token from the content OR the
                            # reasoning/thinking channel: a model that "thinks"
                            # (gemma-QAT via Ollama emits delta.reasoning; vLLM/
                            # DeepSeek use reasoning_content) streams real decode
                            # tokens there. Ignoring it mislabeled all-reasoning
                            # replies as EMPTY and inflated TTFT (it waited past the
                            # whole reasoning phase for the first 'content' token).
                            delta = (_d.get("content") or _d.get("reasoning")
                                     or _d.get("reasoning_content") or "")
                        else:
                            delta = ""
                        # Final chunk (include_usage) has empty choices + usage.
                        usage = data.get("usage")
                        if usage and usage.get("completion_tokens") is not None:
                            usage_tokens = int(usage["completion_tokens"])
                        if usage and usage.get("prompt_tokens") is not None:
                            usage_prompt = int(usage["prompt_tokens"])
                    except Exception:
                        delta = ""
                    if delta:
                        content_times.append(time.perf_counter())

            req_end = time.perf_counter()
            if not content_times:
                # HTTP 200 but 0 content chunks: an EMPTY completion. Diagnose WHY:
                # usage.completion_tokens>0 => server DID generate tokens we missed
                # (parse bug; delta_keys/choice_keys show where the text went);
                # ==0/None with finish_reason=stop => genuinely empty from the model.
                logger.warning(
                    "EMPTY text reply diag: usage.completion_tokens=%s "
                    "finish_reason=%s data_chunks=%s choice_keys=%s delta_keys=%s",
                    usage_tokens, finish_reason, n_data_chunks,
                    sorted(choice_keys), sorted(delta_keys))
                return None

            ttft_ms = (content_times[0] - req_start) * 1000
            e2el_ms = (req_end - req_start) * 1000
            num_tokens = usage_tokens if usage_tokens is not None else len(content_times)
            # Per-request inter-token latencies (ms); empty for a 1-token reply.
            req_itls_ms = [
                (content_times[j] - content_times[j - 1]) * 1000
                for j in range(1, len(content_times))
            ]
            return {
                "ttft_ms": ttft_ms,
                "itls_ms": req_itls_ms,
                "e2el_ms": e2el_ms,
                "num_tokens": num_tokens,
                "prompt_tokens": usage_prompt,
            }

        def _classify_error(e):
            import urllib.error
            import socket
            if isinstance(e, urllib.error.HTTPError):
                return f"http_{e.code}"
            reason = getattr(e, "reason", None)
            if (isinstance(e, (socket.timeout, TimeoutError))
                    or isinstance(reason, socket.timeout)
                    or "timed out" in str(e).lower()):
                return "timeout"
            return type(e).__name__

        async def _worker(req):
            nonlocal completed, empty, failed, total_output_tokens
            async with semaphore:
                try:
                    r = await asyncio.to_thread(_send_single_request, req)
                except Exception as e:
                    # Transport / HTTP error -> a real failure, with a reason.
                    failed += 1
                    error_reasons.append(_classify_error(e))
                    logger.warning(f"Remote benchmark request error: {e}")
                    return
                if r is None:
                    # HTTP 200 with 0 content tokens: an EMPTY completion, tracked
                    # separately from failures (it is not an error).
                    empty += 1
                    return
                all_ttfts.append(r["ttft_ms"])
                all_itl_blocks.append(r["itls_ms"])
                all_e2els.append(r["e2el_ms"])
                all_out_lens.append(r["num_tokens"])
                if r.get("prompt_tokens") is not None:
                    all_prompt_lens.append(r["prompt_tokens"])
                completed += 1
                total_output_tokens += r["num_tokens"]

        tasks = [_worker(req) for req in input_requests]
        await asyncio.gather(*tasks)

        dur = time.perf_counter() - start_time

        def p50(lst):
            return float(np.percentile(lst, 50)) if lst else 0.0
        def p99(lst):
            return float(np.percentile(lst, 99)) if lst else 0.0

        # Per-request TPOT = mean inter-token gap (only requests with >=2 tokens).
        tpots = [sum(b) / len(b) for b in all_itl_blocks if b]
        itls_flat = [v for b in all_itl_blocks for v in b]

        reason_counts = {}
        for _rsn in error_reasons:
            reason_counts[_rsn] = reason_counts.get(_rsn, 0) + 1

        metrics = {
            "request_throughput": completed / dur if dur > 0 else 0.0,
            "output_throughput": total_output_tokens / dur if dur > 0 else 0.0,
            "total_output_tokens": int(total_output_tokens),
            "completed": completed,
            "empty": empty,
            "failed": failed,
            "error_reasons": reason_counts,
            "total": len(input_requests),
            "duration": dur,
            # Raw per-request samples (ms) so the POOLED aggregator recomputes
            # percentiles + bootstrap CIs exactly as for the MM/local paths, and
            # so the text remote path shares ONE estimator + ONE aggregation with
            # them (no legacy scalar fallback -> no mean-as-p50 divergence, no
            # ITL=TPOT copies, and failed requests are counted, not silently lost).
            "raw_ttfts_ms": all_ttfts,
            "raw_itl_blocks_ms": all_itl_blocks,
            "raw_e2els_ms": all_e2els,
            "output_lens": all_out_lens,
            # Server-measured ACTUAL input tokens (usage.prompt_tokens), median
            # over completed requests; None when the server omits usage. Mirrors
            # the MM path so both rows can report actual (not just nominal) input.
            "input_length_actual": (int(np.median(all_prompt_lens))
                                    if all_prompt_lens else None),
        }
        if all_ttfts:
            metrics["mean_ttft_ms"] = float(np.mean(all_ttfts))
            metrics["median_ttft_ms"] = p50(all_ttfts)
            metrics["p99_ttft_ms"] = p99(all_ttfts)
            metrics["mean_e2el_ms"] = float(np.mean(all_e2els))
            metrics["median_e2el_ms"] = p50(all_e2els)
            metrics["p99_e2el_ms"] = p99(all_e2els)
        if tpots:
            metrics["mean_tpot_ms"] = float(np.mean(tpots))
            metrics["median_tpot_ms"] = p50(tpots)
            metrics["p99_tpot_ms"] = p99(tpots)
        if itls_flat:
            metrics["mean_itl_ms"] = float(np.mean(itls_flat))
            metrics["median_itl_ms"] = p50(itls_flat)
            metrics["p99_itl_ms"] = p99(itls_flat)

        return metrics

    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: Optional[str] = None,
        multimodal: bool = False,
        manage_server: bool = True,
    ) -> dict:
        """Run serving benchmark for a specific configuration (text only).

        For multimodal, use run_with_iterations() which routes to
        _run_mm_with_iterations() for real image requests.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)
            batch_size: Number of concurrent requests
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset to use (optional, ignored - we use RandomDataset)
            multimodal: If True, delegates to _run_mm_with_iterations()

        Returns:
            Dictionary with benchmark results

        Raises:
            RuntimeError: If benchmark execution fails
        """
        num_prompts = num_prompts or self.config.num_prompts

        # Route multimodal to the proper MM implementation
        if multimodal:
            return self._run_mm_with_iterations(
                model, format, batch_size, num_prompts, manage_server=manage_server
            )

        lm = self.config.log_manager

        # Generate output filename
        output_file = lm.get_serving_result_path(
            model.short_name, format.value, batch_size, multimodal
        )

        # Start vLLM server (unless the caller owns its lifecycle, e.g. run_all
        # reusing one server across batch sizes)
        if manage_server and not self.config.dry_run and not self.config.remote_endpoint:
            if not self._start_server(model, format):
                raise RuntimeError("Failed to start vLLM server")

        try:
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would run benchmark for {model.short_name}")
                return {"dry_run": True}

            # Log clear benchmark configuration banner
            logger.info(
                f"\n{'='*60}\n"
                f"  BENCHMARK CONFIGURATION\n"
                f"{'='*60}\n"
                f"  Type:           Serving\n"
                f"  Mode:           Text\n"
                f"  Model:          {model.short_name} ({format.value})\n"
                f"  Batch size:     {batch_size}\n"
                f"  Num prompts:    {num_prompts}\n"
                f"{'='*60}"
            )

            # Generate sample requests
            logger.info(f"Generating {num_prompts} sample requests...")
            input_requests, tokenizer = self._generate_sample_requests(
                model, format, num_prompts
            )

            # Run benchmark
            result = asyncio.run(self._run_benchmark_async(
                model, format, batch_size, num_prompts, input_requests, tokenizer
            ))

            # Save result to file
            result["model"] = model.short_name
            result["model_short"] = model.short_name
            result["model_name"] = model.name
            result["format"] = format.value
            result["batch_size"] = batch_size
            result["multimodal"] = multimodal
            result["output_token_throughput"] = result.get("output_throughput", result.get("output_token_throughput", 0.0))
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            # Log key metrics
            req_tput = result.get("request_throughput", "N/A")
            ttft = result.get("mean_ttft_ms", "N/A")
            tpot = result.get("mean_tpot_ms", "N/A")
            logger.info(
                f"Results: request_throughput={req_tput:.2f} req/s, "
                f"TTFT={ttft:.2f}ms, TPOT={tpot:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise RuntimeError(f"Serving benchmark failed: {e}")
        
        finally:
            # Cleanup server unless the caller owns its lifecycle
            if manage_server and not self.config.dry_run and not self.config.remote_endpoint:
                self._cleanup_server()

    def _get_output_path(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        multimodal: bool = False,
    ) -> Path:
        """Generate output path for benchmark results.
        
        Args:
            model: Model configuration
            format: Model format
            batch_size: Batch size used
            multimodal: Whether this is a multimodal benchmark
            
        Returns:
            Path to the output JSON file
        """
        lm = self.config.log_manager
        return lm.get_serving_result_path(
            model.short_name, format.value, batch_size, multimodal
        )

    def run_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: Optional[int] = None,
        dataset: Optional[str] = None,
        multimodal: bool = False,
        manage_server: bool = True,
    ) -> dict:
        """Run benchmark with multiple iterations and statistical analysis.

        For text: uses vLLM's benchmark() API with RandomDataset.
        For multimodal: uses aiohttp + SSE with real images via /v1/chat/completions.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)
            batch_size: Number of concurrent requests
            num_prompts: Number of prompts to test (optional)
            dataset: Dataset to use (optional)
            multimodal: If True, send real image requests

        Returns:
            Dictionary with aggregated results and statistics
        """
        # No single-iteration shortcut: even num_iterations==1 flows through the
        # pooled aggregator so every text-serving result carries pooled
        # percentiles + bootstrap CIs + tail_validity (uniform, self-describing
        # output). The pooled path handles one rep fine (no run-to-run CV).

        # Log clear benchmark configuration banner
        mode_str = "Multimodal" if multimodal else "Text"
        logger.info(
            f"\n{'='*60}\n"
            f"  BENCHMARK CONFIGURATION\n"
            f"{'='*60}\n"
            f"  Type:           Serving\n"
            f"  Mode:           {mode_str}\n"
            f"  Model:          {model.short_name} ({format.value})\n"
            f"  Batch size:     {batch_size}\n"
            f"  Num prompts:    {num_prompts or self.config.num_prompts}\n"
            f"{'='*60}"
        )

        logger.info(
            f"Running {self.config.num_iterations} iterations "
            f"(+{self.config.warmup_iterations} warmup)"
        )

        num_prompts = num_prompts or self.config.num_prompts

        if multimodal:
            return self._run_mm_with_iterations(
                model, format, batch_size, num_prompts, manage_server=manage_server
            )
        else:
            return self._run_text_with_iterations(
                model, format, batch_size, num_prompts, manage_server=manage_server
            )

    def _run_text_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: int,
        manage_server: bool = True,
    ) -> dict:
        """Run text serving benchmark with multiple iterations."""
        # Start server (unless the caller reuses one across batch sizes)
        if manage_server and not self.config.dry_run and not self.config.remote_endpoint:
            if not self._start_server(model, format):
                raise RuntimeError("Failed to start vLLM server")

        try:
            # Generate sample requests once (reused for all iterations)
            logger.info(f"Generating {num_prompts} sample requests...")
            input_requests, tokenizer = self._generate_sample_requests(
                model, format, num_prompts
            )

            # Warmup iterations
            warmup_total = self.config.warmup_iterations
            for i in range(warmup_total):
                progress = "█" * (i + 1) + "░" * (warmup_total - i - 1)
                logger.info(f"Warmup [{progress}] {i+1}/{warmup_total} starting...")
                iter_start = time.time()
                try:
                    warmup_reqs = input_requests[:min(2, len(input_requests))] if self.config.remote_endpoint else input_requests
                    asyncio.run(self._run_benchmark_async(
                        model, format, min(batch_size, len(warmup_reqs)), len(warmup_reqs), warmup_reqs, tokenizer
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  Warmup {i+1} completed in {elapsed:.1f}s")
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.warning(f"Warmup iteration {i+1} failed after {elapsed:.1f}s: {e}")

            # Actual benchmark iterations
            results = []
            iter_total = self.config.num_iterations
            for i in range(iter_total):
                progress = "█" * (i + 1) + "░" * (iter_total - i - 1)
                logger.info(f"Benchmark [{progress}] {i+1}/{iter_total} starting...")
                iter_start = time.time()
                try:
                    result = asyncio.run(self._run_benchmark_async(
                        model, format, batch_size, num_prompts, input_requests, tokenizer
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  Benchmark {i+1} completed in {elapsed:.1f}s")
                    results.append(result)
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.error(f"Iteration {i+1} failed after {elapsed:.1f}s: {e}")
                    continue

            return self._aggregate_serving_results(
                results, model, format, batch_size, multimodal=False
            )

        finally:
            if manage_server and not self.config.dry_run and not self.config.remote_endpoint:
                self._cleanup_server()

    def _assert_mm_images_processed(self, model_path: str, base_url: str,
                                    per_img_tokens: int = 280) -> None:
        """Preflight guard: prove the server actually INGESTS images before we
        trust any multimodal number. Sends one chat request with
        IMAGES_PER_REQUEST images + trivial text and checks that the server's
        reported ``usage.prompt_tokens`` reflects the image expansion. The floor
        is derived from the model's per-image token count (``per_img_tokens``,
        model-specific - 280 for gemma-4) at 0.5x, so it (a) still catches
        text-only drops AND partial ingestion (e.g. 1 of 4 images) on gemma, and
        (b) does not false-abort a non-gemma VLM with a different token count.
        Raises if not, so a mis-configured server (no --limit-mm-per-prompt /
        images silently dropped) can NEVER produce text-only results mislabeled
        as multimodal.

        Returns the SERVER-MEASURED per-image soft-token count (int) = the
        prompt-token gain per image the server actually charges, robust to
        --hf-overrides / --mm-processor-kwargs (e.g. gemma-4 served at 1120, not
        the 280 default). Returns None when the probe is skipped (dry_run / no
        images). Callers may use it to size + report the MM workload truthfully.
        """
        if self.config.dry_run or not self._mm_image_paths:
            return None
        import urllib.request
        n_img = self._images_per_request()

        def _u(p):
            if self.config.remote_endpoint:
                import base64
                with open(p, "rb") as f:
                    return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
            return f"file://{p}"

        # base_url may already include /v1 (remote endpoints like Ollama's
        # http://host:11434/v1); only append /v1 when it isn't there, else the
        # URL becomes /v1/v1/... -> 404.
        _b = base_url.rstrip("/")
        chat_url = f"{_b}/chat/completions" if _b.endswith("/v1") else f"{_b}/v1/chat/completions"
        _timeout = int(getattr(self.config, "mm_probe_timeout", None) or 60)

        def _prompt_tokens(content):
            payload = {"model": model_path, "max_tokens": 1, "stream": False,
                       "messages": [{"role": "user", "content": content}]}
            req = urllib.request.Request(
                chat_url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=_timeout) as resp:
                d = json.loads(resp.read())
            return int((d.get("usage") or {}).get("prompt_tokens", 0))

        TEXT = "Describe."
        urls = [_u(self._mm_image_paths[k % len(self._mm_image_paths)])
                for k in range(n_img)]
        try:
            text_ptok = _prompt_tokens([{"type": "text", "text": TEXT}])
            img_ptok = _prompt_tokens(
                [{"type": "text", "text": TEXT}]
                + [{"type": "image_url", "image_url": {"url": u}} for u in urls])
        except Exception as e:
            raise RuntimeError(f"MM preflight request failed: {e}") from e

        # Self-calibrating: the image(s) must add a meaningful number of prompt
        # tokens vs the SAME text alone. Backend-AGNOSTIC - does NOT assume gemma-4's
        # 280 tok/img: Ollama's GGUF vision uses a different (smaller ~90) count, so
        # an absolute 280-based floor false-flagged a working endpoint. A dropped
        # image adds ~0. (per_img_tokens is now advisory only.) NOTE: with n_img>1
        # this checks the AVERAGE per-image gain, so it won't catch a partial
        # ingestion of a subset - but --personal uses n_img=1 (all-or-nothing) and
        # the local vLLM path processes all images.
        MIN_PER_IMG = 16
        per_img = (img_ptok - text_ptok) / max(1, n_img)
        if per_img < MIN_PER_IMG:
            raise RuntimeError(
                f"MM PREFLIGHT FAILED: images added only ~{per_img:.0f} tok/img "
                f"(text-only={text_ptok}, +{n_img} image(s)={img_ptok}). Images are "
                f"NOT being processed end-to-end - refusing to report text-only "
                f"numbers as multimodal. Check the endpoint accepts images.")
        logger.info(f"MM preflight OK: images add ~{per_img:.0f} tok/img "
                    f"(text {text_ptok} -> {img_ptok} with {n_img} image(s)); "
                    f"vision path confirmed live.")
        return int(round(per_img))

    def _run_mm_with_iterations(
        self,
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        num_prompts: int,
        manage_server: bool = True,
    ) -> dict:
        """Run multimodal serving benchmark with multiple iterations.

        Sets up images, starts server with --allowed-local-media-path,
        runs warmup + benchmark iterations using aiohttp + SSE,
        then aggregates and cleans up. When ``manage_server`` is False the
        caller (run_all) has already set up the images and started one shared
        server, so this method only runs the iterations.
        """
        # Set up images + start the server only when we own the lifecycle.
        if manage_server:
            mm_media_path = self._setup_mm_images(num_images=50)
            if not self.config.dry_run and not self.config.remote_endpoint:
                if not self._start_server(
                    model, format, allowed_local_media_path=mm_media_path,
                    mm_limit_images=self._images_per_request(),
                ):
                    self._cleanup_mm_images()
                    raise RuntimeError("Failed to start vLLM server for MM serving")

        try:
            # Preflight: prove images are actually ingested before any measurement.
            base_url = self.config.remote_endpoint or f"http://127.0.0.1:{self.server_port}"
            self._assert_mm_images_processed(
                self._resolve_model_id(model, format), base_url,
                per_img_tokens=getattr(model, "vision_tokens_per_image", None) or self.PER_IMAGE_SOFT_TOKENS)

            # Warmup iterations
            warmup_total = self.config.warmup_iterations
            for i in range(warmup_total):
                progress = "█" * (i + 1) + "░" * (warmup_total - i - 1)
                logger.info(f"MM Warmup [{progress}] {i+1}/{warmup_total} starting...")
                iter_start = time.time()
                try:
                    # A remote endpoint has no local server to warm - a couple of
                    # requests suffice to prime connections/caches. Mirror the text
                    # path (which slices to 2) so remote MM warmup doesn't run the
                    # full campaign serially (~20 min under --personal on a laptop).
                    warmup_prompts = (min(2, num_prompts)
                                      if self.config.remote_endpoint else num_prompts)
                    asyncio.run(self._run_mm_serving_benchmark(
                        model, format,
                        max_concurrency=min(batch_size, warmup_prompts),
                        num_prompts=warmup_prompts,
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  MM Warmup {i+1} completed in {elapsed:.1f}s")
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.warning(f"MM Warmup {i+1} failed after {elapsed:.1f}s: {e}")

            # Actual benchmark iterations
            results = []
            iter_total = self.config.num_iterations
            for i in range(iter_total):
                progress = "█" * (i + 1) + "░" * (iter_total - i - 1)
                logger.info(f"MM Benchmark [{progress}] {i+1}/{iter_total} starting...")
                iter_start = time.time()
                try:
                    result = asyncio.run(self._run_mm_serving_benchmark(
                        model, format, max_concurrency=batch_size, num_prompts=num_prompts
                    ))
                    elapsed = time.time() - iter_start
                    logger.info(f"  MM Benchmark {i+1} completed in {elapsed:.1f}s")
                    results.append(result)
                except Exception as e:
                    elapsed = time.time() - iter_start
                    logger.error(f"MM Iteration {i+1} failed after {elapsed:.1f}s: {e}")
                    continue

            return self._aggregate_serving_results(
                results, model, format, batch_size, multimodal=True
            )

        finally:
            if manage_server:
                if not self.config.dry_run and not self.config.remote_endpoint:
                    self._cleanup_server()
                self._cleanup_mm_images()

    def _extract_raw_samples(self, result: dict) -> dict:
        """Normalize one iteration's result to raw per-request samples (ms).

        Handles the text path (vLLM benchmark() result: ``ttfts`` in seconds,
        ``itls`` as per-request lists in seconds) and the MM path (``raw_*``
        arrays already in ms). Per-request TPOT is the mean inter-token latency;
        e2el is the measured value (MM) or reconstructed as ttft + sum(itls)
        (text), which is vLLM's own e2el definition.

        Returns:
            {ttft_ms, itl_blocks_ms, e2el_ms, tpot_ms} - ttft/e2el/tpot are one
            value per request; itl_blocks_ms is one list per request.
        """
        if "raw_ttfts_ms" in result:  # MM path (already ms, only successes recorded)
            ttft_ms = [float(t) for t in result.get("raw_ttfts_ms", [])]
            itl_blocks = [[float(v) for v in b]
                          for b in result.get("raw_itl_blocks_ms", [])]
            e2el_ms = [float(v) for v in result.get("raw_e2els_ms", [])]
        else:  # text path (vLLM benchmark()), seconds -> ms
            # This vLLM's benchmark() records ttfts/itls/e2els for SUCCESSFUL
            # requests ONLY (serve.py `if outputs[i].success:`), so failed/aborted
            # requests are already absent here - they never reach the pooled
            # percentiles and do not inflate n. The ttft>0 guard is kept as a
            # defensive backstop (a 0-ttft success from some endpoint). Failure
            # COUNTS are surfaced in _aggregate_serving_results via the result's
            # `completed`/`failed` fields, not inferred from ttft.
            raw_ttfts = result.get("ttfts", [])
            raw_itls = result.get("itls", [])
            ttft_ms, itl_blocks = [], []
            for i, t in enumerate(raw_ttfts):
                if t is None or float(t) <= 0.0:
                    continue
                ttft_ms.append(float(t) * 1000.0)
                gaps = raw_itls[i] if i < len(raw_itls) else []
                itl_blocks.append([float(v) * 1000.0 for v in (gaps or [])])
            e2el_ms = []
        tpot_ms, recon_e2e = [], []
        for i, ttft in enumerate(ttft_ms):
            gaps = itl_blocks[i] if i < len(itl_blocks) else []
            if gaps:
                tpot_ms.append(sum(gaps) / len(gaps))
                recon_e2e.append(ttft + sum(gaps))
            else:
                recon_e2e.append(ttft)
        if not e2el_ms:
            e2el_ms = recon_e2e
        return {"ttft_ms": ttft_ms, "itl_blocks_ms": itl_blocks,
                "e2el_ms": e2el_ms, "tpot_ms": tpot_ms}

    def _aggregate_serving_results(
        self,
        results: list[dict],
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        multimodal: bool,
    ) -> dict:
        """Aggregate multi-iteration serving results the statistically-correct way.

        POOLS the raw per-request samples across all iterations and recomputes
        each percentile ONCE from the pooled distribution (averaging
        per-iteration percentiles is invalid - you cannot average a P99). Reports
        mean + P50/P95/P99/P99.9 with a bootstrap 95% CI on P99 (block bootstrap
        for autocorrelated ITL) and a tail_validity flag per percentile, so an
        under-sampled tail (inevitable at batch_size=1 / long outputs) is
        surfaced with its uncertainty, never passed off as a hard number.
        Throughput rates are averaged across reps (valid - independent rate
        estimates) with CV. Falls back to legacy scalar aggregation for the
        remote lightweight path, which carries no raw samples.
        """
        if not results:
            raise RuntimeError("All iterations failed")

        # Key PRESENCE, not truthiness: a raw-capable result whose sample arrays
        # are EMPTY (e.g. every request returned an empty completion -> raw_ttfts_ms
        # == []) must still enter the pooled path so the all-empty guard below flags
        # it as failed. Testing truthiness dropped such a result to the legacy scalar
        # path, which never sets failed=True, so an all-empty run read as a success.
        valid = [
            r for r in results
            if isinstance(r, dict) and ("ttfts" in r or "raw_ttfts_ms" in r)
        ]
        if not valid:
            return self._aggregate_serving_results_legacy(
                results, model, format, batch_size, multimodal
            )

        # Pool raw per-request samples across all iterations.
        pooled_ttft, pooled_e2e, pooled_tpot, itl_blocks = [], [], [], []
        pooled_out_lens = []
        per_iter_p99_ttft = []
        tot_completed = tot_failed = tot_empty = 0
        merged_reasons = {}
        for r in valid:
            raw = self._extract_raw_samples(r)
            pooled_ttft.extend(raw["ttft_ms"])
            pooled_e2e.extend(raw["e2el_ms"])
            pooled_tpot.extend(raw["tpot_ms"])
            itl_blocks.extend(raw["itl_blocks_ms"])
            pooled_out_lens.extend(r.get("output_lens") or [])
            if raw["ttft_ms"]:
                per_iter_p99_ttft.append(float(np.percentile(raw["ttft_ms"], 99)))
            tot_completed += int(r.get("completed", len(raw["ttft_ms"])) or 0)
            tot_failed += int(r.get("failed", 0) or 0)
            tot_empty += int(r.get("empty", 0) or 0)
            for _rsn, _cnt in (r.get("error_reasons") or {}).items():
                merged_reasons[_rsn] = merged_reasons.get(_rsn, 0) + int(_cnt)

        # Surface outcomes. Percentiles above are success-only (clean). EMPTY
        # completions (HTTP 200, 0 tokens - e.g. a backend that ignores ignore_eos
        # replying to a nonsense prompt) are NOT errors: they are counted and
        # reported separately and EXCLUDED from the failure rate. Only real errors
        # (exceptions / non-200) count as failures.
        offered = tot_completed + tot_empty + tot_failed
        fail_rate = (tot_failed / offered) if offered > 0 else 0.0
        empty_rate = (tot_empty / offered) if offered > 0 else 0.0
        if tot_failed or tot_empty:
            _bits = []
            if tot_failed:
                _bits.append(f"{tot_failed} error"
                             + (f" {merged_reasons}" if merged_reasons else ""))
            if tot_empty:
                _bits.append(f"{tot_empty} empty (0-token)")
            logger.warning(
                f"Serving {model.short_name} batch={batch_size} mm={multimodal}: "
                f"{tot_completed}/{offered} ok; " + ", ".join(_bits)
                + "; latency percentiles reflect successful requests only."
            )

        # Every request failed (no first token) across all iterations: there are
        # no latency samples, so this config genuinely failed. Surface it as a
        # failure (which the exit-code guard + reporting handle) rather than
        # writing a result with throughput but no latency keys.
        if not pooled_ttft:
            _why = ("all requests returned EMPTY (0-token) completions"
                    if tot_empty and not tot_failed
                    else "no successful requests")
            logger.error(
                f"Serving config produced no usable latency samples for "
                f"{model.short_name} batch={batch_size} mm={multimodal}: {_why} "
                f"(ok={tot_completed}, empty={tot_empty}, error={tot_failed})."
            )
            return {
                "failed": True,
                "error": f"no usable serving samples: {_why} "
                         f"(ok={tot_completed}, empty={tot_empty}, error={tot_failed})",
                "completed_requests": tot_completed,
                "empty_requests": tot_empty,
                "failed_requests": tot_failed,
                "error_reasons": merged_reasons,
                "model": model.short_name, "model_short": model.short_name,
                "model_name": model.name, "format": format.value,
                "batch_size": batch_size, "multimodal": multimodal,
            }

        agg: dict = {}
        # Authoritative pooled percentiles + bootstrap CIs + tail validity.
        agg.update(summarize_latency_metric("ttft", pooled_ttft, ci_percentile=99.0))
        agg.update(summarize_latency_metric("tpot", pooled_tpot, ci_percentile=99.0))
        agg.update(summarize_latency_metric("e2el", pooled_e2e, ci_percentile=99.0))
        agg.update(summarize_latency_metric("itl", None, blocks=itl_blocks,
                                            ci_percentile=99.0))

        # Back-compat scalar keys (downstream reporting reads these), sourced from
        # the POOLED distribution - NOT averaged across iterations.
        for m in ("ttft", "tpot", "itl", "e2el"):
            if f"{m}_p50_ms" in agg:
                agg[f"mean_{m}_ms"] = agg.get(f"{m}_mean_ms")
                agg[f"median_{m}_ms"] = agg.get(f"{m}_p50_ms")
                agg[f"p99_{m}_ms"] = agg.get(f"{m}_p99_ms")

        # Per-request DISPERSION (CV over the pooled samples), so the report's
        # high-variance guard has a real signal to fire on. Previously no tpot CV
        # key existed on any path, so the "indicative only ... TPOT CV" caption
        # was dead code; this is the genuine per-request spread the caption means.
        if pooled_tpot:
            agg["tpot_cv_percent"] = compute_statistics(pooled_tpot)["cv_percent"]
        if pooled_ttft:
            agg["ttft_cv_percent"] = compute_statistics(pooled_ttft)["cv_percent"]

        # Throughput: averaging RATES across independent reps IS valid; keep CV.
        for m in ("request_throughput", "output_throughput"):
            vals = [r.get(m) for r in valid if r.get(m) is not None]
            if vals:
                st = compute_statistics(vals)
                agg[m] = st["mean"]
                agg[f"{m}_mean"] = st["mean"]
                agg[f"{m}_std"] = st["std"]
                agg[f"{m}_cv_percent"] = st["cv_percent"]
                is_valid, msg = validate_repeatability(
                    {f"{m}_cv_percent": st["cv_percent"]}, m,
                    self.config.min_acceptable_cv_percent,
                )
                agg[f"{m}_repeatability_valid"] = is_valid
                logger.info(f"{m}: mean={st['mean']:.2f} {msg}")
        agg["output_token_throughput"] = agg.get("output_throughput", 0.0)

        # Run-to-run stability: CV of each iteration's own P99 TTFT (a summary
        # statistic across independent reps - averaging its CV IS valid).
        if len(per_iter_p99_ttft) >= 2:
            agg["p99_ttft_run_to_run_cv_percent"] = \
                compute_statistics(per_iter_p99_ttft)["cv_percent"]

        # Provenance / metadata.
        agg["num_iterations"] = len(valid)
        agg["pooled_sample_count"] = len(pooled_ttft)
        agg["completed_requests"] = tot_completed
        agg["empty_requests"] = tot_empty
        agg["failed_requests"] = tot_failed
        agg["offered_requests"] = offered
        agg["failure_rate"] = fail_rate
        agg["empty_rate"] = empty_rate
        agg["error_reasons"] = merged_reasons
        # Report the input shape so rows aren't unlabeled. For MM this is the ACTUAL
        # prefill (clamped text + 4 image placeholder tokens) carried on the raw
        # result; for text it's the campaign's nominal input length.
        if multimodal:
            _first = valid[0]
            agg["input_length"] = _first.get("input_length")
            agg["input_length_nominal"] = _first.get("input_length_nominal")
            agg["mm_images"] = _first.get("mm_images", self._images_per_request())
            # ACTUAL server-measured prefill (usage.prompt_tokens) when captured,
            # so remote/GGUF backends whose projector != the nominal 280/img
            # budget (Ollama ~56/img) aren't misreported as the nominal figure.
            _act = _first.get("input_length_actual")
            if isinstance(_act, (int, float)) and _act > 0:
                agg["input_length_actual"] = int(_act)
        else:
            agg["input_length"] = (self.config.input_lengths[0]
                                   if self.config.input_lengths else None)
            # Server-measured ACTUAL input tokens (usage.prompt_tokens) when the
            # endpoint reports usage - mirrors MM so text rows can show actual vs
            # the nominal campaign length too.
            _tact = valid[0].get("input_length_actual")
            if isinstance(_tact, (int, float)) and _tact > 0:
                agg["input_length_actual"] = int(_tact)
        agg["output_length"] = (self.config.output_lengths[0]
                                if self.config.output_lengths else None)
        # Realized decode length: a remote backend that ignores ignore_eos (e.g.
        # Ollama) stops at natural EOS, so the nominal output_length is NOT what
        # was generated - record the measured median so the JSON is honest.
        # Exclude non-positive lengths (the LOCAL vLLM benchmark() path records 0
        # for failed/aborted requests; remote/MM already append success-only), so a
        # high-failure local run doesn't drag the realized-length median toward 0.
        _real_lens = [x for x in pooled_out_lens
                      if isinstance(x, (int, float)) and x > 0]
        if _real_lens:
            agg["output_length_realized"] = int(np.median(_real_lens))
        # ignore_eos is only guaranteed honored by a LOCAL vLLM engine; a remote
        # OpenAI-compat endpoint may silently drop it (Ollama does), so the decode
        # shape is deterministic ONLY for local runs. Record requested-vs-honored
        # honestly instead of asserting the backend complied.
        _honored = not bool(self.config.remote_endpoint)
        agg["serving_shape_deterministic"] = _honored
        agg["ignore_eos"] = True                  # requested (back-compat key)
        agg["ignore_eos_honored"] = _honored
        agg["model"] = model.short_name
        agg["model_short"] = model.short_name
        agg["model_name"] = model.name
        agg["format"] = format.value
        agg["batch_size"] = batch_size
        agg["multimodal"] = multimodal

        # Honest headline log: median (solid) + P99 with CI + validity.
        tv = agg.get("ttft_tail_validity", {}).get("p99", {})
        logger.info(
            f"Pooled {len(pooled_ttft)} req over {len(valid)} iter | "
            f"TTFT P50={agg.get('ttft_p50_ms', float('nan')):.1f}ms "
            f"P99={agg.get('ttft_p99_ms', float('nan')):.1f}ms "
            f"[{agg.get('ttft_p99_ci_low_ms', float('nan')):.1f},"
            f"{agg.get('ttft_p99_ci_high_ms', float('nan')):.1f}] "
            f"(P99 valid={tv.get('valid')}, need n>={tv.get('required_n')}) | "
            f"ITL P50={agg.get('itl_p50_ms', float('nan')):.1f}ms "
            f"P99={agg.get('itl_p99_ms', float('nan')):.1f}ms"
        )

        output_file = self._get_output_path(model, format, batch_size, multimodal)
        with open(output_file, 'w') as f:
            json.dump(agg, f, indent=2)
        logger.info(f"Results saved to: {output_file}")
        return agg

    def _aggregate_serving_results_legacy(
        self,
        results: list[dict],
        model: ModelConfig,
        format: ModelFormat,
        batch_size: int,
        multimodal: bool,
    ) -> dict:
        """Scalar aggregation for results without raw samples (remote lightweight
        path). Averages per-iteration scalar metrics; kept only for that path."""
        if not results:
            raise RuntimeError("All iterations failed")

        serving_metrics = [
            "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
            "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
            "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
            "request_throughput", "output_throughput",
        ]
        aggregated = aggregate_benchmark_results(results, serving_metrics)

        for metric in ["request_throughput", "mean_ttft_ms"]:
            is_valid, msg = validate_repeatability(
                aggregated, metric, self.config.min_acceptable_cv_percent
            )
            logger.info(f"{metric}: {msg}")
            aggregated[f"{metric}_repeatability_valid"] = is_valid

        for metric in ["request_throughput", "mean_ttft_ms", "mean_tpot_ms"]:
            summary = format_statistics_summary(aggregated, metric)
            logger.info(summary)

        aggregated["model"] = model.short_name
        aggregated["model_short"] = model.short_name
        aggregated["model_name"] = model.name
        aggregated["format"] = format.value
        aggregated["batch_size"] = batch_size
        aggregated["multimodal"] = multimodal
        aggregated["output_token_throughput"] = aggregated.get(
            "output_throughput_mean", aggregated.get("output_throughput", 0.0))
        output_file = self._get_output_path(model, format, batch_size, multimodal)
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        logger.info(f"Results saved to: {output_file}")
        return aggregated


    def run_all(
        self,
        model: ModelConfig,
        format: ModelFormat,
        multimodal: bool = False,
    ) -> list[dict]:
        """Run all serving benchmark configurations for a model.

        Args:
            model: Model configuration
            format: Model format
            multimodal: If True, use random-mm dataset for image+text

        Returns:
            List of aggregated result dictionaries (includes failed runs with failed=True)
        """
        results = []
        configs = self.config.get_serving_configs()

        # Start ONE server and reuse it across all batch sizes. Batch size only
        # changes client concurrency, not the server config, so reloading the
        # model per batch size was pure waste (~3.5min each for a 26B). Safe now
        # that prefix caching is disabled - no cache leaks between configs. Only
        # when we actually own a local server; remote/dry-run keep the inner
        # methods managing (which no-op for those paths).
        manage_here = not self.config.dry_run and not self.config.remote_endpoint
        mm_media_path = None
        server_up = False
        try:
            if manage_here:
                if multimodal:
                    mm_media_path = self._setup_mm_images(num_images=50)
                if not self._start_server(
                    model, format, allowed_local_media_path=mm_media_path,
                    mm_limit_images=self._images_per_request() if multimodal else 1,
                ):
                    if multimodal:
                        self._cleanup_mm_images()
                    raise RuntimeError("Failed to start vLLM server")
                server_up = True

            for cfg in configs:
                try:
                    # Use iteration-aware runner; when we hold the shared server
                    # tell it not to start/stop its own.
                    result = self.run_with_iterations(
                        model,
                        format,
                        batch_size=cfg["batch_size"],
                        num_prompts=cfg["num_prompts"],
                        dataset=cfg.get("dataset"),
                        multimodal=multimodal,
                        manage_server=not manage_here,
                    )
                    # Add config metadata to result
                    result["batch_size"] = cfg["batch_size"]
                    result["multimodal"] = multimodal
                    # P99 reliability: True when this config was sampled below the
                    # stable-P99 threshold (long-output campaigns) - see
                    # BenchmarkConfig._serving_num_prompts.
                    result["low_confidence"] = cfg.get("low_confidence", False)
                    result["num_prompts"] = cfg.get("num_prompts")
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Failed benchmark for {model.short_name}: {e}"
                    )
                    # Track failure in results
                    results.append({
                        "failed": True,
                        "error": str(e),
                        "model": model.short_name,
                        "format": format.value,
                        "batch_size": cfg["batch_size"],
                        "multimodal": multimodal,
                    })
        finally:
            if server_up:
                self._cleanup_server()
            if manage_here and multimodal:
                self._cleanup_mm_images()

        return results

    def _start_server(
        self,
        model: ModelConfig,
        format: ModelFormat,
        allowed_local_media_path: Optional[str] = None,
        mm_limit_images: int = 1,
    ) -> bool:
        """Start vLLM server for serving benchmarks.

        Args:
            model: Model configuration
            format: Model format
            allowed_local_media_path: If set, allow loading local media from this path

        Returns:
            True if server started successfully, False otherwise
        """
        if self.server_process is not None:
            logger.warning("Server already running, cleaning up first")
            self._cleanup_server()

        logger.info(f"Starting vLLM server for {model.name} ({format.value})")

        # Build server command
        cmd = [
            "vllm",
            "serve",
            model.get_model_path(format),
            "--port",
            str(self.server_port),
        ]

        # Add performance optimization flags
        gpu_mem = self.config.gpu_memory_utilization
        if gpu_mem:
            cmd.extend([
                "--gpu-memory-utilization",
                str(gpu_mem),
            ])
        
        if self.config.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")

        # Disable automatic prefix caching for benchmarking. The synthetic
        # prompts are fixed-seed and replayed across iterations, batch sizes
        # and stress rate-points; with APC on (vLLM's default) the repeats hit
        # the cache and TTFT collapses to a cache-hit value - a measurement
        # artifact, not real prefill. Off = every request pays real prefill
        # (clean latency) AND the server is safe to reuse across configs.
        cmd.append("--no-enable-prefix-caching")

        if self.config.max_num_batched_tokens:
            cmd.extend([
                "--max-num-batched-tokens",
                str(self.config.max_num_batched_tokens),
            ])
        
        # Limit max concurrent sequences for uniform memory usage
        if self.config.max_num_seqs:
            cmd.extend([
                "--max-num-seqs",
                str(self.config.max_num_seqs),
            ])
        
        # Multi-GPU: tensor parallel
        if self.config.tensor_parallel_size and self.config.tensor_parallel_size > 1:
            cmd.extend([
                "--tensor-parallel-size",
                str(self.config.tensor_parallel_size),
            ])

        # GGUF models need explicit tokenizer from HF model ID
        if format == ModelFormat.GGUF:
            cmd.extend(["--tokenizer", model.hf_model_id])
        
        # Uniform context length for fair comparison (overridable via
        # --max-model-len for long-context campaigns)
        max_model_len = get_max_model_len(self.config.max_model_len)
        if max_model_len:
            logger.info(f"Setting max_model_len={max_model_len} for uniform benchmark comparison")
            cmd.extend(["--max-model-len", str(max_model_len)])

        # Allow loading local media files (needed for multimodal benchmarks)
        if allowed_local_media_path:
            cmd.extend(["--allowed-local-media-path", allowed_local_media_path])

        # Allow multiple images per request (default vLLM cap is 1). Needed for
        # multimodal stress, which sends IMAGES_PER_REQUEST images per prompt.
        # vLLM expects a JSON dict value, e.g. {"image": 4} - NOT image=4.
        if mm_limit_images and mm_limit_images > 1:
            cmd.extend(["--limit-mm-per-prompt", json.dumps({"image": mm_limit_images})])
            # Disable the multimodal preprocessor cache so repeated images are
            # re-preprocessed every request - consistent with APC being off, so
            # host-side image preprocessing is actually measured rather than
            # served from a 4GB cache after the first hit.
            cmd.extend(["--mm-processor-cache-gb", "0"])

        if getattr(self.config, "eval_max_soft_tokens", None) is not None:
            cmd.extend([
                "--hf-overrides",
                json.dumps({"max_soft_tokens": self.config.eval_max_soft_tokens}),
            ])

        try:
            # Redirect server output to log file to avoid pipe buffer deadlock.
            # When using PIPE, the OS buffer (~64KB) can fill up during model
            # loading and CUDA graph capture, causing the server to block.
            # Writing to a file avoids this issue while preserving logs.
            lm = self.config.log_manager
            server_log_path = lm.get_server_log_path(model.short_name, format.value)
            logger.info(f"Server logs will be written to: {server_log_path}")
            
            self._server_log_file = open(server_log_path, "a")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=self._server_log_file,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                start_new_session=True,  # New process group for clean killpg
            )


            # Set timeout based on model size and format
            # - 27b: ~2-3 min for torch.compile + CUDA graphs on first run
            # - GGUF: slower loading (~5-6 min)
            # - Others: 2 min is usually sufficient
            timeout = get_server_timeout(model.total_params_b, model.is_moe)
            if format == ModelFormat.GGUF:
                timeout = max(timeout, 600)
            
            # Wait for server to be ready
            if self._wait_for_server_ready(timeout):
                logger.info(f"vLLM server started successfully on port {self.server_port}")
                from gbench.utils import verify_endpoint_functional
                is_up, err_msg, max_model_len = verify_endpoint_functional(f"http://127.0.0.1:{self.server_port}")
                if not is_up:
                    logger.error(f"❌ Local server on port {self.server_port} is not functional: {err_msg}")
                    self._cleanup_server()
                    return False
                logger.info(f"✅ Local server on port {self.server_port} is functional and answering requests.")
                if max_model_len:
                    logger.info(f"   Server reported max_model_len: {max_model_len} tokens")
                return True
            else:
                logger.error(f"Server failed to become ready. Check logs at: {server_log_path}")
                self._cleanup_server()
                return False

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self._cleanup_server()
            return False

    def _wait_for_server_ready(self, timeout: int = 120) -> bool:
        """Wait for vLLM server to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if server is ready, False if timeout
        """
        url = f"http://127.0.0.1:{self.server_port}/health"
        start_time = time.time()

        logger.info("Waiting for server to be ready...")

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass

            # Check if process died
            if self.server_process and self.server_process.poll() is not None:
                logger.error("Server process died during startup")
                # Capture and log server output
                stdout, stderr = self.server_process.communicate()
                if stdout:
                    logger.error(f"Server STDOUT:\n{stdout[-2000:]}")
                if stderr:
                    logger.error(f"Server STDERR:\n{stderr[-2000:]}")
                return False

            time.sleep(2)

        return False

    def _cleanup_server(self):
        """Clean up vLLM server process.
        
        Note: After killing the process, we wait for GPU memory to be 
        reclaimed by the CUDA driver. This typically takes 5-15 seconds.
        """
        if self.server_process is None:
            return

        logger.info("Cleaning up vLLM server...")

        try:
            # Kill the entire process group (API server + EngineCore + workers)
            # vLLM spawns EngineCore as a grandchild process that terminate()
            # cannot reach, leaving zombie GPU-holding processes.
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, forcing kill")
                os.killpg(pgid, signal.SIGKILL)
                self.server_process.wait(timeout=5)

            logger.info("vLLM server stopped")
            
            # Wait for GPU memory to be reclaimed by CUDA driver
            # Use active polling instead of fixed wait for reliability
            self._wait_for_gpu_memory()
            
        except Exception as e:
            logger.error(f"Error cleaning up server: {e}")
        finally:
            self.server_process = None
            # Close the log file handle
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass
                self._server_log_file = None
