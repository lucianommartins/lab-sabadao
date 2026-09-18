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

"""Model registry with auto-enrichment from HuggingFace config.json.

Models are defined as slim entries (short_name, hf_model_id) with optional
overrides. At registry init, each model's config.json is fetched from the
HuggingFace cache to derive: total params, MoE topology, multimodal support,
category, and max context length.

Local paths follow a convention: models/gbench/<short_name> for third-party
models, models/<short_name> for Gemma4 models.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional

from huggingface_hub import hf_hub_download, model_info

logger = logging.getLogger(__name__)

# Workspace root (parent of gbench repo)
# models.py is at gbench/gbench/core/models.py → core/ → gbench/ → gbench/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Get models directory - can be overridden with GEMMA_MODELS_DIR env var
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR = Path(os.getenv("GEMMA_MODELS_DIR", DEFAULT_MODELS_DIR))




def _clean_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()




class ModelFormat(Enum):
    """Supported model formats."""

    HF = "hf"
    GGUF = "gguf"
    REMOTE = "remote-endpoint"


class ModelCategory(Enum):
    """Model categories."""

    TEXT = "text"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"


class Priority(Enum):
    """Benchmark priority levels."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


@dataclass
class ModelConfig:
    """Configuration for a specific model variant.

    Core fields (short_name, hf_model_id) are required. Everything else
    is either auto-derived from config.json or uses sensible defaults.
    """

    # ── Required ─────────────────────────────────────────────
    short_name: str
    hf_model_id: str

    # ── Auto-derived from config.json (populated by _enrich) ─
    total_params_b: float = 0.0
    is_moe: bool = False
    num_experts: int = 0
    num_active_experts: int = 0
    supports_multimodal: bool = False
    vision_tokens_per_image: int = 280   # per-image placeholder/soft tokens; the
                                         # gemma-4 default, overridden from config
                                         # when derivable (never 0/None - a 0 would
                                         # size the MM text budget to ~full context
                                         # and overflow it).
    category: ModelCategory = ModelCategory.TEXT
    max_context_length: int = 128_000

    # ── Optional overrides ───────────────────────────────────
    name: str = ""  # Human-readable name (defaults to short_name)
    local_path: Optional[str] = None
    gguf_model_id: Optional[str] = None
    gguf_file: Optional[str] = None
    priority: Priority = Priority.P0
    supports_audio: bool = False

    def __post_init__(self):
        if not self.name:
            self.name = self.short_name

    def get_model_path(self, format: ModelFormat) -> str:
        """Get model path for specified format.

        For HF models, returns local_path (resolved to absolute) if set,
        otherwise HuggingFace Hub model ID.
        For GGUF models, returns absolute path to local file.
        """
        if format == ModelFormat.REMOTE:
            return self.hf_model_id
        elif format == ModelFormat.HF:
            if self.local_path:
                resolved = PROJECT_ROOT / self.local_path
                if resolved.exists():
                    return str(resolved)
            return self.hf_model_id
        elif format == ModelFormat.GGUF:
            if not self.gguf_file:
                raise ValueError(
                    f"GGUF format not available for {self.short_name}"
                )

            gguf_path = MODELS_DIR / self.gguf_file

            if not gguf_path.exists():
                raise FileNotFoundError(
                    f"GGUF model file not found: {gguf_path}\n\n"
                    f"Download it with:\n"
                    f"  ./utils/download_gemma.sh {self.short_name}\n\n"
                    f"Or set GEMMA_MODELS_DIR environment variable.\n"
                    f"Current GEMMA_MODELS_DIR: {MODELS_DIR}"
                )

            return str(gguf_path)
        raise ValueError(f"Unknown format: {format}")


# ── Auto-enrichment from HuggingFace config.json ────────────

def _enrich_from_config(model: ModelConfig) -> None:
    """Populate auto-derived fields from cached HuggingFace config.json.

    Uses huggingface_hub to locate the cached config.json. If not cached,
    downloads it (~1KB). Sets: total_params_b, is_moe, num_experts,
    num_active_experts, supports_multimodal, category, max_context_length.
    """
    try:
        if model.hf_model_id.startswith("gs://"):
            temp_dir = tempfile.mkdtemp(prefix="gbench_gcs_config_")
            config_path = os.path.join(temp_dir, "config.json")
            gcs_url = model.hf_model_id.rstrip("/") + "/config.json"
            try:
                subprocess.run(
                    ["gcloud", "storage", "cp", gcs_url, config_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                with open(config_path) as f:
                    config = json.load(f)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        elif ":" in model.hf_model_id or "/" not in model.hf_model_id:
            # Model ID is an Ollama tag (e.g. gemma4-qat:4b) or short name, not an HF repo ID
            return
        else:
            try:
                # First try local cache only to avoid network spam/401s on gated models
                path = hf_hub_download(model.hf_model_id, "config.json", local_files_only=True)
            except Exception:
                try:
                    path = hf_hub_download(model.hf_model_id, "config.json")
                except Exception as dl_err:
                    logger.debug(f"Could not fetch config.json for {model.hf_model_id}: {dl_err}")
                    return
            with open(path) as f:
                config = json.load(f)
    except Exception as e:
        logger.debug(
            f"Could not fetch config.json for {model.hf_model_id}: {e}. "
            f"Using defaults."
        )
        return

    tc = config.get("text_config", config)

    # ── Architecture fields ──────────────────────────────────
    hidden = tc.get("hidden_size", 0)
    layers = tc.get("num_hidden_layers", 0)
    heads = tc.get("num_attention_heads", 0)
    kv_heads = tc.get("num_key_value_heads", heads)
    intermediate = tc.get("intermediate_size", 0)
    vocab = tc.get("vocab_size", config.get("vocab_size", 0))

    # MoE detection (field names vary across vendors)
    num_experts = (
        tc.get("num_experts")
        or tc.get("num_local_experts")
        or tc.get("n_routed_experts")
        or 0
    )
    num_active = (
        tc.get("num_experts_per_tok")
        or tc.get("top_k_experts")
        or 0
    )
    moe_intermediate = tc.get("moe_intermediate_size", 0)
    n_shared = tc.get("n_shared_experts", 0)
    is_moe = num_experts > 0

    model.is_moe = is_moe
    model.num_experts = num_experts
    model.num_active_experts = num_active

    # ── Multimodal / audio detection ─────────────────────────
    arch_str = str(config.get("architectures", []))
    # Guard against a present-but-null modality block: gemma-4 ships the audio
    # (and, on text-only variants, vision) slot as a KEY whose VALUE is null when
    # the checkpoint has no tower for it (e.g. gemma-4-26B has "audio_config": null
    # -> no audio tower). A bare `"x_config" in config` key-presence check would
    # read that as capable; test the value is non-null instead.
    has_vision = (
        config.get("vision_config") is not None
        or "ConditionalGeneration" in arch_str
    )
    has_audio = config.get("audio_config") is not None
    # supports_multimodal drives the IMAGE-based MM benchmarks, so it keys off
    # vision. Audio is tracked separately (previously never populated).
    model.supports_multimodal = has_vision
    model.supports_audio = has_audio
    # Per-image token count for MM request sizing + the MM preflight guard. These
    # fields live TOP-LEVEL or under vision_config (NOT in text_config, so read
    # from `config`/`vision_config`, never `tc`). Priority list covers the gemma
    # variants (26B/E2B/E4B/31B: vision_soft_tokens_per_image; 12B:
    # vision_config.num_soft_tokens; gemma-3: mm_tokens_per_image). Keep the
    # conservative default if none is present (dynamic-token VLMs like Qwen2-VL
    # expose no fixed count) - never 0/None.
    _vcfg = config.get("vision_config") or {}
    _vtoks = (config.get("vision_soft_tokens_per_image")
              or (_vcfg.get("num_soft_tokens") if isinstance(_vcfg, dict) else None)
              or config.get("mm_tokens_per_image"))
    if isinstance(_vtoks, int) and _vtoks > 0:
        model.vision_tokens_per_image = _vtoks
    model.category = (
        ModelCategory.MULTIMODAL
        if (has_vision or has_audio)
        else ModelCategory.TEXT
    )

    # ── Max context length ───────────────────────────────────
    max_pos = tc.get(
        "max_position_embeddings",
        config.get("max_position_embeddings", 128_000),
    )
    model.max_context_length = max_pos

    # ── Total params estimation ──────────────────────────────
    # Prefer safetensors metadata (exact), fall back to architecture estimate.
    st_total = None
    if not model.hf_model_id.startswith("gs://"):
        try:
            info = model_info(model.hf_model_id)
            if info.safetensors and info.safetensors.total:
                st_total = info.safetensors.total / 1e9
        except Exception:
            pass

    if st_total:
        model.total_params_b = round(st_total, 1)
    else:
        # Estimate from architecture
        head_dim = hidden // heads if heads else 0
        embed_params = vocab * hidden
        attn_per_layer = (
            hidden * heads * head_dim
            + hidden * kv_heads * head_dim * 2
            + heads * head_dim * hidden
        )
        if is_moe:
            expert_inter = moe_intermediate if moe_intermediate else intermediate
            expert_ffn = 3 * hidden * expert_inter
            ffn_per_layer = num_experts * expert_ffn + n_shared * expert_ffn
        else:
            ffn_per_layer = 3 * hidden * intermediate

        est = (embed_params + layers * (attn_per_layer + ffn_per_layer)) / 1e9
        model.total_params_b = round(est, 1)

    # ── Name fallback ────────────────────────────────────────
    if not model.name:
        model.name = model.short_name


def _hf_id_to_short_name(hf_model_id: str) -> str:
    """Derive a model ID from a HuggingFace model ID.

    Preserves original casing from HuggingFace.
    E.g. 'google/gemma-4-E4B-it'  → 'gemma-4-E4B-it'
         'google/gemma-4-31B-it'  → 'gemma-4-31B-it'
    """
    return hf_model_id.rstrip("/").split("/")[-1]


def enrich_metadata_from(model: ModelConfig, hf_id: str) -> bool:
    """Populate a model's DERIVED metadata (params, MoE, vision, context) from a
    DIFFERENT HuggingFace id, without changing its identity (short_name/hf_model_id).

    For remote endpoints whose served id can't be introspected - e.g. an Ollama
    tag like ``gemma4-qat:4b`` - pass the user's ``--tokenizer`` HF id (e.g.
    ``google/gemma-4-E4B-it``) so the report shows real params + modality instead
    of the 0B/dense/text defaults. Returns True if enrichment resolved params.
    """
    if not hf_id or ":" in hf_id or "/" not in hf_id:
        return False
    probe = ModelConfig(short_name=model.short_name, hf_model_id=hf_id)
    try:
        _enrich_from_config(probe)
    except Exception:
        return False
    if probe.total_params_b and probe.total_params_b > 0:
        model.total_params_b = probe.total_params_b
        model.is_moe = probe.is_moe
        model.num_experts = probe.num_experts
        model.num_active_experts = probe.num_active_experts
        model.supports_multimodal = probe.supports_multimodal
        model.vision_tokens_per_image = probe.vision_tokens_per_image
        model.category = probe.category
        model.max_context_length = probe.max_context_length
        return True
    return False


# ── Slim model definitions ──────────────────────────────────
# Only hf_model_id is required. short_name is derived automatically.
# Optional GGUF fields for models that support quantized formats.

# No built-in model table. Every model is resolved on demand from its --models
# id via ModelRegistry.get() -> register_hf_model() -> _enrich_from_config(),
# which derives all metadata (params, MoE, multimodal, context) from the real
# config.json - a single source of truth, and no stale hardcoded metadata.
# Consequences of an empty default set: running `gbench` with no --models exits
# "No models selected", bare short-names (without the google/ prefix) no longer
# resolve, and the web service's model list is empty until models are staged.
_MODEL_DEFS: list[dict] = []


def _build_models() -> list[ModelConfig]:
    """Build ModelConfig list from slim definitions + HF enrichment."""
    models = []
    for defn in _MODEL_DEFS:
        hf_id = defn["hf_model_id"]
        short = _hf_id_to_short_name(hf_id)

        model = ModelConfig(
            short_name=short,
            local_path=f"models/{short}",
            **defn,
        )
        if not model.total_params_b:
            _enrich_from_config(model)
        models.append(model)
    return models


# Build on import - config.json files are tiny and cached by huggingface_hub
MODELS = _build_models()


class ModelRegistry:
    """Registry for accessing model configurations.

    Built-in models (Gemma3/4) are pre-loaded at import time.
    Any other HuggingFace model can be dynamically registered via
    ``register_hf_model()`` or by passing an HF model ID to ``get()``.
    """

    def __init__(self):
        """Initialize the model registry.

        Lookups are case-insensitive - 'gemma-4-E2B-it' and 'gemma-4-e2b-it'
        both resolve to the same model.
        """
        self._models: dict[str, ModelConfig] = {}
        for model in MODELS:
            self._models[model.short_name.lower()] = model
        self._by_category: dict[ModelCategory, list[ModelConfig]] = {}
        for model in MODELS:
            self._by_category.setdefault(model.category, []).append(model)

    def register_hf_model(self, hf_model_id: str) -> ModelConfig:
        """Dynamically register a model from its HuggingFace model ID.

        Creates a ModelConfig with auto-derived fields from config.json.
        If already registered, returns the existing entry.

        Args:
            hf_model_id: HuggingFace model ID (e.g. 'google/gemma-4-31b-it').

        Returns:
            The registered ModelConfig.
        """
        # Check if already registered (by HF ID)
        for m in self._models.values():
            if m.hf_model_id == hf_model_id:
                return m

        short = _hf_id_to_short_name(hf_model_id)
        key = short.lower()

        # Avoid short_name collision
        if key in self._models:
            return self._models[key]

        model = ModelConfig(
            short_name=short,
            hf_model_id=hf_model_id,
            local_path=f"models/gbench/{short}",
        )
        _enrich_from_config(model)

        self._models[key] = model
        MODELS.append(model)
        self._by_category.setdefault(model.category, []).append(model)

        logger.info(
            f"Registered {short} from {hf_model_id} "
            f"({model.total_params_b:.1f}B, "
            f"{'MoE' if model.is_moe else 'dense'}, "
            f"{'multimodal' if model.supports_multimodal else 'text'})"
        )
        return model

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get model by short name (case-insensitive) or HuggingFace model ID.

        If not found and the name contains '/', automatically registers
        it as a new model from HuggingFace.
        """
        model = self._models.get(name.lower())
        if model is not None:
            return model

        # Auto-register if it looks like an HF model ID
        if "/" in name:
            return self.register_hf_model(name)

        return None

    def get_by_category(
        self, category: ModelCategory
    ) -> list[ModelConfig]:
        """Get all models in a category."""
        return self._by_category.get(category, [])

    def get_by_priority(self, priority: Priority) -> list[ModelConfig]:
        """Get all models at a priority level."""
        return [m for m in self._models.values() if m.priority == priority]

    def list_all(self) -> list[ModelConfig]:
        """Get all registered models."""
        return list(self._models.values())

    def filter(
        self,
        category: Optional[ModelCategory] = None,
        priority: Optional[Priority] = None,
        supports_gguf: bool = False,
    ) -> list[ModelConfig]:
        """Filter models by criteria."""
        filtered = list(self._models.values())

        if category is not None:
            filtered = [m for m in filtered if m.category == category]

        if priority is not None:
            filtered = [m for m in filtered if m.priority == priority]

        if supports_gguf:
            filtered = [
                m for m in filtered if m.gguf_model_id is not None
            ]

        return filtered


# Global registry instance
registry = ModelRegistry()
