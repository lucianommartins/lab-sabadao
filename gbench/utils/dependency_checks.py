"""Dependency validation and optional engine availability checks."""

import os
import sys
from typing import Optional


def check_vllm_available() -> bool:
    """Check if the vllm package is installed and importable."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def require_vllm_engine(feature_name: str = "Local vLLM engine execution") -> None:
    """Ensure vllm is installed.

    If missing, outputs a user-friendly error and exits gracefully.
    """
    if not check_vllm_available():
        sys.stderr.write(
            f"\n[ERROR] {feature_name} requires a local vLLM serving stack, which is not installed.\n\n"
            f"gbench does NOT install vllm/torch itself (they are GPU/CUDA-specific and often a\n"
            f"validated custom build; auto-installing would overwrite it or pull a mismatched CUDA).\n"
            f"Provision a serving stack for your hardware, then re-run:\n"
            f"    pip install vllm        # fresh machine: pulls a matching torch/CUDA\n"
            f"    # or use your existing validated torch + vLLM build\n\n"
            f"Only LOCAL serving needs this; runs against a served model with --remote-endpoint do not.\n\n"
        )
        sys.exit(1)


def safe_get_tokenizer(tokenizer_path: str, custom_tokenizer: Optional[str] = None):
    """Safely load a tokenizer without crashing on Ollama tags or invalid HF repo IDs.

    Prioritizes custom_tokenizer if provided (e.g. via --tokenizer).
    """
    target_path = custom_tokenizer or tokenizer_path
    if not target_path or ":" in target_path or ("/" not in target_path and not os.path.exists(target_path)):
        return None
    try:
        from vllm.tokenizers import get_tokenizer
        return get_tokenizer(target_path)
    except Exception:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(target_path)
        except Exception:
            return None
