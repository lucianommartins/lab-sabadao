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

"""Native MMMU-Pro multimodal vision evaluation suite.

Sampling: this suite asks for temperature 1.0 (nucleus top_p 0.95). The global default is
1.0, the model's shipped `generation_config.json`. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MMMU_PRO_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.

Scope: runs the MMMU-Pro "standard (10 options)" split only; it does NOT run the combined
vision-plus-text config, so the score is not directly comparable to the combined MMMU-Pro
leaderboard.
"""

import ast
import base64
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset

logger = logging.getLogger(__name__)

OPTION_LETTERS = "ABCDEFGHIJ"

PROMPT_FOOTER = (
    "Try to reason about the question step by step. Don't give a final"
    " answer without reasoning. Output the final answer in the format"
    " 'Final Answer: (X)' where X is the correct letter choice. Answer:"
)
def _image_to_base64_url(image) -> str:
    """Convert PIL image to base64 data URL."""
    if image is None:
        raise ValueError("Cannot convert None image to base64 URL.")
    if hasattr(image, "mode") and image.mode != "RGB":
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _load_mmmu_pro_samples(max_soft_tokens: Optional[int] = None, limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MMMU-Pro samples directly from canonical HF Hub dataset ('MMMU/MMMU_Pro')."""
    from datasets import load_dataset
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    if limit is not None:

        if hasattr(ds, "select"):

            ds = limit_dataset(ds, limit, "subject", seed="mmmu_pro")

        else:

            ds = ds[:limit]
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} MMMU-Pro samples from HF Hub ('standard (10 options)').")

    samples = []
    for item in raw_samples:
        question = item["question"]
        options = item.get("options", "")
        if isinstance(options, str):
            try:
                options = ast.literal_eval(options)
            except Exception:
                options = []
        gold_answer = str(item["answer"]).strip().upper()

        letters = list(OPTION_LETTERS[: len(options)]) if options else ["A", "B", "C", "D"]
        options_text = "\n".join(f"({letter}) {opt}" for letter, opt in zip(letters, options)) if options else str(options)
        prompt_text = f"Question: {question}\n\n{options_text}\n\n{PROMPT_FOOTER}"

        content = []
        for i in range(1, 8):
            img = item.get(f"image_{i}")
            if img is not None:
                content.append({"type": "image_url", "image_url": {"url": _image_to_base64_url(img)}})
        if not content and item.get("image") is not None:
            content.append({"type": "image_url", "image_url": {"url": _image_to_base64_url(item["image"])}})
        if not content:
            content = [{"type": "text", "text": prompt_text}]
        else:
            content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        extra = {
            "category": str(item.get("subject", item.get("sub_discipline", ""))),
            # `temperature` is NOT set here: a per-sample payload key beats everything and
            # would make --temperature / GBENCH_MMMU_PRO_TEMPERATURE inert. The suite's
            # protocol value (1.0, the vision-eval sampling default) is passed
            # to run_eval_suite instead, where it sits at the right precedence level.
            "top_p": 0.95,
            # NO per-sample max_tokens: like `temperature` above, a per-sample cap OVERRIDES the
            # sovereign --max-output-tokens (an unconditional 8192 truncated the reasoning before the
            # answer under --thinking, a false 0%/low). The run budget - or base.py's thinking-aware
            # default when unset - governs output length.
        }
        if max_soft_tokens is not None:
            extra["mm_processor_kwargs"] = {"max_soft_tokens": max_soft_tokens}

        samples.append((messages, gold_answer, extra))
    return samples


def _eval_mmmu_pro(response_text: str, gold_answer: str) -> bool:
    """Evaluate MMMU-Pro letter match matching parse_answer from eval_mmmu_pro.py."""
    response = response_text.strip()
    choices = list(OPTION_LETTERS)

    if response.upper() in choices:
        return response.upper() == gold_answer

    matches = re.findall(r"(?i)final\s+answer\s*[:=]\s*\(?([A-J])\)?", response)
    if matches:
        return matches[-1].upper() == gold_answer

    matches = re.findall(r"(?i)answer\s*[:=]\s*\(?([A-J])\)?", response)
    if matches:
        return matches[-1].upper() == gold_answer

    matches = re.findall(r"\(([A-J])\)", response)
    if matches:
        return matches[-1].upper() == gold_answer

    for line in reversed(response.split("\n")):
        line = line.strip().rstrip(".)")
        if line.upper() in choices:
            return line.upper() == gold_answer

    # Last resort: the LAST standalone option letter. The previous fallback was
    # `gold_answer in response.upper()`, a bare single-character substring test - for a
    # gold of "A" that matched the "A" in any ordinary word, crediting wrong answers.
    standalone = re.findall(r"\b([A-J])\b", response.upper())
    if standalone:
        return standalone[-1] == gold_answer
    return False


def run_mmmu_pro(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    eval_max_soft_tokens: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native MMMU-Pro evaluation suite."""
    samples = _load_mmmu_pro_samples(eval_max_soft_tokens, limit=kwargs.get("limit"))
    return run_eval_suite(
        eval_name="mmmu_pro",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_mmmu_pro,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),   # sovereign; base.py defaults when unset
        # Vision eval sampling: nucleus sampling, temp 1.0 / top_p 0.95.
        temperature=1.0,
    )

