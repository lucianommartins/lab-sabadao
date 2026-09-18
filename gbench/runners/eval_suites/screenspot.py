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

"""Native ScreenSpot GUI grounding vision evaluation suite.

Canonical default is ScreenSpot-v2 (HongxinLi/ScreenSpot_v2); a local ScreenSpot-Pro
data dir, when present, is used instead (see docs/evals/screenspot.md).

Sampling: this suite asks for temperature 0.0 (ScreenSpot grounding is scored greedy). The global default is
1.0, the model's shipped `generation_config.json`. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SCREENSPOT_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import stratified_sample, limit_dataset

logger = logging.getLogger(__name__)

MOUSE_CLICK_TOOL = {
    "type": "function",
    "function": {
        "name": "mouse_click",
        "description": "Performs a mouse click.",
        "parameters": {
            "type": "object",
            "properties": {
                "button": {
                    "type": "string",
                    "description": 'The button to click. Either "left", "middle" or "right".',
                },
                "repeats": {
                    "type": "integer",
                    "description": "The number of times to click. Default is 1.",
                    "default": 1,
                },
                "x": {
                    "type": "integer",
                    "description": "The normalized x coordinate within the [0, 1000] range of the image.",
                },
                "y": {
                    "type": "integer",
                    "description": "The normalized y coordinate within the [0, 1000] range of the image.",
                },
            },
            "required": ["button", "y", "x"],
        },
    },
}


def _image_to_base64_url(image) -> str:
    """Convert PIL image to base64 data URL."""
    import io
    if image is None:
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    if hasattr(image, "mode") and image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _load_screenspot_pro_local(data_dir: Optional[str] = None):
    from PIL import Image
    data_dir = (data_dir or os.environ.get("GBENCH_SCREENSPOT_DATA_DIR")
                or os.environ.get("SCREENSPOT_PRO_DATA_DIR", "./screenspot_pro_data"))
    annotations_dir = os.path.join(data_dir, "annotations")
    images_dir = os.path.join(data_dir, "images")
    if not os.path.exists(annotations_dir) or not os.path.exists(images_dir):
        return None

    samples = []
    for ann_file in sorted(os.listdir(annotations_dir)):
        if not ann_file.endswith(".json"):
            continue
        with open(os.path.join(annotations_dir, ann_file), "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            instruction = entry.get("instruction")
            img_filename = entry.get("img_filename")
            bbox = entry.get("bbox")
            if not instruction or not img_filename or not bbox:
                continue
            img_path = os.path.join(images_dir, img_filename)
            if not os.path.exists(img_path):
                continue
            with Image.open(img_path) as img:
                w, h = img.size
            x1_norm = bbox[0] * 1000 / w
            y1_norm = bbox[1] * 1000 / h
            x2_norm = bbox[2] * 1000 / w
            y2_norm = bbox[3] * 1000 / h
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = "png" if img_path.lower().endswith(".png") else "jpeg"
            image_url = f"data:image/{ext};base64,{b64}"
            prompt_text = f'Click on the UI element: "{instruction}".\nUse the computer tool at your disposal.'
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            messages = [{"role": "user", "content": content}]
            extra = {
                "tools": [MOUSE_CLICK_TOOL],
                "tool_choice": "auto",
                # No per-sample max_tokens or temperature: a per-sample payload key beats
                # --max-output-tokens (see mmmu_pro). Coordinates are short, but capping output here
                # OVERRODE the sovereign budget and truncated reasoning under --thinking; the run
                # budget (or base.py's thinking-aware default) governs. The suite temperature goes to
                # run_eval_suite so --temperature / GBENCH_SCREENSPOT_TEMPERATURE can reach it.
            }
            samples.append((messages, (x1_norm, y1_norm, x2_norm, y2_norm), extra))
    return samples


def _load_screenspot_samples(max_soft_tokens: Optional[int] = None, limit: Optional[int] = None) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load samples for the screenspot suite. Canonical default = ScreenSpot-v2
    (HongxinLi/ScreenSpot_v2, per docs/evals/screenspot.md). If a local ScreenSpot-Pro data dir is
    provided (GBENCH_SCREENSPOT_DATA_DIR, legacy alias SCREENSPOT_PRO_DATA_DIR) it is scored instead.
    Returns (samples, dataset_label) so the caller records exactly which dataset was scored and a
    v2 run is never mislabelled as Pro."""
    local_samples = _load_screenspot_pro_local()
    if local_samples is not None and len(local_samples) > 0:
        logger.info(f"Loaded {len(local_samples)} ScreenSpot-Pro samples from the local data dir.")
        dataset_label = "ScreenSpot-Pro (local data dir)"
        # Stratified, not a contiguous head (audit RC-1). Local samples are
        # (messages, target, extra) tuples, so key on the extra dict's data_type
        # (not the tuple itself, which has no .get -> crashed under --eval-limit).
        samples = stratified_sample(
            local_samples, limit,
            lambda s: (s[2] if len(s) > 2 and isinstance(s[2], dict) else {}).get("data_type"),
            seed="screenspot")
    else:
        from datasets import load_dataset
        ds = load_dataset("HongxinLi/ScreenSpot_v2", split="test")
        # Stratified, not a contiguous head (audit RC-1).
        ds = limit_dataset(ds, limit, 'data_type', seed="screenspot")
        raw_samples = list(ds)
        # Canonical default is ScreenSpot-v2 (docs/evals/screenspot.md); the local Pro data dir
        # above is an optional upgrade. Label the dataset actually scored so a v2 run can never
        # masquerade as a Pro number.
        logger.info(f"Loaded {len(raw_samples)} ScreenSpot-v2 samples from HF Hub ('HongxinLi/ScreenSpot_v2').")
        dataset_label = "HongxinLi/ScreenSpot_v2"
        samples = []
        for item in raw_samples:
            instruction = item["instruction"]
            bbox = item["bbox"]
            gx1, gy1, gx2, gy2 = bbox[0] * 1000, bbox[1] * 1000, bbox[2] * 1000, bbox[3] * 1000
            image_url = _image_to_base64_url(item.get("image"))
            prompt_text = f'Click on the UI element: "{instruction}".\nUse the computer tool at your disposal.'
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            messages = [{"role": "user", "content": content}]
            extra = {
                "tools": [MOUSE_CLICK_TOOL],
                "tool_choice": "auto",
                # No per-sample max_tokens or temperature: a per-sample payload key beats
                # --max-output-tokens (see mmmu_pro). Coordinates are short, but capping output here
                # OVERRODE the sovereign budget and truncated reasoning under --thinking; the run
                # budget (or base.py's thinking-aware default) governs. The suite temperature goes to
                # run_eval_suite so --temperature / GBENCH_SCREENSPOT_TEMPERATURE can reach it.
            }
            samples.append((messages, (gx1, gy1, gx2, gy2), extra))

    if max_soft_tokens is not None:
        for idx in range(len(samples)):
            msg, target, extra = samples[idx]
            extra["mm_processor_kwargs"] = {"max_soft_tokens": max_soft_tokens}
            samples[idx] = (msg, target, extra)
    return samples, dataset_label


def parse_click_coordinates(text: str) -> tuple[int | None, int | None]:
    """Extract (x, y) coordinates from model tool call response."""
    patterns = [
        r"mouse_click\{[^}]*x[:\s]*(\d+)[^}]*y[:\s]*(\d+)",
        r"mouse_click\{[^}]*y[:\s]*(\d+)[^}]*x[:\s]*(\d+)",
        r'"x"\s*:\s*(\d+).*?"y"\s*:\s*(\d+)',
        r'"y"\s*:\s*(\d+).*?"x"\s*:\s*(\d+)',
        r"mouse_click\([^)]*x\s*=\s*(\d+)[^)]*y\s*=\s*(\d+)",
        r"mouse_click\([^)]*y\s*=\s*(\d+)[^)]*x\s*=\s*(\d+)",
    ]
    for i, pattern in enumerate(patterns):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            if i in (1, 3, 5):
                return int(m.group(2)), int(m.group(1))
            return int(m.group(1)), int(m.group(2))
    return None, None


def _eval_screenspot(response_text: str, gold_bbox: Any) -> bool:
    """Check if predicted click point (x, y) falls inside the target bbox in [0, 1000] scale."""
    pred_x, pred_y = parse_click_coordinates(response_text)
    if pred_x is None or pred_y is None:
        return False
    x1, y1, x2, y2 = gold_bbox[0], gold_bbox[1], gold_bbox[2], gold_bbox[3]
    return x1 <= pred_x <= x2 and y1 <= pred_y <= y2


def run_screenspot(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    eval_max_soft_tokens: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run the screenspot GUI-grounding suite (ScreenSpot-v2 canonical; ScreenSpot-Pro when a local
    data dir is provided). The dataset actually scored is recorded on the result."""
    samples, dataset_label = _load_screenspot_samples(eval_max_soft_tokens, limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="screenspot",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_screenspot,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        # ScreenSpot is scored greedy (temperature 0.0).
        temperature=0.0,
    )
    result["dataset"] = dataset_label
    return result

