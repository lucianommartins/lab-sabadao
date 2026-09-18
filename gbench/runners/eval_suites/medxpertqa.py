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

"""Native MedXpertQA (Medical Multimodal Expert Exam & Diagnostic QA) evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MEDXPERTQA_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import functools
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite
from .sampling import limit_dataset
from .dataset_utils import build_image_message

logger = logging.getLogger(__name__)

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
DOCS_URL = "docs/evals/medxpertqa.md"


@functools.lru_cache(maxsize=1)
def _medxpertqa_image_dir() -> Optional[str]:
    """Extract MedXpertQA's `images.zip` once and return the dir its image files live in.

    The MM split's `images` column is a LIST OF FILENAMES ('MM-0-a.jpeg'), not image bytes;
    the actual pixels ship in a separate images.zip in the repo. Without resolving it the
    multimodal split is silently text-only (unanswerable).
    """
    try:
        from huggingface_hub import hf_hub_download
        zpath = hf_hub_download("TsinghuaC3I/MedXpertQA", "images.zip", repo_type="dataset")
        extract_root = zpath + "_extracted"
        if not os.path.isdir(extract_root):
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(extract_root)
        for root, _dirs, files in os.walk(extract_root):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return root
        return extract_root
    except Exception as e:
        logger.warning("medxpertqa: could not fetch/extract images.zip (%s); multimodal rows "
                       "will be scored TEXT-ONLY. See %s", e, DOCS_URL)
        return None


def _load_medxpertqa_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MedXpertQA medical exam samples from canonical HF Hub dataset ('TsinghuaC3I/MedXpertQA')."""
    from datasets import load_dataset

    try:
        ds = load_dataset("TsinghuaC3I/MedXpertQA", "MM", split="test")
    except Exception:
        ds = load_dataset("wish6424/MedXpertQA-Diagnosis", split="test")

    # Stratified, not a contiguous head (audit RC-1).
    ds = limit_dataset(ds, limit, "medical_task", seed="medxpertqa")
    raw_samples = list(ds)
    logger.info(f"Loaded {len(raw_samples)} MedXpertQA samples from HF Hub.")

    base_dir = _medxpertqa_image_dir()
    attached_n = total_mm = 0
    samples = []
    for item in raw_samples:
        q_text = item.get("question", "")
        options = item.get("options", {})
        gold_ans = str(item.get("label", item.get("answer", ""))).strip().upper()
        specialty = item.get("medical_task", item.get("body_system", item.get("specialty", "Clinical")))

        # Format multiple choice options
        if isinstance(options, dict):
            options_lines = [f"({k.upper()}) {v}" for k, v in sorted(options.items())]
        elif isinstance(options, list):
            options_lines = [f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options)]
        else:
            options_lines = []

        options_str = "\n".join(options_lines)
        prompt_text = (
            f"Medical Question ({specialty}):\n{q_text}\n\n"
            f"Options:\n{options_str}\n\n"
        )
        if enable_thinking:
            prompt_text += "Let's reason carefully step by step and output the correct option letter in the format: 'Answer: (X)'."
        else:
            prompt_text += "Answer directly with the correct option letter in the format: 'Answer: (X)'."

        # MedXpertQA "MM" exposes `images` (a LIST of filenames resolved against images.zip);
        # some fallbacks give bytes/PIL. build_image_message handles all of them and resolves
        # bare filenames under base_dir. Attach every image the question references.
        imgs = item.get("images")
        if not imgs:
            single = item.get("image")
            imgs = [single] if single is not None else []
        messages, attached = build_image_message(prompt_text, imgs, base_dir=base_dir)
        if imgs:
            total_mm += 1
            attached_n += 1 if attached else 0
        samples.append((messages, gold_ans, {"category": specialty}))

    if total_mm:
        logger.info("medxpertqa: attached images for %d/%d multimodal rows.", attached_n, total_mm)
        if attached_n == 0:
            logger.warning("medxpertqa: NO images attached for any of %d multimodal rows - "
                           "scoring TEXT-ONLY (images.zip unavailable or filenames unresolved). "
                           "See %s", total_mm, DOCS_URL)
    return samples


def _eval_medxpertqa(response_text: str, gold_letter: str) -> bool:
    """Extract predicted option letter and compare with gold answer."""
    if not response_text or not gold_letter:
        return False

    gold = gold_letter.strip().upper()
    if len(gold) > 1:
        match = re.search(r"\b([A-J])\b", gold)
        if match:
            gold = match.group(1)

    resp = response_text.strip()
    # Anchor to the model's FINAL answer in the requested 'Answer: (X)' format and take the LAST
    # occurrence. The old `re.search` (FIRST match) on a loose pattern grabbed stray letters from
    # the reasoning - e.g. the 'C' in '**Answer Choices**' - and coincidental matches inflated the
    # score with false positives (medxpertqa audit 2026-09-12). Require the colon so 'Answer Choices'
    # (no colon) can never match; the prompt explicitly asks for 'Answer: (X)'.
    anchored = re.findall(r"answer\s*:\s*\(?([A-J])\)?", resp, re.IGNORECASE)
    if anchored:
        return anchored[-1].upper() == gold

    # Fallback: last option-letter mention in an answer-ish phrase (still LAST, not first).
    loose = re.findall(r"(?:correct\s+option|final\s+answer|option|choice|answer)\s*(?:is|:)?\s*\(?([A-J])\)?",
                       resp, re.IGNORECASE)
    if loose:
        return loose[-1].upper() == gold

    tokens = re.findall(r"\b([A-J])\b", resp)
    if tokens:
        return tokens[-1].upper() == gold

    return False


def run_medxpertqa(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native MedXpertQA medical multimodal evaluation suite."""
    samples = _load_medxpertqa_samples(
        enable_thinking=enable_thinking,
        limit=kwargs.get("limit"),
    )
    return run_eval_suite(
        eval_name="medxpertqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_medxpertqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192 if enable_thinking else 2048),
        temperature=kwargs.get("temperature"),
    )
