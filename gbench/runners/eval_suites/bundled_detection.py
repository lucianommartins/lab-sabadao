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

"""Native Bundled Object Detection & Grounding evaluation suite.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_BUNDLED_DETECTION_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite

logger = logging.getLogger(__name__)

COCO_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union between two boxes in [ymin, xmin, ymax, xmax] format."""
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2

    inter_ymin = max(ymin1, ymin2)
    inter_xmin = max(xmin1, xmin2)
    inter_ymax = min(ymax1, ymax2)
    inter_xmax = min(xmax1, xmax2)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, xmax1 - xmin1) * max(0.0, ymax1 - ymin1)
    area2 = max(0.0, xmax2 - xmin2) * max(0.0, ymax2 - ymin2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _load_bundled_detection_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load object detection samples from canonical HF Hub dataset ('detection-datasets/coco')."""
    from datasets import load_dataset

    ds = load_dataset("detection-datasets/coco", split="val", streaming=True)
    samples = []
    count = 0

    for item in ds:
        objects = item.get("objects", {})
        bboxes = objects.get("bbox", [])
        categories = objects.get("category", [])
        if not bboxes:
            continue

        width = item.get("width", 640)
        height = item.get("height", 480)

        # Convert gold bboxes [x, y, w, h] to normalized [ymin, xmin, ymax, xmax] in [0, 1000]
        gold_objects = []
        labels_present = set()
        for bbox, cat_id in zip(bboxes, categories):
            if cat_id < len(COCO_CATEGORIES):
                cat_name = COCO_CATEGORIES[cat_id]
            else:
                cat_name = str(cat_id)
            labels_present.add(cat_name)

            # detection-datasets/coco stores bboxes as [xmin, ymin, xmax, ymax], NOT
            # [x, y, w, h]. Treating them as xywh added xmax to xmin (and ymax to ymin),
            # inflating every gold box and corrupting all IoU comparisons.
            x0, y0, x1, y1 = bbox
            xmin = (x0 / width) * 1000.0
            ymin = (y0 / height) * 1000.0
            xmax = (x1 / width) * 1000.0
            ymax = (y1 / height) * 1000.0
            gold_objects.append({
                "label": cat_name,
                "box_2d": [ymin, xmin, ymax, xmax]
            })

        # Image encode
        img_obj = item["image"]
        if hasattr(img_obj, "mode") and img_obj.mode != "RGB":
            img_obj = img_obj.convert("RGB")
        buf = io.BytesIO()
        img_obj.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Do NOT leak which categories are present in THIS image (the old prompt listed
        # sorted(labels_present), handing the model the answer set). Closed-set COCO
        # detection: give the fixed 80-category vocabulary and let the model decide what is
        # present. Ask for a confidence so mAP can rank detections.
        prompt = (
            "Detect every object in this image that belongs to the COCO-80 categories:\n"
            f"{', '.join(COCO_CATEGORIES)}.\n\n"
            "Output ONLY a JSON array; each element has 'label' (one of the categories), "
            "'box_2d' ([ymin, xmin, ymax, xmax] normalized to [0, 1000]), and 'confidence' "
            "(0.0-1.0). Emit one element per detected object."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        primary_cat = sorted(labels_present)[0] if labels_present else "object"
        samples.append((messages, gold_objects, {"category": primary_cat}))
        count += 1
        if limit is not None and count >= limit:
            break

    logger.info(f"Loaded {len(samples)} Bundled Detection samples from HF Hub ('detection-datasets/coco').")
    return samples


def _parse_detections(response_text: str) -> List[Dict[str, Any]]:
    """Parse the model's detections into [{label, box:[ymin,xmin,ymax,xmax] in [0,1000], conf}]."""
    if not response_text:
        return []
    raw: Any = None
    m = re.search(r"\[\s*\{.*\}\s*\]", response_text, re.DOTALL)
    try:
        raw = json.loads(m.group(0) if m else response_text)
    except Exception:
        raw = None
    dets: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for pred in raw:
            if not isinstance(pred, dict):
                continue
            box = pred.get("box_2d") or pred.get("bbox") or pred.get("box")
            if not box or len(box) != 4:
                continue
            try:
                box = [float(x) for x in box]
            except (ValueError, TypeError):
                continue
            if max(box) <= 1.0:                       # [0,1] -> [0,1000]
                box = [x * 1000.0 for x in box]
            conf = pred.get("confidence", pred.get("score", 1.0))
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                conf = 1.0
            dets.append({"label": str(pred.get("label", "")).strip().lower(),
                         "box": box, "conf": max(0.0, min(1.0, conf))})
    return dets


def _eval_bundled_detection(response_text: str, gold_objects: Any) -> bool:
    """Secondary binary (>=1 correctly-classed box at IoU>=0.5). Headline is mAP (run_*)."""
    dets = _parse_detections(response_text)
    for d in dets:
        for gold in gold_objects or []:
            if d["label"] and d["label"] == str(gold.get("label", "")).strip().lower():
                if _compute_iou(d["box"], gold.get("box_2d")) >= 0.5:
                    return True
    return False


def _average_precision(preds: List[Tuple[float, int, List[float]]],
                       gold_by_img: Dict[int, List[List[float]]],
                       iou_thr: float) -> Optional[float]:
    """101-point-interpolated AP for one class at one IoU threshold.

    preds: (conf, image_id, box) across all images for this class, any order.
    gold_by_img: image_id -> gold boxes of this class. Returns None if the class has no gold.
    """
    n_gold = sum(len(v) for v in gold_by_img.values())
    if n_gold == 0:
        return None
    matched: Dict[int, set] = {img: set() for img in gold_by_img}
    order = sorted(preds, key=lambda p: p[0], reverse=True)
    tp = [0.0] * len(order)
    fp = [0.0] * len(order)
    for i, (_conf, img, box) in enumerate(order):
        golds = gold_by_img.get(img, [])
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(golds):
            if j in matched[img]:
                continue
            iou = _compute_iou(box, gbox)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            matched[img].add(best_j)
            tp[i] = 1.0
        else:
            fp[i] = 1.0
    # cumulative precision/recall
    ctp = cfp = 0.0
    recalls, precisions = [], []
    for i in range(len(order)):
        ctp += tp[i]
        cfp += fp[i]
        recalls.append(ctp / n_gold)
        precisions.append(ctp / (ctp + cfp) if (ctp + cfp) else 0.0)
    # 101-point interpolation
    ap = 0.0
    for r in [x / 100.0 for x in range(101)]:
        prec = max((p for p, rec in zip(precisions, recalls) if rec >= r), default=0.0)
        ap += prec / 101.0
    return ap


def _compute_map(traces: List[Dict[str, Any]]) -> Dict[str, float]:
    """COCO mAP@[.5:.95] and mAP@.5 over all detection traces."""
    # class -> preds [(conf, img, box)]; class -> {img: [gold boxes]}
    preds_by_cls: Dict[str, List[Tuple[float, int, List[float]]]] = {}
    gold_by_cls: Dict[str, Dict[int, List[List[float]]]] = {}
    for img_id, t in enumerate(traces):
        for g in (t.get("gold_answer") or []):
            cls = str(g.get("label", "")).strip().lower()
            gold_by_cls.setdefault(cls, {}).setdefault(img_id, []).append(g.get("box_2d"))
        for d in _parse_detections(t.get("response_text") or ""):
            if not d["label"]:
                continue
            preds_by_cls.setdefault(d["label"], []).append((d["conf"], img_id, d["box"]))

    thresholds = [0.5 + 0.05 * i for i in range(10)]     # .50, .55, ... .95
    per_thr_maps: Dict[float, float] = {}
    for thr in thresholds:
        aps = []
        for cls, gimg in gold_by_cls.items():
            ap = _average_precision(preds_by_cls.get(cls, []), gimg, thr)
            if ap is not None:
                aps.append(ap)
        per_thr_maps[thr] = (sum(aps) / len(aps)) if aps else 0.0
    map_5095 = sum(per_thr_maps.values()) / len(per_thr_maps) if per_thr_maps else 0.0
    return {"map": map_5095, "map_50": per_thr_maps.get(0.5, 0.0)}


def run_bundled_detection(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native Bundled Object Detection; headline = COCO mAP@[.5:.95]."""
    samples = _load_bundled_detection_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="bundled_detection",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_bundled_detection,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
    )
    # Canonical COCO detection metric is mAP (per-class AP, greedy IoU matching with
    # confidence ranking, averaged over IoU .5:.95), NOT a per-image "any box right" rate.
    traces = [t for t in (result.get("sample_traces") or []) if t.get("response_text") is not None]
    maps = _compute_map(traces)
    result["hit_rate_any_box"] = result.get("accuracy")
    result["map_50"] = round(maps["map_50"] * 100.0, 2)
    result["map_50_95"] = round(maps["map"] * 100.0, 2)
    result["metric"] = "COCO mAP@[.5:.95] (canonical); map_50 and hit_rate_any_box also reported"
    result["accuracy"] = round(maps["map"] * 100.0, 2)
    # COCO mAP presumes trained per-box confidence scores for the PR ranking, which a chat VLM is
    # not asked to emit (absent confidences default to 1.0, collapsing the curve), and hard request
    # failures are excluded from the denominator; so this is not the published COCO detection number.
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "chat-VLM boxes carry no trained confidence for the mAP PR-ranking; failures excluded from "
        "the denominator - not directly comparable to the published COCO detection mAP")
    return result
