# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: coco_caption
# Description: MS-COCO Captions (COCO-2014 val, lmms-eval 40,504-image) - image captioning, CIDEr-D metric

"""gbench native built-in runner for coco_caption (Multimodal Vision).

MS-COCO image captioning on the lmms-eval COCO-2014 validation split (40,504 images
sent to the VLM). NOTE: this is the lmms-eval coco2014 val set, NOT the Karpathy 5k
test split, so CIDEr here is not directly comparable to Karpathy-test leaderboard
numbers. Scored corpus-level with pycocoevalcap (CIDEr-D primary + BLEU/ROUGE-L,
METEOR/SPICE when their Java assets are available). Requires the 'pycocoevalcap'
package (with Java for the PTB tokenizer); skips cleanly if it is not installed.

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_COCO_CAPTION_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple
from .base import run_eval_suite, strip_thinking_tags
from .sampling import limit_dataset
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Multimodal & Vision"
DOCS_URL = "docs/evals/coco_caption.md"
# COCO 2014 caption val set with embedded images + 5 references (lmms-eval standard).
_DATASET = "lmms-lab/COCO-Caption"
_SPLIT = "val"
# Per-image CIDEr threshold used only for the trace-level pass flag; the reported
# canonical number is the corpus 'cider' score in the result dict.
_PASS_CIDER = 0.5


def check_coco_caption_prerequisites() -> Tuple[bool, str]:
    """Check datasets + Pillow + pycocoevalcap are importable."""
    try:
        import datasets  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        return False, "Python packages 'datasets'/'Pillow' are not installed."
    try:
        from pycocoevalcap.cider.cider import Cider  # noqa: F401
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # noqa: F401
    except ImportError:
        return False, (
            "Python package 'pycocoevalcap' is not installed "
            "(pip install gbench[evals]; needs Java for the PTB tokenizer)."
        )
    return True, ""


def _load_coco_caption_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load MS-COCO Karpathy test captions (with images) from HF Hub; raises on failure."""
    try:
        from datasets import load_dataset
        ds = load_dataset(_DATASET, split=_SPLIT)
    except Exception as e:
        logger.error(f"Failed to load dataset for coco_caption: {e}")
        raise RuntimeError(f"Could not load dataset for coco_caption: {e}") from e

    if limit is not None and limit > 0 and hasattr(ds, "select"):
        # Stratified, not a contiguous head (audit RC-1).
        ds = limit_dataset(ds, limit, None, seed="coco_caption")

    samples = []
    for item in ds:
        image = item.get("image")
        captions = item.get("answer")  # list of 5 reference captions
        cocoid = item.get("id")
        if image is None or not captions or cocoid is None:
            raise RuntimeError(
                "coco_caption: unexpected dataset schema (image/answer/id); "
                "refusing to fabricate sample data"
            )
        refs = [str(c).strip() for c in captions if c]
        if not refs:
            raise RuntimeError("coco_caption: empty reference captions")
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Provide one short caption describing this image in a single sentence. Output only the caption."},
            ],
        }]
        samples.append((messages, refs, {"category": "captioning", "image_id": int(cocoid)}))

    if not samples:
        raise RuntimeError("coco_caption returned empty rows")
    logger.info(f"Loaded {len(samples)} coco_caption samples from HF Hub ('{_DATASET}').")
    return samples


def _make_scorer(metrics: Dict[str, Any]):
    """Build the corpus-level pycocoevalcap async scorer; writes scores into `metrics`."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge

        gts: Dict[Any, List[Dict[str, str]]] = {}
        res: Dict[Any, List[Dict[str, str]]] = {}
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("image_id")
            if iid is None:
                continue
            refs = tr.get("gold_answer") or []
            pred = strip_thinking_tags(tr.get("response_text") or "")
            gts[iid] = [{"caption": r} for r in refs]
            res[iid] = [{"caption": pred}]

        tok = PTBTokenizer()
        gts_t = tok.tokenize(gts)
        res_t = tok.tokenize(res)

        keys = list(gts_t.keys())
        cider_score, cider_per = Cider().compute_score(gts_t, res_t)
        metrics["cider"] = round(float(cider_score), 4)
        try:
            bleu_score, _ = Bleu(4).compute_score(gts_t, res_t)
            metrics["bleu1"], metrics["bleu2"], metrics["bleu3"], metrics["bleu4"] = \
                [round(float(b), 4) for b in bleu_score]
        except Exception as e:
            logger.warning(f"coco_caption BLEU failed: {e}")
        try:
            rouge_score, _ = Rouge().compute_score(gts_t, res_t)
            metrics["rouge_l"] = round(float(rouge_score), 4)
        except Exception as e:
            logger.warning(f"coco_caption ROUGE-L failed: {e}")
        try:  # METEOR needs Java + bundled jar; omit (not fabricate) on failure
            from pycocoevalcap.meteor.meteor import Meteor
            meteor_score, _ = Meteor().compute_score(gts_t, res_t)
            metrics["meteor"] = round(float(meteor_score), 4)
        except Exception as e:
            logger.warning(f"coco_caption METEOR skipped: {e}")

        per_image = dict(zip(keys, list(cider_per)))
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("image_id")
            tr["cider"] = round(float(per_image.get(iid, 0.0)), 4)
            tr["is_correct"] = bool(per_image.get(iid, 0.0) >= _PASS_CIDER)
            tr["status"] = "OK"
    return _score


def run_coco_caption(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native COCO Caption evaluation suite (or skip if pycocoevalcap missing)."""
    ok, reason = check_coco_caption_prerequisites()
    if not ok:
        raise infra_required("coco_caption", reason, DOCS_URL)

    samples = _load_coco_caption_samples(limit=kwargs.get("limit"))
    metrics: Dict[str, Any] = {}
    result = run_eval_suite(
        eval_name="coco_caption",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(metrics),
        declared_scoring_mode="deterministic",  # corpus CIDEr-D metric, not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),   # sovereign; base.py defaults when unset
        temperature=kwargs.get("temperature"),
    )
    result.update(metrics)  # surface corpus CIDEr/BLEU/ROUGE-L/METEOR
    # Promote corpus CIDEr-D (x100, the standard MS-COCO reporting convention) to the headline. The
    # canonical MS-COCO metric is CIDEr-D, NOT the per-image CIDEr>=0.5 pass rate base.py computed as
    # `accuracy` from is_correct: that arbitrary threshold flags a genuinely good caption scoring
    # below 0.5 as a failure (WS10 audit). The per-image pass rate is kept as a secondary field.
    if "cider" in metrics:
        result["per_image_pass_rate"] = result.get("accuracy")
        result["accuracy"] = round(float(metrics["cider"]) * 100.0, 2)
        result["metric"] = "CIDEr-D x100 (canonical MS-COCO metric; a score, not a percentage)"
    return result
