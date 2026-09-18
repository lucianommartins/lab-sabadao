# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
# Evaluation Suite: omnidocbench
# Description: OmniDocBench v1.5 (Multimodal Document Parsing, Layout & Formula Recognition)

"""gbench native built-in runner for omnidocbench (Multimodal & Vision).

Sampling: temperature defaults to 0.0 (greedy) for no-think runs and 1.0 (the model's
shipped `generation_config.json`) with `--thinking`; the measurement is in
base.DEFAULT_TEMPERATURE. Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_OMNIDOCBENCH_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download
from .base import run_eval_suite
from .dataset_utils import extract_lossless_image_b64
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Multimodal & Vision"
DOCS_URL = "docs/evals/omnidocbench.md"

#: The evaluator image (Python 3.10 + TeX Live/CJK + ImageMagick 7 + Ghostscript for the CDM
#: metric). Built LOCALLY from docker/omnidocbench.Dockerfile (gbench never pulls registry
#: images). Scored in a container because OmniDocBench pins Python <3.12 and numpy==1.24.4, which
#: would break the torch/vLLM serving env.
_SCORER_IMAGE_DEFAULT = "gbench-omnidocbench"

#: The end2end config the evaluator runs (canonical per-modality composite): text Edit_dist,
#: display-formula Edit_dist + CDM, table TEDS + Edit_dist, reading-order Edit_dist. Written INSIDE
#: the container; paths are relative to /workspace (the image's WORKDIR).
_END2END_CONFIG = """end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
      cdm_workers: {workers}
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: {workers}
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./gt/gt.json
    prediction:
      data_path: ./data_md/predictions
    match_method: quick_match
    match_workers: {workers}
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""

#: OmniDocBench excludes these from the end2end text metric (page furniture, not content).
_IGNORED_CATS = {"header", "footer", "page_number", "page_footnote", "abandon"}


def _load_omnidocbench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load OmniDocBench benchmark dataset directly from HF Hub (opendatalab/OmniDocBench)."""
    rows = []
    try:
        from datasets import load_dataset, Image
        ds = load_dataset("opendatalab/OmniDocBench", split="train")
        # The split has EXACTLY ONE column, `image` - no filename anywhere - so pairing by
        # `item["image_path"]`/`file_name`/`page_name` always missed and every page fell
        # back to positional pairing against a different page's ground truth. Measured
        # 2026-08-17: 20/20 unpaired, hence a guaranteed 0/20 even once the requests
        # started going through. Decoding drops the source path; the UNDECODED feature
        # keeps it, and its basename ("PPT_1001115_eng_page_003.png") is exactly the key
        # OmniDocBench.json uses in `page_info.image_path`.
        page_names: List[str] = []
        try:
            undecoded = load_dataset("opendatalab/OmniDocBench", split="train").cast_column(
                "image", Image(decode=False))
            page_names = [os.path.basename((undecoded[i]["image"] or {}).get("path") or "")
                          for i in range(len(undecoded))]
        except Exception as e:
            logger.warning("omnidocbench: could not recover page filenames (%s); pairing "
                           "will fall back to position and the ground truth may belong to "
                           "a different page.", e)
        json_file = hf_hub_download(repo_id="opendatalab/OmniDocBench", filename="OmniDocBench.json", repo_type="dataset")
        with open(json_file) as f:
            annotations = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset for omnidocbench: {e}")
        raise RuntimeError(f"Could not load dataset for omnidocbench: {e}") from e

    if not ds:
        raise RuntimeError("Dataset for omnidocbench returned empty rows")

    # Pair image <-> annotation by page filename. Indexing both by position assumes the
    # HF split and OmniDocBench.json enumerate the pages in the same order; when they do
    # not, every page is scored against a different page's ground truth.
    anno_by_name = {}
    for anno in annotations:
        name = str((anno.get("page_info") or {}).get("image_path") or "").strip()
        if name:
            anno_by_name[os.path.basename(name)] = anno

    # Iterate ALL pages and pair by name, collecting up to `limit` PAIRABLE pages. Capping by
    # position first was a bug: the split's first rows are dataset-artwork images
    # (data_diversity.*) with no annotation, so a small --eval-limit scored 0 pages.
    samples = []
    unpaired = 0
    for i in range(len(ds)):
        if limit is not None and limit > 0 and len(samples) >= limit:
            break
        item = ds[i]
        page_name = (page_names[i] if i < len(page_names) else "") or os.path.basename(
            str(item.get("image_path") or item.get("file_name")
                or item.get("page_name") or "").strip())
        anno = anno_by_name.get(page_name)
        if anno is None:
            # No annotation for this page (the split carries a couple of dataset artwork
            # images that are not benchmark pages). Positional fallback would grade the
            # page against a DIFFERENT page's ground truth - a guaranteed-wrong score that
            # looks like a model failure - so drop it instead.
            unpaired += 1
            continue
        image_val = item.get("image")

        layout_dets = anno.get("layout_dets", [])
        page_info = anno.get("page_info", {})
        page_attr = page_info.get("page_attribute", {})
        if isinstance(page_attr, dict):
            doc_type = str(page_attr.get("subset") or page_attr.get("data_source") or "unknown")
        else:
            doc_type = str(page_attr or "unknown")

        # A page's ground truth is not text alone: tables are annotated as `html` and
        # formulas as `latex`. Keeping only `text` asked the model to transcribe the whole
        # page and then graded it against a reference with the tables and equations
        # removed, so a correct table cost the page its score.
        # Order blocks by the annotation's READING ORDER (not JSON order), and exclude
        # ignored blocks + page furniture (header/footer/page number/abandon) which
        # OmniDocBench does not score in the end2end text metric.
        ordered = sorted(layout_dets, key=lambda d: (d.get("order") if isinstance(d.get("order"), int) else 10**9))
        text_parts: List[str] = []
        table_htmls: List[str] = []
        formula_latex: List[str] = []
        all_parts: List[str] = []
        for det in ordered:
            if det.get("ignore"):
                continue
            ct = str(det.get("category_type") or "").lower()
            if ct in _IGNORED_CATS:
                continue
            if ct == "table":
                v = det.get("html")
                if v:
                    table_htmls.append(str(v))
                    all_parts.append(str(v))
            elif "equation" in ct or "formula" in ct:
                v = det.get("latex") or det.get("text")
                if v:
                    formula_latex.append(str(v))
                    all_parts.append(str(v))
            else:
                v = det.get("text") or det.get("html") or det.get("latex")
                if v:
                    text_parts.append(str(v))
                    all_parts.append(str(v))
        # gold carries the per-modality streams (for the composite) plus the concatenated
        # `all` (for the overall edit distance + strict pass).
        gt_text = json.dumps({
            "all": "\n".join(all_parts),
            "text": "\n".join(text_parts),
            "tables": table_htmls,
            "formulas": formula_latex,
            "page_name": page_name,          # image basename; names the .md the evaluator matches
        })

        prompt = (
            "[Document Parsing Task]\n"
            "Carefully transcribe all text, tables, and mathematical formulas from this document image in clean markdown format."
        )
        content_payload: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        b64_str = extract_lossless_image_b64(image_val)
        if b64_str:
            content_payload.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_str}"}
            })

        messages = [{"role": "user", "content": content_payload}]
        samples.append((messages, gt_text, {"category": doc_type}))

    if unpaired:
        logger.warning("[omnidocbench] %d page(s) had no annotation and were EXCLUDED "
                       "(scoring them against another page's ground truth would be a "
                       "guaranteed-wrong result attributed to the model). %d scored.",
                       unpaired, len(samples))
    logger.info(f"Loaded {len(samples)} omnidocbench samples with lossless direct images.")
    return samples


def _normalize_doc_text(text: str) -> str:
    """OmniDocBench text normalization before edit distance.

    Whitespace, markdown emphasis and heading markers are transcription style, not
    transcription accuracy, so they are removed on both sides.
    """
    t = str(text or "")
    # Keep a link's DISPLAY TEXT, not delete the whole link. Deleting `[text](url)` entirely was
    # ASYMMETRIC: a model that renders a bare URL as a markdown link (`[http://x](http://x)`) lost the
    # URL from the prediction while the gold's bare `http://x` was kept, inflating the edit distance
    # and flipping a faithful transcription to FAIL. Reducing to the display text normalizes both
    # sides consistently (`[http://x](http://x)` -> `http://x`; `[Grammarly](url)` -> `Grammarly`).
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", t)       # images/links -> their display text
    t = re.sub(r"[*_`#>|]+", " ", t)                       # markdown emphasis/heading/table pipes
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def normalized_edit_distance(pred: str, gold: str) -> float:
    """Character-level normalized edit distance in [0, 1]; 0 is a perfect transcription.

    This is OmniDocBench's canonical text metric. `rapidfuzz` is used when present (C++,
    linear memory); `difflib` is the fallback so the suite still runs without it, and the
    substitution is recorded on the result rather than hidden.
    """
    p, g = _normalize_doc_text(pred), _normalize_doc_text(gold)
    if not g:
        return 0.0 if not p else 1.0
    try:
        from rapidfuzz.distance import Levenshtein
        return float(Levenshtein.normalized_distance(p, g))
    except ImportError:
        import difflib
        return 1.0 - difflib.SequenceMatcher(None, p, g, autojunk=False).ratio()


def _omnidoc_gold(gold_target: Any) -> Dict[str, Any]:
    """The per-modality gold payload {all, text, tables[], formulas[]} from the JSON gold."""
    try:
        g = json.loads(gold_target)
        if isinstance(g, dict):
            return g
    except Exception:
        pass
    s = str(gold_target or "")
    return {"all": s, "text": s, "tables": [], "formulas": []}


def _eval_omnidocbench(response_text: str, gold_target: str) -> bool:
    """Strict per-page pass: a transcription within 10% edit distance of the annotation.

    The headline metric of the suite is the *continuous* mean edit distance (see
    `run_omnidocbench`); this boolean only feeds the harness' pass count. The previous
    scorer passed a page when 40% of the ground truth's 3+ character word types appeared
    anywhere in the response, in any order - a page of the right document with more than
    half its content missing scored as a correct transcription.
    """
    gold_all = _omnidoc_gold(gold_target).get("all", "")
    if not response_text or not str(gold_all).strip():
        return False
    return normalized_edit_distance(response_text, str(gold_all)) <= 0.10


def _scorer_image() -> str:
    return os.environ.get("GBENCH_OMNIDOCBENCH_IMAGE", _SCORER_IMAGE_DEFAULT)


def _check_prereqs(image: str) -> None:
    """Docker + OmniDocBench's official evaluator image must be present (never a silent skip)."""
    build = (f"OmniDocBench's evaluator (with the CDM TeX Live / ImageMagick 7 / Ghostscript "
             f"runtime) runs in a locally-built image. Build it (context = the OmniDocBench "
             f"checkout):\n  docker build -t {image} -f docker/omnidocbench.Dockerfile "
             "$GBENCH_PREREQS_DIR/OmniDocBench\n"
             "(Override the tag with GBENCH_OMNIDOCBENCH_IMAGE.)")
    if not shutil.which("docker"):
        raise infra_required("omnidocbench", "docker CLI not found. " + build, DOCS_URL)
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise infra_required("omnidocbench", "docker daemon not reachable. " + build, DOCS_URL)
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        raise infra_required("omnidocbench", f"image {image!r} not found. " + build, DOCS_URL)


def _build_command(image: str, gt_json: str, pred_dir: str, out_dir: str, workers: int) -> List[str]:
    """The `docker run` that scores predictions with OmniDocBench's own pipeline in its image.

    Mounts the GT json, the predictions dir and an output dir; writes the end2end config inside
    the container (paths relative to the image's /workspace WORKDIR) and runs pdf_validation.py.
    """
    inner = ("set -e; mkdir -p configs\n"
             "cat > configs/gbench_end2end.yaml <<'GBENCH_EOF'\n"
             + _END2END_CONFIG.format(workers=workers) +
             "GBENCH_EOF\n"
             "python pdf_validation.py --config configs/gbench_end2end.yaml")
    return [
        "docker", "run", "--rm", "--entrypoint", "bash",
        "-v", f"{gt_json}:/workspace/gt/gt.json:ro",
        "-v", f"{pred_dir}:/workspace/data_md/predictions:ro",
        "-v", f"{out_dir}:/workspace/result",
        image, "-lc", inner,
    ]


def _nb_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    return (metrics.get(key) or {}).get("notebook_value")


def _raw_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    return (metrics.get(key) or {}).get("raw")


def _parse_summary(out_dir: str) -> Optional[Dict[str, Any]]:
    """The evaluator writes result/<pred_basename>_quick_match_run_summary.json (save_name =
    basename(pred_dir) + '_quick_match' = 'predictions_quick_match')."""
    path = os.path.join(out_dir, "predictions_quick_match_run_summary.json")
    if not os.path.isfile(path):
        cand = glob.glob(os.path.join(out_dir, "*_run_summary.json"))
        if not cand:
            return None
        path = cand[0]
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("notebook_metric_summary")


def run_omnidocbench(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run canonical OmniDocBench end2end.

    gbench GENERATES the per-page markdown transcriptions against the served model (multimodal
    prompts, in-process), then delegates SCORING to OmniDocBench's own evaluator inside its
    official image - the canonical per-modality composite:
    ``((1 - text_EditDist)*100 + table_TEDS*100 + formula_CDM*100) / 3`` (CDM formula metric, TEDS
    tables, edit-distance text + reading order). Hard-errors (never skips) if Docker/the image is
    absent. Not scored in-process (OmniDocBench pins Python <3.12 + numpy 1.24.4)."""
    image = _scorer_image()
    _check_prereqs(image)

    # Ground truth: OmniDocBench's annotations, matched to predictions by image basename.
    gt_file = hf_hub_download(repo_id="opendatalab/OmniDocBench", filename="OmniDocBench.json",
                             repo_type="dataset")
    with open(gt_file, encoding="utf-8") as f:
        annotations = json.load(f)
    anno_by_name = {}
    for anno in annotations:
        nm = os.path.basename(str((anno.get("page_info") or {}).get("image_path") or "").strip())
        if nm:
            anno_by_name[nm] = anno

    samples = _load_omnidocbench_samples(limit=limit)
    result = run_eval_suite(
        eval_name="omnidocbench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_omnidocbench,     # interim per-page pass count; real score comes from the evaluator
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        # Full-page transcription (text + tables + formulas) overruns small budgets; 8192 covers
        # a dense page.
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
    )

    # Write each response as <image-stem>.md, and a GT json holding exactly the generated pages
    # (so a --eval-limit subset scores only what it generated).
    workdir = tempfile.mkdtemp(prefix="gbench_omnidoc_")
    pred_dir = os.path.join(workdir, "predictions")
    out_dir = os.path.join(workdir, "result")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    kept: List[Dict[str, Any]] = []
    for trace in result.get("sample_traces", []):
        gold = _omnidoc_gold(trace.get("gold_answer"))
        page_name = str(gold.get("page_name") or "").strip()
        if not page_name:
            continue
        stem = os.path.splitext(os.path.basename(page_name))[0]
        with open(os.path.join(pred_dir, stem + ".md"), "w", encoding="utf-8") as f:
            f.write(trace.get("response_text") or "")
        if page_name in anno_by_name:
            kept.append(anno_by_name[page_name])
    if not kept:
        raise RuntimeError("omnidocbench: no predictions could be paired to ground-truth pages")
    gt_json = os.path.join(workdir, "gt.json")
    with open(gt_json, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    workers = max(1, min(16, (os.cpu_count() or 8) // 3))
    cmd = _build_command(image, gt_json, pred_dir, out_dir, workers)
    logger.info("omnidocbench: scoring %d pages via %s", len(kept), image)
    proc = subprocess.run(cmd, capture_output=True, text=True)

    summary = _parse_summary(out_dir)
    if not summary:
        raise RuntimeError(
            "omnidocbench: the evaluator produced no run summary (scoring failed) - a "
            "harness/infra failure, not a 0. Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])

    metrics = summary.get("metrics", {})
    overall = summary.get("overall_notebook")
    result["omnidocbench_overall"] = round(overall, 2) if overall is not None else None
    result["text_edit_distance"] = _raw_value(metrics, "text_block_Edit_dist")
    result["formula_cdm"] = _nb_value(metrics, "display_formula_CDM")             # 0-100, higher better
    result["table_teds"] = _nb_value(metrics, "table_TEDS")                       # 0-100, higher better
    result["reading_order_edit_distance"] = _raw_value(metrics, "reading_order_Edit_dist")
    result["pages_scored"] = len(kept)
    if overall is not None:
        result["accuracy"] = round(overall, 2)          # canonical headline = the composite
        result["correct_answers"] = 0
    else:
        # The composite needs all three of text/table/formula; a small --eval-limit subset may
        # contain no tables or formulas (TEDS/CDM then have no data). Don't pass off the interim
        # per-page pass rate as the score - mark it undefined.
        result["accuracy"] = None
        result["correct_answers"] = 0
        result["omnidocbench_note"] = (
            "composite undefined: the scored pages have no tables and/or display formulas, so "
            "TEDS/CDM have no data. Run the full set (no --eval-limit) for the OmniDocBench "
            "composite; the per-modality values above are over whatever modalities were present.")
    result["metric"] = (
        "OmniDocBench end2end composite = ((1-text_EditDist)*100 + table_TEDS*100 + "
        "formula_CDM*100)/3, computed by the official OmniDocBench evaluator image (CDM formula + "
        "TEDS table + edit-distance text + reading order); per-modality values also reported.")
    # Full dataset via the canonical evaluator AND greedy: a --thinking run (temperature 1.0) is
    # not the greedy leaderboard protocol, so it is not directly comparable either.
    noncanon = []
    if limit:
        noncanon.append(f"--eval-limit subset ({int(limit)} pages), not the full set")
    if enable_thinking:
        noncanon.append("--thinking run (non-greedy temperature), not the greedy leaderboard protocol")
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    return result
