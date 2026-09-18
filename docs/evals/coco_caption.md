# coco_caption setup

Canonical MS-COCO image captioning (`lmms-lab/COCO-Caption`, **2014 validation split**,
not the Karpathy test split most published CIDEr numbers use, so scores are not
directly comparable to those) scored
with the standard COCO metrics (CIDEr-D primary, plus BLEU-1..4, ROUGE-L and,
when available, METEOR) via `pycocoevalcap`.

## Requirements
- `pip install gbench[evals]` (installs `pycocoevalcap`, which pulls in `pycocotools`).
- A **Java runtime** (the PTB tokenizer + METEOR use bundled CoreNLP jars).
- A **vision-capable** endpoint (the image is sent as a base64 data URL to `/v1/chat/completions`).

If `pycocoevalcap` (or Java) is missing the suite hard-errors (`infra_required`); it never skips or emits a fabricated number.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals coco_caption --eval-limit 500
```

## Scoring
Corpus-level metrics are computed post-hoc over all predictions and reported in
the result JSON (`cider`, `bleu4`, `rouge_l`, `meteor`, ...). The per-sample
`is_correct`/`accuracy` is a convenience pass-rate at per-image CIDEr ≥ 0.5; the
headline canonical number is the corpus **CIDEr**.
