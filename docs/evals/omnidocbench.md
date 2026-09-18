# omnidocbench

Canonical OmniDocBench (`opendatalab/OmniDocBench`): multimodal document parsing. The model
reads a page **image** and transcribes it to **markdown** (text + tables as HTML + formulas as
LaTeX). Scoring is OmniDocBench's **end2end per-modality composite**, computed by OmniDocBench's
own evaluator:

**Overall = ((1 − Text Edit Distance)·100 + Table TEDS·100 + Formula CDM·100) / 3**

- **Text** (text_block/title/references/captions) → normalized edit distance
- **Table** → TEDS (tree-edit similarity)
- **Display formula** → **CDM** (Character Detection Matching, renders LaTeX and matches)
- **Reading order** → edit distance

> **On figures:** OmniDocBench does **not** score figure *image content*. The `figure` category
> is in no scored group (only its text caption/footnote is scored). So transcribing a page to
> markdown loses nothing the benchmark measures; it is a text/table/formula parsing benchmark, not
> figure reproduction.

## Why it's containerized

OmniDocBench pins **Python <3.12** and **numpy==1.24.4**, which would break the torch/vLLM serving
env, and its CDM metric needs a TeX Live / ImageMagick 7 / Ghostscript runtime. So gbench does
**not** install it into the gbench serving environment. Instead it runs OmniDocBench's `pdf_validation.py` **inside a
locally-built image**: gbench generates the markdown transcriptions in-process (multimodal prompts
against your endpoint), writes them out, and scores them in the image. gbench hard-errors (never
silently skips) if Docker or the image is missing.

## Setup

gbench builds the evaluator image **locally** (it never pulls registry images). The gbench
Dockerfile mirrors OmniDocBench's own repro runtime (Python 3.10 + Ghostscript + TeX Live with CJK
+ ImageMagick 7, PDF-enabled) and installs the checkout's pinned deps:

```bash
# context = your OmniDocBench checkout
docker build -t gbench-omnidocbench \
    -f docker/omnidocbench.Dockerfile \
    $GBENCH_PREREQS_DIR/OmniDocBench
```
This is a large image (TeX Live); the first build is slow. Override the tag with
`GBENCH_OMNIDOCBENCH_IMAGE`. The dataset (page images + `OmniDocBench.json` annotations) is
pulled from HF Hub automatically at run time.

## Run

```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals omnidocbench \
       --suite-timeout 36000            # CDM rendering is slow; raise the timeout
```

What happens: gbench generates one markdown transcription per page against your endpoint, writes
each as `<image-stem>.md` plus a ground-truth json of exactly the pages it generated, then
`docker run`s the evaluator image (mounting the predictions, GT, and an output dir) to compute the
composite. It reads back `result/predictions_quick_match_run_summary.json` and reports
**`accuracy` = the composite**, with `text_edit_distance`, `table_teds`, `formula_cdm`,
`reading_order_edit_distance` and `pages_scored` alongside.

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_OMNIDOCBENCH_IMAGE` | evaluator image tag (default `gbench-omnidocbench`, the image built LOCALLY from `docker/omnidocbench.Dockerfile` above; gbench never pulls from a registry) |
| `GBENCH_OMNIDOCBENCH_TEMPERATURE` | per-suite sampling temperature override |

## Notes / caveats

- **`leaderboard_comparable`** is set for a **full** run (no `--eval-limit`); a limited run scores
  only the pages it generated and is a subset.
- **Runtime:** CDM renders every predicted formula with pdflatex + ImageMagick. The full run is
  slow. Use `--eval-limit N` for a smoke test.
- **No LLM judge:** scoring is OmniDocBench's static per-modality evaluator (CDM/TEDS/edit
  distance), not a model.
