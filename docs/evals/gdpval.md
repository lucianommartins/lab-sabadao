# gdpval: pairwise win-rate vs the expert reference

Canonical GDPval (`openai/gdpval`): 220 economically-valuable knowledge-work tasks whose
answer is a **produced file deliverable** (`.pdf`, `.xlsx`, `.docx`, `.pptx`). Each task ships
an **expert reference deliverable** (`deliverable_files`) and a long grading rubric.

## Metric: pairwise win-rate (the canonical GDPval measure)

GDPval is scored by comparing a model's deliverable against the human-expert reference in a
blind pairwise judgement (win / tie / loss) and reporting the **win-rate**. `run_gdpval`
reproduces this:

1. The model answers the task (with input attachments; see below).
2. The expert reference (`deliverable_files`) is rendered to text with the same converters.
3. A judge is shown the model answer and the reference as **A / B** (order arbitrary), judged
   against the rubric's content. The comparison is **position-swapped** (model as A, then as B)
   and the two verdicts averaged to cancel order bias.
4. Headline `accuracy` = **win-rate** = mean pairwise score (win 1.0, tie 0.5, loss 0.0) over
   the scored tasks. `pass rate` semantics: a task counts as "won or tied" when its score ≥ 0.5.

The judge is gbench's standard Gemini cascade (a gbench convention). `run_gdpval` hard-errors
(`infra_required`) without `GEMINI_API_KEY`; it never returns a fabricated number.

**`leaderboard_comparable` is always False** for one honest reason: a plain-text `/v1` endpoint
cannot emit an `.xlsx`/`.docx`/`.pdf`, so the expert reference FILE is rendered to text and the
model competes on rendered **content** only - it can never match live formulas/formatting. That
is a limitation of the served interface, not a fabricated score, and it is why this is a
gbench-internal win-rate rather than OpenAI's GDPval number. The Gemini judge is the accepted
gbench grader, not the reason for the flag.

Honest handling of edge cases: a task whose reference cannot be rendered to text (image-only /
unrenderable) or that hits a judge outage is **excluded** from the win-rate (not scored as a
loss); an empty/failed model deliverable **is** counted, as a loss. `tasks_scored`,
`tasks_no_reference`, and `tasks_judge_outage` are reported so the denominator is legible.

## What the benchmark actually demands (measured 2026-08-17)

| | |
| --- | ---: |
| tasks | 220 |
| tasks whose input is one or more **attached files** | **125 (57%)** |
| rubric criteria | 10,453 (**47.5 per task**) |
| criteria that depend on file format/properties | **1,474 (14.1%)**, **15.1%** of all rubric points |
| **criteria that are content-only** | **8,979 (85.9%)**, 84.9% of points |

Input file types (all 261): `xlsx` 86, `pdf` 74, `docx` 67, `png` 9, `wav` 8, `zip` 3,
`jpg` 3, `mp3` 2, `webp` 2, `mp4` 2, `step` 2, `txt` 1, `psd` 1, `pptx` 1.
Expected deliverable types: `pdf` 85, `xlsx` 65, `docx` 64, `pptx` 17, and a small tail.

File-property criteria are a bigger share of points (15.1%) than of criteria (14.1%)
because they are weighted higher: 1.84 points each on average versus 1.69 for content
criteria. GDPval scores "is this even the right kind of artifact" above any single
content point.

So the "it's all file properties" framing this doc used to carry was wrong. ~85% of the
points are reachable by a text endpoint. The larger obstacle is the **input** side, and
that one is fixable.

## The files are downloadable, but NOT by `load_dataset`

This is the detail that keeps catching people. The dataset is two-stage:

```python
# stage 1 - metadata only. Pulls ~2 MB: README + the parquet holding prompts,
# rubrics, and the file PATHS as plain strings. NO binaries.
ds = load_dataset("openai/gdpval", split="train")
ds[0]["reference_files"]      # ['reference_files/cc781e.../Population v2.xlsx']  <- a path

# stage 2 - the actual bytes. One call per file, cached afterwards.
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="openai/gdpval",
                       filename=ds[0]["reference_files"][0],
                       repo_type="dataset")
```

Verified on a clean cache: after `load_dataset` alone, **1 of 261** reference files and
**0 of 248** deliverable files are present locally. The suite's loader only ever did stage
one, which is why the attachments never reached the model.

`deliverable_files` are the **expert reference deliverables** and are now **REQUIRED**: they
are the reference side of the pairwise win-rate. A task whose reference cannot be fetched or
rendered to text is excluded from the win-rate (not scored as a loss), so fetch them before a
graded run (see below).

## Fetching the attachments (do this once)

The `hf` CLI pulls everything in one call. `--dry-run` first if you want the size.

```bash
# 1,668 MB, 302 objects - the INPUTS, without which 125 tasks are unanswerable
hf download openai/gdpval --repo-type dataset --include "reference_files/*"

# 623 MB, 249 objects - the EXPERT REFERENCE deliverables (REQUIRED: the pairwise reference)
hf download openai/gdpval --repo-type dataset --include "deliverable_files/*"
```

Both land in the same HF cache `hf_hub_download` reads, so the loader picks them up with
no further network access. The globs fetch slightly more than the tasks reference (302 vs
261 objects, 249 vs 248) because the repo holds a few unreferenced files, harmless.

> **If the download stalls, disable Xet.** `hf_xet` fetches deduplicated chunks and then
> reassembles them (the `Reconstructing …` progress line). Observed 2026-08-17 on this box:
> wedged at 1.27 GB / 1.31 GB with the process alive but zero throughput, leaving 16
> `.incomplete` blobs. `--force-download` does not help - it re-runs the same backend.
> ```bash
> export HF_HUB_DISABLE_XET=1      # plain HTTPS per file; completed in ~22 s
> ```
> Further knobs if needed: `--max-workers 2`, and `HF_HUB_DOWNLOAD_TIMEOUT` /
> `HF_HUB_ETAG_TIMEOUT` (both default to a rather tight **10 s**).

Validated after download: 261/261 referenced reference files and 248/248 deliverables
present, all with correct magic bytes, 0 missing, 0 zero-byte; **all 220 tasks have their
inputs**. Clean up leftovers with
`find ~/.cache/huggingface/hub/datasets--openai--gdpval -name '*.incomplete' -delete`.

## Converter dependencies

All verified against the real attachments on 2026-08-18: **256 of 261 input files (98%)
read successfully.**

| type | input files | package | verified read |
| --- | ---: | --- | --- |
| `xlsx` | 86 | `openpyxl` 3.1.5 (via pandas) | 1 sheet, (1516, 8) |
| `pdf` | 74 | `PyMuPDF` (`fitz`), `pypdf` | 8 pages, text extracts |
| `docx` | 67 | `python-docx` 1.2.0 | 13 paragraphs, 123 chars |
| `png`/`jpg`/`webp` | 14 | `Pillow` + gbench's `extract_lossless_image_b64` | dimensions + mode |
| `wav`/`mp3` | 10 | `soundfile` | 116.5 s @ 48 kHz |
| `zip` | 3 | stdlib `zipfile` | 16 entries |
| `pptx` | 1 | `python-pptx` 1.0.2 | 6 slides |
| `txt` | 1 | stdlib | 12,423 chars |
| **`mp4`, `step`, `psd`** | **5** | - | **no converter; see below** |

```bash
pip install openpyxl python-docx python-pptx
```

The 5 unreadable files are 2 `.mp4`, 2 `.step` (CAD) and 1 `.psd`. None has a sensible
text or image projection, so the loader should attach an explicit "this input could not be
rendered" note for those tasks rather than sending the prompt as if nothing were missing;
otherwise they look like model failures.

> **Trap: `pip install docx` is the WRONG package.** `docx` 0.2.4 is an abandoned 2014
> Python-2 project that also imports as `docx`, so it shadows the real one and fails at
> import with `ModuleNotFoundError: No module named 'exceptions'` (a Py2 builtin removed in
> Python 3). The maintained package is **`python-docx`**. If `docx` got installed by
> mistake, `pip uninstall -y docx` first; installing `python-docx` alongside it leaves the
> broken module winning the import.

## What is implemented

1. **Input ingestion** (the 125 file-input tasks). The loader resolves each `reference_files`
   path and converts it: pdf → text (PyMuPDF); xlsx → CSV (pandas + openpyxl); docx → text
   (python-docx); pptx → text (python-pptx); png/jpg → the `extract_lossless_image_b64` path;
   wav/mp3 → a note (not transcribed). An unrenderable input attaches an explicit note so the
   task does not read as a model failure.
2. **Reference ingestion + pairwise judge.** The expert `deliverable_files` are rendered to
   text and the model answer is compared against them by the Gemini cascade, position-swapped,
   for the win-rate (see *Metric* above).

The only path to a number directly comparable with OpenAI's published GDPval win-rate is a
**file-producing agent harness** (tool use / code execution that writes real `.xlsx`/`.docx`/
`.pdf` deliverables), so the judge compares like with like. That is a separate project; until
then this is a gbench-internal win-rate and `leaderboard_comparable` stays False.

## Reporting

* `leaderboard_comparable: false`; a 60% win-rate here is not OpenAI's 60% (text vs a file
  deliverable rendered to text).
* `tasks_scored` / `tasks_no_reference` / `tasks_judge_outage` name the win-rate denominator;
  excluded tasks are not counted as losses, an empty model answer is.
* `wins` / `ties_or_split` / `losses` and `consistent_pairs` (how often the two position-swapped
  verdicts agreed) expose the judgement quality.

