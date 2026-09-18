# Third-party datasets: licenses, gated access, redistribution

gbench does **not** vendor or redistribute benchmark data. Each eval suite pulls its dataset at
runtime from its upstream source (mostly the Hugging Face Hub via `datasets.load_dataset`), into
your local HF cache. Running a benchmark and publishing a score is therefore subject to **that
dataset's own license and terms of use**, not gbench's Apache-2.0 license.

This inventory covers the datasets gbench loads via `load_dataset` / `hf_hub_download`. Suites that
instead stand up a container or clone an upstream harness (the SWE-bench family, `spider2`,
`terminal_bench`, `mcp_bench`, `wildclawbench`, `gaia2`, `skillsbench`, the roadmap `swe_lancer` /
`ui_control_osworld`, etc.) carry their own prerequisites and upstream licenses in their
`docs/evals/<suite>.md`.

> Snapshot taken 2026-09-14 by querying the HF Hub. **License tags and gated status can change
> upstream** - re-verify before a release, and have counsel confirm that evaluation use and the
> publication of accuracy numbers are permitted for each dataset your run touches.

## Requires action before running / publishing

### Gated: need a Hugging Face token and acceptance of the dataset's terms
A gated dataset returns HTTP 401/403 until you accept its terms on the dataset page and run with a
token (`huggingface-cli login` or `HF_TOKEN`). `--evals all` will hard-error on these until then.

| Dataset | Suite(s) | License | Note |
|---|---|---|---|
| `Idavidrein/gpqa` | `gpqa_diamond` | cc-by-4.0 | Ships a canary GUID; terms forbid training on / leaking the data. |
| `cais/hle` | `hle` | mit | Gated access; "Humanity's Last Exam". |
| `gaia-benchmark/GAIA` | `gaia` | none declared | Gated; treat as all-rights-reserved (see below). |

### Non-commercial (NC): evaluation/research use only, no commercial use
| Dataset | Suite(s) | License |
|---|---|---|
| `MathArena/hmmt_feb_2025` | `hmmt` | cc-by-nc-sa-4.0 |
| `facebook/CIMemories` | `cimemories` | cc-by-nc-4.0 |

### Copyleft (GPL): redistribution/derivative implications
| Dataset | Suite(s) | License |
|---|---|---|
| `HuggingFaceM4/ChartQA`, `ahmed-masry/ChartQA` | `chartqa` | gpl-3.0 |

### No license declared / unknown: treat as all-rights-reserved
No SPDX tag on the Hub means **no grant of rights**; do not assume redistribution or even
evaluation use is permitted. Confirm with the upstream authors or counsel before relying on these.

`He-Ren/OJBench_testdata` (`ojbench`), `HongxinLi/ScreenSpot_v2` (`screenspot`),
`WildEval/WildBench` (`lmsys_noncoding_hard`), `amitayusht/PutnamBench` (`putnam`),
`causal-nlp/corr2cause` (`causalbench`), `detection-datasets/coco` (`coco_caption`,
`semantic_keypoint`, `bundled_detection`), `lmms-lab/textvqa` (`textvqa`),
`mm-eval/InfographicVQA` (`infographicvqa`), `opendatalab/OmniDocBench` (`omnidocbench`),
`sunblaze-ucb/cybergym` (`cybergym`), `wish6424/MedXpertQA-Diagnosis` (`medxpertqa`),
`livecodebench/execution` (`cruxeval`/lcb family), `wmt/wmt19` (`i18n_translate`, tag "unknown").

## Permissive: attribution as the license requires
apache-2.0 / mit / cc-by-4.0 / cc-by-sa-4.0. These still require attribution (and share-alike for
cc-by-sa); comply with each license's notice terms.

| Dataset | License |
|---|---|
| `AI-MO/aimo-validation-aime`, `AI-MO/aimo-validation-amc` | apache-2.0 |
| `ArtificialAnalysis/AA-LCR`, `MMMU/MMMU_Pro`, `SciCode1/SciCode` | apache-2.0 |
| `google/IFEval`, `google/frames-benchmark` | apache-2.0 |
| `gorilla-llm/APIBench`, `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | apache-2.0 |
| `ibm-research/nestful`, `smolagents/browse_comp`, `zai-org/LongBench-v2` | apache-2.0 |
| `lmms-lab-encoder/DocVQA` | apache-2.0 |
| `TIGER-Lab/MMLU-Pro`, `TsinghuaC3I/MedXpertQA`, `basicv8vc/SimpleQA` | mit |
| `cais/mmlu`, `cais/wmdp`, `casey-martin/Seal-Tools`, `cruxeval-org/cruxeval` | mit |
| `nuprl/MultiPL-E`, `openai/MMMLU`, `openai/gsm8k`, `openai/healthbench`, `openai/mrcr` | mit |
| `walledai/CyberSecEval`, `xbench/DeepSearch-2510` | mit |
| `edinburgh-dawg/mmlu-redux`, `ScaleAI/MCP-Atlas` | cc-by-4.0 |
| `Mohammadta/BEAM`, `futurehouse/lab-bench`, `princeton-nlp/CharXiv` | cc-by-sa-4.0 |
| `livecodebench/test_generation` | cc |

## Serving workload dataset
- **ShareGPT** (`serving.py`, `--campaign chat-like`): `anon8231489123/ShareGPT_Vicuna_unfiltered` is
  downloaded just-in-time (to `~/.cache/gbench/`, or `SHAREGPT_PATH`) for the ShareGPT serving
  workload. Note it is a scraped dataset with contested license/provenance; `--dataset random` needs
  no external data.
