# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC

"""Canonical-sync provenance registry.

One source of truth for when each eval was last reconciled against its upstream benchmark, and
against what exact version. The dispatcher stamps `canonical_sync` onto every eval result, and
`gbench --eval-provenance` prints the table.

Each entry records:
  synced    the reconcile date (YYYY-MM-DD)
  upstream  the upstream project or package (repo, PyPI name, or "roadmap"/"internal")
  revision  the pinned commit, tag, or version reconciled against (optional)
  dataset   the dataset id and revision the loader reads (optional)
  checksum  a content fingerprint that backs `synced` with a VERIFIABLE hash rather than a
            self-attested date: the sha256 of the pinned dataset snapshot, or the upstream
            tree/commit sha (optional). `gbench --eval-provenance` reports an entry as
            "version-pinned" when it carries a `revision` or `checksum`, else "date-only".
  method    how it was reconciled (for example "delegated to pinned harness", "metric reconciled
            via trace-walk", "CLI flags verified against the wheel")

A suite with no entry is reported as status "unverified": it has not yet been formally reconciled,
so `gbench --eval-provenance` doubles as the remaining reconcile worklist. Add entries as suites are
reconciled; do not invent a date or a commit that was not actually checked.
"""

from typing import Any, Dict

# Suites reconciled against a pinned external harness (delegated container / adapter builds).
CANONICAL_SYNC: Dict[str, Dict[str, Any]] = {
    "gaia2": {
        "synced": "2026-09-09", "upstream": "meta-agents-research-environments (PyPI)",
        "revision": "1.2.0", "dataset": "meta-agents-research-environments/gaia2",
        "method": "delegated to are-benchmark; CLI flags verified against the 1.2.0 wheel",
    },
    "wildclawbench": {
        "synced": "2026-09-09", "upstream": "github.com/internlm/WildClawBench",
        "revision": "316334ccc4a87b9b5635ad73da99b4dfc0b3887e", "dataset": "internlm/WildClawBench",
        "method": "delegated to eval/run_batch.py (OpenClaw); full real-model E2E verified",
    },
    "skillsbench": {
        "synced": "2026-09-08", "upstream": "github.com/benchflow-ai/skillsbench",
        "revision": "9a1f4dd5f7659f75707435da3ce854b6e48321d1",
        "method": "delegated to BenchFlow 0.6.3 (deterministic verifier); oracle pipeline verified",
    },
    "mcp_bench": {
        "synced": "2026-09-08", "upstream": "github.com/Accenture/mcp-bench",
        "revision": "7a8eaeae", "method": "delegated in-container; 28/28 servers spawn-verified",
    },
    "toolbench": {
        "synced": "2026-09-08", "upstream": "StableToolBench",
        "revision": "aa4ed9f", "method": "delegated qa_pipeline DFSDT; SoPR/SoWR judge = Gemini cascade",
    },
    "multipl_e": {
        "synced": "2026-09-08", "upstream": "github.com/nuprl/MultiPL-E",
        "revision": "3025a531", "dataset": "nuprl/MultiPL-E",
        "method": "delegated to the official evaluator image (24 language toolchains)",
    },
    "swe_bench_pro": {
        "synced": "2026-09-08", "upstream": "github.com/ScaleAI/SWE-bench_Pro",
        "dataset": "ScaleAI/SWE-bench_Pro", "method": "agentic mini-swe-agent + Pro exec scorer",
    },
    "complexfuncbench": {
        "synced": "2026-09-08", "upstream": "THUDM/ComplexFuncBench",
        "dataset": "THUDM/ComplexFuncBench",
        "method": "in-process ComplexEval port (golden loop + 4-tier CompareFC); judge = Gemini cascade",
    },
    "gaia": {
        "synced": "2026-09-08", "upstream": "gaia-benchmark/GAIA",
        "dataset": "gaia-benchmark/GAIA", "method": "web_search tool loop; GEMINI search backend",
    },
    # Suites whose metric/prompt/extraction were reconciled in the Wave 2 fidelity pass.
    "simpleqa": {"synced": "2026-09-07", "dataset": "basicv8vc/SimpleQA",
                 "method": "Wave 2: GRADER_TEMPLATE rubric + F-score/NOT_ATTEMPTED via trace-walk"},
    "mmlu": {"synced": "2026-09-07", "dataset": "cais/mmlu",
             "method": "Wave 2: canonical 5-shot via dev split"},
    "mmlu_pro": {"synced": "2026-09-07", "dataset": "TIGER-Lab/MMLU-Pro",
                 "method": "Wave 2: honors --eval-n-shot (5-shot CoT)"},
    "hmmt": {"synced": "2026-09-07", "dataset": "MathArena/hmmt_feb_2025",
             "method": "Wave 2: sympy/math_verify grader + boxed-answer format instruction"},
    "aime": {"synced": "2026-09-07", "dataset": "AI-MO/aimo-validation-aime",
             "method": "Wave 2: pass@k recomputed post-cutoff-only"},
    "amc_aime": {"synced": "2026-09-07", "dataset": "AI-MO/aimo-validation-amc",
                 "method": "Wave 2: per-competition accuracy; contamination note"},
    "cruxeval": {"synced": "2026-09-07", "dataset": "cruxeval-org/cruxeval",
                 "method": "Wave 2: canonical [PYTHON]/[ANSWER] one-shot prompt + assert-RHS extraction"},
    "multilingual_mmlu": {"synced": "2026-09-07", "dataset": "openai/MMMLU",
                          "method": "Wave 2: swapped to human-translated MMMLU; CJK/Arabic extraction"},
    "docvqa": {"synced": "2026-09-07", "dataset": "lmms-lab-encoder/DocVQA",
               "method": "Wave 2: mean-ANLS; removed silent smoke-fixture fallback"},
    "infographicvqa": {"synced": "2026-09-07", "dataset": "mm-eval/InfographicVQA",
                       "method": "Wave 2: mean-ANLS"},
    "semantic_keypoint": {"synced": "2026-09-07", "dataset": "HongxinLi/ScreenSpot_v2",
                          "method": "Wave 2: point-in-bbox (not fixed-radius)"},
    "charxiv": {"synced": "2026-09-07", "dataset": "princeton-nlp/CharXiv",
                "method": "Wave 2: Gemini-cascade judge (relaxed-match fallback)"},
    "bundled_detection": {"synced": "2026-09-07", "dataset": "detection-datasets/coco",
                          "method": "Wave 2: full COCO mAP; prompt no longer leaks classes"},
    "omnidocbench": {"synced": "2026-09-07", "dataset": "opendatalab/OmniDocBench",
                     "method": "Wave 2: per-modality TEDS + formula edit distance; reading-order"},
    "bfcl": {"synced": "2026-09-07", "method": "Wave 2: bijective matching + unexpected-param; turn0 fix"},
    "bfcl_v3_live": {"synced": "2026-09-07", "method": "Wave 2: structural possible-answer + bijective name-set"},
    "seal_tools": {"synced": "2026-09-07", "dataset": "casey-martin/Seal-Tools",
                   "method": "Wave 2: pooled micro P/R/F1 over all gold calls"},
    "nestful": {"synced": "2026-09-07", "dataset": "ibm-research/nestful",
                "method": "Wave 2: position-aligned Seq Match + F1-Func/F1-Param"},
    "acebench": {"synced": "2026-09-10", "dataset": "oliveirabruno01/acebench",
                 "revision": "5c549d5217ba756ddfecb996666a3e6e67872af4",
                 "method": "ACEBench-en Normal+Special subset (agent category + Chinese half excluded; "
                           "leaderboard_comparable=False); error_param VALUE check confirmed vs official checker"},
    "mrcr": {"synced": "2026-09-07", "dataset": "openai/mrcr",
             "method": "Wave 2: autojunk canonical; failures count as 0"},
    "custom_jsonl": {"synced": "2026-09-07", "method": "Wave 2: first-present-key gold; keeps falsy golds"},
    "cyberseceval": {"synced": "2026-09-07", "dataset": "walledai/CyberSecEval",
                     "method": "Wave 2: whole-language ICD; all 8 instruct splits; no injected suffix"},
    "wmdp": {"synced": "2026-09-07", "dataset": "cais/wmdp",
             "method": "Wave 2: added wmdp-cyber subset (loglikelihood MCQA on roadmap)"},
    "api_bank": {"synced": "2026-09-07", "dataset": "liminghao1630/API-Bank",
                 "method": "Wave 2: Level-1 + Level-2 (L3 + ROUGE-L on roadmap)"},
    "humanitys_last_exam": {"synced": "2026-09-07", "dataset": "cais/hle",
                            "method": "Wave 2: image attach (data-URI) + calibration error"},
    "medxpertqa": {"synced": "2026-09-07", "dataset": "TsinghuaC3I/MedXpertQA",
                   "method": "Wave 2: images.zip -> base_dir attach; honest text-only warning"},
    "deepsearch_qa": {"synced": "2026-09-07", "dataset": "xbench/DeepSearch-2510",
                      "method": "Wave 2: added Gemini-cascade judge"},
    "frames": {"synced": "2026-09-07", "dataset": "google/frames-benchmark",
               "method": "Wave 2: canonical FRAMES rubric prompt"},
    "lmsys_noncoding_hard": {"synced": "2026-09-15", "dataset": "WildEval/WildBench",
                             "method": "WB-Score 1-10 judge; filter coding tasks; canonical "
                                       "(mean_raw-5)*20 rescale, range -80..+100"},
    "livebench": {"synced": "2026-09-07",
                  "dataset": "livebench/{coding,data_analysis,instruction_following,language,math,reasoning} "
                             "(6 core categories; agentic_coding excluded)",
                  "revision": "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb",
                  "method": "6 core categories via LiveBench's own CLI; removed LLM judge (canonical "
                            "forbids it), deterministic-only; monthly-refreshed - revision marks the "
                            "reconciled release (livebench/math)"},
    "beam_128k": {"synced": "2026-09-07", "dataset": "Mohammadta/BEAM",
                  "method": "Wave 2: rubric-nugget coverage judge"},
    "cimemories": {"synced": "2026-09-07", "dataset": "facebook/CIMemories",
                   "method": "Wave 2: removed privacy-injection prompt (Violation@n split on roadmap)"},
    "codeforces": {"synced": "2026-09-07", "dataset": "open-r1/codeforces-cots",
                   "method": "Wave 2: checker argv order + verdict from stdout not exit code"},
    "gsm8k": {"synced": "2026-09-15", "dataset": "openai/gsm8k",
              "method": "8-shot CoT with canonical fixed Wei et al. 2022 exemplars (lm-eval "
                        "gsm8k_cot); shared anchored_span extractor (last textual anchor)"},

    # WS1 back-fill: reconciled at the code level (loader dataset/split/config + scorer confirmed
    # against the canonical metric documented in each runner and against docs/RESOURCES.md).
    # Pillar 1: general knowledge and scientific reasoning.
    "gpqa": {"synced": "2026-09-09", "dataset": "Idavidrein/gpqa (gpqa_main, train)",
             "method": "zero-shot CoT; deterministic per-item choice shuffle; 'Final Answer: (X)' accuracy"},
    "gpqa_diamond": {"synced": "2026-09-09", "dataset": "Idavidrein/gpqa (gpqa_diamond, train)",
                     "method": "zero-shot CoT; deterministic per-item choice shuffle; accuracy"},
    "arc_agi": {"synced": "2026-09-09", "dataset": "dataartist/arc-agi (ARC-AGI-1 eval)",
                "method": "grid-serialized tasks; exact integer-grid match (2 attempts)"},
    "mmlu_redux": {"synced": "2026-09-09", "dataset": "edinburgh-dawg/mmlu-redux (per-subject, test)",
                   "method": "error-corrected MMLU MCQA; boxed-letter extraction; accuracy"},
    "lab_bench": {"synced": "2026-09-09", "dataset": "futurehouse/lab-bench (ProtocolQA, train)",
                  "method": "biology/chemistry MCQA; boxed-letter extraction; accuracy"},
    "healthbench": {"synced": "2026-09-09", "dataset": "openai/healthbench (oss_eval.jsonl)",
                    "method": "weighted physician-rubric grader (Gemini cascade judge); per-conversation rubric score"},
    "causalbench": {"synced": "2026-09-09", "dataset": "causal-nlp/corr2cause (test)",
                    "revision": "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53",
                    "method": "corr2cause causal-relation classification; headline = positive-class F1 "
                              "(majority-No set; native prompt, leaderboard_comparable=False)"},
    "i18n_translate": {"synced": "2026-09-09", "dataset": "wmt/wmt19 (per language pair, validation)",
                       "method": "WMT19 translation; chrF via sacrebleu (local chrF fallback)"},
    "ifeval": {"synced": "2026-09-15", "dataset": "google/IFEval (train)",
               "method": "programmatic per-instruction checkers (canonical two-regex highlight "
                         "counter); prompt-level STRICT accuracy"},
    "cybergym": {"synced": "2026-09-09", "dataset": "sunblaze-ucb/cybergym (per-CVE files)",
                 "method": "vulnerability/exploit reasoning; Gemini-cascade judge"},

    # Pillar 2: mathematics and proofs.
    "imo_answer_bench": {"synced": "2026-09-09", "dataset": "OpenEvals/IMO-AnswerBench (train)",
                         "method": "short-final-answer olympiad; boxed extraction + Gemini-cascade equivalence judge"},
    "putnam": {"synced": "2026-09-09", "dataset": "amitayusht/PutnamBench (train)",
               "method": "natural-language proof track; Gemini-cascade proof judge"},
    "putnam_formal": {"synced": "2026-09-09", "dataset": "amitayusht/PutnamBench (train)",
                      "method": "Lean 4 formal track; Docker sandbox compile-check (infra_required)"},

    # Pillar 3: coding and algorithmic design.
    "aider_polyglot": {"synced": "2026-09-09", "upstream": "github.com/Aider-AI/polyglot-benchmark",
                       "method": "delegated to aider Coder over 225 Exercism exercises (6 langs); diff-edit pass@2 (infra_required)"},
    "bigcodebench": {"synced": "2026-09-09", "dataset": "bigcode/bigcodebench (instruct)",
                     "method": "sanitize+calibrate; execution-based calibrated pass@1 (infra_required)"},
    "lcb": {"synced": "2026-09-09", "dataset": "livecodebench/code_generation_lite (release_v6)",
            "method": "LiveCodeBench execution-based pass@1; contamination-pruned release tag"},
    "scicode": {"synced": "2026-09-09", "dataset": "SciCode1/SciCode (test) + SciCode test-data",
                "method": "sequential per-sub-step gencode protocol; execution scoring (infra_required)"},
    "ojbench": {"synced": "2026-09-09", "dataset": "He-Ren/OJBench_testdata (prompts/full.jsonl)",
                "method": "online-judge Pass@1 (all test cases) via DMOJ sandbox (infra_required)"},
    "copilot_bench_swe": {"synced": "2026-09-08", "dataset": "princeton-nlp/SWE-bench_Verified",
                          "upstream": "swebench Docker harness (vanilla)",
                          "method": "delegated resolved-rate via shared swebench_common; leaderboard_comparable gate (infra_required)"},
    "swe_bench_live": {"synced": "2026-09-10", "dataset": "SWE-bench-Live/SWE-bench-Live",
                       "upstream": "github.com/SWE-bench-Live/SWE-bench-Live",
                       "revision": "a145aa87c62361532dacaa243398978164b234b7",
                       "method": "rollout in-process; scoring runs the pinned SWE-bench-Live fork in a "
                                 "LOCAL image (isolated from upstream swebench), DooD resolved-rate (infra_required)"},
    "swe_bench_multilingual": {"synced": "2026-09-08", "dataset": "SWE-bench/SWE-bench_Multilingual",
                               "upstream": "swebench Docker harness (vanilla)",
                               "method": "Group C: delegated resolved-rate via shared swebench_common (infra_required)"},
    "multi_swe_bench": {"synced": "2026-09-09", "dataset": "ByteDance-Seed/Multi-SWE-bench",
                        "upstream": "github.com/multi-swe-bench/multi-swe-bench",
                        "method": "delegated to the project's own multi_swe_bench harness; resolved-rate (infra_required)"},
    "spider2": {"synced": "2026-09-08", "upstream": "github.com/xlang-ai/Spider2",
                "dataset": "xlangai/spider2-lite (repo metadata + schema injected)",
                "revision": "ce2e836654acc548dad7d348b1ad0e5ad0564f9e",
                "method": "Group C: Spider 2.0-lite execution accuracy vs SQLite; upstream vectors_match (infra_required)"},

    # Pillar 4: long context and retrieval.
    "aa_lcr": {"synced": "2026-09-09", "dataset": "ArtificialAnalysis/AA-LCR (csv + extracted-text zip)",
               "method": "long-context reasoning over attached documents; Gemini-cascade judge; accuracy"},
    "culer": {"synced": "2026-09-09", "dataset": "zai-org/LongBench-v2 (train)",
              "method": "long-context 4-choice MCQA, left-cropped to 128k; accuracy"},
    "loft_x_arxiv": {"synced": "2026-09-09", "upstream": "github.com/google-deepmind/loft",
                     "dataset": "LOFT SciFact bundle (loft-bench GCS)",
                     "method": "LOFT in-context retrieval on SciFact passages; positional recall"},
    "ruler": {"synced": "2026-09-09", "upstream": "github.com/NVIDIA/RULER",
              "method": "drives RULER prepare.py (GBENCH_RULER_DIR) for per-tokenizer synthetic data; Avg over 13 tasks x 6 bands (infra_required)"},

    # Pillar 5: tool use and agentic workflows.
    "agent_dojo": {"synced": "2026-09-09", "upstream": "github.com/ethz-spylab/agentdojo",
                   "dataset": "ffuuugor/agentdojo-dump",
                   "method": "utility + prompt-injection security in the tool-exec env (infra_required)"},
    "bfcl_v4_agentic": {"synced": "2026-09-09", "upstream": "gorilla-llm bfcl-eval (v4 agentic)",
                        "method": "delegated to Berkeley bfcl-eval stateful multi-turn tool-exec harness (infra_required)"},
    "browsecomp": {"synced": "2026-09-09", "dataset": "smolagents/browse_comp (test)",
                   "method": "per-row XOR-decrypt (canary); Gemini-cascade judge; accuracy"},
    "gorilla_apibench": {"synced": "2026-09-10", "dataset": "gorilla-llm/APIBench (huggingface_eval.json)",
                         "method": "single-call AST match, stricter exact-gold-call (conservative); canonical domain-match on roadmap (verified vs ast_eval_hf.py)"},
    "mcp_atlas": {"synced": "2026-09-09", "dataset": "ScaleAI/MCP-Atlas (train)",
                  "method": "multi-server MCP tool orchestration; Gemini-cascade judge"},
    "nexus_function_calling": {"synced": "2026-09-09", "dataset": "Nexusflow/NexusRaven_API_evaluation (CC-BY-NC-4.0)",
                               "method": "zero-shot single-call; AST/structural call match"},
    "t_eval": {"synced": "2026-09-08", "dataset": "lovesnowbest/T-Eval",
               "method": "tool-use sub-skills; import blocker fixed (Group B); 6-dimension graded scoring on roadmap"},
    "tau2": {"synced": "2026-09-09", "upstream": "github.com/sierra-research/tau2-bench",
             "method": "delegated to tau2 harness; pass^k over airline/retail/telecom (infra_required)"},
    "tau3": {"synced": "2026-09-09", "upstream": "github.com/sierra-research/tau2-bench (tau3)",
             "method": "banking_knowledge RAG domain; delegated harness + Gemini-cascade judge (infra_required)"},

    # Pillar 6: multimodal vision and grounding.
    "chartqa": {"synced": "2026-09-09", "dataset": "ahmed-masry/ChartQA (test)",
                "method": "relaxed accuracy (numeric within 5%, else exact) via vqa_common.eval_relaxed"},
    "coco_caption": {"synced": "2026-09-09", "dataset": "lmms-lab/COCO-Caption (Karpathy 5k test)",
                     "method": "corpus-level CIDEr-D primary (pycocoevalcap) + BLEU/ROUGE (infra_required: java)"},
    "mmmu_pro": {"synced": "2026-09-09", "dataset": "MMMU/MMMU_Pro (standard, 10 options, test)",
                 "method": "10-option multimodal MCQA; accuracy; temperature 1.0 per Gemma4 IT vision eval"},
    "screenspot": {"synced": "2026-09-09", "dataset": "HongxinLi/ScreenSpot_v2 (test)",
                   "method": "GUI coordinate grounding; predicted point-in-gold-bbox accuracy"},
    "textvqa": {"synced": "2026-09-09", "dataset": "lmms-lab/textvqa (validation)",
                "method": "VQA soft-accuracy min(matching-annotators/3, 1); OCR VQA"},

    # Cross-pillar: economic knowledge work + sandboxed terminal.
    "gdpval": {"synced": "2026-09-14", "dataset": "openai/gdpval (train + file attachments + deliverable_files)",
               "method": "220 economic knowledge-work tasks; canonical PAIRWISE WIN-RATE of the model "
                         "deliverable vs the expert reference (deliverable_files), Gemini-cascade judge, "
                         "position-swapped; text endpoint competes on rendered content only"},
    "terminal_bench": {"synced": "2026-09-08", "dataset": "terminal-bench/terminal-bench-2-1",
                       "upstream": "Harbor framework (Terminal-Bench 2.1)",
                       "revision": "terminal-bench-2.1",  # Harbor task-registry version (not an HF dataset)
                       "method": "Group C: delegated to Harbor Docker sandboxes; resolved-rate; 1 trial/task "
                                 "(TB2.1 leaderboard uses >=5), leaderboard_comparable=False; --thinking temp bug fixed (infra_required)"},
}


def canonical_sync_for(eval_name: str) -> Dict[str, Any]:
    """Provenance record for a suite, or an honest 'unverified' marker if it has no entry yet."""
    entry = CANONICAL_SYNC.get(eval_name)
    if entry is None:
        return {"status": "unverified",
                "note": "not yet formally reconciled against upstream (see canonical_sync.py)"}
    return {"status": "reconciled", **entry}
