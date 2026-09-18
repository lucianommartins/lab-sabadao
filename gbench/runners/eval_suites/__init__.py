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

"""Native built-in evaluation benchmark suites package for gbench."""

from .loader import discover_and_register_plugins, CUSTOM_PILLARS

from .aa_lcr import run_aa_lcr
from .acebench import run_acebench
from .agent_dojo import run_agent_dojo
from .aider_polyglot import run_aider_polyglot
from .aime import run_aime
from .api_bank import run_api_bank
from .arc_agi import run_arc_agi
from .beam_128k import run_beam_128k
from .bfcl import run_bfcl
from .bfcl_v3_live import run_bfcl_v3_live
from .bfcl_v4_agentic import run_bfcl_v4_agentic
from .bigcodebench import run_bigcodebench
from .browsecomp import run_browsecomp
from .bundled_detection import run_bundled_detection
from .causalbench import run_causalbench
from .chartqa import run_chartqa
from .charxiv import run_charxiv
from .cimemories import run_cimemories
from .coco_caption import run_coco_caption
from .codeforces import run_codeforces
from .complexfuncbench import run_complexfuncbench
from .copilot_bench_swe import run_copilot_bench_swe
from .cruxeval import run_cruxeval
from .culer import run_culer
from .custom_jsonl import run_custom_jsonl
from .cyberseceval import run_cyberseceval
from .cybergym import run_cybergym
from .deepsearch_qa import run_deepsearch_qa
from .docvqa import run_docvqa
from .frames import run_frames
from .gaia import run_gaia
from .gaia2 import run_gaia2
from .gdpval import run_gdpval
from .gorilla_apibench import run_gorilla_apibench
from .gpqa import run_gpqa
from .gpqa_diamond import run_gpqa_diamond
from .gsm8k import run_gsm8k
from .healthbench import run_healthbench
from .hmmt import run_hmmt
from .humanitys_last_exam import run_humanitys_last_exam
from .i18n_translate import run_i18n_translate
from .ifeval import run_ifeval
from .imo_answer_bench import run_imo_answer_bench
from .infographicvqa import run_infographicvqa
from .lab_bench import run_lab_bench
from .lcb import run_lcb
from .livebench import run_livebench
from .lmsys_noncoding_hard import run_lmsys_noncoding_hard
from .loft_x_arxiv import run_loft_x_arxiv
from .mcp_atlas import run_mcp_atlas
from .mcp_bench import run_mcp_bench
from .medxpertqa import run_medxpertqa
from .mmlu import run_mmlu
from .mmlu_pro import run_mmlu_pro
from .mmlu_redux import run_mmlu_redux
from .mmmu_pro import run_mmmu_pro
from .mrcr import run_mrcr
from .multi_swe_bench import run_multi_swe_bench
from .multilingual_mmlu import run_multilingual_mmlu
from .multipl_e import run_multipl_e
from .nestful import run_nestful
from .amc_aime import run_amc_aime
from .nexus_function_calling import run_nexus_function_calling
from .ojbench import run_ojbench
from .omnidocbench import run_omnidocbench
from .putnam import run_putnam
from .putnam_formal import run_putnam_formal
from .ruler import run_ruler
from .scicode import run_scicode
from .screenspot import run_screenspot
from .seal_tools import run_seal_tools
from .semantic_keypoint import run_semantic_keypoint
from .simpleqa import run_simpleqa
from .skillsbench import run_skillsbench
from .spider2 import run_spider2
from .swe_bench_live import run_swe_bench_live
from .swe_bench_multilingual import run_swe_bench_multilingual
from .swe_bench_pro import run_swe_bench_pro
from .t_eval import run_t_eval
from .tau2 import run_tau2
from .tau3 import run_tau3
from .terminal_bench import run_terminal_bench
from .textvqa import run_textvqa
from .toolbench import run_toolbench
from .wildclawbench import run_wildclawbench
from .wmdp import run_wmdp

SUITES = {
    "aa_lcr": run_aa_lcr,
    "acebench": run_acebench,
    "agent_dojo": run_agent_dojo,
    "aider_polyglot": run_aider_polyglot,
    "aime": run_aime,
    "api_bank": run_api_bank,
    "arc_agi": run_arc_agi,
    "beam_128k": run_beam_128k,
    "bfcl": run_bfcl,
    "bfcl_v3_live": run_bfcl_v3_live,
    "bfcl_v4_agentic": run_bfcl_v4_agentic,
    "bigcodebench": run_bigcodebench,
    "browsecomp": run_browsecomp,
    "bundled_detection": run_bundled_detection,
    "causalbench": run_causalbench,
    "chartqa": run_chartqa,
    "charxiv": run_charxiv,
    "cimemories": run_cimemories,
    "coco_caption": run_coco_caption,
    "codeforces": run_codeforces,
    "complexfuncbench": run_complexfuncbench,
    "copilot_bench_swe": run_copilot_bench_swe,
    "cruxeval": run_cruxeval,
    "culer": run_culer,
    "custom_jsonl": run_custom_jsonl,
    "cyberseceval": run_cyberseceval,
    "cybergym": run_cybergym,
    "deepsearch_qa": run_deepsearch_qa,
    "docvqa": run_docvqa,
    "frames": run_frames,
    "gaia": run_gaia,
    "gaia2": run_gaia2,
    "gdpval": run_gdpval,
    "gorilla_apibench": run_gorilla_apibench,
    "gpqa": run_gpqa,
    "gpqa_diamond": run_gpqa_diamond,
    "gsm8k": run_gsm8k,
    "healthbench": run_healthbench,
    "hmmt": run_hmmt,
    "humanitys_last_exam": run_humanitys_last_exam,
    "i18n_translate": run_i18n_translate,
    "ifeval": run_ifeval,
    "imo_answer_bench": run_imo_answer_bench,
    "infographicvqa": run_infographicvqa,
    "lab_bench": run_lab_bench,
    "lcb": run_lcb,
    "livebench": run_livebench,
    "lmsys_noncoding_hard": run_lmsys_noncoding_hard,
    "loft_x_arxiv": run_loft_x_arxiv,
    "mcp_atlas": run_mcp_atlas,
    "mcp_bench": run_mcp_bench,
    "medxpertqa": run_medxpertqa,
    "mmlu": run_mmlu,
    "mmlu_pro": run_mmlu_pro,
    "mmlu_redux": run_mmlu_redux,
    "mmmu_pro": run_mmmu_pro,
    "mrcr": run_mrcr,
    "multi_swe_bench": run_multi_swe_bench,
    "multilingual_mmlu": run_multilingual_mmlu,
    "multipl_e": run_multipl_e,
    "nestful": run_nestful,
    "amc_aime": run_amc_aime,
    "nexus_function_calling": run_nexus_function_calling,
    "ojbench": run_ojbench,
    "omnidocbench": run_omnidocbench,
    "putnam": run_putnam,
    "putnam_formal": run_putnam_formal,
    "ruler": run_ruler,
    "scicode": run_scicode,
    "screenspot": run_screenspot,
    "seal_tools": run_seal_tools,
    "semantic_keypoint": run_semantic_keypoint,
    "simpleqa": run_simpleqa,
    "skillsbench": run_skillsbench,
    "spider2": run_spider2,
    "swe_bench_live": run_swe_bench_live,
    "swe_bench_multilingual": run_swe_bench_multilingual,
    "swe_bench_pro": run_swe_bench_pro,
    "t_eval": run_t_eval,
    "tau2": run_tau2,
    "tau3": run_tau3,
    "terminal_bench": run_terminal_bench,
    "textvqa": run_textvqa,
    "toolbench": run_toolbench,
    "wildclawbench": run_wildclawbench,
    "wmdp": run_wmdp,
}

__all__ = [
    "SUITES",
    "CUSTOM_PILLARS",
    "discover_and_register_plugins",
    "run_aa_lcr",
    "run_acebench",
    "run_agent_dojo",
    "run_aider_polyglot",
    "run_aime",
    "run_api_bank",
    "run_arc_agi",
    "run_beam_128k",
    "run_bfcl",
    "run_bfcl_v3_live",
    "run_bfcl_v4_agentic",
    "run_bigcodebench",
    "run_browsecomp",
    "run_bundled_detection",
    "run_causalbench",
    "run_chartqa",
    "run_charxiv",
    "run_cimemories",
    "run_coco_caption",
    "run_codeforces",
    "run_complexfuncbench",
    "run_copilot_bench_swe",
    "run_cruxeval",
    "run_culer",
    "run_custom_jsonl",
    "run_cyberseceval",
    "run_cybergym",
    "run_deepsearch_qa",
    "run_docvqa",
    "run_frames",
    "run_gaia",
    "run_gaia2",
    "run_gdpval",
    "run_gorilla_apibench",
    "run_gpqa",
    "run_gpqa_diamond",
    "run_gsm8k",
    "run_healthbench",
    "run_hmmt",
    "run_humanitys_last_exam",
    "run_i18n_translate",
    "run_ifeval",
    "run_imo_answer_bench",
    "run_infographicvqa",
    "run_lab_bench",
    "run_lcb",
    "run_livebench",
    "run_lmsys_noncoding_hard",
    "run_loft_x_arxiv",
    "run_mcp_atlas",
    "run_mcp_bench",
    "run_medxpertqa",
    "run_mmlu",
    "run_mmlu_pro",
    "run_mmlu_redux",
    "run_mmmu_pro",
    "run_mrcr",
    "run_multi_swe_bench",
    "run_multilingual_mmlu",
    "run_multipl_e",
    "run_nestful",
    "run_amc_aime",
    "run_nexus_function_calling",
    "run_ojbench",
    "run_omnidocbench",
    "run_putnam",
    "run_putnam_formal",
    "run_ruler",
    "run_scicode",
    "run_screenspot",
    "run_seal_tools",
    "run_semantic_keypoint",
    "run_simpleqa",
    "run_skillsbench",
    "run_spider2",
    "run_swe_bench_live",
    "run_swe_bench_multilingual",
    "run_swe_bench_pro",
    "run_t_eval",
    "run_tau2",
    "run_tau3",
    "run_terminal_bench",
    "run_textvqa",
    "run_toolbench",
    "run_wildclawbench",
    "run_wmdp",
]
