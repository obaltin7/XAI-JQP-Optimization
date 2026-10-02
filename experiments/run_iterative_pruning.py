"""
Pruning-ratio sweep with iterative XAI-JQP and selector controls (GPU; --dry-run runs on CPU).

Builds on the shared infrastructure of run_ablation_tests.py / run_e2e_pipeline.py (Logger, atomic
ResultStore, environment/argument logging, clean model loading, quantization with a bitsandbytes
fallback, DryRun). For every ratio f (0.2/0.4/0.6 = fraction of attention heads, i.e. blocks, not of
parameters) all configurations run under the same budget: the number of pruned heads and the number
of INT4 modules in the within-type percentile plan with tier_fractions = (f, (1-f)/2, (1-f)/2),
computed from the stored attribution scores (--scores). For each configuration the model is loaded
cleanly, the plan is applied, and WikiText-2 perplexity, an MMLU subset (eval_mmlu, plain prompts, on
by default), measured size, wall time and peak VRAM are recorded; the model is then fully released.

Default configurations (--configs selects a subset; run for every ratio):
  xai_single      One shot with the stored scores: allocate -> mask heads -> INT4. At f=0.2 the result
                  is checked against the `both` configuration of the ablation run
                  (results/ablation_gun6.json, 6.3171) within +-0.01; a mismatch logs "CHECK FAILED"
                  and the run continues.
  xai_iter        n_rounds rounds of pruning. The head budget is split across rounds (205 = 69+68+68);
                  round k selects its share with allocate(exclude_heads=heads pruned earlier), masks it
                  and recomputes the scores on the masked FP16 model with xai_engine (same settings as
                  the stored scores: 16 passages, n_steps=8, same passages), written to
                  results/gun7_scores/f{ratio}_round{k}.json. The next round selects from these scores.
                  The INT4 plan is derived from the last round's scores and applied once. Attribution
                  calls = number of rounds; per-round times are stored in the JSON.
  xai_iter_fixedq Same iterative pruning as xai_iter, but the INT4 plan is copied from xai_single (as for
                  the selector controls): size and quantization are identical to xai_single, only the
                  pruning selection differs. Round scores go to f{ratio}_xai_iter_fixedq_round{k}.json
                  (ignored by compute_drift.py).
  prune_wanda     Same head budget, lowest compressor.head_wanda_scores; INT4 plan identical to
                  xai_single (only the selector changes).
  prune_taylor    Same, with compressor.head_taylor_scores (sum |G*W|, LLM-Pruner family).
  prune_random    Same head budget, seeded random selection (--n-repeats repeats); same INT4 plan.
  az_buda_cok_kuantize  f=0.2 only: tier_fractions=(0.1, 0.5, 0.4) (prune/int4/fp16), one shot with the
                  stored scores; nominal bits/param and size are recorded for an equal-size comparison
                  with xai_single.
  Magnitude and inverse-score controls at f=0.2 are covered by run_ablation_tests.py and not repeated.

Opt-in configurations (not in the default list; request them with --configs):
  prune_wanda_ln  Same budget; Wanda score z-normalized within each layer (diagnose_wanda.py showed that
                  raw scores differ in scale across layers and prune early layers wholesale). Raw and
                  normalized scores are written to f{ratio}_prune_wanda_ln_criterion.json.
  prune_attnconf  Same budget; criterion compressor.head_attention_confidence_scores (Voita et al. 2019
                  confidence: mean maximum attention probability; the lowest are pruned). Confidence,
                  entropy and first-key share are written to f{ratio}_prune_attnconf_criterion.json.
                  INT4 plan identical to xai_single.
  xai_single_4tier / xai_iter_fixedq_4tier  f=0.2 only. Four regions: prune / INT4 / INT8 / FP16. Head
                  fractions (0.2, 0.3, 0.3, 0.2), MLP (int4, int8, fp16) = (0.4, 0.3, 0.3); INT8 =
                  bitsandbytes LLM.int8 (Linear8bitLt, threshold 6.0); median rule over the four labels.
                  The pruned set follows the xai_single / xai_iter_fixedq procedure; only the quantization
                  plan differs (with the stored scores: 91 INT4 + 102 INT8 modules, nominal size ratio
                  0.506 vs 0.549 with three tiers). The int8_modules field exists only for these configs.
  mixed_xai_k5 / mixed_xai_k10 / mixed_xai_k20 / mixed_random_k10 / mixed_magnitude_k10 / mixed_wanda_ln_k10
                  (experiment D-1) XAI-guided mixed precision WITHOUT pruning. A module is the attention
                  block (q/k/v/o_proj) or the MLP block of a layer; attention module score = lower median
                  of the layer's head scores (consistent with the median rule), MLP module score = MLP
                  score. Within each type the top round(k/100*32) modules stay FP16 and all other decoder
                  Linear layers become NF4 (nf4_uniform settings; lm_head / embeddings FP16). Controls use
                  the same number of blocks: random (seeds 42/43/44) and magnitude (Frobenius norm). No
                  attribution. The k=0 reference is nf4_uniform in results/baselines_gun7.json;
                  mixed_xai_k0 reproduces it through this script's load -> quantize path (difference
                  <= 0.01; a mismatch logs "CHECK FAILED" and the run continues). mixed_wanda_ln_k10
                  remains available but is excluded from the standard command: the layer median of a
                  within-layer z-score is ~0 and carries no information for cross-layer module selection.
                  These configs write by default to results/mixed_precision_gun11.json and
                  log_gun11_mixed.txt under the ratio key "0.2" (they are ratio-independent).

--rescore-after (on by default): for xai_single / prune_* / az_buda_cok_kuantize the scores are
recomputed once after pruning and BEFORE quantization and written to
results/gun7_scores/f{ratio}_{config}_after[_seed{s}].json (drift analysis: compute_drift.py). For
xai_iter the last round's scores serve this purpose. Cost ~6 min per configuration.

--calib-dataset c4 (out-of-domain calibration): round-0 scores are read from
results/importance_scores_c4.json (produced by run_xai_on_mistral.py --calib-dataset c4); round, rescore
and criterion passages come from the same C4 selection (sha1-verified). Outputs are kept separate
(results/calib_c4_gun9.json, results/gun9_scores/c4/, log_gun9_c4.txt) and the reproduction check against
the ablation run is skipped. Evaluation is unchanged: WikiText-2 test perplexity + MMLU.

Approximate runtime on one 48 GB L40S (model cached):
  * FP16 rerun: load ~13 s + perplexity ~66 s + MMLU ~2 min                       -> ~4 min (once)
  * xai_single / az_buda_cok_kuantize: load + apply + ppl ~2.5 min + MMLU ~3 min + rescore ~6 min -> ~11.5 min
  * xai_iter / xai_iter_fixedq (3 rounds): 3 x ~6 min attribution + ~2.5 min + MMLU ~3 min -> ~23.5 min each
  * prune_wanda / prune_taylor: calibration <10 s + ~11.5 min                      -> ~12 min
  * prune_random: n_repeats x ~12 min (n_repeats=3 -> ~36 min)
  * per ratio (n_repeats=3): ~119 min; 3 ratios + az_buda_cok_kuantize + FP16 ~ 6.2 h (n_repeats=1: ~5 h)
  * peak VRAM: attribution ~20 GB, other steps ~15-17 GB.
  * --no-rescore-after saves ~6 min per configuration (drift is then measured only for xai_iter).

Usage:
    python experiments/run_iterative_pruning.py --preflight               # check inputs/budgets without loading the model
    nohup python experiments/run_iterative_pruning.py --n-repeats 3 > run.out 2>&1 &
    python src/compute_drift.py                                   # after the run, no GPU needed
    python experiments/run_iterative_pruning.py --fractions 0.2 --configs xai_single,xai_iter,xai_iter_fixedq
    python experiments/run_iterative_pruning.py --configs prune_wanda_ln --n-repeats 1 --skip-baseline --output <file>
    python experiments/run_iterative_pruning.py --configs prune_attnconf --n-repeats 1 --skip-baseline --output <file>
    python experiments/run_iterative_pruning.py --fractions 0.2 --configs xai_single_4tier,xai_iter_fixedq_4tier --skip-baseline --output <file>
    python experiments/run_iterative_pruning.py --configs mixed_xai_k0,mixed_xai_k5,mixed_xai_k10,mixed_xai_k20,mixed_random_k10,mixed_magnitude_k10 --n-repeats 3 --skip-baseline
    python experiments/run_iterative_pruning.py --calib-dataset c4 --fractions 0.2 --configs xai_single,xai_iter_fixedq,prune_random --n-repeats 1 --no-rescore-after
    python experiments/run_iterative_pruning.py --dry-run                 # mini model on CPU (outputs in dryrun_out/)
    python experiments/run_iterative_pruning.py --resume                  # skip completed (ratio, config) pairs

Outputs (defaults): results/iterative_gun7.json (per-ratio budget, per-config repeats, summaries and
comparisons), log_gun7.txt, and round / rescore score files in results/gun7_scores/.
"""

# HF_HOME must be set BEFORE transformers/datasets are imported (see run_e2e_pipeline.py)
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compressor
from compressor import (
    LayerPlan,
    allocate_compression_tiers,
    allocate_mixed_precision,
    apply_structural_pruning,
    build_compression_plan,
    estimate_compression_budget,
    head_attention_confidence_scores,
    head_taylor_scores,
    head_wanda_scores,
    load_scores,
    mixed_precision_plan,
    mlp_wanda_scores,
    module_importance_scores,
    module_magnitude_scores,
    parse_block_key,
)
# Shared components of the earlier experiment scripts (imported unchanged)
from run_ablation_tests import (
    CALIB,
    DryRun,
    HeadList,
    _StoreView,
    all_heads,
    heads_from_tiers,
    load_calibration_batches,
    merge_plan,
    prune_plan_from_heads,
    select_heads_by_score,
    select_heads_random,
    set_seed,
)
from run_baselines import ensure_clean_gpu, release_gpu
from run_e2e_pipeline import (
    BASELINE_FILE,
    MODEL_NAME,
    SCORES_FILE,
    Logger,
    should_log_to_file,
    ResultStore,
    cuda_gb,
    env_info,
    load_model,
    load_wikitext_text,
    module_bytes,
    perplexity_tools,
    plan_summary,
    quantize_with_fallback,
    tier_counts,
)
from xai_engine import calculate_importance_scores

OUTPUT_FILE = os.path.join("results", "iterative_gun7.json")
LOG_FILE = "log_gun7.txt"
SCORES_DIR = os.path.join("results", "gun7_scores")
GUN6_FILE = os.path.join("results", "ablation_gun6.json")
GUN6_BOTH_PPL_FALLBACK = 6.3171
REPRO_TOL = 0.01
DRYRUN_DIR = "dryrun_out"  # --dry-run outputs (git-ignored); real results/ files are never touched
# Default paths of the out-of-domain calibration run (--calib-dataset c4)
C4_SCORES_FILE = os.path.join("results", "importance_scores_c4.json")
C4_OUTPUT_FILE = os.path.join("results", "calib_c4_gun9.json")
C4_LOG_FILE = "log_gun9_c4.txt"
C4_SCORES_DIR = os.path.join("results", "gun9_scores", "c4")

# Attribution settings of the stored scores (run_xai_on_mistral.py defaults); calibration = CALIB (16 passages, B=4, T=256)
ATTRIBUTION = dict(n_steps=8, internal_batch_size=2, max_batches=4)
DRY_ATTRIBUTION = dict(n_steps=2, internal_batch_size=None, max_batches=None)
AZ_BUDA_FRACTIONS = (0.1, 0.5, 0.4)
AZ_BUDA_ONLY_FRACTION = 0.2
# Four-tier variant (f=0.2 only)
FOUR_TIER_FRACTIONS = (0.2, 0.3, 0.3, 0.2)       # head: prune / int4 / int8 / fp16
FOUR_TIER_MLP_FRACTIONS = (0.4, 0.3, 0.3)        # MLP: int4 / int8 / fp16 (MLP is never pruned)
FOUR_TIER_CONFIGS = ("xai_single_4tier", "xai_iter_fixedq_4tier")
# D-1: XAI-guided mixed precision without pruning, FP16/NF4; ratio-independent (runs once under the "0.2" key)
MIXED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mixed_xai_k0": {"k": 0, "selector": "xai"},       # optional check: this script's load->quantize path reproduces nf4_uniform (5.3065)
    "mixed_xai_k5": {"k": 5, "selector": "xai"},
    "mixed_xai_k10": {"k": 10, "selector": "xai"},
    "mixed_xai_k20": {"k": 20, "selector": "xai"},
    "mixed_random_k10": {"k": 10, "selector": "random"},
    "mixed_magnitude_k10": {"k": 10, "selector": "magnitude"},
    "mixed_wanda_ln_k10": {"k": 10, "selector": "wanda_ln"},
}
K0_TOL = 0.01  # ppl tolerance: mixed_xai_k0 (FP16 load -> per-module NF4) vs nf4_uniform in baselines_gun7.json (HF load_in_4bit)
MIXED_OUTPUT_FILE = os.path.join("results", "mixed_precision_gun11.json")
MIXED_LOG_FILE = "log_gun11_mixed.txt"
WANDA_LN_CRITERION_FILE = os.path.join(SCORES_DIR, "f0.2_prune_wanda_ln_criterion.json")  # criterion computed on the unpruned model, hence identical across ratios
BASELINES_FILE = os.path.join("results", "baselines_gun7.json")  # k=0 reference (nf4_uniform; not re-run)
ONLY_20_CONFIGS = ("az_buda_cok_kuantize",) + FOUR_TIER_CONFIGS + tuple(MIXED_CONFIGS)

CONFIG_DESCRIPTIONS: Dict[str, str] = {
    "xai_single": "one shot with the stored scores: allocate -> mask heads -> INT4; at f=0.2 checked against the ablation run's `both` config",
    "xai_iter": "n_rounds pruning rounds; rescoring on the masked FP16 model after each round (same attribution settings), "
                "selection with exclude_heads; INT4 plan from the last round's scores, applied once at the end",
    "xai_iter_fixedq": "same iterative pruning as xai_iter (n_rounds rounds, rescoring each round); INT4 plan identical to xai_single "
                       "(identical size and quantization, only the pruning selection differs)",
    "prune_wanda": "same head budget; criterion head_wanda_scores (|W| × o_proj input norm), lowest N; INT4 plan identical to xai_single",
    "prune_taylor": "same head budget; criterion head_taylor_scores (Σ|G⊙W|, LLM-Pruner family), lowest N; INT4 plan identical to xai_single",
    "prune_random": "same head budget; seeded random selection (n_repeats repeats); INT4 plan identical to xai_single",
    "az_buda_cok_kuantize": f"f=0.2 only: tier_fractions={AZ_BUDA_FRACTIONS} (prune/int4/fp16), one shot with the stored scores",
    # Wanda diagnosis (diagnose_wanda.py): opt-in, must be requested explicitly with --configs
    "prune_wanda_ln": "same head budget; criterion head_wanda_scores(normalize='layer_zscore') (within-layer z-score, removes the "
                      "cross-layer scale), lowest N; INT4 plan identical to xai_single; raw and normalized scores "
                      "are written to f{ratio}_prune_wanda_ln_criterion.json",
    # Fourth control-criterion family (attention pattern): opt-in
    "prune_attnconf": "same head budget; criterion head_attention_confidence_scores (Voita et al. 2019 confidence = mean maximum attention "
                      "probability, same 16 calibration passages), lowest N; INT4 plan identical to xai_single; confidence, entropy "
                      "and first-key share are written to f{ratio}_prune_attnconf_criterion.json",
    # Four regions (prune / INT4 / INT8 / FP16): f=0.2 only, opt-in
    "xai_single_4tier": f"f=0.2 only: 4 tiers, head tier_fractions={FOUR_TIER_FRACTIONS} (prune/int4/int8/fp16), MLP "
                        f"{FOUR_TIER_MLP_FRACTIONS} (int4/int8/fp16); one shot with the stored scores; INT8 = bitsandbytes LLM.int8 "
                        "(Linear8bitLt, threshold 6.0); pruned set IDENTICAL to xai_single (lowest 205), only the quantization plan differs",
    "xai_iter_fixedq_4tier": "f=0.2 only: same iterative pruning as xai_iter_fixedq; the quantization plan is the fixed "
                             "4-tier plan of xai_single_4tier",
}
# D-1: mixed precision without pruning; opt-in
_MIXED_SELECTOR_TEXT = {
    "xai": "module score = stored XAI scores (attn: lower median of the layer's head scores, MLP: MLP score)",
    "random": "seeded random module selection (n_repeats repeats; same count per type)",
    "magnitude": "module score = joint Frobenius norm of the Linear weights (compressor.module_magnitude_scores)",
    "wanda_ln": "attn module score = lower median of the within-layer normalized head scores in the stored prune_wanda_ln criterion "
                "file; MLP module score = compressor.mlp_wanda_scores (the file has no MLP scores; computed at run time, same 16 passages)",
}
for _name, _spec in MIXED_CONFIGS.items():
    CONFIG_DESCRIPTIONS[_name] = (f"mixed precision WITHOUT pruning: within each type the top {_spec['k']}% of modules stay FP16, all other decoder "
                                  f"Linear layers NF4 (same settings as nf4_uniform: double quantization, fp16 compute; lm_head/embeddings FP16); "
                                  f"{_MIXED_SELECTOR_TEXT[_spec['selector']]}; no attribution")
EXTRA_CONFIGS = ["prune_wanda_ln", "prune_attnconf", "xai_single_4tier", "xai_iter_fixedq_4tier"] + list(MIXED_CONFIGS)  # opt-in; not part of the default run (the default config list is unchanged)
ALL_CONFIGS = [c for c in CONFIG_DESCRIPTIONS if c not in EXTRA_CONFIGS]
WANDA_LN_NORMALIZE = "layer_zscore"
STOCHASTIC_CONFIGS = {"prune_random", "mixed_random_k10"}
SELECTOR_CONFIGS = {"prune_wanda", "prune_taylor", "prune_random", "prune_wanda_ln", "prune_attnconf"}  # INT4 plan copied from xai_single
ITERATIVE_CONFIGS = {"xai_iter", "xai_iter_fixedq", "xai_iter_fixedq_4tier"}  # n_rounds pruning rounds, rescoring after each
FIXED_INT4_CONFIGS = SELECTOR_CONFIGS | {"xai_single", "xai_iter_fixedq"}  # budget guarantee: the INT4 module SET equals that of xai_single


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pruning-ratio sweep with iterative XAI-JQP and selector controls")
    p.add_argument("--fractions", default="0.2,0.4,0.6",
                   help="comma-separated block ratios (the lowest-scoring fraction f of the heads is pruned)")
    p.add_argument("--configs", default=",".join(ALL_CONFIGS),
                   help="comma-separated config list (default: all default configs; az_buda_cok_kuantize runs only at 20%%)")
    p.add_argument("--n-rounds", type=int, default=3, help="number of rounds for the iterative configs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=3,
                   help="number of repeats for stochastic configs (prune_random); seeds seed, seed+1, ...")
    p.add_argument("--no-mmlu", action="store_true", help="skip the MMLU subset (default: evaluate it)")
    p.add_argument("--mmlu-batch-size", type=int, default=8)
    p.add_argument("--no-rescore-after", action="store_true",
                   help="skip rescoring after pruning / before quantization for one-shot configs (default: rescore)")
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--log", default=LOG_FILE)
    p.add_argument("--scores-dir", default=SCORES_DIR, help="directory of the round / rescore score JSON files")
    p.add_argument("--scores", default=SCORES_FILE, help="stored attribution scores (round 0)")
    p.add_argument("--baseline", default=BASELINE_FILE)
    p.add_argument("--gun6-json", default=GUN6_FILE, help="ablation-run result (config `both`) for the xai_single 20%% reproduction check")
    p.add_argument("--skip-baseline", action="store_true", help="do not re-measure FP16 perplexity/MMLU")
    p.add_argument("--resume", action="store_true", help="skip (ratio, config) pairs with status=completed in --output")
    p.add_argument("--preflight", action="store_true", help="check inputs, budgets and round shares without loading the model, then exit")
    p.add_argument("--dry-run", action="store_true", help="mini Mistral + mock quantization/perplexity/MMLU, CPU")
    p.add_argument("--calib-dataset", choices=("wikitext2", "c4"), default="wikitext2",
                   help="source of the round/rescore/criterion calibration passages; c4 = out-of-domain calibration test "
                        "(with scores from run_xai_on_mistral.py --calib-dataset c4). Evaluation is always WikiText-2 test + MMLU. "
                        "Default paths with c4: --scores results/importance_scores_c4.json, --output results/calib_c4_gun9.json, "
                        "--log log_gun9_c4.txt, --scores-dir results/gun9_scores/c4 (the default output files are left untouched)")
    p.add_argument("--wanda-ln-scores", default=WANDA_LN_CRITERION_FILE,
                   help="stored prune_wanda_ln criterion file for mixed_wanda_ln_k10 (raw + normalized head scores)")
    p.add_argument("--save-window-nll", action="store_true",
                   help="(D-1) also store WikiText-2 window NLLs for every run (rep['ppl']['wikitext2'], E-5 schema; for the paired "
                        "bootstrap; +~70 s per run and dataset). Off by default = bit-identical output")
    p.add_argument("--window-nll-datasets", default="wikitext2,c4",
                   help="datasets measured when --save-window-nll is set (C4 = fixed subset, "
                        "sha1-verified). Can be narrowed to 'wikitext2' without network access")
    p.add_argument("--rescore-nan-fallback", choices=("off", "bf16", "fp32"), default="off",
                   help="(E-3) if an iterative round's rescore contains NaN/inf, recompute THAT round at this precision (parameters are cast "
                        "temporarily and restored BIT-IDENTICALLY from a CPU copy; needs ~model size of CPU RAM). Default off = original behavior")
    args = p.parse_args(argv)
    args.rescore_after = not args.no_rescore_after
    args.mmlu = not args.no_mmlu
    if uses_mixed_configs(args.configs):  # do not overwrite the default result/log files (dryrun_out/ in dry-run)
        if args.output == OUTPUT_FILE:
            args.output = os.path.join(DRYRUN_DIR, "mixed_precision_gun11_dry.json") if args.dry_run else MIXED_OUTPUT_FILE
        if args.log == LOG_FILE and args.dry_run:
            args.log = os.path.join(DRYRUN_DIR, "log_gun11_mixed_dry.txt")
        elif args.log == LOG_FILE:
            args.log = MIXED_LOG_FILE
        if args.fractions == "0.2,0.4,0.6" and all(c in MIXED_CONFIGS for c in _split_configs(args.configs)):
            args.fractions = "0.2"  # ratio-independent configs: avoid empty entries for the other ratios
    if args.calib_dataset == "c4" and not args.dry_run:  # do not overwrite the default result/score files
        if args.scores == SCORES_FILE:
            args.scores = C4_SCORES_FILE
        if args.output == OUTPUT_FILE:
            args.output = C4_OUTPUT_FILE
        if args.log == LOG_FILE:
            args.log = C4_LOG_FILE
        if args.scores_dir == SCORES_DIR:
            args.scores_dir = C4_SCORES_DIR
    if args.dry_run:  # never overwrite real results/ files
        if args.output == OUTPUT_FILE:
            args.output = os.path.join(DRYRUN_DIR, "iterative_gun7_dry.json")
        if args.log == LOG_FILE:
            args.log = os.path.join(DRYRUN_DIR, "log_gun7_dry.txt")
        if args.scores_dir == SCORES_DIR:
            args.scores_dir = os.path.join(DRYRUN_DIR, "gun7_scores")
    return args


def _split_configs(text: str) -> List[str]:
    return [c.strip() for c in text.split(",") if c.strip()]


def uses_mixed_configs(configs_text: str) -> bool:
    """Whether --configs contains a mixed-precision config without pruning (redirects the default output/log)."""
    return any(c in MIXED_CONFIGS for c in _split_configs(configs_text))


def default_log_file(args: argparse.Namespace) -> str:
    """Monitored default log for should_log_to_file: log_gun11_mixed.txt for mixed configs, otherwise log_gun7.txt."""
    return MIXED_LOG_FILE if uses_mixed_configs(args.configs) else LOG_FILE


def parse_fractions(text: str) -> List[float]:
    out: List[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        f = float(tok)
        if not 0.0 < f < 1.0:
            raise ValueError(f"ratio must lie in (0, 1): {tok}")
        out.append(f)
    if not out:
        raise ValueError("--fractions is empty")
    return out


def fraction_key(fraction: float) -> str:
    return f"{fraction:g}"


# --------------------------------------------------------------------------- #
# Budget / round helpers (no GPU needed; called directly by the tests)
# --------------------------------------------------------------------------- #
def tier_fractions_for(fraction: float) -> Tuple[float, float, float]:
    """Ratio f -> (f, (1-f)/2, (1-f)/2): 20% -> (0.2, 0.4, 0.4), 40% -> (0.4, 0.3, 0.3), 60% -> (0.6, 0.2, 0.2)."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie in (0, 1): {fraction}")
    rest = (1.0 - fraction) / 2.0
    return (fraction, rest, rest)


def split_budget(n_budget: int, n_rounds: int) -> List[int]:
    """Split the head budget across rounds; the total equals the budget, the remainder goes to the first rounds (205, 3 -> [69, 68, 68]; 2, 3 -> [1, 1, 0])."""
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")
    if n_budget < 0:
        raise ValueError("n_budget must be >= 0")
    base, extra = divmod(n_budget, n_rounds)
    return [base + (1 if r < extra else 0) for r in range(n_rounds)]


def head_names(heads: HeadList) -> List[str]:
    return [f"layer_{l}.attn.head_{h}" for l, h in heads]


def select_round_heads(scores: Dict[str, float], share: int, exclude: Sequence[str]) -> HeadList:
    """
    Select this round's share: the REMAINING pool (allocate_compression_tiers with exclude_heads =
    heads pruned in earlier rounds) is tiered with prune fraction share/|remaining| and the "prune"
    heads are returned (the share lowest-scoring heads; ties broken by structural order). Excluded
    heads (which score exactly 0 after rescoring) never enter the ranking. If float rounding makes
    the prune count deviate from share, the selection is corrected with the same ordering
    (select_heads_by_score).
    """
    if share < 0:
        raise ValueError("share must be >= 0")
    if share == 0:
        return []
    excluded = set(exclude)
    remaining = [k for k in scores if parse_block_key(k).kind == "attn" and k not in excluded]
    if share > len(remaining):
        raise ValueError(f"share {share} > number of remaining heads {len(remaining)}")
    p = share / len(remaining)
    tiers = allocate_compression_tiers(scores, 3, tier_fractions=(p, (1.0 - p) / 2.0, (1.0 - p) / 2.0),
                                       exclude_heads=sorted(excluded), mlp_min_tier="int4")
    chosen = heads_from_tiers(tiers)
    if len(chosen) != share:  # safety net (same ordering: score, then structural order)
        chosen = select_heads_by_score({k: scores[k] for k in remaining}, share, lowest=True)
    assert not (set(head_names(chosen)) & excluded)
    return chosen


def int4_module_count(plan: Dict[int, LayerPlan]) -> int:
    return sum(4 for p in plan.values() if p.attn_quant == "int4") + sum(3 for p in plan.values() if p.mlp_quant == "int4")


def quant_module_count(plan: Dict[int, LayerPlan], tier: str) -> int:
    """Number of Linear modules in tier `tier` ("int4" | "int8") of the plan (attention block: 4 modules, MLP block: 3)."""
    return sum(4 for p in plan.values() if p.attn_quant == tier) + sum(3 for p in plan.values() if p.mlp_quant == tier)


def four_tier_plan(scores: Dict[str, float]) -> Tuple[Dict[str, str], Dict[int, LayerPlan]]:
    """Four-tier assignment and layer plan for f=0.2 (median rule over the 4 labels; same build_compression_plan code)."""
    tiers = allocate_compression_tiers(scores, 4, tier_fractions=FOUR_TIER_FRACTIONS, mlp_min_tier="int4",
                                       mlp_tier_fractions=FOUR_TIER_MLP_FRACTIONS)
    return tiers, build_compression_plan(tiers)


def int4_module_set(plan: Dict[int, LayerPlan]) -> List[str]:
    out: List[str] = []
    for i, p in sorted(plan.items()):
        if p.attn_quant == "int4":
            out += [f"layer_{i}.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        if p.mlp_quant == "int4":
            out += [f"layer_{i}.mlp.{n}" for n in ("gate_proj", "up_proj", "down_proj")]
    return out


def fp16_module_set(plan: Dict[int, LayerPlan]) -> List[str]:
    """Decoder Linear modules that REMAIN FP16 in the plan (complement of int4_module_set; for plans without pruning)."""
    out: List[str] = []
    for i, p in sorted(plan.items()):
        if p.attn_quant == "fp16":
            out += [f"layer_{i}.self_attn.{n}" for n in compressor.ATTN_LINEARS]
        if p.mlp_quant == "fp16":
            out += [f"layer_{i}.mlp.{n}" for n in compressor.MLP_LINEARS]
    return out


def nf4_uniform_check(ppl: Optional[float], expected: Any) -> Optional[Dict[str, Any]]:
    """Whether the mixed_xai_k0 perplexity is within K0_TOL of nf4_uniform in baselines_gun7.json; None (check skipped) without a reference."""
    if not isinstance(expected, (int, float)) or not isinstance(ppl, (int, float)):
        return None
    diff = abs(ppl - expected)
    return {"expected_nf4_uniform": expected, "abs_diff": diff, "tolerance": K0_TOL, "within_tolerance": diff <= K0_TOL}


def load_wanda_ln_head_scores(path: str) -> Dict[str, float]:
    """Within-layer normalized head scores from a stored prune_wanda_ln criterion file ({"normalized": {head: z}})."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = data.get("normalized")
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"{path}: no 'normalized' head scores (expected a prune_wanda_ln criterion file)")
    return {k: float(v) for k, v in scores.items()}


def mixed_module_scores(selector: str, model: nn.Module, tokenizer, ctx: Dict[str, Any], log: Logger) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Module scores of a mixed-precision selector and their source. xai: stored scores (no attribution); magnitude:
    from the loaded FP16 model; wanda_ln: attn = stored criterion file (computed on the mini model in dry-run, where the
    file does not exist), MLP = mlp_wanda_scores (at run time); random: no scores (None).
    """
    args, dry = ctx["args"], ctx["dry"]
    if selector == "xai":
        return module_importance_scores(ctx["scores"]), f"xai: {args.scores}"
    if selector == "random":
        return None, "random (seed)"
    if selector == "magnitude":
        return module_magnitude_scores(model), "magnitude: sqrt(sum ||W||_F^2), loaded FP16 model"
    if selector == "wanda_ln":
        batches = calibration_batches(ctx, tokenizer)
        if dry:
            head_scores, src = head_wanda_scores(model, batches, normalize=WANDA_LN_NORMALIZE), "wanda_ln: from the mini model (dry-run)"
        else:
            if not os.path.exists(args.wanda_ln_scores):
                raise FileNotFoundError(f"mixed_wanda_ln: criterion file not found: {args.wanda_ln_scores} (run prune_wanda_ln first)")
            head_scores, src = load_wanda_ln_head_scores(args.wanda_ln_scores), f"wanda_ln: {args.wanda_ln_scores} (normalized)"
        n_expected = len(all_heads(ctx["scores"]))
        if len(head_scores) != n_expected:
            raise ValueError(f"mixed_wanda_ln: {len(head_scores)} head scores, expected {n_expected}")
        scores = module_importance_scores(head_scores)  # attention modules only
        scores.update(mlp_wanda_scores(model, batches))
        log(f"mixed_wanda_ln: attn scores {src}; MLP scores mlp_wanda_scores ({len(batches)} batches)")
        return {k: scores[k] for k in module_importance_scores(ctx["scores"])}, src + " + mlp_wanda_scores (run time)"
    raise ValueError(f"unknown mixed-precision selector: {selector}")


def nominal_budget(plan: Dict[int, LayerPlan], model: Optional[nn.Module] = None) -> Dict[str, float]:
    if model is None:
        return estimate_compression_budget(plan)
    cfg = model.config
    return estimate_compression_budget(
        plan, hidden_size=cfg.hidden_size, head_dim=compressor._head_dim(model),
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size)


def fraction_budget(scores: Dict[str, float], fraction: float, dry: bool) -> Dict[str, Any]:
    """One-shot plan and budget definition for ratio f (stored scores; no model needed)."""
    tf = tier_fractions_for(fraction)
    tiers = allocate_compression_tiers(scores, 3, tier_fractions=tf, mlp_min_tier="int4")
    plan = build_compression_plan(tiers)
    heads = heads_from_tiers(tiers)
    n_total = len(all_heads(scores))
    return {
        "fraction": fraction, "tier_fractions": list(tf), "tiers": tiers, "single_plan": plan, "xai_heads": heads,
        "n_prune_heads": len(heads), "n_heads_total": n_total,
        "int4_modules_single": int4_module_count(plan), "int4_module_set_single": int4_module_set(plan),
        "int4_layers": {"attn": sum(p.attn_quant == "int4" for p in plan.values()),
                        "mlp": sum(p.mlp_quant == "int4" for p in plan.values())},
        "tier_counts": tier_counts(tiers),
        "budget_nominal_single": None if dry else estimate_compression_budget(plan),
    }


def budget_json(b: Dict[str, Any], n_rounds: int) -> Dict[str, Any]:
    return {
        "definition": "budget = NUMBER of pruned heads and NUMBER of INT4 modules in this ratio's within-type percentile plan "
                      "(stored scores, tier_fractions=(f,(1-f)/2,(1-f)/2)); the selector configs (wanda/taylor/random) and xai_iter_fixedq "
                      "copy the INT4 plan from xai_single; xai_iter derives its INT4 plan from the last-round scores with the same fractions "
                      "(equal tier counts; the module set may differ because of the median rule and is reported in the JSON)",
        "tier_fractions": b["tier_fractions"], "n_prune_heads": b["n_prune_heads"], "n_heads_total": b["n_heads_total"],
        "nominal_block_prune_ratio": b["n_prune_heads"] / b["n_heads_total"],
        "int4_modules_single": b["int4_modules_single"], "int4_layers_single": b["int4_layers"],
        "tier_counts_single": b["tier_counts"], "budget_nominal_single": b["budget_nominal_single"],
        "round_shares": split_budget(b["n_prune_heads"], n_rounds),
        "plan_summary_single": plan_summary(b["single_plan"]),
    }


# --------------------------------------------------------------------------- #
# Rescoring (xai_engine, same settings as the stored scores) and score files
# --------------------------------------------------------------------------- #
def rescore(model: nn.Module, batches: Sequence[Dict[str, torch.Tensor]], dry: bool) -> Tuple[Dict[str, float], float]:
    settings = DRY_ATTRIBUTION if dry else ATTRIBUTION
    t0 = time.time()
    scores = calculate_importance_scores(model, batches, verbose=False, **settings)
    return scores, time.time() - t0


FALLBACK_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def nonfinite_score_keys(scores: Dict[str, float]) -> List[str]:
    return [k for k, v in scores.items() if not math.isfinite(float(v))]


def rescore_high_precision(model: nn.Module, batches: Sequence[Dict[str, torch.Tensor]], dry: bool,
                           dtype_name: str) -> Tuple[Dict[str, float], float]:
    """E-3: rescore a round at higher precision when fp16 attribution overflows (NaN/inf). xai_engine is untouched: only
    PARAMETERS are cast temporarily (buffers such as the fp32 inv_freq stay as they are, as with a native bf16 load); since
    fp16 -> bf16 -> fp16 is lossy, the original weights are restored BIT-IDENTICALLY from a CPU copy (later measurements are unchanged)."""
    dtype = FALLBACK_DTYPES[dtype_name]
    backup = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}
    try:
        for p in model.parameters():
            if p.is_floating_point():
                p.data = p.data.to(dtype)
        return rescore(model, batches, dry)
    finally:
        for n, p in model.named_parameters():
            p.data = backup[n].to(p.device)


def save_scores(path: str, scores: Dict[str, float], meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {**meta, "created_at": datetime.now().isoformat(timespec="seconds"), "n_scores": len(scores), "scores": scores}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def scores_meta(fkey: str, config: str, pruned: List[str], dry: bool, **extra: Any) -> Dict[str, Any]:
    return {"fraction": fkey, "config": config, "model": "mini (dry-run)" if dry else MODEL_NAME,
            "attribution": {**(DRY_ATTRIBUTION if dry else ATTRIBUTION), "calibration": None if dry else dict(CALIB),
                            "method": "captum.LayerIntegratedGradients (attribute_to_layer_input)"},
            "pruned_heads": list(pruned), "n_pruned_heads": len(pruned), **extra}


def calibration_batches(ctx: Dict[str, Any], tokenizer) -> List[Dict[str, torch.Tensor]]:
    dry: Optional[DryRun] = ctx["dry"]
    if dry:
        return dry.calibration_batches()
    if "calib" not in ctx:  # same 16 passages as the stored scores; the tokenizer is identical across loads -> build once
        if ctx["args"].calib_dataset == "c4":
            ctx["calib"], ctx["calib_passages"] = load_c4_calibration_batches(tokenizer, ctx["args"].scores)
        else:
            ctx["calib"] = load_calibration_batches(tokenizer)
    return ctx["calib"]


def ensure_pad_token(tokenizer):
    """
    SAME rule as run_xai_on_mistral.main / run_e2e_pipeline.load_model: without a pad_token use eos_token, pad on the right.
    The Mistral tokenizer has no pad token; since build_calibration_batches is called with padding=True, a tokenizer without a
    pad token raises ValueError even for unpadded C4 passages that fill the window. A tokenizer from load_model (already
    configured) is left unchanged.
    """
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_c4_calibration_batches(tokenizer, scores_path: Optional[str] = None):
    """
    C4 calibration batches: the SAME selection as run_xai_on_mistral.py --calib-dataset c4 (first 16 documents with >= T
    tokens, deterministic stream). If the score file at scores_path contains passage identifiers (sha1), an exact match is
    verified; a mismatch raises, since it would otherwise silently rescore on different passages.
    """
    from run_xai_on_mistral import build_calibration_batches, load_calibration_passages

    ensure_pad_token(tokenizer)  # the preflight tokenizer does not pass through load_model -> "Asking to pad but the tokenizer does not have a padding token"
    passages, ids = load_calibration_passages(tokenizer, "c4", CALIB["n_passages"], CALIB["min_chars"], CALIB["max_length"])
    if len(passages) < CALIB["n_passages"]:
        raise ValueError(f"only {len(passages)} passages could be selected from the C4 stream (required {CALIB['n_passages']})")
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, "r", encoding="utf-8") as f:
            stored = (json.load(f).get("calibration") or {}).get("passages")
        if stored and [p["sha1"] for p in stored] != [p["sha1"] for p in ids]:
            raise ValueError(f"C4 passages do not match the identifiers in {scores_path} (the dataset/tokenizer version may have changed)")
    return build_calibration_batches(tokenizer, passages, CALIB["batch_size"], CALIB["max_length"]), ids


def calib_extra(args: argparse.Namespace) -> Dict[str, Any]:
    """Extra score-file metadata; empty for wikitext2 (keeps the default score-file schema unchanged)."""
    return {} if args.calib_dataset == "wikitext2" else {"calib_dataset": args.calib_dataset}


def reproduction_check_applies(args: argparse.Namespace) -> bool:
    """The xai_single 20% check against the ablation run's `both` config is meaningful only with the stored WikiText-2 scores and calibration."""
    return args.calib_dataset == "wikitext2" and os.path.normpath(args.scores) == os.path.normpath(SCORES_FILE)


def measure_mmlu(model: nn.Module, tokenizer, ctx: Dict[str, Any], log: Logger) -> Dict[str, Any]:
    import eval_mmlu  # lazy import: never loaded with --no-mmlu

    args = ctx["args"]
    if ctx["dry"]:
        return eval_mmlu.evaluate_mmlu_dry(model, args.seed, log=lambda m: log(m, tag="MMLU"))
    return eval_mmlu.evaluate_mmlu(model, tokenizer, subset=ctx["mmlu_subset"], prompt_style="plain",
                                   batch_size=args.mmlu_batch_size, log=lambda m: log(m, tag="MMLU"))


# --------------------------------------------------------------------------- #
# Single-config run
# --------------------------------------------------------------------------- #
def apply_and_measure(name: str, seed: int, fb: Dict[str, Any], ctx: Dict[str, Any], rep: Dict[str, Any],
                      store: ResultStore, log: Logger, box: Dict[str, Any]) -> None:
    """Apply the config to the model (held in box) and measure it; the model reference ends with this frame."""
    args, dry = ctx["args"], ctx["dry"]
    model, tokenizer = box["model"], box["tokenizer"]
    fkey = fraction_key(fb["fraction"])
    scores0: Dict[str, float] = ctx["scores"]
    n_layers = len(compressor._decoder_layers(model))
    n_budget: int = fb["n_prune_heads"]
    single_plan: Dict[int, LayerPlan] = fb["single_plan"]
    xai_heads: HeadList = fb["xai_heads"]
    rep["model_bytes_before"] = module_bytes(model)
    tag = f"f={fkey} {name}"

    # --- head selection and plan ---
    if name == "xai_single":
        heads = xai_heads
        plan = merge_plan(prune_plan_from_heads(heads, n_layers), single_plan)
        assert all(plan[i].pruned_heads == single_plan[i].pruned_heads for i in single_plan)
    elif name == "xai_single_4tier":
        tiers, plan = four_tier_plan(scores0)
        heads = heads_from_tiers(tiers)
        assert heads == xai_heads, "the 4-tier pruned set must equal that of xai_single (same 20% percentile)"
        rep["tier_fractions"] = {"heads": list(FOUR_TIER_FRACTIONS), "mlp": list(FOUR_TIER_MLP_FRACTIONS)}
        rep["tier_counts"] = tier_counts(tiers)
    elif name == "az_buda_cok_kuantize":
        tiers = allocate_compression_tiers(scores0, 3, tier_fractions=AZ_BUDA_FRACTIONS, mlp_min_tier="int4")
        plan = build_compression_plan(tiers)
        heads = heads_from_tiers(tiers)
        rep["tier_fractions"] = list(AZ_BUDA_FRACTIONS)
        rep["tier_counts"] = tier_counts(tiers)
        rep["overlap_with_xai_heads"] = len(set(heads) & set(xai_heads))
    elif name in ("prune_wanda", "prune_taylor", "prune_wanda_ln", "prune_attnconf"):
        t0 = time.time()
        batches = calibration_batches(ctx, tokenizer)
        if name == "prune_attnconf":  # attention confidence (Voita et al. 2019); entropy and first-key share are only recorded
            crit, attn_details = head_attention_confidence_scores(model, batches, return_details=True)
        elif name == "prune_wanda_ln":  # the raw score is also computed (diagnosis: 7B layer profile); selection uses the normalized score
            raw_crit = head_wanda_scores(model, batches)
            crit = head_wanda_scores(model, batches, normalize=WANDA_LN_NORMALIZE)
        else:
            crit = head_wanda_scores(model, batches) if name == "prune_wanda" else head_taylor_scores(model, batches)
        heads = select_heads_by_score(crit, n_budget, lowest=True)
        rep["criterion"] = {"seconds": time.time() - t0, "n_batches": len(batches),
                            "score_range": [min(crit.values()), max(crit.values())]}
        if name == "prune_wanda_ln":
            os.makedirs(args.scores_dir, exist_ok=True)
            crit_path = os.path.join(args.scores_dir, f"f{fkey}_{name}_criterion.json")
            with open(crit_path, "w", encoding="utf-8") as f:
                json.dump({"fraction": fkey, "config": name, "normalize": WANDA_LN_NORMALIZE, "n_batches": len(batches),
                           "created_at": datetime.now().isoformat(timespec="seconds"), "raw": raw_crit, "normalized": crit},
                          f, indent=1, ensure_ascii=False)
            rep["criterion"].update({"normalize": WANDA_LN_NORMALIZE, "scores_file": crit_path,
                                     "raw_score_range": [min(raw_crit.values()), max(raw_crit.values())],
                                     "overlap_with_raw_wanda_heads": len(set(heads) & set(select_heads_by_score(raw_crit, n_budget, lowest=True)))})
        if name == "prune_attnconf":
            os.makedirs(args.scores_dir, exist_ok=True)
            crit_path = os.path.join(args.scores_dir, f"f{fkey}_{name}_criterion.json")
            with open(crit_path, "w", encoding="utf-8") as f:
                json.dump({"fraction": fkey, "config": name, "definition": "Voita et al. 2019 confidence: mean maximum attention probability "
                           "over valid query tokens (t>0); the lowest N are pruned", "n_batches": len(batches),
                           "n_query_tokens": attn_details["n_query_tokens"], "created_at": datetime.now().isoformat(timespec="seconds"),
                           "confidence": crit, "entropy": attn_details["entropy"], "first_key_share": attn_details["first_key_share"]},
                          f, indent=1, ensure_ascii=False)
            pruned_names = set(head_names(heads))
            ent = attn_details["entropy"]
            rep["criterion"].update({
                "scores_file": crit_path, "n_query_tokens": attn_details["n_query_tokens"],
                "mean_entropy_pruned": float(np.mean([ent[k] for k in pruned_names])) if pruned_names else None,
                "mean_entropy_kept": float(np.mean([v for k, v in ent.items() if k not in pruned_names])),
                "mean_first_key_share_pruned": float(np.mean([attn_details["first_key_share"][k] for k in pruned_names])) if pruned_names else None})
        rep["overlap_with_xai_heads"] = len(set(heads) & set(xai_heads))
        plan = merge_plan(prune_plan_from_heads(heads, n_layers), single_plan)
        log(f"{tag}: criterion {rep['criterion']['seconds']:.1f} s, overlap with XAI {rep['overlap_with_xai_heads']}/{n_budget}")
    elif name == "prune_random":
        heads = select_heads_random(all_heads(scores0), n_budget, seed)
        rep["overlap_with_xai_heads"] = len(set(heads) & set(xai_heads))
        plan = merge_plan(prune_plan_from_heads(heads, n_layers), single_plan)
    elif name in MIXED_CONFIGS:  # D-1: no pruning; top k% modules FP16, the rest NF4; no attribution
        spec = MIXED_CONFIGS[name]
        heads = []
        t0 = time.time()
        # dry-run: the mini model has 2 layers (round(0.1·2) = 0), so k > 0 configs use an effective k = 50 to exercise the FP16 path
        k_eff = spec["k"] if not dry or spec["k"] == 0 else 50
        xai_modules = module_importance_scores(scores0)
        mod_scores, source = mixed_module_scores(spec["selector"], model, tokenizer, ctx, log)
        module_tiers = allocate_mixed_precision(mod_scores if mod_scores is not None else xai_modules, k_eff,
                                                random_seed=seed if spec["selector"] == "random" else None)
        plan = mixed_precision_plan(module_tiers)
        if len(plan) != n_layers:
            raise ValueError(f"{name}: plan has {len(plan)} layers, model has {n_layers} (score file from a different model?)")
        fp16_blocks = [k for k, t in module_tiers.items() if t == "fp16"]
        xai_fp16 = {k for k, t in allocate_mixed_precision(xai_modules, k_eff).items() if t == "fp16"}
        rep["mixed_precision"] = {
            "k_percent": spec["k"], "k_percent_effective": k_eff, "selector": spec["selector"], "score_source": source, "selection_seconds": time.time() - t0,
            "n_modules": {kind: sum(1 for k in module_tiers if k.endswith("." + kind)) for kind in ("attn", "mlp")},
            "n_fp16_blocks": {kind: sum(1 for k in fp16_blocks if k.endswith("." + kind)) for kind in ("attn", "mlp")},
            "fp16_blocks": fp16_blocks, "fp16_modules": fp16_module_set(plan), "n_fp16_modules": len(fp16_module_set(plan)),
            "overlap_with_xai_fp16_blocks": len(set(fp16_blocks) & xai_fp16), "module_scores": mod_scores}
        log(f"{tag}: {k_eff}% -> FP16 blocks {rep['mixed_precision']['n_fp16_blocks']} ({rep['mixed_precision']['n_fp16_modules']} Linear), "
            f"overlap with the XAI selection {rep['mixed_precision']['overlap_with_xai_fp16_blocks']}/{len(fp16_blocks)}: {fp16_blocks}")
    elif name in ITERATIVE_CONFIGS:
        # fixedq round files use a distinct name: they do not overwrite xai_iter's and are not double-counted by compute_drift (FILE_RE does not match)
        round_prefix = "" if name == "xai_iter" else f"{name}_"
        shares = split_budget(n_budget, args.n_rounds)
        pruned: List[str] = []
        cur = scores0
        rounds: List[Dict[str, Any]] = []
        for k, share in enumerate(shares, 1):
            if share == 0:  # budget smaller than the number of rounds (mini/dry-run only): no attribution spent
                log(f"{tag}: round {k}/{len(shares)} has share 0, skipped")
                continue
            t_round = time.time()
            chosen = select_round_heads(cur, share, pruned)
            apply_structural_pruning(model, prune_plan_from_heads(chosen, n_layers), verbose=False)
            pruned += head_names(chosen)
            batches = calibration_batches(ctx, tokenizer)
            new_scores, attr_s = rescore(model, batches, bool(dry))
            fallback_extra: Dict[str, Any] = {}
            bad = nonfinite_score_keys(new_scores)
            if bad:  # E-3: runs only when NaN/inf appear, so clean runs stay bit-identical
                fb_dtype = getattr(args, "rescore_nan_fallback", "off")
                log(f"{tag}: round {k} rescore has {len(bad)} NaN/inf ({sum(1 for b in bad if b in pruned)} pruned, "
                    f"{sum(1 for b in bad if b not in pruned)} surviving blocks; e.g. {bad[:4]}); --rescore-nan-fallback={fb_dtype}")
                if fb_dtype != "off":
                    new_scores, attr_hp = rescore_high_precision(model, batches, bool(dry), fb_dtype)
                    still = nonfinite_score_keys(new_scores)
                    fallback_extra = {"rescore_fallback": {"dtype": fb_dtype, "n_nonfinite_before": len(bad), "nonfinite_before": bad,
                                                           "n_nonfinite_after": len(still), "first_attempt_seconds": attr_s, "seconds": attr_hp}}
                    attr_s += attr_hp
                    log(f"{tag}: round {k} rescored in {fb_dtype} ({attr_hp:.1f} s); remaining NaN/inf {len(still)}; weights restored bit-identically")
                    if still:
                        raise ValueError(f"{tag}: round {k} scores still contain NaN/inf at {fb_dtype} precision ({len(still)} blocks, e.g. {still[:4]})")
            path = os.path.join(args.scores_dir, f"f{fkey}_{round_prefix}round{k}.json")
            save_scores(path, new_scores, scores_meta(fkey, name, pruned, bool(dry), round=k, share=share,
                                                       round_pruned_heads=head_names(chosen), seconds=attr_s, **fallback_extra, **calib_extra(args)))
            rounds.append({"round": k, "share": share, "pruned_heads": [list(h) for h in chosen],
                           "n_pruned_cumulative": len(pruned), "attribution_seconds": attr_s,
                           "n_batches": len(batches), "pruned_heads_zero_score": sum(1 for h in pruned if new_scores[h] == 0.0),
                           "scores_file": path, "seconds": time.time() - t_round, **fallback_extra})
            rep["rounds"] = rounds
            cur = new_scores
            log(f"{tag}: round {k}/{len(shares)} share {share} -> {len(pruned)} heads in total; rescore {attr_s:.1f} s "
                f"({rounds[-1]['pruned_heads_zero_score']}/{len(pruned)} pruned heads score 0) -> {path}")
            store.save(f"{tag} round {k}")
        heads = sorted((parse_block_key(h).layer, parse_block_key(h).index) for h in pruned)
        final_tiers = allocate_compression_tiers(cur, 3, tier_fractions=tuple(fb["tier_fractions"]), mlp_min_tier="int4")
        # xai_iter: INT4 plan from the last-round scores; fixedq: copied from xai_single (as for the selector configs)
        if name == "xai_iter_fixedq":
            quant_plan, plan_source = single_plan, "xai_single"
        elif name == "xai_iter_fixedq_4tier":  # fixed 4-tier plan (from the round-0 scores; identical to xai_single_4tier)
            four_tiers, quant_plan = four_tier_plan(scores0)
            plan_source = "xai_single_4tier"
            rep["tier_fractions_4tier"] = {"heads": list(FOUR_TIER_FRACTIONS), "mlp": list(FOUR_TIER_MLP_FRACTIONS)}
            rep["tier_counts_4tier"] = tier_counts(four_tiers)
        else:
            quant_plan, plan_source = build_compression_plan(final_tiers), "last_round_scores"
        plan = merge_plan(prune_plan_from_heads(heads, n_layers), quant_plan)
        rep["int4_plan_source"] = plan_source
        rep["final_prune_tier_matches_pruned"] = set(heads_from_tiers(final_tiers)) == set(heads)
        rep["final_tier_counts"] = tier_counts(final_tiers)
        rep["attribution_calls"] = len(rounds)
        rep["attribution_seconds_total"] = sum(r["attribution_seconds"] for r in rounds)
        rep["round_shares"] = shares
        rep["overlap_with_xai_heads"] = len(set(heads) & set(xai_heads))
        rep["last_scores_file"] = rounds[-1]["scores_file"] if rounds else None
    else:
        raise ValueError(f"unknown config: {name}")

    rep["pruned_heads"] = [list(h) for h in heads]
    rep["n_pruned_heads"] = len(heads)
    rep["plan_summary"] = plan_summary(plan)
    rep["budget_nominal"] = nominal_budget(plan, model)
    rep["int4_modules_planned"] = int4_module_count(plan)
    if name in FOUR_TIER_CONFIGS:
        rep["int8_modules_planned"] = quant_module_count(plan, "int8")
    rep["int4_module_set_equals_single"] = int4_module_set(plan) == fb["int4_module_set_single"]
    if name in FIXED_INT4_CONFIGS:
        assert rep["int4_module_set_equals_single"] and len(heads) == n_budget, "budget guarantee violated"
    log(f"{tag}: {len(heads)} heads to prune (budget {n_budget}), planned INT4 modules {rep['int4_modules_planned']} "
        f"(xai_single {fb['int4_modules_single']}, same set={rep['int4_module_set_equals_single']}); "
        f"nominal size ratio {rep['budget_nominal']['size_ratio']:.3f}, mean bits/param {rep['budget_nominal']['avg_bits_per_param']:.2f}")

    # --- pruning (already applied for iterative configs; re-zeroing is idempotent) -> rescore -> quantization ---
    t0 = time.time()
    rep["pruning"] = apply_structural_pruning(model, plan, verbose=False)
    if args.rescore_after and name not in ITERATIVE_CONFIGS and name != "xai_single_4tier" and name not in MIXED_CONFIGS:  # 4tier: pruned set = xai_single (same after-scores); mixed: no pruning, no attribution
        batches = calibration_batches(ctx, tokenizer)
        after, attr_s = rescore(model, batches, bool(dry))
        suffix = f"_seed{seed}" if name in STOCHASTIC_CONFIGS else ""
        path = os.path.join(args.scores_dir, f"f{fkey}_{name}_after{suffix}.json")
        names = head_names(heads)
        save_scores(path, after, scores_meta(fkey, name, names, bool(dry), seed=seed, seconds=attr_s,
                                             note="after pruning, before quantization (rescore-after)", **calib_extra(args)))
        rep["rescore_after"] = {"scores_file": path, "attribution_seconds": attr_s, "n_batches": len(batches),
                                "pruned_heads_zero_score": sum(1 for h in names if after[h] == 0.0)}
        log(f"{tag}: rescore-after {attr_s:.1f} s -> {path}")
        store.save(f"{tag} rescore")
    rep["quantization"] = quantize_with_fallback(model, plan, log, _StoreView(store, rep), "cfg")
    rep["apply_seconds"] = time.time() - t0
    rep["int4_modules"] = rep["quantization"]["int4_modules"]
    if name in FOUR_TIER_CONFIGS:
        rep["int8_modules"] = rep["quantization"]["int8_modules"]
        assert rep["int8_modules"] == rep["int8_modules_planned"] and rep["int4_modules"] == rep["int4_modules_planned"], "the 4-tier plan could not be applied"
    if name in MIXED_CONFIGS:
        assert rep["int4_modules"] == rep["int4_modules_planned"] and rep["pruning"]["pruned_heads"] == 0, "the mixed-precision plan could not be applied"
    rep["int4_modules_equals_single"] = rep["int4_modules"] == fb["int4_modules_single"]
    rep["model_bytes_after"] = module_bytes(model)
    rep["measured_model_bytes_ratio"] = rep["model_bytes_after"]["bytes"] / rep["model_bytes_before"]["bytes"]
    log(f"{tag}: applied ({rep['apply_seconds']:.1f} s) pruning={rep['pruning']} quantization={rep['quantization']} "
        f"size {rep['model_bytes_before']['gb']:.2f} -> {rep['model_bytes_after']['gb']:.2f} GB")
    store.save(f"{tag} applied")

    # --- perplexity + MMLU ---
    t0 = time.time()
    if dry:
        ppl = dry.perplexity(model)
    else:
        ppl = ctx["compute_perplexity"](model, tokenizer, ctx["text"], next(model.parameters()).device)
    rep["perplexity"] = ppl
    rep["perplexity_seconds"] = time.time() - t0
    rep["perplexity_finite"] = math.isfinite(ppl)
    log(f"{tag}: perplexity = {ppl:.4f} ({rep['perplexity_seconds']:.1f} s)")
    store.save(f"{tag} perplexity")
    if getattr(args, "save_window_nll", False):  # D-1: opt-in; does not affect the rep["perplexity"] measurement
        import eval_ppl_windows as epw

        w = epw.record_window_nll(rep, model, tokenizer, ctx["window_nll_texts"], bool(dry))
        log(f"{tag}: window NLL stored: " + ", ".join(f"{ds} ppl {v['perplexity']:.4f} ({v['n_windows']} windows, {v['seconds']:.1f} s)"
                                                           for ds, v in w.items()))
        store.save(f"{tag} window NLL")
    if args.mmlu:
        rep["mmlu"] = measure_mmlu(model, tokenizer, ctx, log)
        rep["mmlu_subset_acc"] = rep["mmlu"]["mmlu_subset_acc"]


def run_one(name: str, seed: int, fb: Dict[str, Any], ctx: Dict[str, Any], rep: Dict[str, Any],
            store: ResultStore, log: Logger) -> None:
    """Clean load -> apply/measure -> FULL release (memory pattern: the model lives in a single dict that is cleared at the end)."""
    dry: Optional[DryRun] = ctx["dry"]
    rep.update({"status": "running", "seed": seed, "fraction": fb["fraction"],
                "started_at": datetime.now().isoformat(timespec="seconds")})
    store.save(f"{name} started")
    t_cfg = time.time()
    set_seed(seed)
    rep["cuda_allocated_before_load_gb"] = ensure_clean_gpu(log, name)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    box: Dict[str, Any] = {}
    try:
        box["tokenizer"], box["model"], rep["model_load_seconds"] = dry.load_model(log) if dry else load_model(log)
        apply_and_measure(name, seed, fb, ctx, rep, store, log, box)
        rep["status"] = "completed"
    finally:
        rep["peak_vram_gb"] = cuda_gb("peak")
        box.clear()
        rep["cuda_allocated_after_free_gb"] = release_gpu()
        rep["seconds"] = time.time() - t_cfg
        log(f"{name}: model released; CUDA allocated={rep['cuda_allocated_after_free_gb']} GB, "
            f"peak VRAM={rep['peak_vram_gb']} GB, {rep['seconds']:.0f} s")
        store.save(f"{name} finished")


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summarize_config(cfg: Dict[str, Any], fp16_ppl: Optional[float], fp16_acc: Optional[float]) -> None:
    reps = [r for r in cfg["repeats"] if r.get("status") == "completed"]
    ppls = [r["perplexity"] for r in reps]
    cfg["n_completed_repeats"] = len(reps)
    cfg["perplexity_values"] = ppls
    cfg["perplexity_mean"] = float(np.mean(ppls)) if ppls else None
    cfg["perplexity_std"] = float(np.std(ppls, ddof=1)) if len(ppls) > 1 else (0.0 if ppls else None)
    if reps:
        r0 = reps[0]
        for k in ("pruned_heads", "n_pruned_heads", "int4_modules", "int4_modules_equals_single",
                  "int4_module_set_equals_single", "measured_model_bytes_ratio", "budget_nominal"):
            cfg[k] = r0.get(k)
        cfg["model_bytes_after_gb"] = r0["model_bytes_after"]["gb"]
        if "int8_modules" in r0:  # 4-tier configs only; the JSON schema of the others is unchanged
            cfg["int8_modules"] = r0["int8_modules"]
        if "mixed_precision" in r0:  # mixed_* configs only
            mp = r0["mixed_precision"]
            cfg["mixed_precision"] = {"k_percent": mp["k_percent"], "selector": mp["selector"], "n_fp16_blocks": mp["n_fp16_blocks"],
                                      "n_fp16_modules": mp["n_fp16_modules"],
                                      "fp16_blocks_per_repeat": [r["mixed_precision"]["fp16_blocks"] for r in reps],
                                      "fp16_modules": mp["fp16_modules"],
                                      "overlap_with_xai_fp16_blocks": [r["mixed_precision"]["overlap_with_xai_fp16_blocks"] for r in reps]}
            sizes = [r["model_bytes_after"]["gb"] for r in reps]
            cfg["model_bytes_after_gb_values"] = sizes  # random selection keeps the per-type count, so the size must be equal across repeats
        cfg["pruned_heads_identical_across_repeats"] = all(r["pruned_heads"] == r0["pruned_heads"] for r in reps)
        cfg["seconds"] = sum(r["seconds"] for r in reps)
        peaks = [r["peak_vram_gb"] for r in reps if r.get("peak_vram_gb") is not None]
        cfg["peak_vram_gb"] = max(peaks) if peaks else None
        if "overlap_with_xai_heads" in r0:
            cfg["overlap_with_xai_heads"] = [r["overlap_with_xai_heads"] for r in reps]
        if "rounds" in r0:
            cfg["attribution_calls"] = r0.get("attribution_calls")
            cfg["attribution_seconds_total"] = r0.get("attribution_seconds_total")
            cfg["round_shares"] = r0.get("round_shares")
            cfg["round_pruned_heads"] = [r["pruned_heads"] for r in r0["rounds"]]
            cfg["round_seconds"] = [r["seconds"] for r in r0["rounds"]]
        else:
            cfg["attribution_calls"] = 1 if "rescore_after" in r0 else 0
            cfg["attribution_seconds_total"] = r0["rescore_after"]["attribution_seconds"] if "rescore_after" in r0 else 0.0
        accs = [r["mmlu_subset_acc"] for r in reps if "mmlu_subset_acc" in r]
        if accs:
            cfg["mmlu_subset_acc_values"] = accs
            cfg["mmlu_subset_acc_mean"] = float(np.mean(accs))
            if fp16_acc is not None:
                cfg["mmlu_delta_vs_fp16"] = cfg["mmlu_subset_acc_mean"] - fp16_acc
    if fp16_ppl is not None and cfg.get("perplexity_mean") is not None:
        cfg["delta_vs_fp16"] = cfg["perplexity_mean"] - fp16_ppl
        cfg["ratio_vs_fp16"] = cfg["perplexity_mean"] / fp16_ppl


def fraction_comparison(fr: Dict[str, Any]) -> Dict[str, Any]:
    """Within-ratio comparisons: iterative − one shot, selectors − xai_single (ppl and MMLU)."""
    cfgs = fr["configs"]

    def ppl(n: str) -> Optional[float]:
        c = cfgs.get(n)
        return c.get("perplexity_mean") if c else None

    def acc(n: str) -> Optional[float]:
        c = cfgs.get(n)
        return c.get("mmlu_subset_acc_mean") if c else None

    out: Dict[str, Any] = {}
    base = ppl("xai_single")
    if base is not None:
        for n in ("xai_iter", "xai_iter_fixedq", "prune_wanda", "prune_wanda_ln", "prune_attnconf", "prune_taylor", "prune_random",
                  "az_buda_cok_kuantize", "xai_single_4tier", "xai_iter_fixedq_4tier"):
            if ppl(n) is not None:
                out[f"{n}_minus_xai_single_ppl"] = ppl(n) - base
            if acc(n) is not None and acc("xai_single") is not None:
                out[f"{n}_minus_xai_single_mmlu"] = acc(n) - acc("xai_single")
    # D-1: selector comparison at equal budget (k=10): control − XAI
    if ppl("mixed_xai_k10") is not None:
        for n in ("mixed_random_k10", "mixed_magnitude_k10", "mixed_wanda_ln_k10"):
            if ppl(n) is not None:
                out[f"{n}_minus_mixed_xai_k10_ppl"] = ppl(n) - ppl("mixed_xai_k10")
            if acc(n) is not None and acc("mixed_xai_k10") is not None:
                out[f"{n}_minus_mixed_xai_k10_mmlu"] = acc(n) - acc("mixed_xai_k10")
    # same pruning procedure, different INT4 plan: contribution of the quantization plan to xai_iter's gain
    if ppl("xai_iter") is not None and ppl("xai_iter_fixedq") is not None:
        out["xai_iter_fixedq_minus_xai_iter_ppl"] = ppl("xai_iter_fixedq") - ppl("xai_iter")
        if acc("xai_iter") is not None and acc("xai_iter_fixedq") is not None:
            out["xai_iter_fixedq_minus_xai_iter_mmlu"] = acc("xai_iter_fixedq") - acc("xai_iter")
    return out


def format_summary_table(fractions: Dict[str, Any], fp16_ppl: Optional[float], fp16_acc: Optional[float]) -> str:
    def f(v: Any, spec: str) -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    lines = [f"{'ratio':<6}{'config':<22}{'ppl':>9}{'±std':>8}{'Δfp16':>9}{'mmlu':>7}{'head':>6}{'int4':>6}{'GB':>7}{'attr':>5}{'s':>7}  status"]
    if fp16_ppl is not None:
        lines.append(f"{'-':<6}{'fp16':<22}{fp16_ppl:>9.4f}{'':>8}{0.0:>9.4f}{f(fp16_acc, '.3f'):>7}{0:>6}{0:>6}{'':>7}{'':>5}{'':>7}  reference")
    for fkey, fr in fractions.items():
        for name, c in fr["configs"].items():
            lines.append(f"{fkey:<6}{name:<22}{f(c.get('perplexity_mean'), '.4f'):>9}{f(c.get('perplexity_std'), '.4f'):>8}"
                         f"{f(c.get('delta_vs_fp16'), '+.4f'):>9}{f(c.get('mmlu_subset_acc_mean'), '.3f'):>7}"
                         f"{str(c.get('n_pruned_heads', '-')):>6}{str(c.get('int4_modules', '-')):>6}"
                         f"{f(c.get('model_bytes_after_gb'), '.2f'):>7}{str(c.get('attribution_calls', '-')):>5}"
                         f"{f(c.get('seconds'), '.0f'):>7}  {c.get('status')}")
    return "\n".join(lines)


def estimate_minutes(fractions: Sequence[float], config_names: Sequence[str], n_rounds: int, n_repeats: int,
                     mmlu: bool, rescore_after: bool, skip_baseline: bool) -> Dict[str, Any]:
    """Rough GPU runtime estimate (units from the module docstring); written to the JSON and the preflight log."""
    base, mmlu_min, attr = 2.5, (3.0 if mmlu else 0.0), 6.0
    per: Dict[str, float] = {}
    for fr in fractions:
        total = 0.0
        for n in config_names:
            if n in ONLY_20_CONFIGS and not math.isclose(fr, AZ_BUDA_ONLY_FRACTION):
                continue
            if n in ITERATIVE_CONFIGS:
                total += n_rounds * attr + base + mmlu_min
            else:
                one = base + mmlu_min + (attr if rescore_after and n not in MIXED_CONFIGS else 0.0)  # mixed: no attribution
                total += one * (n_repeats if n in STOCHASTIC_CONFIGS else 1)
        per[fraction_key(fr)] = total
    fp16 = 0.0 if skip_baseline else (1.5 + (2.0 if mmlu else 0.0))
    return {"per_fraction_minutes": per, "fp16_minutes": fp16, "total_minutes": fp16 + sum(per.values()),
            "assumptions": "load+apply+ppl 2.5 min, MMLU 3 min, attribution 6 min per call (measured in earlier runs)"}


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log = Logger(args.log, to_file=should_log_to_file(args, default_log_file(args)))  # dry-run/preflight never write to the monitored default log
    store = ResultStore(args.output, log)
    t_start = time.time()
    fractions = parse_fractions(args.fractions)
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in config_names if c not in CONFIG_DESCRIPTIONS]
    if unknown:
        raise SystemExit(f"unknown config(s): {unknown}; valid: {ALL_CONFIGS} + opt-in: {EXTRA_CONFIGS}")
    if args.n_rounds < 1:
        raise SystemExit("--n-rounds must be >= 1")
    log(f"===== Ratio sweep starting: fractions={fractions} configs={config_names} n_rounds={args.n_rounds} "
        f"seed={args.seed} n_repeats={args.n_repeats} mmlu={args.mmlu} rescore_after={args.rescore_after} "
        f"dry_run={args.dry_run} =====", tag="G7")

    # --- inputs (no GPU needed) ---
    dry: Optional[DryRun] = DryRun(args.seed) if args.dry_run else None
    if dry:
        scores = dry.scores
        baseline_gun1, gun6_both = None, None
        # write the round-0 scores to a file (for the compute_drift.py dry-run check)
        args.scores = os.path.join(os.path.dirname(args.output) or ".", "importance_scores_dry.json")
        save_scores(args.scores, scores, {"model": "mini (dry-run)", "note": "round 0 (mock random scores)"})
    else:
        scores = load_scores(args.scores)
        baseline_gun1 = None
        if os.path.exists(args.baseline):
            with open(args.baseline, "r", encoding="utf-8") as f:
                baseline_gun1 = float(json.load(f)["perplexity"])
        gun6_both = GUN6_BOTH_PPL_FALLBACK
        if os.path.exists(args.gun6_json):
            with open(args.gun6_json, "r", encoding="utf-8") as f:
                gun6_both = float(json.load(f).get("configs", {}).get("both", {}).get("perplexity_mean", GUN6_BOTH_PPL_FALLBACK))
    budgets = {fraction_key(fr): fraction_budget(scores, fr, bool(dry)) for fr in fractions}
    estimate = estimate_minutes(fractions, config_names, args.n_rounds, args.n_repeats, args.mmlu, args.rescore_after,
                                args.skip_baseline)

    previous = None
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            previous = json.load(f)
        log(f"--resume: read {args.output}; completed (ratio, config) pairs will be skipped", tag="G7")

    def configs_for(fr: float) -> List[str]:
        return [n for n in config_names if n not in ONLY_20_CONFIGS or math.isclose(fr, AZ_BUDA_ONLY_FRACTION)]

    store.data = {
        "run": {"started_at": datetime.now().isoformat(timespec="seconds"), "status": "running", "args": vars(args),
                "model": "mini (dry-run)" if dry else MODEL_NAME, "env": env_info(), "fractions": fractions,
                "config_order": config_names, "time_estimate": estimate},
        "reference": {
            "fp16_baseline_gun1": baseline_gun1, "gun6_both_perplexity": gun6_both, "reproduction_tolerance": REPRO_TOL,
            "scores_file": args.scores, "scores_dir": args.scores_dir,
            "round0_scores": "stored attribution scores (scores_file); round-k file = rescore on the masked FP16 model AFTER the round-k pruning",
            "attribution": {**(DRY_ATTRIBUTION if dry else ATTRIBUTION), "calibration": None if dry else dict(CALIB)},
            "rescore_after": args.rescore_after, "n_rounds": args.n_rounds,
            "size_note": "model_bytes = parameter + buffer bytes; excludes the bnb quant_state overhead (same measurement as run_e2e_pipeline.py / run_ablation_tests.py)",
            "peak_vram_note": "reset_peak_memory_stats per config; memory allocated before loading is in cuda_allocated_before_load_gb",
        },
        "fp16_rerun": (previous or {}).get("fp16_rerun", {"status": "skipped" if args.skip_baseline else "pending"}),
        "fractions": {fk: {"budget": budget_json(b, args.n_rounds),
                           "configs": {n: {"description": CONFIG_DESCRIPTIONS[n], "status": "pending", "repeats": []}
                                       for n in configs_for(b["fraction"])},
                           "comparison": {}}
                      for fk, b in budgets.items()},
    }
    if args.calib_dataset != "wikitext2":  # added only for out-of-domain calibration (the default JSON schema stays unchanged)
        store.data["reference"].update({
            "calib_dataset": args.calib_dataset,
            "calib_note": "round-0 scores, round/rescore rescoring and the Taylor/Wanda criteria use this calibration set; "
                          "EVALUATION is WikiText-2 test perplexity + MMLU (out-of-domain calibration test); xai_single = the ablation run's `both` path",
            "reproduction_check": "not applied (scores are not the stored WikiText-2 scores)"})
        log(f"Calibration set: {args.calib_dataset}; scores {args.scores}; output {args.output}; score dir {args.scores_dir}", tag="G7")
    mixed_names = [c for c in config_names if c in MIXED_CONFIGS]
    if mixed_names:  # added only when mixed_* configs are requested (the default JSON schema stays unchanged)
        k0 = None
        if not dry and os.path.exists(BASELINES_FILE):
            with open(BASELINES_FILE, "r", encoding="utf-8") as f:
                k0 = (json.load(f).get("summary", {}).get("configs", {}) or {}).get("nf4_uniform")
        xai_modules = module_importance_scores(scores)
        store.data["reference"]["mixed_precision"] = {
            "definition": "NO pruning; module = a layer's attention block (q/k/v/o_proj) or MLP block (gate/up/down_proj); module score: attn = "
                          "LOWER median of the layer's head scores, MLP = MLP score; WITHIN each type the top round_half_up(k/100·N_type) "
                          "modules stay FP16, all other decoder Linear layers NF4 (double quantization, fp16 compute); lm_head and embeddings FP16 "
                          "(same as nf4_uniform); no attribution (stored scores)",
            "configs": {n: MIXED_CONFIGS[n] for n in mixed_names},
            "n_modules": {kind: sum(1 for k in xai_modules if k.endswith("." + kind)) for kind in ("attn", "mlp")},
            "k0_reference": {"source": f"{BASELINES_FILE} summary.configs.nf4_uniform (not re-run)", "values": k0},
            "fraction_key_note": "configs are ratio-independent; they run under the '0.2' key to keep the default JSON structure",
            "wanda_ln_scores_file": args.wanda_ln_scores if "mixed_wanda_ln_k10" in mixed_names else None,
        }
        for n in mixed_names:
            if MIXED_CONFIGS[n]["selector"] == "xai":
                blocks = [k for k, t in allocate_mixed_precision(xai_modules, MIXED_CONFIGS[n]["k"]).items() if t == "fp16"]
                log(f"{n}: {len(blocks)} blocks stay FP16 (stored scores): {blocks}", tag="G7")
    env = store.data["run"]["env"]
    log(f"Environment: torch {env['torch']}, transformers {env['transformers']}, bitsandbytes {env.get('bitsandbytes')}, "
        f"CUDA={env['cuda_available']} ({env.get('gpu')})", tag="G7")
    for fk, b in budgets.items():
        bj = store.data["fractions"][fk]["budget"]
        log(f"ratio {fk}: tier_fractions={b['tier_fractions']} budget {b['n_prune_heads']}/{b['n_heads_total']} heads, "
            f"INT4 layers attn/mlp={b['int4_layers']} (= {b['int4_modules_single']} modules), round shares {bj['round_shares']}, "
            f"configs {list(store.data['fractions'][fk]['configs'])}", tag="G7")
    log(f"Time estimate (min): {json.dumps(estimate['per_fraction_minutes'])} + fp16 {estimate['fp16_minutes']} "
        f"= total ~{estimate['total_minutes']:.0f} min", tag="G7")
    log(f"FP16 baseline={baseline_gun1}; ablation `both`={gun6_both}", tag="G7")

    if args.preflight:
        problems: List[str] = []
        if not dry and not torch.cuda.is_available():
            problems.append("CUDA not available")
        if not dry and baseline_gun1 is None:
            problems.append(f"FP16 baseline could not be read: {args.baseline}")
        if not dry and not os.path.exists(args.gun6_json):
            log(f"WARNING: ablation JSON not found, the check will use the default {GUN6_BOTH_PPL_FALLBACK}: {args.gun6_json}", tag="G7")
        if not dry and "mixed_wanda_ln_k10" in config_names:
            try:
                n_w = len(load_wanda_ln_head_scores(args.wanda_ln_scores))
                if n_w != len(all_heads(scores)):
                    problems.append(f"wanda_ln criterion file has {n_w} head scores, expected {len(all_heads(scores))}")
            except Exception as e:
                problems.append(f"mixed_wanda_ln_k10: criterion file could not be read ({args.wanda_ln_scores}): {e!r}")
        if mixed_names and not any(math.isclose(fr, AZ_BUDA_ONLY_FRACTION) for fr in fractions):
            problems.append("mixed_* configs run under the '0.2' ratio key; --fractions must include 0.2")
        if any(b["n_prune_heads"] < args.n_rounds for b in budgets.values()):
            log("WARNING: a ratio's budget is smaller than the number of rounds; some rounds get share 0 (skipped)", tag="G7")
        if not dry and args.calib_dataset == "c4":  # C4 stream + tokenizer access; do the passages match the score file?
            try:
                from transformers import AutoTokenizer

                _, ids = load_c4_calibration_batches(AutoTokenizer.from_pretrained(MODEL_NAME), args.scores)
                log(f"C4 calibration passages ready: {len(ids)} passages, stream indices {[p['stream_index'] for p in ids]} "
                    f"(matched the identifiers in the score file)", tag="G7")
            except Exception as e:
                problems.append(f"C4 calibration passages could not be prepared: {e!r}")
        if not dry and args.mmlu:
            try:
                import eval_mmlu

                subset = eval_mmlu.load_or_build_subset(log=lambda m: log(m, tag="MMLU"))
                log(f"MMLU subset ready: {len(subset['records'])} questions", tag="MMLU")
            except Exception as e:
                problems.append(f"MMLU subset could not be loaded: {e!r}")
        if args.save_window_nll:  # D-1: dataset names and the C4 subset (sha1) are verified BEFORE the run
            import eval_ppl_windows as epw

            wn = epw.parse_ppl_datasets(args.window_nll_datasets)
            log(f"Window NLL datasets: {wn} (+~70 s per run and dataset)", tag="G7")
            if not dry and "c4" in wn:
                try:
                    _, wn_meta = epw.load_ppl_texts(["c4"])
                    log(f"C4 window-NLL subset ready: {wn_meta['c4_ppl_subset']['n_docs']} documents, sha1 verified", tag="G7")
                except Exception as e:
                    problems.append(f"C4 window-NLL subset could not be prepared: {e!r}")
        store.data["run"]["status"] = "preflight_only"
        if problems:
            log("Preflight FAILED (model not loaded): " + "; ".join(problems), tag="ERR")
            return 1
        log("Preflight completed (model not loaded). Nothing was written to the result file.", tag="G7")
        return 0
    if not torch.cuda.is_available() and not dry:
        log("WARNING: CUDA not available — the 7B model + bitsandbytes cannot run on CPU. Run on a GPU (or use --dry-run).", tag="G7")
    os.makedirs(args.scores_dir, exist_ok=True)
    store.save("start")

    ctx: Dict[str, Any] = {"args": args, "dry": dry, "scores": scores}
    exit_code = 0
    fp16_ppl: Optional[float] = baseline_gun1
    fp16_acc: Optional[float] = None
    try:
        if not dry:
            log("Loading the WikiText-2 test set...", tag="G7")
            ctx["text"] = load_wikitext_text()
            ctx["compute_perplexity"], max_len, stride = perplexity_tools()
            store.data["reference"]["perplexity_settings"] = {"max_length": max_len, "stride": stride,
                                                              "dataset": "wikitext-2-raw-v1 (test split)"}
            if args.mmlu:
                import eval_mmlu

                ctx["mmlu_subset"] = eval_mmlu.load_or_build_subset(log=lambda m: log(m, tag="MMLU"))
                store.data["reference"]["mmlu_n_questions"] = len(ctx["mmlu_subset"]["records"])

        if args.save_window_nll:  # D-1: two-dataset window NLL; the WikiText text was loaded above, the C4 subset is sha1-verified
            import eval_ppl_windows as epw

            wn = epw.parse_ppl_datasets(args.window_nll_datasets)
            if dry:
                ctx["window_nll_texts"] = {d: None for d in wn}
            else:
                ctx["window_nll_texts"], wn_meta = epw.load_ppl_texts(wn, ctx["text"])
                store.data["reference"]["window_nll"] = {"datasets": wn, **wn_meta}
                log(f"Window NLL datasets ready: {wn}", tag="G7")

        # --- FP16 re-measurement (once, clean load) ---
        fp = store.data["fp16_rerun"]
        if args.skip_baseline:
            log(f"FP16 rerun SKIPPED (--skip-baseline); reference baseline={fp16_ppl}", tag="G7")
        elif fp.get("status") == "completed":
            fp16_ppl, fp16_acc = fp["perplexity"], fp.get("mmlu_subset_acc")
            log(f"FP16 rerun taken from the previous run (--resume): ppl={fp16_ppl} mmlu={fp16_acc}", tag="G7")
        else:
            fp.update({"status": "running"})
            set_seed(args.seed)
            t0 = time.time()
            ensure_clean_gpu(log, "fp16")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            box: Dict[str, Any] = {}
            try:
                box["tokenizer"], box["model"], fp["model_load_seconds"] = dry.load_model(log) if dry else load_model(log)
                fp["model_bytes"] = module_bytes(box["model"])
                if dry:
                    ppl0 = dry.perplexity(box["model"])
                else:
                    ppl0 = ctx["compute_perplexity"](box["model"], box["tokenizer"], ctx["text"], next(box["model"].parameters()).device)
                fp["perplexity"] = ppl0
                if args.mmlu:
                    fp["mmlu"] = measure_mmlu(box["model"], box["tokenizer"], ctx, log)
                    fp["mmlu_subset_acc"] = fp["mmlu"]["mmlu_subset_acc"]
            finally:
                fp["peak_vram_gb"] = cuda_gb("peak")
                box.clear()
                release_gpu()
            fp.update({"seconds": time.time() - t0, "status": "completed",
                       "minus_gun1": (ppl0 - baseline_gun1) if baseline_gun1 is not None else None})
            fp16_ppl, fp16_acc = ppl0, fp.get("mmlu_subset_acc")
            log(f"FP16 RERUN DONE: perplexity = {ppl0:.4f} (baseline: {baseline_gun1}, diff {fp['minus_gun1']}), "
                f"mmlu={fp16_acc}, {fp['seconds']:.1f} s", tag="G7")
            store.save("fp16 rerun")

        # --- ratios × configs ---
        n_total = sum(len(fr["configs"]) for fr in store.data["fractions"].values())
        ci = 0
        for fk, fr in store.data["fractions"].items():
            fb = budgets[fk]
            for name, cfg in fr["configs"].items():
                ci += 1
                prev_cfg = (previous or {}).get("fractions", {}).get(fk, {}).get("configs", {}).get(name)
                if prev_cfg and prev_cfg.get("status") == "completed":
                    fr["configs"][name] = prev_cfg
                    fr["configs"][name]["resumed"] = True
                    log(f"CONFIG {ci}/{n_total} f={fk} {name}: --resume, taken from the previous run "
                        f"(ppl={prev_cfg.get('perplexity_mean')})", tag="G7")
                    continue
                n_rep = args.n_repeats if name in STOCHASTIC_CONFIGS else 1
                cfg.update({"status": "running", "n_repeats": n_rep, "stochastic": name in STOCHASTIC_CONFIGS})
                log(f"===== CONFIG {ci}/{n_total}: f={fk} {name} ({n_rep} repeats) — {cfg['description']} =====", tag="G7")
                try:
                    for r in range(n_rep):
                        rep: Dict[str, Any] = {}
                        cfg["repeats"].append(rep)
                        run_one(name, args.seed + r, fb, ctx, rep, store, log)
                    summarize_config(cfg, fp16_ppl, fp16_acc)
                    cfg["status"] = "completed"
                    if (name == "xai_single" and math.isclose(fb["fraction"], 0.2) and gun6_both is not None
                            and reproduction_check_applies(args)):
                        diff = abs(cfg["perplexity_mean"] - gun6_both)
                        cfg["reproduction_check"] = {"expected_gun6_both": gun6_both, "abs_diff": diff,
                                                     "within_tolerance": diff <= REPRO_TOL}
                        log(("CHECK OK" if diff <= REPRO_TOL else "CHECK FAILED") +
                            f": xai_single(20%)={cfg['perplexity_mean']:.4f} vs ablation `both` {gun6_both:.4f} "
                            f"(diff {diff:.4f}, tolerance {REPRO_TOL})", tag="CHECK")
                    if name == "mixed_xai_k0":  # does this script's load->quantize path reproduce nf4_uniform? (the run does NOT stop on a mismatch)
                        k0_ref = ((store.data["reference"].get("mixed_precision") or {}).get("k0_reference") or {}).get("values") or {}
                        expected = k0_ref.get("perplexity")
                        check = nf4_uniform_check(cfg["perplexity_mean"], expected)
                        if check is not None:
                            cfg["nf4_uniform_check"] = check
                            log(("CHECK OK" if check["within_tolerance"] else "CHECK FAILED") +
                                f": mixed_xai_k0={cfg['perplexity_mean']:.4f} vs nf4_uniform {expected:.4f} (diff {check['abs_diff']:.4f}, "
                                f"tolerance {K0_TOL})", tag="CHECK")
                        else:
                            log(f"CHECK SKIPPED: no nf4_uniform reference ({BASELINES_FILE})", tag="CHECK")
                    log(f"CONFIG {ci}/{n_total} DONE: f={fk} {name} ppl={cfg['perplexity_mean']:.4f} ± {cfg['perplexity_std']:.4f} "
                        f"(Δfp16={cfg.get('delta_vs_fp16')}) mmlu={cfg.get('mmlu_subset_acc_mean')} {cfg['seconds']:.0f} s", tag="G7")
                except Exception as e:  # a single config failure must not abort the others
                    summarize_config(cfg, fp16_ppl, fp16_acc)
                    cfg.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
                    log(f"CONFIG f={fk} {name} ERROR: {e!r} — moving on to the next config", tag="ERR")
                    log(cfg["traceback"], tag="ERR")
                    exit_code = 1
                fr["comparison"] = fraction_comparison(fr)
                if "calib_passages" in ctx:  # c4: identifiers of the passages used
                    store.data["reference"]["calibration_passages"] = ctx["calib_passages"]
                store.save(f"f={fk} {name} summary")
        store.data["run"]["status"] = "completed" if exit_code == 0 else "completed_with_failures"
    except BaseException as e:  # including KeyboardInterrupt / SystemExit: do not lose partial results
        store.data["run"]["status"] = "failed"
        store.data["run"]["error"] = repr(e)
        store.data["run"]["traceback"] = traceback.format_exc()
        log(f"ERROR: {e!r} — writing partial results to disk", tag="ERR")
        log(store.data["run"]["traceback"], tag="ERR")
        exit_code = 1
    finally:
        for fr in store.data["fractions"].values():
            for cfg in fr["configs"].values():
                if cfg.get("repeats") and cfg.get("status") not in ("completed", "failed"):
                    summarize_config(cfg, fp16_ppl, fp16_acc)  # partial: summarize the completed repeats
            fr["comparison"] = fraction_comparison(fr)
        store.data["run"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        store.data["run"]["total_seconds"] = time.time() - t_start
        store.save("final")
        log("SUMMARY TABLE:\n" + format_summary_table(store.data["fractions"], fp16_ppl, fp16_acc), tag="G7")
        log(f"===== Finished: status={store.data['run']['status']}, total {store.data['run']['total_seconds']:.1f} s =====", tag="G7")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
