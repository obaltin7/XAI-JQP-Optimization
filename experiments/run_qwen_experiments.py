"""
Second-model generalisation run (default: Qwen/Qwen2.5-7B-Instruct; GPU, or CPU with --dry-run).

Applies the pipeline of run_ablation_tests / run_iterative_pruning (Logger, atomic ResultStore, env/args record, clean
reload per configuration, bitsandbytes quantisation with fallback, DryRun, full memory release) to another decoder model.
No Mistral-specific dimension is hard-coded: the number of layers / heads / KV heads, head_dim and the q/k/v bias are read
from the model config and modules (Qwen2.5-7B: 28 layers x 28 heads = 784 heads, 4 KV heads (GQA group 7), head_dim 128,
q/k/v_proj with bias; 20% = 157 heads).

Stages:
  baseline     FP16 (or --dtype bf16) model: WikiText-2 perplexity + MMLU (500-question subset) + size + time + peak VRAM
  attribution  xai_engine with the Mistral settings (same 16 WikiText-2 passages, T=256, B=4, n_steps 8,
               internal_batch_size 2) -> results/gun9_scores/importance_scores_<model>.json (schema of run_xai_on_mistral.py;
               read by compare_scores / compressor). Not recomputed if --scores is given, or with --resume if the file exists.
  plan         within-type percentiles, tier_fractions = (f, (1-f)/2, (1-f)/2) (default f=0.2 -> (0.2, 0.4, 0.4)); MLP blocks
               are never pruned; attention INT4 decided by the per-layer median rule (allocate_compression_tiers +
               build_compression_plan, as for Mistral)
  configs      (each from a clean reload; perplexity + MMLU with per-question records + measured size + time + peak VRAM)
    quant_only      no pruning; INT4 modules of the plan
    prune_only      plan heads selected by xAI are pruned; no quantisation
    both            xAI pruning + INT4 = single-shot XAI-JQP (same as `xai_single` in run_iterative_pruning.py)
    prune_random    same head count, seeded random selection (--n-repeats seeds); no quantisation
    prune_reverse   same head count, HIGHEST xAI scores; no quantisation
    prune_taylor    same head count, lowest head_taylor_scores + the INT4 plan of `both`
    xai_iter_fixedq n_rounds pruning rounds, rescoring the masked model each round (round scores in results/gun9_scores/);
                    INT4 plan identical to `both` (final configuration)
  Each configuration therefore has the same definition as its Mistral counterpart (results/ablation_gun6.json:
  quant_only/prune_only/both/random/reverse; results/iterative_gun7.json: taylor/fixedq), as used by the make_figures
  model2 table. Each configuration in the JSON carries a "quantized" field.

IG baseline token (--baseline-token auto): config.pad_token_id -> tokenizer.pad_token_id -> 0. For Mistral this resolves to
0 (<unk>); Qwen2.5 has no pad token in its config and the tokenizer pad is <|endoftext|> (id 0 = "!" is a real token and is
therefore not used). The resolved id is written to the JSON.

--smoke: ~5 min health check before the full run -> <output>_smoke.json: (1) finite forward pass (fp16 activation overflow
risk for Qwen2; use --dtype bf16 on NaN/Inf), (2) short perplexity, (3) attribution on 1 batch x n_steps 2 (finite, > 0,
timing -> full attribution estimate), (4) masked vs physical logit difference for 3 heads (PrunedHeadAttention, real
weights), (5) finite logits after INT4 on layer 0.

Runtime estimate on an L40S (scaled from the Mistral runs; Qwen2.5-7B has 7.6B parameters, 56 blocks, 152k vocabulary):
  baseline ~4 min; attribution ~6 min/call; unquantised config ~5 min, quantised ~5.5 min; prune_random 3 seeds ~15 min;
  xai_iter_fixedq 3 rounds ~23 min => total ~75-90 min. Peak VRAM: fp16 model ~15.3 GB, attribution/Taylor ~21-23 GB.

Usage:
    python experiments/run_qwen_experiments.py --dry-run      # mini Qwen2 on CPU (outputs in dryrun_out/)
    python experiments/run_qwen_experiments.py --preflight    # CUDA, config access, dimensions, budget, MMLU subset
    python experiments/run_qwen_experiments.py --smoke        # short health check (loads the model)
    python experiments/run_qwen_experiments.py                # full run -> results/model2_qwen_gun9.json, log_gun9_model2.txt
    python experiments/run_qwen_experiments.py --resume       # skip completed stages/configurations
"""

# HF_HOME must be set BEFORE transformers/datasets are imported (see run_e2e_pipeline.py)
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import json
import math
import re
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    apply_quantization,
    apply_structural_pruning,
    build_compression_plan,
    estimate_compression_budget,
    head_taylor_scores,
    load_scores,
    model_budget_dims,
    physically_prune_heads,
)
# Shared building blocks of the Mistral ablation / iterative-pruning scripts (imported unchanged)
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
    quant_only_plan,
    select_heads_by_score,
    select_heads_random,
    set_seed,
)
from run_baselines import ensure_clean_gpu, release_gpu
from run_e2e_pipeline import (
    SMOKE_TEXT,
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
from run_iterative_pruning import (
    ATTRIBUTION,
    DRY_ATTRIBUTION,
    fraction_key,
    head_names,
    int4_module_count,
    int4_module_set,
    save_scores,
    select_round_heads,
    split_budget,
    summarize_config,
    tier_fractions_for,
)
from xai_engine import calculate_importance_scores

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_FILE = os.path.join("results", "model2_qwen_gun9.json")
LOG_FILE = "log_gun9_model2.txt"
SCORES_DIR = os.path.join("results", "gun9_scores")
DRYRUN_DIR = "dryrun_out"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

CONFIG_DESCRIPTIONS: Dict[str, str] = {
    "quant_only": "no pruning; the attn/MLP INT4 tiers of the plan are applied as is",
    "prune_only": "prune heads of the plan selected by xAI (LIG) are pruned; no quantisation",
    "both": "xAI pruning + INT4 = single-shot XAI-JQP (same path as xai_single in run_iterative_pruning.py)",
    "prune_random": "same head count; seeded random selection (n_repeats repeats); no quantisation",
    "prune_reverse": "same head count; heads with the HIGHEST xAI scores are pruned (reverse control); no quantisation",
    "prune_taylor": "same head count; criterion head_taylor_scores (Σ|G⊙W|), lowest N; INT4 plan identical to both",
    "xai_iter_fixedq": "n_rounds pruning rounds, masked model rescored each round (exclude_heads); INT4 plan identical "
                       "to both (final configuration)",
}
ALL_CONFIGS = list(CONFIG_DESCRIPTIONS)
STOCHASTIC_CONFIGS = {"prune_random"}
QUANTIZED_CONFIGS = {"quant_only", "both", "prune_taylor", "xai_iter_fixedq"}
FIXED_INT4_CONFIGS = {"both", "prune_taylor", "xai_iter_fixedq"}  # INT4 module SET must equal that of both


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Second model (Qwen2.5-7B-Instruct): attribution + 20% plan + control configurations")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (Mistral/Llama/Qwen2-style decoder)")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="fp16",
                   help="load precision; use bf16 if fp16 produces NaN/Inf (reported by --smoke)")
    p.add_argument("--fraction", type=float, default=0.2, help="fraction of head BLOCKS to prune (not a parameter ratio)")
    p.add_argument("--configs", default=",".join(ALL_CONFIGS), help="comma-separated configuration list (default: all)")
    p.add_argument("--n-rounds", type=int, default=3, help="number of xai_iter_fixedq rounds")
    p.add_argument("--n-steps", type=int, default=ATTRIBUTION["n_steps"], help="number of IG Riemann steps (default 8)")
    p.add_argument("--baseline-token", default="auto",
                   help="IG baseline token id; auto = config.pad_token_id -> tokenizer.pad_token_id -> 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=3, help="number of prune_random repeats; seeds seed, seed+1, ...")
    p.add_argument("--scores", default=None, help="precomputed importance-score JSON (skips the attribution stage)")
    p.add_argument("--scores-dir", default=SCORES_DIR, help="directory for attribution and per-round score JSONs")
    p.add_argument("--no-mmlu", action="store_true", help="do NOT evaluate the MMLU subset (default: evaluate)")
    p.add_argument("--mmlu-batch-size", type=int, default=8)
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--log", default=LOG_FILE)
    p.add_argument("--skip-baseline", action="store_true", help="SKIP the uncompressed-model measurement")
    p.add_argument("--resume", action="store_true", help="reuse completed stages/configurations in --output and an existing score file")
    p.add_argument("--preflight", action="store_true", help="check config/budget/MMLU without loading the model, then exit")
    p.add_argument("--smoke", action="store_true", help="short health check (loads the model, ~5 min) -> <output>_smoke.json")
    p.add_argument("--dry-run", action="store_true", help="mini Qwen2 + fake quantisation/perplexity/MMLU on CPU")
    args = p.parse_args(argv)
    args.mmlu = not args.no_mmlu
    if not 0.0 < args.fraction < 1.0:
        raise SystemExit("--fraction must be in (0, 1)")
    if args.n_rounds < 1:
        raise SystemExit("--n-rounds must be >= 1")
    if args.dry_run:  # never overwrite the real results/ files
        if args.output == OUTPUT_FILE:
            args.output = os.path.join(DRYRUN_DIR, "model2_gun9_dry.json")
        if args.log == LOG_FILE:
            args.log = os.path.join(DRYRUN_DIR, "log_gun9_model2_dry.txt")
        if args.scores_dir == SCORES_DIR:
            args.scores_dir = os.path.join(DRYRUN_DIR, "gun9_scores")
    if args.smoke:
        root, ext = os.path.splitext(args.output)
        args.output = f"{root}_smoke{ext}"
    return args


def model_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def scores_path_for(args: argparse.Namespace) -> str:
    return args.scores or os.path.join(args.scores_dir, f"importance_scores_{model_slug(args.model)}.json")


# --------------------------------------------------------------------------- #
# Model info / plan (read from the config; no Mistral-specific dimensions)
# --------------------------------------------------------------------------- #
def config_info(cfg) -> Dict[str, Any]:
    n_heads = cfg.num_attention_heads
    return {"model_type": getattr(cfg, "model_type", None), "n_layers": cfg.num_hidden_layers, "n_heads": n_heads,
            "n_kv_heads": getattr(cfg, "num_key_value_heads", None) or n_heads,
            "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads, "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size, "vocab_size": cfg.vocab_size,
            "n_heads_total": cfg.num_hidden_layers * n_heads, "pad_token_id": getattr(cfg, "pad_token_id", None)}


def model_info(model: nn.Module) -> Dict[str, Any]:
    attn = compressor._decoder_layers(model)[0].self_attn
    info = config_info(model.config)
    info.update({"attention_class": type(attn).__name__,
                 "attention_bias": {n: getattr(getattr(attn, n), "bias", None) is not None for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
                 "gqa_group": info["n_heads"] // info["n_kv_heads"],
                 "n_params": sum(p.numel() for p in model.parameters()),
                 "attn_implementation": getattr(model.config, "_attn_implementation", None)})
    return info


def expected_prune_count(n_heads_total: int, fraction: float) -> int:
    return compressor._tier_counts(n_heads_total, tier_fractions_for(fraction))[0]


def build_plan(scores: Dict[str, float], fraction: float) -> Dict[str, Any]:
    tf = tier_fractions_for(fraction)
    tiers = allocate_compression_tiers(scores, 3, tier_fractions=tf, mlp_min_tier="int4")
    plan = build_compression_plan(tiers)
    heads = heads_from_tiers(tiers)
    return {"tier_fractions": list(tf), "tiers": tiers, "full_plan": plan, "xai_heads": heads, "n_prune_heads": len(heads),
            "n_heads_total": len(all_heads(scores)), "int4_modules": int4_module_count(plan), "int4_module_set": int4_module_set(plan),
            "int4_layers": {"attn": sum(p.attn_quant == "int4" for p in plan.values()),
                            "mlp": sum(p.mlp_quant == "int4" for p in plan.values())}}


def resolve_baseline_token(spec: str, model: nn.Module, tokenizer) -> int:
    if spec != "auto":
        return int(spec)
    for v in (getattr(model.config, "pad_token_id", None), getattr(tokenizer, "pad_token_id", None)):
        if v is not None:
            return int(v)
    return 0


# --------------------------------------------------------------------------- #
# Loading / measurement helpers
# --------------------------------------------------------------------------- #
def make_dry(seed: int) -> DryRun:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
    from test_qwen2_mini import build_mini_qwen2

    return DryRun(seed, build_model=build_mini_qwen2)


def load(ctx: Dict[str, Any], log: Logger):
    dry: Optional[DryRun] = ctx["dry"]
    args = ctx["args"]
    if dry:
        return dry.load_model(log)
    return load_model(log, dtype=DTYPES[args.dtype], model_name=args.model)


def calibration_batches(ctx: Dict[str, Any], tokenizer) -> List[Dict[str, torch.Tensor]]:
    if ctx["dry"]:
        return ctx["dry"].calibration_batches()
    if "calib" not in ctx:  # same 16 WikiText-2 passages as for Mistral, tokenised with this model's tokenizer
        ctx["calib"] = load_calibration_batches(tokenizer)
    return ctx["calib"]


def attribution_settings(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if ctx["dry"]:
        return dict(DRY_ATTRIBUTION)
    return {**ATTRIBUTION, "n_steps": ctx["args"].n_steps}


def attribute(model: nn.Module, tokenizer, ctx: Dict[str, Any]):
    batches = calibration_batches(ctx, tokenizer)
    baseline_id = resolve_baseline_token(ctx["args"].baseline_token, model, tokenizer)
    t0 = time.time()
    scores = calculate_importance_scores(model, batches, verbose=False, baseline_token_id=baseline_id, **attribution_settings(ctx))
    return scores, time.time() - t0, baseline_id, batches


def measure_ppl(model: nn.Module, tokenizer, ctx: Dict[str, Any]) -> float:
    if ctx["dry"]:
        return ctx["dry"].perplexity(model)
    return ctx["compute_perplexity"](model, tokenizer, ctx["text"], next(model.parameters()).device)


def measure_mmlu(model: nn.Module, tokenizer, ctx: Dict[str, Any], log: Logger) -> Dict[str, Any]:
    import eval_mmlu  # lazy: never imported with --no-mmlu

    args = ctx["args"]
    mlog = lambda m: log(m, tag="MMLU")  # noqa: E731
    if ctx["dry"]:
        return eval_mmlu.evaluate_mmlu_dry(model, args.seed, log=mlog, record_questions=True)
    return eval_mmlu.evaluate_mmlu(model, tokenizer, subset=ctx["mmlu_subset"], prompt_style="plain",
                                   batch_size=args.mmlu_batch_size, log=mlog, record_questions=True)


def quantize(model: nn.Module, plan: Dict[int, LayerPlan], ctx: Dict[str, Any], rep: Dict[str, Any],
             store: ResultStore, log: Logger) -> Dict[str, int]:
    dtype = DTYPES[ctx["args"].dtype]
    if ctx["dry"] or dtype == torch.float16:  # same path as the Mistral runs (primary + version-tolerant fallback)
        return quantize_with_fallback(model, plan, log, _StoreView(store, rep), "cfg")
    rep["quantization_path"] = f"primary (compressor._quantize_linear, compute_dtype={dtype})"
    return apply_quantization(model, plan, compute_dtype=dtype, verbose=True)


# --------------------------------------------------------------------------- #
# Stage: attribution
# --------------------------------------------------------------------------- #
def run_attribution(ctx: Dict[str, Any], store: ResultStore, log: Logger) -> Dict[str, float]:
    args = ctx["args"]
    path = scores_path_for(args)
    st = store.data["attribution"]
    st["scores_file"] = path
    if os.path.exists(path) and (args.scores or args.resume):
        scores = load_scores(path)
        st.update({"status": "completed", "source": "loaded", "n_scores": len(scores)})
        log(f"Attribution skipped: scores loaded from file ({len(scores)} blocks): {path}", tag="G9")
        return scores
    if args.scores:
        raise SystemExit(f"--scores file not found: {args.scores}")
    st.update({"status": "running", "source": "computed"})
    store.save("attribution started")
    set_seed(args.seed)
    ensure_clean_gpu(log, "attribution")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    box: Dict[str, Any] = {}
    try:
        box["tokenizer"], box["model"], load_s = load(ctx, log)
        store.data.setdefault("model_info", model_info(box["model"]))
        scores, attr_s, baseline_id, batches = attribute(box["model"], box["tokenizer"], ctx)
        n_tokens = sum(int(b["attention_mask"].sum()) for b in batches[: attribution_settings(ctx).get("max_batches") or len(batches)])
        n_blocks = 2 * len(compressor._decoder_layers(box["model"]))
    finally:
        peak = cuda_gb("peak")
        box.clear()
        release_gpu()
    bad = [k for k, v in scores.items() if not math.isfinite(v)]
    if bad:
        raise RuntimeError(f"attribution: NaN/Inf in {len(bad)} scores (first: {bad[:3]}); retry with --dtype bf16")
    settings = attribution_settings(ctx)
    payload_meta = {  # output schema of run_xai_on_mistral.py (compatible with compare_scores.py / compressor.load_scores)
        "model": "mini (dry-run)" if ctx["dry"] else args.model, "precision": args.dtype,
        "dataset": "wikitext-2-raw-v1 (test split)",
        "calibration": None if ctx["dry"] else {**CALIB, "n_passages": CALIB["n_passages"], "n_batches_used": len(batches),
                                                "passage_offset": 0, "n_valid_tokens": n_tokens},
        "attribution": {"method": "captum.LayerIntegratedGradients (attribute_to_layer_input)", "n_steps": settings["n_steps"],
                        "internal_batch_size": settings["internal_batch_size"], "baseline_token_id": baseline_id},
        "timing": {"model_load_seconds": load_s, "attribution_seconds": attr_s, "seconds_per_block": attr_s / n_blocks},
        "peak_vram_gb": peak, "seed": args.seed,
    }
    save_scores(path, scores, payload_meta)
    st.update({"status": "completed", "n_scores": len(scores), "attribution_seconds": attr_s, "seconds_per_block": attr_s / n_blocks,
               "peak_vram_gb": peak, "baseline_token_id": baseline_id, "n_valid_tokens": n_tokens, "settings": settings})
    log(f"ATTRIBUTION DONE: {len(scores)} blocks, {attr_s:.1f} s ({attr_s / n_blocks:.2f} s/block), baseline token {baseline_id}, "
        f"peak VRAM {peak} GB -> {path}", tag="G9")
    store.save("attribution")
    return scores


# --------------------------------------------------------------------------- #
# Single configuration run
# --------------------------------------------------------------------------- #
def apply_and_measure(name: str, seed: int, ctx: Dict[str, Any], rep: Dict[str, Any], store: ResultStore, log: Logger,
                      box: Dict[str, Any]) -> None:
    args, dry = ctx["args"], ctx["dry"]
    model, tokenizer = box["model"], box["tokenizer"]
    scores0: Dict[str, float] = ctx["scores"]
    pb: Dict[str, Any] = ctx["plan"]
    full_plan: Dict[int, LayerPlan] = pb["full_plan"]
    xai_heads: HeadList = pb["xai_heads"]
    n_budget: int = pb["n_prune_heads"]
    n_layers = len(compressor._decoder_layers(model))
    fkey = fraction_key(args.fraction)
    rep["model_bytes_before"] = module_bytes(model)
    rep["quantized"] = name in QUANTIZED_CONFIGS

    # --- head selection (same budget for every configuration) ---
    if name in ("prune_only", "both"):
        heads = xai_heads
    elif name == "quant_only":
        heads = []
    elif name == "prune_random":
        heads = select_heads_random(all_heads(scores0), n_budget, seed)
    elif name == "prune_reverse":
        heads = select_heads_by_score(scores0, n_budget, lowest=False)
    elif name == "prune_taylor":
        t0 = time.time()
        batches = calibration_batches(ctx, tokenizer)
        crit = head_taylor_scores(model, batches)
        heads = select_heads_by_score(crit, n_budget, lowest=True)
        rep["criterion"] = {"seconds": time.time() - t0, "n_batches": len(batches), "score_range": [min(crit.values()), max(crit.values())]}
    elif name == "xai_iter_fixedq":
        shares = split_budget(n_budget, args.n_rounds)
        pruned: List[str] = []
        cur = scores0
        rounds: List[Dict[str, Any]] = []
        for k, share in enumerate(shares, 1):
            if share == 0:  # budget smaller than the number of rounds (mini/dry-run only)
                log(f"{name}: round {k}/{len(shares)} has share 0, skipped", tag="G9")
                continue
            t_round = time.time()
            chosen = select_round_heads(cur, share, pruned)
            apply_structural_pruning(model, prune_plan_from_heads(chosen, n_layers), verbose=False)
            pruned += head_names(chosen)
            cur, attr_s, baseline_id, batches = attribute(model, tokenizer, ctx)
            path = os.path.join(args.scores_dir, f"f{fkey}_{name}_round{k}.json")
            save_scores(path, cur, {"fraction": fkey, "config": name, "model": "mini (dry-run)" if dry else args.model,
                                    "attribution": {**attribution_settings(ctx), "baseline_token_id": baseline_id},
                                    "pruned_heads": list(pruned), "n_pruned_heads": len(pruned), "round": k, "share": share,
                                    "round_pruned_heads": head_names(chosen), "seconds": attr_s})
            rounds.append({"round": k, "share": share, "pruned_heads": [list(h) for h in chosen], "n_pruned_cumulative": len(pruned),
                           "attribution_seconds": attr_s, "n_batches": len(batches),
                           "pruned_heads_zero_score": sum(1 for h in pruned if cur[h] == 0.0), "scores_file": path,
                           "seconds": time.time() - t_round})
            rep["rounds"] = rounds
            log(f"{name}: round {k}/{len(shares)} share {share} -> {len(pruned)} heads in total; rescoring {attr_s:.1f} s "
                f"({rounds[-1]['pruned_heads_zero_score']}/{len(pruned)} pruned heads score 0) -> {path}", tag="G9")
            store.save(f"{name} round {k}")
        heads = sorted((compressor.parse_block_key(h).layer, compressor.parse_block_key(h).index) for h in pruned)
        rep.update({"int4_plan_source": "both", "attribution_calls": len(rounds), "round_shares": shares,
                    "attribution_seconds_total": sum(r["attribution_seconds"] for r in rounds)})
    else:
        raise ValueError(f"unknown configuration: {name}")
    if name not in ("prune_only", "both", "quant_only"):
        rep["overlap_with_xai_heads"] = len(set(heads) & set(xai_heads))

    quant_plan = None
    if name == "quant_only":
        quant_plan = quant_only_plan(full_plan, log)
    elif name in QUANTIZED_CONFIGS:
        quant_plan = full_plan
    plan = merge_plan(prune_plan_from_heads(heads, n_layers), quant_plan)
    rep["pruned_heads"] = [list(h) for h in heads]
    rep["n_pruned_heads"] = len(heads)
    rep["plan_summary"] = plan_summary(plan)
    rep["budget_nominal"] = estimate_compression_budget(plan, **model_budget_dims(model))
    rep["int4_modules_planned"] = int4_module_count(plan)
    rep["int4_module_set_equals_single"] = int4_module_set(plan) == pb["int4_module_set"]
    if name in FIXED_INT4_CONFIGS:
        assert rep["int4_module_set_equals_single"] and len(heads) == n_budget, "budget guarantee violated"
    if name != "quant_only":
        assert len(heads) == n_budget, "head budget violated"
    log(f"{name}: {len(heads)} heads to prune (budget {n_budget}), planned INT4 modules {rep['int4_modules_planned']}; nominal size "
        f"ratio {rep['budget_nominal']['size_ratio']:.3f}, pruned parameter ratio {rep['budget_nominal']['pruned_ratio']:.3f}", tag="G9")

    # --- pruning (already applied in the iterative case; re-zeroing is idempotent) -> quantisation ---
    t0 = time.time()
    rep["pruning"] = apply_structural_pruning(model, plan, verbose=False)
    if quant_plan is not None:
        rep["quantization"] = quantize(model, plan, ctx, rep, store, log)
    else:
        rep["quantization"] = {"int4_modules": 0, "int8_modules": 0}
        rep["quantization_path"] = None
    rep["apply_seconds"] = time.time() - t0
    rep["int4_modules"] = rep["quantization"]["int4_modules"]
    rep["int4_modules_equals_single"] = rep["int4_modules"] == pb["int4_modules"]
    rep["model_bytes_after"] = module_bytes(model)
    rep["measured_model_bytes_ratio"] = rep["model_bytes_after"]["bytes"] / rep["model_bytes_before"]["bytes"]
    log(f"{name}: applied ({rep['apply_seconds']:.1f} s) pruning={rep['pruning']} quantization={rep['quantization']} "
        f"size {rep['model_bytes_before']['gb']:.2f} -> {rep['model_bytes_after']['gb']:.2f} GB", tag="G9")
    store.save(f"{name} applied")

    # --- perplexity + MMLU ---
    t0 = time.time()
    ppl = measure_ppl(model, tokenizer, ctx)
    rep.update({"perplexity": ppl, "perplexity_seconds": time.time() - t0, "perplexity_finite": math.isfinite(ppl)})
    log(f"{name}: perplexity = {ppl:.4f} ({rep['perplexity_seconds']:.1f} s)", tag="G9")
    store.save(f"{name} perplexity")
    if args.mmlu:
        rep["mmlu"] = measure_mmlu(model, tokenizer, ctx, log)
        rep["mmlu_subset_acc"] = rep["mmlu"]["mmlu_subset_acc"]


def run_one(name: str, seed: int, ctx: Dict[str, Any], rep: Dict[str, Any], store: ResultStore, log: Logger) -> None:
    """Clean load -> apply/measure -> FULL release of GPU memory (one model in memory at a time)."""
    rep.update({"status": "running", "seed": seed, "fraction": ctx["args"].fraction,
                "started_at": datetime.now().isoformat(timespec="seconds")})
    store.save(f"{name} started")
    t_cfg = time.time()
    set_seed(seed)
    rep["cuda_allocated_before_load_gb"] = ensure_clean_gpu(log, name)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    box: Dict[str, Any] = {}
    try:
        box["tokenizer"], box["model"], rep["model_load_seconds"] = load(ctx, log)
        apply_and_measure(name, seed, ctx, rep, store, log, box)
        rep["status"] = "completed"
    finally:
        rep["peak_vram_gb"] = cuda_gb("peak")
        box.clear()
        rep["cuda_allocated_after_free_gb"] = release_gpu()
        rep["seconds"] = time.time() - t_cfg
        log(f"{name}: model released from memory; peak VRAM={rep['peak_vram_gb']} GB, {rep['seconds']:.0f} s", tag="G9")
        store.save(f"{name} finished")


# --------------------------------------------------------------------------- #
# Summary / time estimate
# --------------------------------------------------------------------------- #
def comparison(configs: Dict[str, Any], base_ppl: Optional[float]) -> Dict[str, Any]:
    def ppl(n: str) -> Optional[float]:
        c = configs.get(n)
        return c.get("perplexity_mean") if c else None

    out: Dict[str, Any] = {"baseline": base_ppl}
    p, q, b = ppl("prune_only"), ppl("quant_only"), ppl("both")
    if base_ppl is not None and None not in (p, q, b):
        out.update({"prune_delta": p - base_ppl, "quant_delta": q - base_ppl, "both_delta": b - base_ppl,
                    "interaction": (b - base_ppl) - (p - base_ppl) - (q - base_ppl)})
    if p is not None:  # unquantised selector controls (compared with prune_only)
        for n in ("prune_random", "prune_reverse"):
            if ppl(n) is not None:
                out[f"{n}_minus_prune_only"] = ppl(n) - p
    if b is not None:  # same INT4 plan as both (compared with both)
        for n in ("prune_taylor", "xai_iter_fixedq"):
            if ppl(n) is not None:
                out[f"{n}_minus_both"] = ppl(n) - b
    return out


def format_summary_table(configs: Dict[str, Any], base: Dict[str, Any]) -> str:
    def f(v: Any, spec: str) -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    lines = [f"{'config':<18}{'ppl':>10}{'±std':>9}{'Δbase':>10}{'mmlu':>7}{'head':>6}{'int4':>6}{'GB':>8}{'VRAM':>7}{'s':>7}  status"]
    if base.get("perplexity") is not None:
        lines.append(f"{'baseline':<18}{base['perplexity']:>10.4f}{'':>9}{0.0:>10.4f}{f(base.get('mmlu_subset_acc'), '.3f'):>7}{0:>6}{0:>6}"
                     f"{f(base.get('model_bytes', {}).get('gb'), '.2f'):>8}{f(base.get('peak_vram_gb'), '.1f'):>7}{f(base.get('seconds'), '.0f'):>7}  reference")
    for name, c in configs.items():
        lines.append(f"{name:<18}{f(c.get('perplexity_mean'), '.4f'):>10}{f(c.get('perplexity_std'), '.4f'):>9}"
                     f"{f(c.get('delta_vs_fp16'), '+.4f'):>10}{f(c.get('mmlu_subset_acc_mean'), '.3f'):>7}{str(c.get('n_pruned_heads', '-')):>6}"
                     f"{str(c.get('int4_modules', '-')):>6}{f(c.get('model_bytes_after_gb'), '.2f'):>8}{f(c.get('peak_vram_gb'), '.1f'):>7}"
                     f"{f(c.get('seconds'), '.0f'):>7}  {c.get('status')}")
    return "\n".join(lines)


def estimate_minutes(args: argparse.Namespace, config_names: List[str], need_attribution: bool) -> Dict[str, Any]:
    """Rough L40S runtime estimate in minutes (see the module docstring; scaled from the Mistral runs)."""
    base, mmlu_min, attr = 2.0, (3.0 if args.mmlu else 0.0), 6.0 * args.n_steps / 8.0
    per: Dict[str, float] = {}
    for n in config_names:
        one = base + mmlu_min + (0.5 if n in QUANTIZED_CONFIGS else 0.0)
        if n == "xai_iter_fixedq":
            one += args.n_rounds * attr
        per[n] = one * (args.n_repeats if n in STOCHASTIC_CONFIGS else 1)
    fixed = (0.0 if args.skip_baseline else 1.5 + mmlu_min) + (attr + 0.5 if need_attribution else 0.0)
    return {"per_config_minutes": per, "baseline_plus_attribution_minutes": fixed, "total_minutes": fixed + sum(per.values()),
            "assumptions": "load+apply+ppl 2-2.5 min, MMLU 3 min, attribution 6 min/call (n_steps 8); excludes first download (~15 GB)"}


# --------------------------------------------------------------------------- #
# Stage: smoke (short health check)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _smoke_logits(model: nn.Module, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).logits.float().cpu()


def run_smoke(ctx: Dict[str, Any], store: ResultStore, log: Logger) -> int:
    args, dry = ctx["args"], ctx["dry"]
    res: Dict[str, Any] = store.data.setdefault("smoke", {})
    res.update({"status": "running", "started_at": datetime.now().isoformat(timespec="seconds")})
    t_all = time.time()
    box: Dict[str, Any] = {}
    problems: List[str] = []
    try:
        box["tokenizer"], box["model"], res["model_load_seconds"] = load(ctx, log)
        model, tok = box["model"], box["tokenizer"]
        store.data["model_info"] = info = model_info(model)
        log(f"Model: {info}", tag="SMK")
        device = next(model.parameters()).device
        if dry:
            ids = torch.randint(1, model.config.vocab_size, (1, 24), generator=torch.Generator().manual_seed(args.seed))
            enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        else:
            enc = {k: v.to(device) for k, v in tok(SMOKE_TEXT, return_tensors="pt").items() if k in ("input_ids", "attention_mask")}
        # (1) finite forward pass
        ref = _smoke_logits(model, enc)
        res["forward"] = {"n_tokens": int(enc["input_ids"].shape[1]), "finite": bool(torch.isfinite(ref).all()),
                          "abs_max": float(ref.abs().max())}
        if not res["forward"]["finite"]:
            problems.append(f"NaN/Inf in the {args.dtype} forward pass; use --dtype bf16")
        # (2) short perplexity
        t0 = time.time()
        res["short_perplexity"] = {"value": measure_ppl(model, tok, {**ctx, "text": ctx.get("text", "")[:40000]}),
                                   "seconds": time.time() - t0, "note": "first 40k characters of WikiText-2 test (not the full measurement)"}
        if not math.isfinite(res["short_perplexity"]["value"]):
            problems.append("short perplexity is not finite")
        log(f"(1) forward finite={res['forward']['finite']} |logit|max={res['forward']['abs_max']:.1f}; "
            f"(2) short perplexity={res['short_perplexity']['value']:.4f}", tag="SMK")
        # (3) mini attribution: 1 batch, n_steps 2
        batches = calibration_batches(ctx, tok)[:1]
        baseline_id = resolve_baseline_token(args.baseline_token, model, tok)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        sc = calculate_importance_scores(model, batches, n_steps=2, internal_batch_size=None if dry else 2, verbose=False,
                                         baseline_token_id=baseline_id)
        attr_s = time.time() - t0
        vals = list(sc.values())
        full_factor = (args.n_steps / 2.0) * (1 if dry else ATTRIBUTION["max_batches"])
        res["attribution"] = {"n_scores": len(sc), "finite": all(math.isfinite(v) for v in vals), "n_zero": sum(v == 0.0 for v in vals),
                              "seconds": attr_s, "estimated_full_seconds": attr_s * full_factor, "baseline_token_id": baseline_id,
                              "peak_vram_gb": cuda_gb("peak"), "head_score_range": [min(v for k, v in sc.items() if ".attn." in k),
                                                                                    max(v for k, v in sc.items() if ".attn." in k)]}
        if not res["attribution"]["finite"]:
            problems.append("NaN/Inf in attribution scores; use --dtype bf16")
        if len(sc) != info["n_layers"] * (info["n_heads"] + 1):
            problems.append(f"unexpected number of scores: {len(sc)}")
        log(f"(3) attribution 1 batch × n_steps 2: {attr_s:.1f} s -> full-run estimate ~{attr_s * full_factor / 60:.1f} min; "
            f"finite={res['attribution']['finite']}, zero scores={res['attribution']['n_zero']}, peak VRAM={res['attribution']['peak_vram_gb']}", tag="SMK")
        # (4) masked vs physical (sequentially on the same model; cutting the masked weights must not change the output)
        heads = [(0, 1), (info["n_layers"] // 2, info["n_heads"] - 1), (info["n_layers"] - 1, 0)]
        apply_structural_pruning(model, prune_plan_from_heads(heads, info["n_layers"]), verbose=False)
        masked = _smoke_logits(model, enc)
        # The wrapper uses the SAME attention kernel as the masked model ("auto": sdpa if the model was loaded with sdpa). With an
        # eager wrapper the difference was the native sdpa-vs-eager rounding gap, also present without the wrapper (Qwen2.5:
        # fp16 9.4, bf16 17.9; not a logic error; see tools/diagnostics/diagnose_qwen_kernel.py).
        phys = physically_prune_heads(model, heads, verbose=False, attn_kernel="auto")
        physical = _smoke_logits(model, enc)
        diff = (masked - physical).abs()
        res["masked_vs_physical"] = {"heads": [list(h) for h in heads], "max_abs_logit_diff": float(diff.max()),
                                     "mean_abs_logit_diff": float(diff.mean()), "top1_agreement": float((masked.argmax(-1) == physical.argmax(-1)).float().mean()),
                                     "params_removed": phys["params_removed"], "params_removed_expected": phys["params_removed_expected"],
                                     "changed_vs_unpruned": float((ref - masked).abs().max()),
                                     "attn_kernel": next((m.attn_kernel for m in model.modules() if hasattr(m, "attn_kernel")), None)}
        if phys["params_removed"] != phys["params_removed_expected"]:
            problems.append("parameters removed by physical pruning do not match the expected count")
        if not torch.isfinite(physical).all():
            problems.append("NaN/Inf after physical pruning")
        log(f"(4) masked vs physical: |Δlogit|max={res['masked_vs_physical']['max_abs_logit_diff']:.3e}, "
            f"top-1 agreement={res['masked_vs_physical']['top1_agreement']:.3f}, parameters removed={phys['params_removed']:,}", tag="SMK")
        # (5) INT4 on layer 0 (including modules with bias) -> finite logits
        res["int4_layer0"] = {}
        counts = quantize(model, {0: LayerPlan(0, attn_quant="int4", mlp_quant="int4")}, ctx, res["int4_layer0"], store, log)
        q_logits = _smoke_logits(model, enc)
        res["int4_layer0"].update({"counts": counts, "finite": bool(torch.isfinite(q_logits).all()),
                                   "max_abs_logit_diff_vs_physical": float((q_logits - physical).abs().max()),
                                   "top1_agreement": float((q_logits.argmax(-1) == physical.argmax(-1)).float().mean())})
        if not res["int4_layer0"]["finite"] or counts.get("int4_modules") != 7:
            problems.append("problem after INT4 on layer 0 (not finite or not 7 modules)")
        log(f"(5) layer 0 INT4: {counts}, finite={res['int4_layer0']['finite']}, top-1 agreement={res['int4_layer0']['top1_agreement']:.3f}", tag="SMK")
        res["status"] = "completed" if not problems else "completed_with_problems"
    except BaseException as e:
        res.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
        log(f"SMOKE ERROR: {e!r}", tag="ERR")
        log(res["traceback"], tag="ERR")
        problems.append(repr(e))
    finally:
        res["peak_vram_gb"] = cuda_gb("peak")
        box.clear()
        release_gpu()
        res["problems"] = problems
        res["seconds"] = time.time() - t_all
        store.data["run"]["status"] = "smoke_" + res["status"]
        store.save("smoke")
    log(("SMOKE OK" if not problems else "SMOKE FAILED: " + "; ".join(problems)) + f" ({res['seconds']:.0f} s)", tag="SMK")
    return 0 if not problems else 1


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log = Logger(args.log, to_file=should_log_to_file(args, LOG_FILE))  # dry-run/preflight never write to the default log
    store = ResultStore(args.output, log)
    t_start = time.time()
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in config_names if c not in CONFIG_DESCRIPTIONS]
    if unknown:
        raise SystemExit(f"unknown configuration(s): {unknown}; valid: {ALL_CONFIGS}")
    dry: Optional[DryRun] = make_dry(args.seed) if args.dry_run else None
    spath = scores_path_for(args)
    need_attr = not (os.path.exists(spath) and (args.scores or args.resume))
    estimate = estimate_minutes(args, config_names, need_attr)
    log(f"===== Second-model run starting: model={'mini Qwen2 (dry-run)' if dry else args.model} dtype={args.dtype} fraction={args.fraction} "
        f"configs={config_names} n_rounds={args.n_rounds} n_steps={args.n_steps} seed={args.seed} n_repeats={args.n_repeats} "
        f"mmlu={args.mmlu} dry_run={args.dry_run} smoke={args.smoke} =====", tag="G9")

    previous = None
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            previous = json.load(f)
        log(f"--resume: read {args.output}; completed stages/configurations will be skipped", tag="G9")

    store.data = {
        "run": {"started_at": datetime.now().isoformat(timespec="seconds"), "status": "running", "args": vars(args),
                "model": "mini Qwen2 (dry-run)" if dry else args.model, "dtype": args.dtype, "env": env_info(),
                "config_order": config_names, "time_estimate": estimate},
        "reference": {"fraction": args.fraction, "tier_fractions": list(tier_fractions_for(args.fraction)), "mlp_min_tier": "int4",
                      "attn_rule": "median", "scores_file": spath, "scores_dir": args.scores_dir,
                      "quantized_configs": sorted(QUANTIZED_CONFIGS),
                      "config_semantics": "quant_only/prune_only/both/prune_random/prune_reverse as in run_ablation_tests.py (random/reverse unquantised); "
                                          "prune_taylor/xai_iter_fixedq as in run_iterative_pruning.py (INT4 plan identical to both)",
                      "size_note": "model_bytes = parameter + buffer bytes; excludes bnb quant_state (same measurement as the Mistral runs)"},
        "attribution": (previous or {}).get("attribution", {"status": "pending"}),
        "baseline": (previous or {}).get("baseline", {"status": "skipped" if args.skip_baseline else "pending", "precision": args.dtype}),
        "configs": {n: {"description": CONFIG_DESCRIPTIONS[n], "status": "pending", "quantized": n in QUANTIZED_CONFIGS, "repeats": []}
                    for n in config_names},
        "comparison": {},
    }
    if previous and previous.get("model_info"):
        store.data["model_info"] = previous["model_info"]
    env = store.data["run"]["env"]
    log(f"Environment: torch {env['torch']}, transformers {env['transformers']}, bitsandbytes {env.get('bitsandbytes')}, "
        f"CUDA={env['cuda_available']} ({env.get('gpu')})", tag="G9")
    log(f"Time estimate (min): {json.dumps(estimate['per_config_minutes'])} + baseline/attribution "
        f"{estimate['baseline_plus_attribution_minutes']:.1f} = total ~{estimate['total_minutes']:.0f} min", tag="G9")

    if args.preflight:
        problems: List[str] = []
        if not dry and not torch.cuda.is_available():
            problems.append("CUDA not available")
        try:
            if dry:
                _, m, _ = dry.load_model(log)
                info = model_info(m)
            else:
                from transformers import AutoConfig

                info = config_info(AutoConfig.from_pretrained(args.model))
            n_prune = expected_prune_count(info["n_heads_total"], args.fraction)
            log(f"Model config: {info}", tag="G9")
            log(f"Budget: {info['n_layers']} × {info['n_heads']} = {info['n_heads_total']} heads; fraction {args.fraction} -> {n_prune} heads pruned "
                f"(tier_fractions {tier_fractions_for(args.fraction)}); round shares {split_budget(n_prune, args.n_rounds)}; "
                f"GQA group {info['n_heads'] // info['n_kv_heads']}", tag="G9")
        except Exception as e:
            problems.append(f"could not read model config ({args.model}): {e!r}")
        if os.path.exists(spath):
            try:
                pb = build_plan(load_scores(spath), args.fraction)
                log(f"Existing score file: {spath} -> {pb['n_prune_heads']}/{pb['n_heads_total']} heads, INT4 modules {pb['int4_modules']} "
                    f"(attn/mlp layers {pb['int4_layers']})" + ("" if (args.scores or args.resume) else " — WARNING: no --resume/--scores, scores will be recomputed"), tag="G9")
            except Exception as e:
                problems.append(f"could not read score file: {e!r}")
        else:
            log(f"No score file, attribution will run: {spath}", tag="G9")
        if not dry and args.mmlu:
            try:
                import eval_mmlu

                subset = eval_mmlu.load_or_build_subset(log=lambda m: log(m, tag="MMLU"))
                log(f"MMLU subset ready: {len(subset['records'])} questions", tag="MMLU")
            except Exception as e:
                problems.append(f"could not load MMLU subset: {e!r}")
        if not dry:
            try:
                import datasets  # noqa: F401
            except Exception as e:
                problems.append(f"could not import datasets: {e!r}")
        if problems:
            log("Preflight FAILED (model not loaded): " + "; ".join(problems), tag="ERR")
            return 1
        log("Preflight completed (model not loaded). Nothing written to the result file.", tag="G9")
        return 0
    if not torch.cuda.is_available() and not dry:
        log("WARNING: CUDA not available; a 7B model + bitsandbytes does not run on CPU. Run on a GPU (or use --dry-run).", tag="G9")
    os.makedirs(args.scores_dir, exist_ok=True)
    store.save("start")

    ctx: Dict[str, Any] = {"args": args, "dry": dry}
    exit_code = 0
    base_ppl: Optional[float] = None
    base_acc: Optional[float] = None
    try:
        if not dry:
            log("Loading the WikiText-2 test set...", tag="G9")
            ctx["text"] = load_wikitext_text()
            ctx["compute_perplexity"], max_len, stride = perplexity_tools()
            store.data["reference"]["perplexity_settings"] = {"max_length": max_len, "stride": stride,
                                                              "dataset": "wikitext-2-raw-v1 (test split)"}
            if args.mmlu and not args.smoke:
                import eval_mmlu

                ctx["mmlu_subset"] = eval_mmlu.load_or_build_subset(log=lambda m: log(m, tag="MMLU"))
                store.data["reference"]["mmlu_n_questions"] = len(ctx["mmlu_subset"]["records"])
        if args.smoke:
            return run_smoke(ctx, store, log)

        # --- uncompressed model (once, clean load) ---
        bl = store.data["baseline"]
        if args.skip_baseline:
            log("Baseline SKIPPED (--skip-baseline)", tag="G9")
        elif bl.get("status") == "completed":
            base_ppl, base_acc = bl["perplexity"], bl.get("mmlu_subset_acc")
            log(f"Baseline taken from the previous run (--resume): ppl={base_ppl} mmlu={base_acc}", tag="G9")
        else:
            bl.update({"status": "running", "precision": args.dtype})
            set_seed(args.seed)
            t0 = time.time()
            ensure_clean_gpu(log, "baseline")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            box: Dict[str, Any] = {}
            try:
                box["tokenizer"], box["model"], bl["model_load_seconds"] = load(ctx, log)
                store.data["model_info"] = model_info(box["model"])
                log(f"Model: {store.data['model_info']}", tag="G9")
                bl["model_bytes"] = module_bytes(box["model"])
                t1 = time.time()
                bl["perplexity"] = measure_ppl(box["model"], box["tokenizer"], ctx)
                bl["perplexity_seconds"] = time.time() - t1
                if args.mmlu:
                    bl["mmlu"] = measure_mmlu(box["model"], box["tokenizer"], ctx, log)
                    bl["mmlu_subset_acc"] = bl["mmlu"]["mmlu_subset_acc"]
            finally:
                bl["peak_vram_gb"] = cuda_gb("peak")
                box.clear()
                release_gpu()
            bl.update({"seconds": time.time() - t0, "status": "completed"})
            base_ppl, base_acc = bl["perplexity"], bl.get("mmlu_subset_acc")
            if not math.isfinite(base_ppl):
                raise RuntimeError(f"baseline perplexity is not finite ({base_ppl}); retry with --dtype bf16")
            log(f"BASELINE DONE ({args.dtype}): perplexity = {base_ppl:.4f}, mmlu={base_acc}, {bl['model_bytes']['gb']:.3f} GB, "
                f"peak VRAM {bl['peak_vram_gb']} GB, {bl['seconds']:.1f} s", tag="G9")
            store.save("baseline")

        # --- attribution + plan ---
        ctx["scores"] = scores = run_attribution(ctx, store, log)
        ctx["plan"] = pb = build_plan(scores, args.fraction)
        info = store.data.get("model_info") or {}
        if info and len(scores) != info["n_layers"] * (info["n_heads"] + 1):
            raise RuntimeError(f"number of scores ({len(scores)}) does not match the model dimensions ({info['n_layers']}×({info['n_heads']}+1)): {spath}")
        store.data["reference"].update({
            "budget_heads": {"n_prune_heads": pb["n_prune_heads"], "n_heads_total": pb["n_heads_total"],
                             "nominal_block_prune_ratio": pb["n_prune_heads"] / pb["n_heads_total"],
                             "definition": "in every pruning configuration the number of pruned heads = number of prune heads in the plan"},
            "int4_layers_in_plan": pb["int4_layers"], "int4_modules_in_plan": pb["int4_modules"], "tier_counts": tier_counts(pb["tiers"]),
            "round_shares": split_budget(pb["n_prune_heads"], args.n_rounds), "plan_summary_both": plan_summary(pb["full_plan"])})
        log(f"Plan: {pb['n_prune_heads']}/{pb['n_heads_total']} heads pruned, INT4 layers attn/mlp={pb['int4_layers']} "
            f"(= {pb['int4_modules']} modules), round shares {split_budget(pb['n_prune_heads'], args.n_rounds)}", tag="G9")
        store.save("plan")

        # --- configurations ---
        for ci, name in enumerate(config_names, 1):
            cfg = store.data["configs"][name]
            prev_cfg = (previous or {}).get("configs", {}).get(name)
            if prev_cfg and prev_cfg.get("status") == "completed":
                store.data["configs"][name] = {**prev_cfg, "resumed": True}
                log(f"CONFIG {ci}/{len(config_names)} {name}: --resume, taken from the previous run (ppl={prev_cfg.get('perplexity_mean')})", tag="G9")
                continue
            n_rep = args.n_repeats if name in STOCHASTIC_CONFIGS else 1
            cfg.update({"status": "running", "n_repeats": n_rep, "stochastic": name in STOCHASTIC_CONFIGS})
            log(f"===== CONFIG {ci}/{len(config_names)}: {name} ({n_rep} repeats) — {cfg['description']} =====", tag="G9")
            try:
                for r in range(n_rep):
                    rep: Dict[str, Any] = {}
                    cfg["repeats"].append(rep)
                    run_one(name, args.seed + r, ctx, rep, store, log)
                summarize_config(cfg, base_ppl, base_acc)
                cfg["status"] = "completed"
                log(f"CONFIG {ci}/{len(config_names)} DONE: {name} ppl={cfg['perplexity_mean']:.4f} ± {cfg['perplexity_std']:.4f} "
                    f"(Δbase={cfg.get('delta_vs_fp16')}) mmlu={cfg.get('mmlu_subset_acc_mean')} {cfg['seconds']:.0f} s", tag="G9")
            except Exception as e:  # a failing configuration must not abort the others
                summarize_config(cfg, base_ppl, base_acc)
                cfg.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
                log(f"CONFIG {name} ERROR: {e!r} — continuing with the next configuration", tag="ERR")
                log(cfg["traceback"], tag="ERR")
                exit_code = 1
            store.data["comparison"] = comparison(store.data["configs"], base_ppl)
            store.save(f"{name} summary")
        store.data["run"]["status"] = "completed" if exit_code == 0 else "completed_with_failures"
    except BaseException as e:  # including KeyboardInterrupt / SystemExit: never lose partial results
        store.data["run"].update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
        log(f"ERROR: {e!r} — writing partial results to disk", tag="ERR")
        log(store.data["run"]["traceback"], tag="ERR")
        exit_code = 1
    finally:
        if not args.smoke:
            store.data["comparison"] = comparison(store.data["configs"], base_ppl)
            store.data["run"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            store.data["run"]["total_seconds"] = time.time() - t_start
            store.save("final")
            log("SUMMARY TABLE:\n" + format_summary_table(store.data["configs"], store.data["baseline"]), tag="G9")
            log(f"Comparison: {json.dumps(store.data['comparison'], ensure_ascii=False)}", tag="G9")
            log(f"===== Finished: status={store.data['run']['status']}, total {store.data['run']['total_seconds']:.1f} s =====", tag="G9")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
