"""
Decomposition and control experiments (GPU required; CPU with --dry-run).

Decomposes the perplexity loss of the end-to-end prototype (run_e2e_pipeline.py)
into its pruning and quantization components, and tests whether the xAI (LIG) head
selection carries information by comparing it with control selections at the same
budget (the 205 prune heads of the tier plan). For each configuration the model is
loaded cleanly, the plan is applied, WikiText-2 perplexity is measured and the model
is released.

Configurations (select a subset with --configs):
  prune_only      205 heads pruned by xAI selection, no quantization
  quant_only      no pruning, INT4 plan applied as is (129 modules)
  both            exact reproduction of the end-to-end prototype (check: 6.3171 ± 0.01)
  prune_magnitude 205 heads with the lowest o_proj column-block L2 norm
  prune_random    205 seeded random heads (--n-repeats repeats, mean ± std)
  prune_reverse   205 heads with the HIGHEST xAI score (reverse control)
  both_biascorr   both + pruning compensation: the mean o_proj input slice of each
                  pruned head over the calibration passages is multiplied by W_o and
                  added to o_proj.bias (created if absent); quantization AFTER compensation

The JSON also contains "median_rule": how many heads whose own tier is fp16 were
downgraded to int4 by the per-layer median rule of the plan (and vice versa); no GPU needed.

Usage:
    python experiments/run_ablation_tests.py --preflight
    python experiments/run_ablation_tests.py --n-repeats 3
    python experiments/run_ablation_tests.py --configs both,prune_only --skip-baseline
    python experiments/run_ablation_tests.py --dry-run                # mini model on CPU, no GPU
    python experiments/run_ablation_tests.py --resume                 # skip completed configurations
    python experiments/run_ablation_tests.py --mmlu                   # also measure the MMLU subset per configuration

Outputs: results/ablation_gun6.json (dry-run: dryrun_out/ablation_gun6_dry.json), log_gun6.txt.
"""

# HF_HOME must be set BEFORE transformers/datasets are imported (see run_e2e_pipeline.py)
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import json
import math
import random
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
    apply_structural_pruning,
    build_compression_plan,
    estimate_compression_budget,
    head_magnitude_scores,
    load_scores,
    parse_block_key,
)
# Shared components of the end-to-end prototype, imported unchanged
from run_e2e_pipeline import (
    BASELINE_FILE,
    MODEL_NAME,
    SCORES_FILE,
    TIERS_FILE,
    Logger,
    should_log_to_file,
    ResultStore,
    cuda_gb,
    env_info,
    free_model,
    load_model,
    load_tiers_json,
    load_wikitext_text,
    module_bytes,
    perplexity_tools,
    plan_summary,
    quantize_with_fallback,
    tier_counts,
)

OUTPUT_FILE = os.path.join("results", "ablation_gun6.json")
LOG_FILE = "log_gun6.txt"
GUN5_FILE = os.path.join("results", "e2e_prototype_gun5.json")
GUN5_BOTH_PPL_FALLBACK = 6.3171  # used if results/e2e_prototype_gun5.json cannot be read
REPRO_TOL = 0.01

# Calibration settings of the xAI scoring run (run_xai_on_mistral.py defaults): 16 passages, T=256, B=4
CALIB = dict(n_passages=16, min_chars=200, batch_size=4, max_length=256)

CONFIG_DESCRIPTIONS: Dict[str, str] = {
    "prune_only": "prune heads of the tier plan removed by xAI (LIG) selection; no quantization",
    "quant_only": "no pruning; attn/MLP INT4 tiers of the plan applied as is",
    "both": "exact reproduction of the end-to-end prototype: xAI pruning + INT4 (check: 6.3171 ± 0.01)",
    "prune_magnitude": "same head count; criterion: o_proj column-block L2 norm (lowest N); no quantization",
    "prune_random": "same head count; seeded random selection (n_repeats repeats); no quantization",
    "prune_reverse": "same head count; heads with the HIGHEST xAI score pruned (reverse control); no quantization",
    "both_biascorr": "both + pruning compensation (o_proj.bias += W_o[:,h] @ mean(x_h), calibration: the 16 xAI scoring passages); "
                     "quantization after compensation",
}
ALL_CONFIGS = list(CONFIG_DESCRIPTIONS)
STOCHASTIC_CONFIGS = {"prune_random"}
HeadList = List[Tuple[int, int]]


# --------------------------------------------------------------------------- #
# Arguments / seed
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decomposition + control experiments (same budget, clean load per configuration)")
    p.add_argument("--configs", default=",".join(ALL_CONFIGS),
                   help="comma-separated configuration list (default: all, in order)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=1,
                   help="number of repeats for stochastic configurations (prune_random); seed, seed+1, ...")
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--log", default=LOG_FILE)
    p.add_argument("--skip-baseline", action="store_true", help="do NOT re-measure FP16 perplexity")
    p.add_argument("--dry-run", action="store_true", help="mini Mistral + fake quantization/perplexity, CPU")
    p.add_argument("--preflight", action="store_true", help="check inputs/plan without loading the model, then exit")
    p.add_argument("--resume", action="store_true", help="skip configurations with status=completed in --output")
    p.add_argument("--tiers", default=TIERS_FILE)
    p.add_argument("--scores", default=SCORES_FILE)
    p.add_argument("--baseline", default=BASELINE_FILE)
    p.add_argument("--gun5-json", default=GUN5_FILE, help="end-to-end prototype result used for the 'both' reproduction check")
    p.add_argument("--mmlu", action="store_true",
                   help="also measure the MMLU subset after perplexity in every configuration and the FP16 rerun (eval_mmlu.py; off by default)")
    args = p.parse_args(argv)
    if args.dry_run and args.output == OUTPUT_FILE:  # mini-model results must not overwrite results/ablation_gun6.json
        args.output = os.path.join("dryrun_out", "ablation_gun6_dry.json")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Plan helpers (no GPU needed)
# --------------------------------------------------------------------------- #
def heads_from_tiers(tiers: Dict[str, str], label: str = "prune") -> HeadList:
    out: HeadList = []
    for name, t in tiers.items():
        bk = parse_block_key(name)
        if bk.kind == "attn" and t == label:
            out.append((bk.layer, bk.index))
    return sorted(out)


def all_heads(tiers_or_scores: Dict[str, Any]) -> HeadList:
    return sorted((bk.layer, bk.index) for bk in map(parse_block_key, tiers_or_scores) if bk.kind == "attn")


def select_heads_by_score(scores: Dict[str, float], n: int, *, lowest: bool) -> HeadList:
    """Returns the n lowest-scoring (lowest=True) or highest-scoring heads (ties broken by structural order)."""
    items = [(parse_block_key(k), float(v)) for k, v in scores.items() if parse_block_key(k).kind == "attn"]
    order = sorted(range(len(items)), key=lambda i: (items[i][1], i), reverse=not lowest)
    return sorted((items[i][0].layer, items[i][0].index) for i in order[:n])


def select_heads_random(candidates: HeadList, n: int, seed: int) -> HeadList:
    rng = random.Random(seed)
    return sorted(rng.sample(list(candidates), n))


def prune_plan_from_heads(heads: HeadList, n_layers: int) -> Dict[int, LayerPlan]:
    plan = {i: LayerPlan(i) for i in range(n_layers)}
    for layer, h in heads:
        plan[layer].pruned_heads.append(h)
    for p in plan.values():
        p.pruned_heads.sort()
    return plan


def quant_only_plan(full_plan: Dict[int, LayerPlan], log: Logger) -> Dict[int, LayerPlan]:
    """Keeps the quantization decisions of the tier plan and removes the pruning."""
    out: Dict[int, LayerPlan] = {}
    for i, p in full_plan.items():
        aq = p.attn_quant
        if aq == "prune":  # all heads were pruned; without pruning the block stays fp16 (does not occur in the Mistral plan)
            log(f"WARNING: layer_{i} attn_quant='prune' (all heads were pruned); left in fp16 for quant_only")
            aq = "fp16"
        out[i] = LayerPlan(i, pruned_heads=[], attn_quant=aq, mlp_pruned=False,
                           mlp_quant="fp16" if p.mlp_pruned else p.mlp_quant)
    return out


def merge_plan(prune_plan: Dict[int, LayerPlan], quant_plan: Optional[Dict[int, LayerPlan]]) -> Dict[int, LayerPlan]:
    """Pruning decisions from prune_plan, quantization decisions from quant_plan (None = fp16)."""
    out: Dict[int, LayerPlan] = {}
    for i, p in prune_plan.items():
        q = quant_plan.get(i) if quant_plan else None
        out[i] = LayerPlan(i, pruned_heads=list(p.pruned_heads),
                           attn_quant=q.attn_quant if q else "fp16",
                           mlp_pruned=p.mlp_pruned, mlp_quant=q.mlp_quant if q else "fp16")
    return out


def median_rule_counts(tiers: Dict[str, str], plan: Dict[int, LayerPlan]) -> Dict[str, Any]:
    """
    Effect of the per-layer median rule: for every unpruned head, its own tier is compared
    with the tier applied to the layer's attn block. "downgraded" = the head's tier has higher
    precision (e.g. fp16) but the block was lowered (int4); "upgraded" = the opposite.
    """
    labels = compressor._labels_for(tiers)
    counts = {"downgraded": 0, "upgraded": 0, "unchanged": 0}
    per_layer: Dict[int, Dict[str, int]] = {}
    pairs: Dict[str, int] = {}
    for name, label in tiers.items():
        bk = parse_block_key(name)
        if bk.kind != "attn" or label == "prune":
            continue
        applied = plan[bk.layer].attn_quant
        key = "unchanged"
        if labels.index(label) > labels.index(applied):
            key = "downgraded"
        elif labels.index(label) < labels.index(applied):
            key = "upgraded"
        counts[key] += 1
        per_layer.setdefault(bk.layer, {"downgraded": 0, "upgraded": 0, "unchanged": 0})[key] += 1
        pk = f"head={label}->layer={applied}"
        pairs[pk] = pairs.get(pk, 0) + 1
    n_kept = sum(counts.values())
    return {
        "definition": "unpruned heads; head tier vs the attn tier applied to the layer (build_compression_plan median rule)",
        "n_unpruned_heads": n_kept,
        **counts,
        "downgraded_ratio": counts["downgraded"] / n_kept if n_kept else None,
        "pairs": pairs,
        "per_layer": {str(i): v for i, v in sorted(per_layer.items())},
    }


# --------------------------------------------------------------------------- #
# Pruning compensation (bias / mean-activation correction)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_o_proj_input_means(model: nn.Module, batches: Sequence[Dict[str, torch.Tensor]]) -> Tuple[Dict[int, torch.Tensor], int]:
    """
    Returns the mean of the self_attn.o_proj INPUT (head context vectors, [B,T,H*d]) over valid
    tokens for every layer: {layer: [H*d] float64}. Measured in a single pass on the unpruned
    model (standard "mean replacement"; not sequential layer by layer).
    """
    layers = compressor._decoder_layers(model)
    device = next(model.parameters()).device
    sums = {i: torch.zeros(layers[i].self_attn.o_proj.in_features, dtype=torch.float64) for i in range(len(layers))}
    state: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(i: int):
        def hook(_mod, inputs):
            x = inputs[0]
            m = state["mask"].to(x.device).unsqueeze(-1).float()
            sums[i] += (x.float() * m).sum(dim=(0, 1)).double().cpu()
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(i)))
    n_tokens = 0
    was_training = model.training
    model.eval()
    try:
        for b in batches:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            state["mask"] = mask
            n_tokens += int(mask.sum().item())
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()
    if n_tokens == 0:
        raise ValueError("No valid tokens in the calibration batches.")
    return {i: s / n_tokens for i, s in sums.items()}, n_tokens


@torch.no_grad()
def compute_bias_corrections(model: nn.Module, means: Dict[int, torch.Tensor], heads: HeadList) -> Dict[int, torch.Tensor]:
    """W_o[:, h] @ mean(x_h) for each pruned head h; must be called BEFORE the o_proj weights are zeroed."""
    layers = compressor._decoder_layers(model)
    d = compressor._head_dim(model)
    out: Dict[int, torch.Tensor] = {}
    for layer, h in heads:
        W = layers[layer].self_attn.o_proj.weight.detach().float()
        mu = means[layer].to(W.device).float()
        sl = slice(h * d, (h + 1) * d)
        corr = W[:, sl] @ mu[sl]
        out[layer] = out[layer] + corr if layer in out else corr
    return out


@torch.no_grad()
def apply_bias_corrections(model: nn.Module, corrections: Dict[int, torch.Tensor]) -> Dict[str, Any]:
    """Adds the correction to o_proj.bias (creating the bias if absent). Called BEFORE quantization."""
    layers = compressor._decoder_layers(model)
    created = 0
    abs_vals: List[float] = []
    for layer, corr in corrections.items():
        o = layers[layer].self_attn.o_proj
        c = corr.to(o.weight.device, o.weight.dtype)
        if o.bias is None:
            o.bias = nn.Parameter(c.clone(), requires_grad=False)
            created += 1
        else:
            o.bias.data += c
        abs_vals.append(float(corr.abs().mean()))
    return {"n_layers_corrected": len(corrections), "n_bias_created": created,
            "mean_abs_bias_per_layer": abs_vals,
            "max_abs_bias": max(float(c.abs().max()) for c in corrections.values()) if corrections else 0.0}


def load_calibration_batches(tokenizer) -> List[Dict[str, torch.Tensor]]:
    """The same 16 passages used for xAI scoring (run_xai_on_mistral.select_passages / build_calibration_batches)."""
    from datasets import load_dataset  # lazy import (datasets may be absent locally)
    from run_xai_on_mistral import build_calibration_batches, select_passages

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    passages = select_passages(dataset["text"], CALIB["n_passages"], CALIB["min_chars"])
    return build_calibration_batches(tokenizer, passages, CALIB["batch_size"], CALIB["max_length"])


# --------------------------------------------------------------------------- #
# Dry-run infrastructure (mini Mistral, CPU, no bitsandbytes/datasets)
# --------------------------------------------------------------------------- #
class FakeLinear4bit(nn.Linear):
    """Dry-run only: fake 'INT4' Linear that rounds weights row-wise to 16 levels (CPU)."""


def _fake_quantize_linear(linear: nn.Linear, tier: str, compute_dtype: torch.dtype = torch.float16) -> nn.Module:
    if tier not in ("int4", "int8"):
        raise ValueError(f"Tier cannot be quantized: {tier!r}")
    levels = 7.0 if tier == "int4" else 127.0
    new = FakeLinear4bit(linear.in_features, linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / levels
        new.weight.copy_((w / scale).round().clamp(-levels, levels) * scale)
        if linear.bias is not None:
            new.bias.copy_(linear.bias.detach())
    return new.to(linear.weight.device)


class DryRun:
    """Mini model, fake tiers/scores, fake perplexity and calibration batches."""

    def __init__(self, seed: int, build_model=None):
        """build_model: builder for another mini architecture (e.g. tests/test_qwen2_mini.build_mini_qwen2); None = mini Mistral."""
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        from test_compressor_mini import build_random_scores
        from test_xai_engine_mini import build_dummy_dataloader, build_mini_model

        if build_model is not None:
            build_mini_model = build_model
        self._build_model = build_mini_model
        self._batches = build_dummy_dataloader(seed=seed + 1)
        m = build_mini_model()
        self.scores = build_random_scores(seed=seed, n_layers=m.config.num_hidden_layers,
                                          n_heads=m.config.num_attention_heads)
        self.tiers = allocate_compression_tiers(self.scores, 3, tier_fractions=(0.25, 0.25, 0.5))
        g = torch.Generator().manual_seed(seed + 2)
        self._eval_ids = torch.randint(1, m.config.vocab_size, (2, 32), generator=g)
        compressor._quantize_linear = _fake_quantize_linear  # type: ignore[attr-defined]

    def load_model(self, log: Logger):
        t0 = time.time()
        model = self._build_model()
        log(f"[DRY] mini model built ({sum(p.numel() for p in model.parameters()):,} parameters)")
        return None, model, time.time() - t0

    @torch.no_grad()
    def perplexity(self, model: nn.Module) -> float:
        out = model(input_ids=self._eval_ids, labels=self._eval_ids)
        return float(torch.exp(out.loss))

    def calibration_batches(self) -> List[Dict[str, torch.Tensor]]:
        return self._batches


def measure_mmlu(model: nn.Module, tokenizer, dry: Optional["DryRun"], log: Logger) -> Dict[str, Any]:
    """--mmlu: MMLU subset (eval_mmlu is imported lazily and never loaded when the flag is off)."""
    import eval_mmlu

    if dry:
        return eval_mmlu.evaluate_mmlu_dry(model, log=log)
    return eval_mmlu.evaluate_mmlu(model, tokenizer, log=log)


# --------------------------------------------------------------------------- #
# Single-configuration run
# --------------------------------------------------------------------------- #
class _StoreView:
    """Redirects the store.data[key] interface expected by quantize_with_fallback to a repeat dictionary."""

    def __init__(self, store: ResultStore, rep: Dict[str, Any]):
        self._store = store
        self.data = {"cfg": rep}

    def save(self, note: str = "") -> None:
        self._store.save(note)


def run_one(name: str, seed: int, ctx: Dict[str, Any], rep: Dict[str, Any], store: ResultStore, log: Logger) -> None:
    """Loads the model cleanly, applies the configuration, measures perplexity, releases the model. Results go into rep."""
    dry: Optional[DryRun] = ctx["dry"]
    tiers, scores, full_plan = ctx["tiers"], ctx["scores"], ctx["full_plan"]
    n_budget: int = ctx["n_prune_heads"]
    rep.update({"status": "running", "seed": seed, "started_at": datetime.now().isoformat(timespec="seconds")})
    store.save(f"{name} started")
    t_cfg = time.time()
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer, model, load_s = dry.load_model(log) if dry else load_model(log)
    rep["model_load_seconds"] = load_s
    try:
        n_layers = len(compressor._decoder_layers(model))
        rep["model_bytes_before"] = module_bytes(model)

        # --- head selection (same budget) ---
        xai_heads = heads_from_tiers(tiers)
        if name in ("prune_only", "both", "both_biascorr"):
            heads = xai_heads
        elif name == "quant_only":
            heads = []
        elif name == "prune_magnitude":
            mag = head_magnitude_scores(model)
            heads = select_heads_by_score(mag, n_budget, lowest=True)
            rep["magnitude_score_range"] = [min(mag.values()), max(mag.values())]
            rep["overlap_with_xai_heads"] = len(set(heads) & set(xai_heads))
        elif name == "prune_random":
            heads = select_heads_random(all_heads(tiers), n_budget, seed)
            rep["overlap_with_xai_heads"] = len(set(heads) & set(xai_heads))
        elif name == "prune_reverse":
            heads = select_heads_by_score(scores, n_budget, lowest=False)
            rep["overlap_with_xai_heads"] = len(set(heads) & set(xai_heads))
        else:
            raise ValueError(f"unknown configuration: {name}")

        quant_plan = None
        if name == "quant_only":
            quant_plan = quant_only_plan(full_plan, log)
        elif name in ("both", "both_biascorr"):
            quant_plan = full_plan
        plan = merge_plan(prune_plan_from_heads(heads, n_layers), quant_plan)
        if name in ("both", "both_biascorr"):  # same plan as the end-to-end prototype? (pruned set + tiers)
            assert all(plan[i].pruned_heads == full_plan[i].pruned_heads and plan[i].attn_quant == full_plan[i].attn_quant
                       and plan[i].mlp_quant == full_plan[i].mlp_quant for i in full_plan)
        rep["pruned_heads"] = [list(h) for h in heads]
        rep["n_pruned_heads"] = len(heads)
        rep["plan_summary"] = plan_summary(plan)
        cfg = model.config
        rep["budget_nominal"] = estimate_compression_budget(
            plan, hidden_size=cfg.hidden_size, head_dim=compressor._head_dim(model),
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size)
        log(f"{name}: {len(heads)} heads to prune, INT4 layers (attn/mlp)="
            f"{sum(p.attn_quant == 'int4' for p in plan.values())}/{sum(p.mlp_quant == 'int4' for p in plan.values())}")

        # --- bias compensation: mean activations on the unpruned model, before o_proj is zeroed ---
        corrections = None
        if name == "both_biascorr":
            t0 = time.time()
            batches = dry.calibration_batches() if dry else load_calibration_batches(tokenizer)
            means, n_tok = collect_o_proj_input_means(model, batches)
            corrections = compute_bias_corrections(model, means, heads)
            rep["bias_correction"] = {"calibration": dict(CALIB, n_batches=len(batches), n_valid_tokens=n_tok),
                                      "seconds": time.time() - t0}
            log(f"{name}: calibration {len(batches)} batches / {n_tok} tokens, corrections computed for "
                f"{len(corrections)} layers ({rep['bias_correction']['seconds']:.1f} s)")

        # --- pruning -> (compensation) -> quantization (same order as the end-to-end prototype) ---
        t0 = time.time()
        rep["pruning"] = apply_structural_pruning(model, plan, verbose=True)
        if corrections is not None:
            rep["bias_correction"].update(apply_bias_corrections(model, corrections))
            log(f"{name}: bias compensation applied to {rep['bias_correction']['n_layers_corrected']} layers, "
                f"max|bias|={rep['bias_correction']['max_abs_bias']:.4f}")
        if quant_plan is not None:
            rep["quantization"] = quantize_with_fallback(model, plan, log, _StoreView(store, rep), "cfg")
        else:
            rep["quantization"] = {"int4_modules": 0, "int8_modules": 0}
            rep["quantization_path"] = None
        rep["apply_seconds"] = time.time() - t0
        rep["int4_modules"] = rep["quantization"]["int4_modules"]
        rep["model_bytes_after"] = module_bytes(model)
        rep["measured_model_bytes_ratio"] = rep["model_bytes_after"]["bytes"] / rep["model_bytes_before"]["bytes"]
        log(f"{name}: applied ({rep['apply_seconds']:.1f} s) pruning={rep['pruning']} quantization={rep['quantization']} "
            f"size {rep['model_bytes_before']['gb']:.2f} -> {rep['model_bytes_after']['gb']:.2f} GB")
        store.save(f"{name} applied")

        # --- perplexity ---
        t0 = time.time()
        if dry:
            ppl = dry.perplexity(model)
        else:
            ppl = ctx["compute_perplexity"](model, tokenizer, ctx["text"], next(model.parameters()).device)
        rep["perplexity"] = ppl
        rep["perplexity_seconds"] = time.time() - t0
        rep["perplexity_finite"] = math.isfinite(ppl)
        if ctx["args"].mmlu:
            rep["mmlu"] = measure_mmlu(model, tokenizer, dry, log)
        rep["status"] = "completed"
        log(f"{name} DONE: perplexity = {ppl:.4f} ({rep['perplexity_seconds']:.1f} s)")
    finally:
        rep["peak_vram_gb"] = cuda_gb("peak")
        rep["seconds"] = time.time() - t_cfg
        free_model(model, log)
        store.save(f"{name} finished")


# --------------------------------------------------------------------------- #
# Configuration summary / decomposition
# --------------------------------------------------------------------------- #
def summarize_config(cfg: Dict[str, Any], fp16_ppl: Optional[float]) -> None:
    reps = [r for r in cfg["repeats"] if r.get("status") == "completed"]
    ppls = [r["perplexity"] for r in reps]
    cfg["n_completed_repeats"] = len(reps)
    cfg["perplexity_values"] = ppls
    cfg["perplexity_mean"] = float(np.mean(ppls)) if ppls else None
    cfg["perplexity_std"] = float(np.std(ppls, ddof=1)) if len(ppls) > 1 else (0.0 if ppls else None)
    if reps:
        r0 = reps[0]
        cfg["pruned_heads"] = r0["pruned_heads"]
        cfg["pruned_heads_identical_across_repeats"] = all(r["pruned_heads"] == r0["pruned_heads"] for r in reps)
        cfg["n_pruned_heads"] = r0["n_pruned_heads"]
        cfg["int4_modules"] = r0["int4_modules"]
        cfg["model_bytes_after_gb"] = r0["model_bytes_after"]["gb"]
        cfg["measured_model_bytes_ratio"] = r0["measured_model_bytes_ratio"]
        cfg["seconds"] = sum(r["seconds"] for r in reps)
        if "overlap_with_xai_heads" in r0:
            cfg["overlap_with_xai_heads"] = [r["overlap_with_xai_heads"] for r in reps]
        accs = [r["mmlu"]["mmlu_subset_acc"] for r in reps if "mmlu" in r]
        if accs:  # only with --mmlu
            cfg["mmlu_subset_acc_values"] = accs
            cfg["mmlu_subset_acc_mean"] = float(np.mean(accs))
    if fp16_ppl is not None and cfg["perplexity_mean"] is not None:
        cfg["delta_vs_fp16"] = cfg["perplexity_mean"] - fp16_ppl
        cfg["ratio_vs_fp16"] = cfg["perplexity_mean"] / fp16_ppl


def decomposition(configs: Dict[str, Any], fp16_ppl: Optional[float]) -> Dict[str, Any]:
    def ppl(n: str) -> Optional[float]:
        c = configs.get(n)
        return c.get("perplexity_mean") if c else None

    out: Dict[str, Any] = {"fp16": fp16_ppl}
    p, q, b = ppl("prune_only"), ppl("quant_only"), ppl("both")
    if fp16_ppl is not None:
        if p is not None:
            out["prune_delta"] = p - fp16_ppl
        if q is not None:
            out["quant_delta"] = q - fp16_ppl
        if b is not None:
            out["both_delta"] = b - fp16_ppl
        if None not in (p, q, b):
            out["interaction"] = (b - fp16_ppl) - (p - fp16_ppl) - (q - fp16_ppl)
            out["note"] = "interaction = both_delta - prune_delta - quant_delta (0 = additive)"
    if p is not None:
        for n in ("prune_magnitude", "prune_random", "prune_reverse"):
            if ppl(n) is not None:
                out[f"{n}_minus_prune_only"] = ppl(n) - p
    if b is not None and ppl("both_biascorr") is not None:
        out["biascorr_gain"] = b - ppl("both_biascorr")
    return out


def format_summary_table(configs: Dict[str, Any], fp16_ppl: Optional[float]) -> str:
    lines = [f"{'config':<16}{'ppl':>9}{'±std':>8}{'Δfp16':>9}{'head':>6}{'int4':>6}{'GB':>7}{'s':>7}  status"]
    if fp16_ppl is not None:
        lines.append(f"{'fp16':<16}{fp16_ppl:>9.4f}{'':>8}{0.0:>9.4f}{0:>6}{0:>6}{'':>7}{'':>7}  reference")
    for name, c in configs.items():
        m = c.get("perplexity_mean")
        s_ppl = f"{m:.4f}" if m is not None else "-"
        s_std = f"{(c.get('perplexity_std') or 0):.4f}" if m is not None else ""
        s_delta = f"{c['delta_vs_fp16']:+.4f}" if "delta_vs_fp16" in c else "-"
        s_gb = f"{c['model_bytes_after_gb']:.2f}" if "model_bytes_after_gb" in c else "-"
        s_sec = f"{c['seconds']:.0f}" if "seconds" in c else "-"
        lines.append(f"{name:<16}{s_ppl:>9}{s_std:>8}{s_delta:>9}{str(c.get('n_pruned_heads', '-')):>6}"
                     f"{str(c.get('int4_modules', '-')):>6}{s_gb:>7}{s_sec:>7}  {c.get('status')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log = Logger(args.log, to_file=should_log_to_file(args, LOG_FILE))  # dry-run/preflight does not write to the tracked log
    store = ResultStore(args.output, log)
    t_start = time.time()
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in config_names if c not in CONFIG_DESCRIPTIONS]
    if unknown:
        raise SystemExit(f"unknown configuration(s): {unknown}; valid: {ALL_CONFIGS}")
    log(f"===== Decomposition/control experiments starting: configs={config_names} seed={args.seed} "
        f"n_repeats={args.n_repeats} dry_run={args.dry_run} skip_baseline={args.skip_baseline} =====")

    # --- inputs (no GPU needed) ---
    dry: Optional[DryRun] = DryRun(args.seed) if args.dry_run else None
    if dry:
        tiers, scores = dry.tiers, dry.scores
        tiers_meta: Dict[str, Any] = {"n_tiers": 3, "tier_fractions": [0.25, 0.25, 0.5], "mlp_min_tier": "int4"}
        baseline_gun1, gun5_both = None, None
    else:
        tiers_json = load_tiers_json(args.tiers)
        tiers, tiers_meta = tiers_json["tiers"], tiers_json
        scores = load_scores(args.scores)
        baseline_gun1 = None
        if os.path.exists(args.baseline):
            with open(args.baseline, "r", encoding="utf-8") as f:
                baseline_gun1 = float(json.load(f)["perplexity"])
        gun5_both = GUN5_BOTH_PPL_FALLBACK
        if os.path.exists(args.gun5_json):
            with open(args.gun5_json, "r", encoding="utf-8") as f:
                gun5_both = float(json.load(f).get("full", {}).get("perplexity_compressed", GUN5_BOTH_PPL_FALLBACK))
    full_plan = build_compression_plan(tiers)
    n_prune_heads = len(heads_from_tiers(tiers))
    n_heads_total = len(all_heads(tiers))
    median = median_rule_counts(tiers, full_plan)
    regenerated = allocate_compression_tiers(scores, tiers_meta.get("n_tiers", 3), tier_fractions=tiers_meta.get("tier_fractions"),
                                             mlp_min_tier=tiers_meta.get("mlp_min_tier", "int4"))
    n_diff = sum(1 for k in tiers if regenerated.get(k) != tiers[k])

    previous = None
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            previous = json.load(f)
        log(f"--resume: read {args.output}; completed configurations will be skipped")

    store.data = {
        "run": {"started_at": datetime.now().isoformat(timespec="seconds"), "status": "running",
                "args": vars(args), "model": "mini (dry-run)" if dry else MODEL_NAME, "env": env_info(),
                "config_order": config_names},
        "reference": {
            "fp16_baseline_gun1": baseline_gun1, "gun5_both_perplexity": gun5_both, "reproduction_tolerance": REPRO_TOL,
            "tiers_file": None if dry else args.tiers, "scores_file": None if dry else args.scores,
            "n_tiers": tiers_meta.get("n_tiers"), "tier_fractions": tiers_meta.get("tier_fractions"),
            "mlp_min_tier": tiers_meta.get("mlp_min_tier"), "tier_counts": tier_counts(tiers),
            "tiers_regenerated_diff": n_diff,
            "budget_heads": {"n_prune_heads": n_prune_heads, "n_heads_total": n_heads_total,
                             "nominal_block_prune_ratio": n_prune_heads / n_heads_total,
                             "definition": "number of pruned heads in every configuration = number of prune heads in the tier plan"},
            "int4_layers_in_plan": {"attn": sum(p.attn_quant == "int4" for p in full_plan.values()),
                                    "mlp": sum(p.mlp_quant == "int4" for p in full_plan.values())},
            "budget_nominal_both": estimate_compression_budget(full_plan) if not dry else None,
            "plan_summary_both": plan_summary(full_plan),
        },
        "median_rule": median,
        "fp16_rerun": (previous or {}).get("fp16_rerun", {"status": "skipped" if args.skip_baseline else "pending"}),
        "configs": {n: {"description": CONFIG_DESCRIPTIONS[n], "status": "pending", "repeats": []} for n in config_names},
        "decomposition": {},
    }
    env = store.data["run"]["env"]
    log(f"Environment: torch {env['torch']}, transformers {env['transformers']}, bitsandbytes {env.get('bitsandbytes')}, "
        f"CUDA={env['cuda_available']} ({env.get('gpu')})")
    log(f"Tiers: {tier_counts(tiers)}; budget {n_prune_heads}/{n_heads_total} heads; JSON vs regenerated diff={n_diff}; "
        f"INT4 layers attn/mlp={store.data['reference']['int4_layers_in_plan']}; FP16 baseline={baseline_gun1}; prototype both={gun5_both}")
    log(f"Median rule: of {median['n_unpruned_heads']} unpruned heads, {median['downgraded']} downgraded "
        f"(own tier has higher precision), {median['upgraded']} upgraded, {median['unchanged']} unchanged; pairs={median['pairs']}")
    if n_diff and not dry:
        log(f"WARNING: {n_diff} tiers in the JSON differ from those regenerated from the scores; using the JSON.")

    if args.preflight:
        store.data["run"]["status"] = "preflight_only"
        log("Preflight completed (no model loaded). Nothing was written to the result file.")
        return 0
    if not torch.cuda.is_available() and not dry:
        log("WARNING: no CUDA; the 7B model + bitsandbytes cannot run on CPU. Run on a GPU machine (or use --dry-run).")
    store.save("start")

    ctx: Dict[str, Any] = {"args": args, "dry": dry, "tiers": tiers, "scores": scores, "full_plan": full_plan,
                           "n_prune_heads": n_prune_heads}
    exit_code = 0
    fp16_ppl: Optional[float] = baseline_gun1
    try:
        if not dry:
            log("Loading WikiText-2 test set...")
            ctx["text"] = load_wikitext_text()
            ctx["compute_perplexity"], max_len, stride = perplexity_tools()
            store.data["reference"]["perplexity_settings"] = {"max_length": max_len, "stride": stride,
                                                              "dataset": "wikitext-2-raw-v1 (test split)"}

        # --- FP16 re-measurement (once, clean load) ---
        fp = store.data["fp16_rerun"]
        if args.skip_baseline:
            log(f"FP16 rerun SKIPPED (--skip-baseline); reference FP16 baseline={fp16_ppl}")
        elif fp.get("status") == "completed":
            fp16_ppl = fp["perplexity"]
            log(f"FP16 rerun taken from the previous run via --resume: {fp16_ppl}")
        else:
            fp.update({"status": "running"})
            set_seed(args.seed)
            t0 = time.time()
            tokenizer, model, load_s = dry.load_model(log) if dry else load_model(log)
            try:
                fp["model_load_seconds"] = load_s
                fp["model_bytes"] = module_bytes(model)
                if dry:
                    ppl0 = dry.perplexity(model)
                else:
                    ppl0 = ctx["compute_perplexity"](model, tokenizer, ctx["text"], next(model.parameters()).device)
                if args.mmlu:
                    fp["mmlu"] = measure_mmlu(model, tokenizer, dry, log)
            finally:
                free_model(model, log)
            fp.update({"perplexity": ppl0, "seconds": time.time() - t0, "status": "completed",
                       "minus_gun1": (ppl0 - baseline_gun1) if baseline_gun1 is not None else None})
            fp16_ppl = ppl0
            log(f"FP16 RERUN DONE: perplexity = {ppl0:.4f} (baseline file: {baseline_gun1}, diff {fp['minus_gun1']}), {fp['seconds']:.1f} s")
            store.save("fp16 rerun")

        # --- configurations ---
        for ci, name in enumerate(config_names, 1):
            cfg = store.data["configs"][name]
            prev_cfg = (previous or {}).get("configs", {}).get(name)
            if prev_cfg and prev_cfg.get("status") == "completed":
                store.data["configs"][name] = prev_cfg
                log(f"CONFIG {ci}/{len(config_names)} {name}: --resume, taken from the previous run (ppl={prev_cfg.get('perplexity_mean')})")
                continue
            n_rep = args.n_repeats if name in STOCHASTIC_CONFIGS else 1
            cfg.update({"status": "running", "n_repeats": n_rep, "stochastic": name in STOCHASTIC_CONFIGS})
            log(f"===== CONFIG {ci}/{len(config_names)}: {name} ({n_rep} repeats) — {cfg['description']} =====")
            for r in range(n_rep):
                rep: Dict[str, Any] = {}
                cfg["repeats"].append(rep)
                run_one(name, args.seed + r, ctx, rep, store, log)
            summarize_config(cfg, fp16_ppl)
            cfg["status"] = "completed"
            if name == "both" and gun5_both is not None:
                diff = abs(cfg["perplexity_mean"] - gun5_both)
                cfg["reproduction_check"] = {"expected_gun5": gun5_both, "abs_diff": diff, "within_tolerance": diff <= REPRO_TOL}
                log(("CHECK OK" if diff <= REPRO_TOL else "CHECK FAILED") +
                    f": both={cfg['perplexity_mean']:.4f} vs prototype {gun5_both:.4f} (diff {diff:.4f}, tolerance {REPRO_TOL})",
                    tag="CHECK")
            store.data["decomposition"] = decomposition(store.data["configs"], fp16_ppl)
            log(f"CONFIG {ci}/{len(config_names)} DONE: {name} ppl={cfg['perplexity_mean']:.4f} ± {cfg['perplexity_std']:.4f} "
                f"(Δfp16={cfg.get('delta_vs_fp16')}), {cfg['seconds']:.0f} s")
            store.save(f"{name} summary")
        store.data["run"]["status"] = "completed"
    except BaseException as e:  # includes KeyboardInterrupt / SystemExit so partial results are not lost
        store.data["run"]["status"] = "failed"
        store.data["run"]["error"] = repr(e)
        store.data["run"]["traceback"] = traceback.format_exc()
        log(f"ERROR: {e!r}; writing partial results to disk", tag="ERR")
        log(store.data["run"]["traceback"], tag="ERR")
        exit_code = 1
    finally:
        for cfg in store.data["configs"].values():
            if cfg.get("repeats") and cfg.get("status") != "completed":
                summarize_config(cfg, fp16_ppl)  # partial: summarize the completed repeats
        store.data["decomposition"] = decomposition(store.data["configs"], fp16_ppl)
        store.data["run"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        store.data["run"]["total_seconds"] = time.time() - t_start
        store.save("final")
        log("SUMMARY TABLE:\n" + format_summary_table(store.data["configs"], fp16_ppl))
        log(f"Decomposition: {json.dumps(store.data['decomposition'], ensure_ascii=False)}")
        log(f"===== Done: status={store.data['run']['status']}, total {store.data['run']['total_seconds']:.1f} s =====")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
