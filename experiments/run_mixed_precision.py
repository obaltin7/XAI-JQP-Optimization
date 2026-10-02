"""
D-2: sub-4-bit, xAI-guided bit allocation with HQQ (GPU; --dry-run runs on CPU).

Question: at the same AVERAGE bit budget (B in {3.0, 3.5} bits/param, parameter-weighted over the decoder Linear modules), does
distributing bits by xAI importance (more bits to important modules, fewer to unimportant ones) beat a uniform bit width and
random / magnitude-based allocation? No pruning and no FP16 tier. Quantization: HQQ (Half-Quadratic Quantization, Badri & Shaji
2023; calibration-free), group_size 64, axis 1, nbits in {2, 3, 4, 8}; lm_head and the embeddings stay FP16 (same treatment as
nf4_uniform / D-1).

Policy:
  * module = the attention block (q/k/v/o_proj) or the MLP block (gate/up/down_proj) of a layer, with the SAME module score as D-1
    (compressor.module_importance_scores: attn = lower median of the head scores, MLP = MLP score; stored importance scores,
    no attribution is run).
  * Ranking is WITHIN each module kind: the most important blocks get 8 bits, then 4, 3, and the least important get 2 bits. Tier
    shares are solved to meet the budget. B=3.5 template: p8 = 0.05 fixed, p4 = p3 = (B - 2 - 6*p8)/3, p2 = remainder
    (0.05/0.40/0.40/0.15). B=3.0 template: no 8-bit tier, equal 4/3/2 shares (1/3 each).
    Integer block counts are solved per kind: average bits <= B, closest to the budget, then closest to the template
    (32 blocks, 8/4/3/2 bits: B=3.0 -> 0/11/10/11, B=3.5 -> 2/12/12/6; both meet the budget EXACTLY). Since blocks within a kind
    have equal size and both kinds meet the same B, the parameter-weighted average is also B.
  * Controls use the SAME counts: hqq_random (seeded random ranking, --n-repeats), hqq_magnitude (module Frobenius norm);
    hqq_uniform_3bit (uniform counterpart of B=3.0), hqq_uniform_4bit (closest uniform width to B=3.5; budget 4.0, a generous control).

Size: GB = parameters + buffers + the scale and zero-point tensors of the HQQ modules (hqq keeps these in its `meta` dict, not as
parameters/buffers, so they are added explicitly) -> W_q + scale + zero are counted. For the bnb rows (nf4_uniform, XAI-JQP) the
quant_state is not counted, so the comparison is slightly biased in favour of bnb (noted under the table). Nominal average bits
(nbits) and EFFECTIVE average bits (W_q packing + scale + zero; with fp16 meta 2->2.5, 3->~3.7 (10 values per int32), 4->4.5,
8->8.5) are reported separately; the budget is defined on NOMINAL bits.

Configs: hqq_uniform_3bit, hqq_uniform_4bit, hqq_{xai,random,magnitude}_b3.0, hqq_{xai,random,magnitude}_b3.5.

Usage (requires: pip install hqq==0.2.8.post1):
    python experiments/run_mixed_precision.py --dry-run          # mini model + hqq (fake group quantization if hqq is missing), CPU
    python experiments/run_mixed_precision.py --preflight        # CUDA, hqq version + 2/3/4/8-bit smoke test, budget solution, MMLU
    python experiments/run_mixed_precision.py --with-tasks       # full run; --with-tasks adds HellaSwag/ARC for uniform/xai/random seed 42
    python experiments/run_mixed_precision.py --resume           # skip completed configs

Outputs: results/mixed_bits_gun11.json (per-config repeats, bit plans, sizes, perplexity, MMLU, optional tasks and window NLLs)
and the log file log_gun11_bits.txt.
"""

# HF_HOME must be set BEFORE transformers/datasets are imported (see run_e2e_pipeline.py)
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compressor
from compressor import load_scores, module_importance_scores, module_magnitude_scores, rank_modules
# Shared building blocks of the earlier experiment scripts (imported unchanged)
from run_ablation_tests import DryRun, set_seed
from run_baselines import ensure_clean_gpu, release_gpu
from run_e2e_pipeline import (
    MODEL_NAME,
    SCORES_FILE,
    Logger,
    ResultStore,
    cuda_gb,
    env_info,
    load_model,
    load_wikitext_text,
    module_bytes,
    perplexity_tools,
    should_log_to_file,
)

OUTPUT_FILE = os.path.join("results", "mixed_bits_gun11.json")
LOG_FILE = "log_gun11_bits.txt"
BASELINES_FILE = os.path.join("results", "baselines_gun7.json")
DRYRUN_DIR = "dryrun_out"
HQQ_PIN = "0.2.8.post1"  # requirements-quant.txt; verified with torch 2.4.1 + transformers 4.44.2
GROUP_SIZE, AXIS = 64, 1
BIT_TIERS: Tuple[int, ...] = (8, 4, 3, 2)  # most important to least important
TEMPLATE_P8 = 0.05
NO_8BIT_BUDGETS: Tuple[float, ...] = (3.0,)  # B=3.0 has no 8-bit tier and equal 4/3/2 shares; the B=3.5 template is unchanged
BUDGETS: Tuple[float, ...] = (3.0, 3.5)
SELECTORS = ("xai", "random", "magnitude")


def budget_key(budget: float) -> str:
    return f"{budget:.1f}"


def build_configs() -> Dict[str, Dict[str, Any]]:
    cfgs: Dict[str, Dict[str, Any]] = {
        "hqq_uniform_3bit": {"selector": "uniform", "nbits": 3, "budget": 3.0, "label": "HQQ uniform 3-bit",
                             "description": "all decoder Linears HQQ 3-bit (uniform counterpart of B=3.0)"},
        "hqq_uniform_4bit": {"selector": "uniform", "nbits": 4, "budget": 4.0, "label": "HQQ uniform 4-bit",
                             "description": "all decoder Linears HQQ 4-bit (closest uniform width to B=3.5; budget 4.0, a generous control)"},
    }
    text = {"xai": "module score = stored xAI importance scores (attn: lower median of the head scores, MLP: MLP score)",
            "random": "seeded random ranking (n_repeats repeats; same tier counts per kind)",
            "magnitude": "module score = joint Frobenius norm of the Linear weights"}
    names = {"xai": "xAI", "random": "random", "magnitude": "magnitude"}
    for b in BUDGETS:
        for sel in SELECTORS:
            cfgs[f"hqq_{sel}_b{budget_key(b)}"] = {
                "selector": sel, "budget": b, "label": f"HQQ {names[sel]} B={budget_key(b)}",
                "description": f"HQQ bit allocation, average {budget_key(b)} bits/param: within each kind the most important get 8, then 4, 3, 2 bits; {text[sel]}"}
    return cfgs


CONFIGS = build_configs()
ALL_CONFIGS = list(CONFIGS)
STOCHASTIC_SELECTORS = {"random"}


# --------------------------------------------------------------------------- #
# Budget solution and bit allocation (no GPU needed; called directly by the tests)
# --------------------------------------------------------------------------- #
def template_shares(budget: float, p8: float = TEMPLATE_P8) -> Dict[int, float]:
    """
    Tier share template: p8 fixed, p4 = p3, p2 = remainder; 8*p8 + 4*p4 + 3*p3 + 2*p2 = budget.
    For budgets in NO_8BIT_BUDGETS (B=3.0) there is no 8-bit tier and 4/3/2 get EQUAL shares (1/3 each; average exactly 3.0).
    The p8 template at B=3.0 would push ~50% of the parameters to 2 bits (2/6/8/16); the equal-share solution is 0/11/10/11
    (34% at 2 bits).
    """
    if any(math.isclose(budget, b) for b in NO_8BIT_BUDGETS):
        return {8: 0.0, 4: 1.0 / 3.0, 3: 1.0 / 3.0, 2: 1.0 / 3.0}
    p43 = (budget - 2.0 - 6.0 * p8) / 3.0
    shares = {8: p8, 4: p43, 3: p43, 2: 1.0 - p8 - 2.0 * p43}
    if min(shares.values()) < -1e-12:  # valid range with p8 = 0.05: 2.3 <= B <= 3.725
        raise ValueError(f"budget {budget} cannot be met with this template (p8={p8}) (valid range {2 + 6 * p8:g}–{(19 - 7 * p8) / 5:g}): {shares}")
    return shares


def solve_tier_counts(n: int, budget: float, p8: float = TEMPLATE_P8) -> Dict[int, int]:
    """
    Split n equal-size blocks into (8, 4, 3, 2)-bit tiers subject to average bits <= budget, choosing (1) the solution CLOSEST to
    the budget, then (2) the smallest L1 distance to the template counts (template_shares*n), then (3) on ties the one with larger
    high-bit tiers. Exhaustive search (n=32: ~6k candidates).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    target = {b: s * n for b, s in template_shares(budget, p8).items()}
    best: Optional[Tuple[Tuple[float, float, int, int], Dict[int, int]]] = None
    for n8 in range(n + 1):
        for n4 in range(n - n8 + 1):
            for n3 in range(n - n8 - n4 + 1):
                n2 = n - n8 - n4 - n3
                total = 8 * n8 + 4 * n4 + 3 * n3 + 2 * n2
                if total > budget * n + 1e-9:
                    continue
                l1 = abs(n8 - target[8]) + abs(n4 - target[4]) + abs(n3 - target[3]) + abs(n2 - target[2])
                key = (round(budget * n - total, 9), round(l1, 9), -n8, -n4)
                if best is None or key < best[0]:
                    best = (key, {8: n8, 4: n4, 3: n3, 2: n2})
    assert best is not None  # budget >= 2 -> all-2-bit is always feasible
    return best[1]


def assign_bits(module_scores: Dict[str, float], budget: float, *, random_seed: Optional[int] = None) -> Dict[str, int]:
    """
    Module -> nbits. WITHIN each kind (attn / mlp separately), the solve_tier_counts counts are assigned 8, 4, 3, 2 bits in order
    of importance. If random_seed is given the scores are ignored and the same counts follow a seeded random ranking. The returned
    dict follows the order of module_scores.
    """
    out: Dict[str, int] = {}
    for kind, names in compressor._module_pools(module_scores).items():
        counts = solve_tier_counts(len(names), budget)
        seed = None if random_seed is None else random_seed + (0 if kind == "attn" else 1)
        ranked = rank_modules(module_scores, names, random_seed=seed)
        pos = 0
        for bits in BIT_TIERS:
            for name in ranked[pos:pos + counts[bits]]:
                out[name] = bits
            pos += counts[bits]
    return {n: out[n] for n in module_scores}


def uniform_bits(module_scores: Dict[str, float], nbits: int) -> Dict[str, int]:
    return {n: int(nbits) for n in module_scores}


def bit_counts(bit_plan: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {"attn": {}, "mlp": {}}
    for name, bits in bit_plan.items():
        kind = name.rsplit(".", 1)[-1]
        out[kind][str(bits)] = out[kind].get(str(bits), 0) + 1
    return out


def module_param_counts(model: nn.Module) -> Dict[str, int]:
    """{"layer_{i}.attn": q+k+v+o weight count, "layer_{i}.mlp": gate+up+down} (bias excluded; must be called BEFORE quantization)."""
    return {key: sum(int(lin.weight.numel()) for _, lin in linears) for key, linears in compressor.module_linears(model).items()}


def average_bits(bit_plan: Dict[str, int], param_counts: Dict[str, int]) -> float:
    """Parameter-weighted NOMINAL average bits (over the decoder Linear modules)."""
    total = sum(param_counts[k] for k in bit_plan)
    return sum(param_counts[k] * bit_plan[k] for k in bit_plan) / total


# --------------------------------------------------------------------------- #
# HQQ application and size
# --------------------------------------------------------------------------- #
def hqq_available() -> bool:
    return importlib.util.find_spec("hqq") is not None


def hqq_version() -> Optional[str]:
    try:
        return importlib.metadata.version("hqq")
    except importlib.metadata.PackageNotFoundError:
        return None


class FakeHQQLinear(nn.Linear):
    """Dry-run stand-in when hqq is NOT installed: fake quantized Linear with per-group (group_size) min-max rounding (CPU; code path only)."""

    nbits: int = 16

    def quant_bytes(self, group_size: int) -> int:  # W_q (nominal packing) + fp16 scale + fp16 zero point
        n = self.weight.numel()
        return int(math.ceil(n * self.nbits / 8)) + 2 * 2 * int(math.ceil(n / group_size))


def _fake_hqq_linear(linear: nn.Linear, nbits: int, group_size: int) -> nn.Module:
    new = FakeHQQLinear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    new.nbits = int(nbits)
    with torch.no_grad():
        w = linear.weight.detach().float()
        g = w.reshape(-1, group_size) if w.numel() % group_size == 0 else w.reshape(1, -1)
        lo, hi = g.amin(dim=1, keepdim=True), g.amax(dim=1, keepdim=True)
        scale = ((hi - lo) / (2 ** nbits - 1)).clamp_min(1e-8)
        new.weight.copy_((((g - lo) / scale).round().clamp(0, 2 ** nbits - 1) * scale + lo).reshape(w.shape))
        if linear.bias is not None:
            new.bias.copy_(linear.bias.detach())
    return new.to(linear.weight.device)


def _hqq_linear(linear: nn.Linear, nbits: int, group_size: int, axis: int, compute_dtype: torch.dtype) -> nn.Module:
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    cfg = BaseQuantizeConfig(nbits=int(nbits), group_size=group_size, axis=axis)
    return HQQLinear(linear, quant_config=cfg, compute_dtype=compute_dtype, device=str(linear.weight.device), del_orig=True)


def is_hqq_module(mod: nn.Module) -> bool:
    return hasattr(mod, "W_q") and isinstance(getattr(mod, "meta", None), dict)


def apply_hqq(model: nn.Module, bit_plan: Dict[str, int], *, group_size: int = GROUP_SIZE, axis: int = AXIS,
              compute_dtype: Optional[torch.dtype] = None, fake: bool = False) -> Dict[str, Any]:
    """
    Convert ALL Linears of every module in bit_plan (attention / MLP block) to HQQ with that module's nbits (in place). lm_head and
    the embeddings are left untouched. fake=True: dry-run without hqq (FakeHQQLinear). Returns {"modules_by_bits": {"3": count, ...}, "n_modules"}.
    """
    layers = compressor._decoder_layers(model)
    expected = {f"layer_{i}.{k}" for i in range(len(layers)) for k in ("attn", "mlp")}
    if set(bit_plan) != expected:
        raise ValueError(f"bit plan does not match the model: {len(bit_plan)} modules in the plan, {len(expected)} in the model")
    if compute_dtype is None:
        compute_dtype = next(model.parameters()).dtype
    by_bits: Dict[str, int] = {}
    for i, layer in enumerate(layers):
        for key, parent, names in ((f"layer_{i}.attn", layer.self_attn, compressor.ATTN_LINEARS),
                                   (f"layer_{i}.mlp", layer.mlp, compressor.MLP_LINEARS)):
            nbits = int(bit_plan[key])
            if nbits not in BIT_TIERS:
                raise ValueError(f"{key}: nbits={nbits} is not supported; {BIT_TIERS}")
            for name in names:
                lin = getattr(parent, name)
                if not isinstance(lin, nn.Linear) or isinstance(lin, FakeHQQLinear):
                    raise ValueError(f"{key}.{name}: not an nn.Linear ({type(lin).__name__}); HQQ must be applied to an uncompressed model")
                new = _fake_hqq_linear(lin, nbits, group_size) if fake else _hqq_linear(lin, nbits, group_size, axis, compute_dtype)
                setattr(parent, name, new)
                by_bits[str(nbits)] = by_bits.get(str(nbits), 0) + 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"modules_by_bits": by_bits, "n_modules": sum(by_bits.values()), "backend": "fake (hqq not installed)" if fake else "hqq"}


def _tensor_bytes(t: Any) -> int:
    return int(t.numel() * t.element_size()) if torch.is_tensor(t) else 0


def hqq_model_bytes(model: nn.Module, group_size: int = GROUP_SIZE) -> Dict[str, Any]:
    """
    Model size: parameters + buffers (module_bytes; in HQQ, W_q is a parameter) + the scale / zero-point tensors in the `meta` dict
    of the HQQ modules + any non-parameter bias. quant_bytes = sum over the quantized Linears only (W_q + scale + zero), used for
    the effective-bit computation. For FakeHQQLinear an analytic estimate is used (nominal packing + fp16 meta).
    """
    base = module_bytes(model)
    extra = quant = fake_adjust = 0
    for mod in model.modules():
        if is_hqq_module(mod):
            meta = sum(_tensor_bytes(v) for v in mod.meta.values())
            bias = getattr(mod, "bias", None)
            extra += meta + (_tensor_bytes(bias) if not isinstance(bias, nn.Parameter) else 0)
            quant += _tensor_bytes(mod.W_q) + meta
        elif isinstance(mod, FakeHQQLinear):
            q = mod.quant_bytes(group_size)
            quant += q
            fake_adjust += q - _tensor_bytes(mod.weight)  # nominal packed size instead of the fp32 fake weight
    total = base["bytes"] + extra + fake_adjust
    return {"bytes": total, "gb": total / 1e9, "param_buffer_bytes": base["bytes"], "hqq_meta_bytes": extra, "quant_bytes": quant,
            "by_dtype_bytes": base["by_dtype_bytes"]}


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D-2: sub-4-bit, xAI-guided bit allocation (HQQ)")
    p.add_argument("--configs", default=",".join(ALL_CONFIGS), help="comma-separated list of configs (default: all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=3, help="number of repeats for the random control; seed, seed+1, ...")
    p.add_argument("--no-mmlu", action="store_true", help="do NOT evaluate the MMLU subset (default: evaluate)")
    p.add_argument("--mmlu-batch-size", type=int, default=8)
    p.add_argument("--with-tasks", action="store_true",
                   help="also evaluate HellaSwag + ARC-Challenge (eval_tasks, per-question records; off by default). ONLY for uniform, "
                        "xai and the FIRST seed (--seed) of the random control; not measured for magnitude or the other random seeds (runtime)")
    p.add_argument("--tasks-batch-size", type=int, default=16)
    p.add_argument("--scores", default=SCORES_FILE, help="stored importance scores (no attribution is run)")
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--log", default=LOG_FILE)
    p.add_argument("--resume", action="store_true", help="skip configs with status=completed in --output")
    p.add_argument("--save-window-nll", action="store_true",
                   help="(D-2) also store WikiText-2 window NLLs for every run (rep['ppl']['wikitext2'], E-5 schema; for paired "
                        "bootstrap; +~70 s/run/dataset). Default OFF = byte-identical output")
    p.add_argument("--window-nll-datasets", default="wikitext2,c4",
                   help="datasets measured when --save-window-nll is on (two domains; C4 = the fixed C4 subset, "
                        "sha1-verified). Can be narrowed to 'wikitext2' when offline")
    p.add_argument("--preflight", action="store_true", help="check CUDA/hqq/budget/MMLU without loading the model, then exit")
    p.add_argument("--dry-run", action="store_true", help="mini Mistral + hqq (fake group quantization if missing) + fake perplexity/MMLU, CPU")
    args = p.parse_args(argv)
    args.mmlu = not args.no_mmlu
    if args.dry_run:  # never touch the real results/ or the tracked log files
        if args.output == OUTPUT_FILE:
            args.output = os.path.join(DRYRUN_DIR, "mixed_bits_gun11_dry.json")
        if args.log == LOG_FILE:
            args.log = os.path.join(DRYRUN_DIR, "log_gun11_bits_dry.txt")
    return args


# --------------------------------------------------------------------------- #
# Single config run
# --------------------------------------------------------------------------- #
def plan_for(name: str, seed: int, module_scores: Dict[str, float], model: Optional[nn.Module]) -> Tuple[Dict[str, int], str]:
    spec = CONFIGS[name]
    sel = spec["selector"]
    if sel == "uniform":
        return uniform_bits(module_scores, spec["nbits"]), "uniform"
    if sel == "xai":
        return assign_bits(module_scores, spec["budget"]), "xai (stored scores)"
    if sel == "random":
        return assign_bits(module_scores, spec["budget"], random_seed=seed), f"random (seed {seed})"
    if sel == "magnitude":
        if model is None:
            raise ValueError("the magnitude plan requires a loaded model")
        return assign_bits(module_magnitude_scores(model), spec["budget"]), "magnitude (loaded FP16 model)"
    raise ValueError(f"unknown selector: {sel}")


def tasks_apply(args: argparse.Namespace, name: str, seed: int) -> bool:
    """--with-tasks scope: uniform + xai always; the random control only for its first seed (args.seed = 42); magnitude never."""
    sel = CONFIGS[name]["selector"]
    return bool(getattr(args, "with_tasks", False)) and (sel in ("uniform", "xai") or (sel == "random" and seed == args.seed))


def measure(name: str, seed: int, ctx: Dict[str, Any], rep: Dict[str, Any], store: ResultStore, log: Logger, box: Dict[str, Any]) -> None:
    args, dry = ctx["args"], ctx["dry"]
    model, tokenizer = box["model"], box["tokenizer"]
    rep["model_bytes_before"] = module_bytes(model)
    params = module_param_counts(model)
    bit_plan, source = plan_for(name, seed, ctx["module_scores"], model)
    xai_plan = assign_bits(ctx["module_scores"], CONFIGS[name]["budget"]) if CONFIGS[name]["selector"] != "uniform" else None
    rep.update({"bit_plan": bit_plan, "bit_counts": bit_counts(bit_plan), "plan_source": source,
                "avg_bits_nominal": average_bits(bit_plan, params),
                "same_bits_as_xai": None if xai_plan is None else sum(1 for k in bit_plan if bit_plan[k] == xai_plan[k])})
    log(f"{name}: tier counts {rep['bit_counts']}, nominal avg. {rep['avg_bits_nominal']:.4f} bits/param (budget {CONFIGS[name]['budget']})")
    t0 = time.time()
    rep["quantization"] = apply_hqq(model, bit_plan, fake=ctx["fake"])
    rep["apply_seconds"] = time.time() - t0
    size = hqq_model_bytes(model)
    rep["model_bytes_after"] = size
    rep["measured_model_bytes_ratio"] = size["bytes"] / rep["model_bytes_before"]["bytes"]
    rep["avg_bits_effective"] = 8.0 * size["quant_bytes"] / sum(params.values())
    log(f"{name}: HQQ applied ({rep['apply_seconds']:.1f} s) {rep['quantization']['modules_by_bits']}; size "
        f"{rep['model_bytes_before']['gb']:.3f} -> {size['gb']:.3f} GB; effective avg. {rep['avg_bits_effective']:.3f} bits/param")
    store.save(f"{name} applied")
    t0 = time.time()
    if dry:
        ppl = dry.perplexity(model)
    else:
        ppl = ctx["compute_perplexity"](model, tokenizer, ctx["text"], next(model.parameters()).device)
    rep.update({"perplexity": ppl, "perplexity_seconds": time.time() - t0, "perplexity_finite": math.isfinite(ppl)})
    log(f"{name}: perplexity = {ppl:.4f} ({rep['perplexity_seconds']:.1f} s)")
    store.save(f"{name} perplexity")
    if getattr(args, "save_window_nll", False):  # opt-in; does not affect the rep["perplexity"] measurement
        import eval_ppl_windows as epw

        w = epw.record_window_nll(rep, model, tokenizer, ctx["window_nll_texts"], bool(dry))
        log(f"{name}: window NLL recorded: " + ", ".join(f"{ds} ppl {v['perplexity']:.4f} ({v['n_windows']} windows, {v['seconds']:.1f} s)"
                                                            for ds, v in w.items()))
        store.save(f"{name} window NLL")
    if args.mmlu:
        import eval_mmlu  # lazy: never loaded with --no-mmlu

        mlog = lambda m: log(m, tag="MMLU")  # noqa: E731
        if dry:
            rep["mmlu"] = eval_mmlu.evaluate_mmlu_dry(model, args.seed, log=mlog, record_questions=True)
        else:
            rep["mmlu"] = eval_mmlu.evaluate_mmlu(model, tokenizer, subset=ctx["mmlu_subset"], prompt_style="plain",
                                                  batch_size=args.mmlu_batch_size, log=mlog, record_questions=True)
        rep["mmlu_subset_acc"] = rep["mmlu"]["mmlu_subset_acc"]
    if tasks_apply(args, name, seed):  # below 4 bits MMLU can collapse onto a single answer letter -> second task family
        import eval_tasks

        tlog = lambda m: log(m, tag="TASK")  # noqa: E731
        if dry:
            rep["tasks"] = eval_tasks.evaluate_tasks_dry(model, args.seed, log=tlog)
        else:
            rep["tasks"] = eval_tasks.evaluate_tasks(model, tokenizer, subsets=ctx["task_subsets"], batch_size=args.tasks_batch_size, log=tlog)
        rep["tasks_summary"] = eval_tasks.summarize_tasks(rep["tasks"])
        log(f"{name}: tasks = " + ", ".join(f"{t} acc {v['acc']:.3f} / acc_norm {v['acc_norm']:.3f}" for t, v in rep["tasks_summary"].items()))


def run_one(name: str, seed: int, ctx: Dict[str, Any], rep: Dict[str, Any], store: ResultStore, log: Logger) -> None:
    """Clean load -> HQQ -> measurement -> FULL release (same memory pattern as run_iterative_pruning.py)."""
    dry: Optional[DryRun] = ctx["dry"]
    rep.update({"status": "running", "seed": seed, "started_at": datetime.now().isoformat(timespec="seconds")})
    store.save(f"{name} started")
    t0 = time.time()
    set_seed(seed)
    rep["cuda_allocated_before_load_gb"] = ensure_clean_gpu(log, name)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    box: Dict[str, Any] = {}
    try:
        box["tokenizer"], box["model"], rep["model_load_seconds"] = dry.load_model(log) if dry else load_model(log)
        measure(name, seed, ctx, rep, store, log, box)
        rep["status"] = "completed"
    finally:
        rep["peak_vram_gb"] = cuda_gb("peak")
        box.clear()
        rep["cuda_allocated_after_free_gb"] = release_gpu()
        rep["seconds"] = time.time() - t0
        log(f"{name}: model released from memory; peak VRAM={rep['peak_vram_gb']} GB, {rep['seconds']:.0f} s")
        store.save(f"{name} finished")


def summarize_config(cfg: Dict[str, Any], fp16_ppl: Optional[float], fp16_acc: Optional[float]) -> None:
    reps = [r for r in cfg["repeats"] if r.get("status") == "completed"]
    ppls = [r["perplexity"] for r in reps]
    cfg["n_completed_repeats"] = len(reps)
    cfg["perplexity_values"] = ppls
    cfg["perplexity_mean"] = float(np.mean(ppls)) if ppls else None
    cfg["perplexity_std"] = float(np.std(ppls, ddof=1)) if len(ppls) > 1 else (0.0 if ppls else None)
    if not reps:
        return
    r0 = reps[0]
    cfg.update({"bit_counts": r0["bit_counts"], "avg_bits_nominal": r0["avg_bits_nominal"], "avg_bits_effective": r0["avg_bits_effective"],
                "model_gb": r0["model_bytes_after"]["gb"], "model_gb_values": [r["model_bytes_after"]["gb"] for r in reps],
                "measured_model_bytes_ratio": r0["measured_model_bytes_ratio"], "seconds": sum(r["seconds"] for r in reps),
                "same_bits_as_xai": [r["same_bits_as_xai"] for r in reps]})
    with_tasks = [r for r in reps if "tasks_summary" in r]
    if with_tasks:  # only for configs in --with-tasks scope (a single seed for random)
        cfg["tasks_summary"] = with_tasks[0]["tasks_summary"]
        cfg["tasks_seed"] = with_tasks[0]["seed"]
    peaks = [r["peak_vram_gb"] for r in reps if r.get("peak_vram_gb") is not None]
    cfg["peak_vram_gb"] = max(peaks) if peaks else None
    accs = [r["mmlu_subset_acc"] for r in reps if "mmlu_subset_acc" in r]
    if accs:
        cfg["mmlu_subset_acc_values"] = accs
        cfg["mmlu_subset_acc_mean"] = float(np.mean(accs))
        if fp16_acc is not None:
            cfg["mmlu_delta_vs_fp16"] = cfg["mmlu_subset_acc_mean"] - fp16_acc
    if fp16_ppl is not None and cfg["perplexity_mean"] is not None:
        cfg["delta_vs_fp16"] = cfg["perplexity_mean"] - fp16_ppl
        cfg["ratio_vs_fp16"] = cfg["perplexity_mean"] / fp16_ppl


def comparison(configs: Dict[str, Any]) -> Dict[str, Any]:
    """Same-budget comparison: control - xAI (ppl, MMLU); uniform 3-bit for B=3.0, uniform 3-bit and 4-bit corners for B=3.5."""
    def val(n: str, k: str) -> Optional[float]:
        return (configs.get(n) or {}).get(k)

    out: Dict[str, Any] = {}
    for b in BUDGETS:
        bk = budget_key(b)
        xai = f"hqq_xai_b{bk}"
        others = [f"hqq_random_b{bk}", f"hqq_magnitude_b{bk}", "hqq_uniform_3bit"] + (["hqq_uniform_4bit"] if b > 3.0 else [])
        for n in others:
            for metric, key in (("ppl", "perplexity_mean"), ("mmlu", "mmlu_subset_acc_mean")):
                if val(n, key) is not None and val(xai, key) is not None:
                    out[f"b{bk}_{n}_minus_xai_{metric}"] = val(n, key) - val(xai, key)
    return out


def format_summary_table(configs: Dict[str, Any]) -> str:
    def f(v: Any, spec: str) -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    lines = [f"{'config':<24}{'ppl':>10}{'±std':>9}{'Δfp16':>10}{'mmlu':>7}{'bit nom.':>10}{'bit eff.':>11}{'GB':>8}{'VRAM':>7}{'s':>7}  status"]
    for name, c in configs.items():
        lines.append(f"{name:<24}{f(c.get('perplexity_mean'), '.4f'):>10}{f(c.get('perplexity_std'), '.4f'):>9}{f(c.get('delta_vs_fp16'), '+.4f'):>10}"
                     f"{f(c.get('mmlu_subset_acc_mean'), '.3f'):>7}{f(c.get('avg_bits_nominal'), '.3f'):>10}{f(c.get('avg_bits_effective'), '.3f'):>11}"
                     f"{f(c.get('model_gb'), '.3f'):>8}{f(c.get('peak_vram_gb'), '.1f'):>7}{f(c.get('seconds'), '.0f'):>7}  {c.get('status')}")
    return "\n".join(lines)


def hqq_smoke(device: str) -> Dict[str, Any]:
    """Preflight: convert a 64x64 Linear to HQQ at every bit width and check that the forward pass is finite (hqq + torch path)."""
    out: Dict[str, Any] = {}
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    for nbits in BIT_TIERS:
        torch.manual_seed(0)
        lin = nn.Linear(64, 64, bias=False).to(device=device, dtype=dtype)
        x = torch.randn(2, 64, device=device, dtype=dtype)
        ref = lin(x).float()
        q = _hqq_linear(lin, nbits, GROUP_SIZE, AXIS, dtype)
        y = q(x).float()
        out[str(nbits)] = {"finite": bool(torch.isfinite(y).all()), "rel_err": float((y - ref).abs().mean() / ref.abs().mean())}
    return out


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log = Logger(args.log, to_file=should_log_to_file(args, LOG_FILE))
    store = ResultStore(args.output, log)
    t_start = time.time()
    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in names if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown config(s): {unknown}; valid: {ALL_CONFIGS}")
    dry: Optional[DryRun] = DryRun(args.seed) if args.dry_run else None
    scores = dry.scores if dry else load_scores(args.scores)
    module_scores = module_importance_scores(scores)
    fake = bool(dry) and not hqq_available()
    n_runs = sum(args.n_repeats if CONFIGS[n]["selector"] in STOCHASTIC_SELECTORS else 1 for n in names)
    n_task_runs = sum(1 for n in names if CONFIGS[n]["selector"] in ("uniform", "xai", "random")) if args.with_tasks else 0
    est_min = n_runs * (0.25 + 1.5 + 2.5 + (4.0 if args.mmlu else 0.0)) + 3.5 * n_task_runs
    log(f"===== D-2 (HQQ bit allocation) starting: configs={names} seed={args.seed} n_repeats={args.n_repeats} mmlu={args.mmlu} "
        f"dry_run={args.dry_run} hqq={hqq_version()} =====", tag="D2")

    n_kind = {kind: len(v) for kind, v in compressor._module_pools(module_scores).items()}
    budgets_ref: Dict[str, Any] = {}
    for b in BUDGETS:
        counts = {kind: {str(k): v for k, v in solve_tier_counts(n, b).items()} for kind, n in n_kind.items()}
        planned = {kind: sum(int(k) * v for k, v in c.items()) / n_kind[kind] for kind, c in counts.items()}
        budgets_ref[budget_key(b)] = {"template_shares": {str(k): v for k, v in template_shares(b).items()}, "tier_counts": counts,
                                      "avg_bits_per_kind": planned}
        log(f"budget B={budget_key(b)}: tier counts per kind (8/4/3/2 bits) {counts} -> avg. bits {planned}", tag="D2")
    log(f"Time estimate: {n_runs} runs × ~{est_min / max(n_runs, 1):.1f} min ≈ {est_min:.0f} min", tag="D2")

    baselines = None
    if not dry and os.path.exists(BASELINES_FILE):
        with open(BASELINES_FILE, "r", encoding="utf-8") as f:
            baselines = (json.load(f).get("summary") or {}).get("configs")
    fp16_ppl = ((baselines or {}).get("fp16") or {}).get("perplexity")
    fp16_acc = ((baselines or {}).get("fp16") or {}).get("mmlu_subset_acc")

    previous = None
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            previous = json.load(f)
        log(f"--resume: read {args.output}; completed configs will be skipped", tag="D2")

    store.data = {
        "run": {"started_at": datetime.now().isoformat(timespec="seconds"), "status": "running", "args": vars(args),
                "model": "mini (dry-run)" if dry else MODEL_NAME, "env": env_info(), "hqq_version": hqq_version(),
                "hqq_backend": "fake (hqq not installed; dry-run only)" if fake else "hqq (HQQLinear, default PYTORCH backend)",
                "config_order": names, "time_estimate": {"n_runs": n_runs, "total_minutes": est_min}},
        "reference": {
            "scores_file": None if dry else args.scores,
            "module_score": "attn = LOWER median of the layer's head scores, MLP = MLP score (compressor.module_importance_scores); "
                            "ranking within each kind; no attribution is run",
            "hqq": {"group_size": GROUP_SIZE, "axis": AXIS, "bit_tiers": list(BIT_TIERS), "quant_zero": False, "quant_scale": False,
                    "pinned_version": HQQ_PIN, "untouched": "lm_head, embeddings, norms (FP16)"},
            "budget_definition": "B = parameter-weighted NOMINAL average bits (nbits) over the decoder Linear modules; the scale/zero-point "
                                 "overhead and 3-bit packing (10 values per int32 = 3.2 bits) are NOT part of the budget and are reported in avg_bits_effective",
            "budgets": budgets_ref, "n_modules": n_kind,
            "size_note": "model_gb = parameters + buffers + HQQ meta (scale + zero point) tensors: W_q + scale + zero are counted; for the bnb "
                         "rows (nf4_uniform, XAI-JQP) quant_state is not counted, so the comparison is slightly biased in favour of bnb",
            "baselines_gun7": baselines,
            "peak_vram_note": "reset_peak_memory_stats per config; peak = FP16 load + per-module quantization temporaries",
        },
        "configs": {n: {"description": CONFIGS[n]["description"], "label": CONFIGS[n]["label"], "selector": CONFIGS[n]["selector"],
                        "budget_bits": CONFIGS[n]["budget"], "stochastic": CONFIGS[n]["selector"] in STOCHASTIC_SELECTORS,
                        "status": "pending", "repeats": []} for n in names},
        "comparison": {},
    }

    if args.preflight:
        problems: List[str] = []
        if not dry and not torch.cuda.is_available():
            problems.append("no CUDA")
        if not hqq_available():
            if not dry:
                problems.append(f"hqq is not installed (pip install hqq=={HQQ_PIN})")
            else:
                log("WARNING: hqq is not installed; the dry run uses fake group quantization", tag="D2")
        else:
            if hqq_version() != HQQ_PIN:
                log(f"WARNING: hqq {hqq_version()} is installed; the verified version is {HQQ_PIN}", tag="D2")
            try:
                smoke = hqq_smoke("cuda" if torch.cuda.is_available() else "cpu")
                log(f"hqq {hqq_version()} smoke test (64×64 Linear): {smoke}", tag="D2")
                if not all(v["finite"] for v in smoke.values()):
                    problems.append(f"hqq smoke test is not finite: {smoke}")
            except Exception as e:
                problems.append(f"hqq smoke test raised an error: {e!r}")
        if not dry and args.mmlu:
            try:
                import eval_mmlu

                log(f"MMLU subset ready: {len(eval_mmlu.load_or_build_subset(log=lambda m: log(m, tag='MMLU'))['records'])} questions", tag="MMLU")
            except Exception as e:
                problems.append(f"could not load the MMLU subset: {e!r}")
        if not dry and args.with_tasks:
            try:
                import eval_tasks

                for t in eval_tasks.ALL_TASKS:
                    log(f"{t} subset ready: {len(eval_tasks.load_or_build_task_subset(t, log=lambda m: log(m, tag='TASK'))['records'])} examples", tag="TASK")
            except Exception as e:
                problems.append(f"could not load the HellaSwag/ARC subsets: {e!r}")
        if args.save_window_nll:  # dataset names + C4 subset (sha1) are validated BEFORE the run
            import eval_ppl_windows as epw

            wn = epw.parse_ppl_datasets(args.window_nll_datasets)
            log(f"Window NLL datasets: {wn} (+~70 s/run/dataset)", tag="D2")
            if not dry and "c4" in wn:
                try:
                    _, wn_meta = epw.load_ppl_texts(["c4"])
                    log(f"C4 window NLL subset ready: {wn_meta['c4_ppl_subset']['n_docs']} documents, sha1 verified", tag="D2")
                except Exception as e:
                    problems.append(f"could not prepare the C4 window NLL subset: {e!r}")
        if problems:
            log("Preflight FAILED (model not loaded): " + "; ".join(problems), tag="ERR")
            return 1
        log("Preflight completed (model not loaded). Nothing written to the result file.", tag="D2")
        return 0
    if not dry and not hqq_available():
        raise SystemExit(f"hqq is not installed: pip install hqq=={HQQ_PIN}")
    if not torch.cuda.is_available() and not dry:
        log("WARNING: no CUDA — a 7B model is impractical on CPU. Run on a GPU machine (or use --dry-run).", tag="D2")
    store.save("start")

    ctx: Dict[str, Any] = {"args": args, "dry": dry, "module_scores": module_scores, "fake": fake}
    exit_code = 0
    try:
        if not dry:
            log("Loading the WikiText-2 test set...", tag="D2")
            ctx["text"] = load_wikitext_text()
            ctx["compute_perplexity"], max_len, stride = perplexity_tools()
            store.data["reference"]["perplexity_settings"] = {"max_length": max_len, "stride": stride, "dataset": "wikitext-2-raw-v1 (test split)"}
            if args.mmlu:
                import eval_mmlu

                ctx["mmlu_subset"] = eval_mmlu.load_or_build_subset(log=lambda m: log(m, tag="MMLU"))
            if args.with_tasks:
                import eval_tasks

                ctx["task_subsets"] = {t: eval_tasks.load_or_build_task_subset(t, log=lambda m: log(m, tag="TASK")) for t in eval_tasks.ALL_TASKS}
        if args.save_window_nll:  # two-domain window NLL; the WikiText text was loaded above, the C4 subset is sha1-verified
            import eval_ppl_windows as epw

            wn = epw.parse_ppl_datasets(args.window_nll_datasets)
            if dry:
                ctx["window_nll_texts"] = {d: None for d in wn}
            else:
                ctx["window_nll_texts"], wn_meta = epw.load_ppl_texts(wn, ctx["text"])
                store.data["reference"]["window_nll"] = {"datasets": wn, **wn_meta}
                log(f"Window NLL datasets ready: {wn}", tag="D2")
        for ci, name in enumerate(names, 1):
            cfg = store.data["configs"][name]
            prev = ((previous or {}).get("configs") or {}).get(name)
            if prev and prev.get("status") == "completed":
                store.data["configs"][name] = {**prev, "resumed": True}
                log(f"CONFIG {ci}/{len(names)} {name}: --resume, taken from the previous run (ppl={prev.get('perplexity_mean')})", tag="D2")
                continue
            n_rep = args.n_repeats if cfg["stochastic"] else 1
            cfg.update({"status": "running", "n_repeats": n_rep})
            log(f"===== CONFIG {ci}/{len(names)}: {name} ({n_rep} repeats) — {cfg['description']} =====", tag="D2")
            try:
                for r in range(n_rep):
                    rep: Dict[str, Any] = {}
                    cfg["repeats"].append(rep)
                    run_one(name, args.seed + r, ctx, rep, store, log)
                summarize_config(cfg, fp16_ppl, fp16_acc)
                cfg["status"] = "completed"
                log(f"CONFIG {ci}/{len(names)} DONE: {name} ppl={cfg['perplexity_mean']:.4f} ± {cfg['perplexity_std']:.4f} "
                    f"mmlu={cfg.get('mmlu_subset_acc_mean')} GB={cfg['model_gb']:.3f} {cfg['seconds']:.0f} s", tag="D2")
            except Exception as e:  # a failure in one config must not abort the others
                summarize_config(cfg, fp16_ppl, fp16_acc)
                cfg.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
                log(f"CONFIG {name} ERROR: {e!r} — moving on to the next config", tag="ERR")
                log(cfg["traceback"], tag="ERR")
                exit_code = 1
            store.data["comparison"] = comparison(store.data["configs"])
            store.save(f"{name} summary")
        store.data["run"]["status"] = "completed" if exit_code == 0 else "completed_with_failures"
    except BaseException as e:  # including KeyboardInterrupt / SystemExit: keep partial results
        store.data["run"].update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
        log(f"ERROR: {e!r} — writing partial results to disk", tag="ERR")
        log(store.data["run"]["traceback"], tag="ERR")
        exit_code = 1
    finally:
        for cfg in store.data["configs"].values():
            if cfg.get("repeats") and cfg.get("status") not in ("completed", "failed"):
                summarize_config(cfg, fp16_ppl, fp16_acc)
        store.data["comparison"] = comparison(store.data["configs"])
        store.data["run"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        store.data["run"]["total_seconds"] = time.time() - t_start
        store.save("final")
        log("SUMMARY TABLE:\n" + format_summary_table(store.data["configs"]), tag="D2")
        log(f"===== Done: status={store.data['run']['status']}, total {store.data['run']['total_seconds']:.1f} s =====", tag="D2")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
