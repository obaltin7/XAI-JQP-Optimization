"""
End-to-end XAI-JQP prototype (GPU required).

Applies the n_tiers=3 tier assignment (results/tier_analysis_gun4.json) to
Mistral-7B and compares WikiText-2 perplexity against the FP16 baseline
(results/baseline_fp16.json, 5.2200). Two stages:

  smoke : small-scale safety check. The planned head pruning plus attention/MLP
          INT4 quantization is applied to a single layer (default layer_0); the
          bitsandbytes version/API is logged, and the forward pass is checked for
          NaN/Inf and compared with the FP16 reference logits. The model is then
          released so that the full stage starts from a clean load.
  full  : full end-to-end prototype. FP16 perplexity is re-measured in the same
          environment (default), the plan is applied to the whole model
          (pruning -> quantization), compressed perplexity is measured and
          generation samples (before/after) are recorded.

Prune ratio: the lowest 20% of blocks (here attention heads) by within-type
percentile are pruned. The realised parameter/size ratio differs and is reported
separately in the result JSON ("nominal_block_prune_ratio" vs
"actual_parameter_prune_ratio").

The result file (results/e2e_prototype_gun5.json) is written atomically after each
major step and in a try/finally on error or interruption; timestamped lines are
appended to log_gun5.txt.

Usage:
    python experiments/run_e2e_pipeline.py                    # both stages
    python experiments/run_e2e_pipeline.py --preflight        # input/plan/bnb check without loading the model
    python experiments/run_e2e_pipeline.py --stage smoke      # single-layer check only
    python experiments/run_e2e_pipeline.py --stage full       # full prototype only
"""

# HF_HOME must be set BEFORE transformers/datasets are imported; otherwise the
# model and dataset caches go to the pod's non-persistent disk (/root/.cache) and
# Mistral-7B (~14GB) is re-downloaded after every restart. /workspace is the
# persistent volume.
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import gc
import inspect
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    load_scores,
    parse_block_key,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_FILE = os.path.join("results", "e2e_prototype_gun5.json")
TIERS_FILE = os.path.join("results", "tier_analysis_gun4.json")
SCORES_FILE = os.path.join("results", "importance_scores_gun3.json")
BASELINE_FILE = os.path.join("results", "baseline_fp16.json")
LOG_FILE = "log_gun5.txt"

SMOKE_TEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
    "France. It is named after the engineer Gustave Eiffel, whose company designed and "
    "built the tower from 1887 to 1889. Locally nicknamed the Iron Lady, it was "
    "constructed as the centrepiece of the 1889 World's Fair."
)
GEN_PROMPTS = [
    "The main causes of the French Revolution were",
    "In computer science, a hash table is",
]


# --------------------------------------------------------------------------- #
# Logging and partial-result store
# --------------------------------------------------------------------------- #
class Logger:
    """Writes timestamped lines to stdout and appends them to a log file."""

    def __init__(self, path: str, to_file: bool = True):
        self.path = path
        self.to_file = to_file  # False = stdout only, so dry-run/preflight probes do not pollute tracked logs

    def __call__(self, msg: str, tag: str = "E2E") -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{tag}] {msg}"
        print(line, flush=True)
        if not self.to_file:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def should_log_to_file(args: Any, default_log: str) -> bool:
    """
    In --dry-run / --preflight mode, nothing is written to the DEFAULT log path (a tracked run log such as
    log_gun6.txt); lines go to stdout only. An explicit --log, or a script redirecting its dry-run log to
    dryrun_out/, is still written. Always True for normal runs.
    """
    probing = bool(getattr(args, "dry_run", False) or getattr(args, "preflight", False))
    return not (probing and getattr(args, "log", None) == default_log)


def _json_default(o: Any):
    if isinstance(o, (torch.dtype, torch.device)):
        return str(o)
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


class ResultStore:
    """
    Holds the result dictionary and writes it ATOMICALLY with `save()` (.tmp, then
    os.replace), so an interruption mid-write never corrupts the previous valid file.
    Called after every intermediate step.
    """

    def __init__(self, path: str, log: Logger):
        self.path = path
        self.log = log
        self.data: Dict[str, Any] = {}

    def save(self, note: str = "") -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, default=_json_default)
        os.replace(tmp, self.path)
        self.log(f"Result file updated ({note}): {self.path}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end XAI-JQP prototype (20%, n_tiers=3)")
    p.add_argument("--stage", choices=("smoke", "full", "both"), default="both",
                   help="smoke: single-layer safety check; full: full prototype; both: smoke then full")
    p.add_argument("--preflight", action="store_true",
                   help="check inputs (tiers/plan/bnb/CUDA) without loading the model, then exit; no GPU needed")
    p.add_argument("--skip-baseline", action="store_true",
                   help="do NOT re-measure FP16 perplexity in the full stage (default: re-measure)")
    p.add_argument("--tiers", default=TIERS_FILE, help="tier assignment JSON (the 'tiers' dictionary is read from it)")
    p.add_argument("--scores", default=SCORES_FILE, help="importance scores (for reporting and consistency check)")
    p.add_argument("--baseline", default=BASELINE_FILE, help="FP16 baseline JSON (reference perplexity)")
    p.add_argument("--output", default=OUTPUT_FILE, help="result JSON path")
    p.add_argument("--log", default=LOG_FILE, help="timestamped log file")
    p.add_argument("--smoke-layer", type=int, default=0, help="layer used by the smoke stage")
    p.add_argument("--gen-tokens", type=int, default=48, help="generation sample length (0 = no generation)")
    return p.parse_args(argv)


def env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import bitsandbytes as bnb

        info["bitsandbytes"] = getattr(bnb, "__version__", "?")
        info["bnb_Params4bit_signature"] = str(inspect.signature(bnb.nn.Params4bit.__new__))
        info["bnb_Int8Params_signature"] = str(inspect.signature(bnb.nn.Int8Params.__new__))
        info["bnb_Linear4bit_signature"] = str(inspect.signature(bnb.nn.Linear4bit.__init__))
    except Exception as e:  # pragma: no cover
        info["bitsandbytes"] = None
        info["bitsandbytes_error"] = repr(e)
    return info


def cuda_gb(kind: str = "allocated") -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    fn = {"allocated": torch.cuda.memory_allocated, "peak": torch.cuda.max_memory_allocated}[kind]
    return fn() / 1e9


def module_bytes(module: nn.Module) -> Dict[str, Any]:
    """Parameter + buffer bytes (packed uint8 quantized tensors are counted at their real size)."""
    total = 0
    by_dtype: Dict[str, int] = {}
    for t in list(module.parameters()) + list(module.buffers()):
        b = t.numel() * t.element_size()
        total += b
        by_dtype[str(t.dtype)] = by_dtype.get(str(t.dtype), 0) + b
    return {"bytes": total, "gb": total / 1e9, "by_dtype_bytes": by_dtype}


def finite_stats(t: torch.Tensor) -> Dict[str, Any]:
    return {
        "has_nan": bool(torch.isnan(t).any()),
        "has_inf": bool(torch.isinf(t).any()),
        "abs_max": float(t.abs().max()) if t.numel() else None,
    }


@torch.no_grad()
def forward_logits(model: nn.Module, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return out.logits.float().cpu()


def compare_logits(ref: torch.Tensor, new: torch.Tensor, input_ids: torch.Tensor) -> Dict[str, Any]:
    """Difference, top-1 agreement and nll/ppl between reference (untouched FP16) and new logits."""
    ids = input_ids.cpu()
    diff = (new - ref).abs()
    top1_ref, top1_new = ref.argmax(-1), new.argmax(-1)
    nll_ref = F.cross_entropy(ref[0, :-1], ids[0, 1:]).item()
    nll_new = F.cross_entropy(new[0, :-1], ids[0, 1:]).item()
    return {
        **finite_stats(new),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "top1_agreement": float((top1_ref == top1_new).float().mean()),
        "nll_ref": nll_ref,
        "nll_new": nll_new,
        "ppl_ref": math.exp(nll_ref),
        "ppl_new": math.exp(nll_new) if math.isfinite(nll_new) else None,
    }


def load_model(log: Logger, dtype: torch.dtype = torch.float16, model_name: str = MODEL_NAME):
    """model_name allows a second model (e.g. Qwen/Qwen2.5-7B-Instruct); defaults to Mistral-7B."""
    log(f"Loading model: {model_name} (dtype={dtype})")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    model.eval()
    load_s = time.time() - t0
    log(f"Model loaded ({load_s:.1f} s), device={next(model.parameters()).device}, CUDA allocated={cuda_gb()} GB")
    return tokenizer, model, load_s


def free_model(model: nn.Module, log: Logger) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Model released from memory; CUDA allocated={cuda_gb()} GB")


def load_wikitext_text() -> str:
    from datasets import load_dataset  # lazy import: local preflight must not require datasets

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(dataset["text"])


def perplexity_tools():
    """Lazily imports baseline_eval (it imports datasets at module level); same settings/code as the FP16 baseline."""
    from tools.evaluation.baseline_eval import MAX_LENGTH, STRIDE, compute_perplexity

    return compute_perplexity, MAX_LENGTH, STRIDE


def load_tiers_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "tiers" not in data or not isinstance(data["tiers"], dict):
        raise ValueError(f"{path}: 'tiers' dictionary not found")
    return data


def plan_summary(plan: Dict[int, LayerPlan]) -> List[Dict[str, Any]]:
    return [
        {"layer": i, "n_pruned_heads": len(p.pruned_heads), "pruned_heads": list(p.pruned_heads),
         "attn_quant": p.attn_quant, "mlp_quant": p.mlp_quant, "mlp_pruned": p.mlp_pruned}
        for i, p in sorted(plan.items())
    ]


def tier_counts(tiers: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {"attn": {}, "mlp": {}}
    for name, label in tiers.items():
        kind = parse_block_key(name).kind
        out[kind][label] = out[kind].get(label, 0) + 1
    return out


def layer_module_types(layer: nn.Module) -> Dict[str, str]:
    names = {"self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"), "mlp": ("gate_proj", "up_proj", "down_proj")}
    return {f"{parent}.{n}": type(getattr(getattr(layer, parent), n)).__name__ for parent, ns in names.items() for n in ns}


@torch.no_grad()
def generate_samples(model: nn.Module, tokenizer, prompts: List[str], max_new_tokens: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if max_new_tokens <= 0:
        return out
    device = next(model.parameters()).device
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        out.append({"prompt": prompt,
                    "completion": tokenizer.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)})
    return out


# --------------------------------------------------------------------------- #
# bitsandbytes compatibility fallback
# --------------------------------------------------------------------------- #
def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drops keyword arguments that are not in fn's signature (handles version differences)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _quantize_linear_compat(linear: nn.Linear, tier: str, compute_dtype: torch.dtype = torch.float16) -> nn.Module:
    """
    Version-tolerant fallback for compressor._quantize_linear (used only if the primary
    path fails). Differences:
      * Linear4bit/Params4bit/Int8Params receive only the arguments present in their signatures.
      * Quantization is triggered at parameter level (Params4bit(...).to(device)), as in the
        HF integration, instead of a module-level .to(device), and then assigned to the module.
    """
    import bitsandbytes as bnb

    device = linear.weight.device
    has_bias = linear.bias is not None
    w = linear.weight.data.to(compute_dtype).contiguous()
    if tier == "int4":
        new = bnb.nn.Linear4bit(linear.in_features, linear.out_features, bias=has_bias,
                                **_filter_kwargs(bnb.nn.Linear4bit.__init__,
                                                 dict(compute_dtype=compute_dtype, compress_statistics=True,
                                                      quant_type="nf4")))
        w_q = bnb.nn.Params4bit(w.cpu(), **_filter_kwargs(bnb.nn.Params4bit.__new__,
                                                          dict(requires_grad=False, quant_type="nf4",
                                                               compress_statistics=True, module=new))).to(device)
    elif tier == "int8":
        new = bnb.nn.Linear8bitLt(linear.in_features, linear.out_features, bias=has_bias,
                                  **_filter_kwargs(bnb.nn.Linear8bitLt.__init__,
                                                   dict(has_fp16_weights=False, threshold=6.0)))
        w_q = bnb.nn.Int8Params(w.cpu(), **_filter_kwargs(bnb.nn.Int8Params.__new__,
                                                          dict(requires_grad=False, has_fp16_weights=False))).to(device)
    else:
        raise ValueError(f"Tier cannot be quantized: {tier!r}")
    new.weight = w_q
    if has_bias:
        new.bias = nn.Parameter(linear.bias.data.to(compute_dtype).to(device), requires_grad=False)
    return new


def quantize_with_fallback(model: nn.Module, plan: Dict[int, LayerPlan], log: Logger,
                           store: ResultStore, key: str) -> Dict[str, int]:
    """
    Tries compressor.apply_quantization first. On failure, the error and traceback are
    logged and written to the result file, `_quantize_linear` is replaced by the fallback
    and the same plan is retried (already converted modules are skipped).
    """
    try:
        counts = apply_quantization(model, plan, verbose=True)
        store.data[key]["quantization_path"] = "primary (compressor._quantize_linear)"
        return counts
    except Exception as e:
        tb = traceback.format_exc()
        log(f"WARNING: primary quantization path failed: {e!r}", tag="BNB")
        log(tb, tag="BNB")
        store.data[key]["quantization_primary_error"] = repr(e)
        store.data[key]["quantization_primary_traceback"] = tb
        store.save("primary quantization error")
    log("Trying fallback path: version-tolerant _quantize_linear_compat", tag="BNB")
    compressor._quantize_linear = _quantize_linear_compat  # type: ignore[attr-defined]
    counts = apply_quantization(model, plan, verbose=True)
    store.data[key]["quantization_path"] = "fallback (_quantize_linear_compat)"
    return counts


# --------------------------------------------------------------------------- #
# Stage: smoke (single layer)
# --------------------------------------------------------------------------- #
def run_smoke(args, plan: Dict[int, LayerPlan], store: ResultStore, log: Logger) -> None:
    L = args.smoke_layer
    if L not in plan:
        raise ValueError(f"layer_{L} is not in the plan")
    res: Dict[str, Any] = store.data.setdefault("smoke", {})
    res.update({"status": "running", "layer": L, "started_at": datetime.now().isoformat(timespec="seconds")})
    store.save("smoke started")
    t_stage = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer, model, load_s = load_model(log)
    res["model_load_seconds"] = load_s
    try:
        layers = compressor._decoder_layers(model)
        device = next(model.parameters()).device
        enc = tokenizer(SMOKE_TEXT, return_tensors="pt").to(device)
        res["text_tokens"] = int(enc["input_ids"].shape[1])
        ref = forward_logits(model, enc)
        res["reference"] = finite_stats(ref)
        res["reference"]["ppl_text"] = compare_logits(ref, ref, enc["input_ids"])["ppl_ref"]
        log(f"SMOKE: reference FP16 forward pass OK ({res['text_tokens']} tokens, ppl={res['reference']['ppl_text']:.3f})")

        # 1) planned head pruning (this layer only)
        p = plan[L]
        prune_plan = {L: LayerPlan(L, pruned_heads=list(p.pruned_heads), attn_quant="fp16", mlp_quant="fp16")}
        res["pruning"] = apply_structural_pruning(model, prune_plan, verbose=True)
        res["after_pruning"] = compare_logits(ref, forward_logits(model, enc), enc["input_ids"])
        ap = res["after_pruning"]
        log(f"SMOKE STEP 1/2 DONE: layer_{L} {len(p.pruned_heads)} heads pruned; NaN={ap['has_nan']} "
            f"Inf={ap['has_inf']} top1 agreement={ap['top1_agreement']:.3f} ppl {ap['ppl_ref']:.3f}->{ap['ppl_new']}")
        store.save("smoke pruning")

        # 2) attention + MLP INT4 (the MLP path is exercised even if the plan keeps the MLP in fp16)
        q_plan = {L: LayerPlan(L, pruned_heads=[], attn_quant="int4", mlp_quant="int4")}
        res["layer_bytes_before"] = module_bytes(layers[L])
        t0 = time.time()
        res["quantization"] = quantize_with_fallback(model, q_plan, log, store, "smoke")
        res["quantization_seconds"] = time.time() - t0
        res["layer_bytes_after"] = module_bytes(layers[L])
        res["layer_bytes_ratio"] = res["layer_bytes_after"]["bytes"] / res["layer_bytes_before"]["bytes"]
        res["module_types_after"] = layer_module_types(layers[L])
        res["after_quant"] = compare_logits(ref, forward_logits(model, enc), enc["input_ids"])
        aq = res["after_quant"]
        ok = not aq["has_nan"] and not aq["has_inf"]
        res["forward_ok"] = ok
        log(f"SMOKE STEP 2/2 DONE: layer_{L} INT4 ({res['quantization']}); path={res['quantization_path']}; "
            f"layer size ratio={res['layer_bytes_ratio']:.3f}; modules={res['module_types_after']}; "
            f"NaN={aq['has_nan']} Inf={aq['has_inf']} max|dlogit|={aq['max_abs_diff']:.4f} "
            f"top1 agreement={aq['top1_agreement']:.3f} ppl {aq['ppl_ref']:.3f}->{aq['ppl_new']}")
        if not ok:
            raise RuntimeError("SMOKE FAILED: forward pass with the quantized layer produced NaN/Inf; not proceeding to the full stage.")
        res["status"] = "completed"
    finally:
        res["peak_vram_gb"] = cuda_gb("peak")
        res["stage_seconds"] = time.time() - t_stage
        free_model(model, log)
        store.save("smoke finished")
    log(f"SMOKE STAGE COMPLETED ({res['stage_seconds']:.1f} s, peak VRAM {res['peak_vram_gb']} GB)")


# --------------------------------------------------------------------------- #
# Stage: full (whole model)
# --------------------------------------------------------------------------- #
def run_full(args, scores: Dict[str, float], tiers: Dict[str, str], plan: Dict[int, LayerPlan],
             store: ResultStore, log: Logger) -> None:
    res: Dict[str, Any] = store.data.setdefault("full", {})
    res.update({"status": "running", "started_at": datetime.now().isoformat(timespec="seconds"),
                "baseline_remeasured": not args.skip_baseline})
    store.save("full started")
    t_stage = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ref_gun1 = store.data["reference"].get("fp16_baseline_gun1")

    tokenizer, model, load_s = load_model(log)
    res["model_load_seconds"] = load_s
    try:
        device = next(model.parameters()).device
        res["model_bytes_before"] = module_bytes(model)
        res["cuda_allocated_gb_before"] = cuda_gb()
        log(f"FULL: model size (param+buffer) {res['model_bytes_before']['gb']:.2f} GB, "
            f"CUDA allocated {res['cuda_allocated_gb_before']} GB")

        log("Loading WikiText-2 test set...")
        text = load_wikitext_text()
        compute_perplexity, MAX_LENGTH, STRIDE = perplexity_tools()
        res["perplexity_settings"] = {"max_length": MAX_LENGTH, "stride": STRIDE,
                                      "dataset": "wikitext-2-raw-v1 (test split)"}

        # 1) FP16 baseline re-measurement (default)
        if not args.skip_baseline:
            t0 = time.time()
            ppl0 = compute_perplexity(model, tokenizer, text, device)
            res["perplexity_fp16_rerun"] = ppl0
            res["perplexity_fp16_rerun_seconds"] = time.time() - t0
            res["fp16_rerun_minus_gun1"] = (ppl0 - ref_gun1) if ref_gun1 is not None else None
            log(f"FULL STEP 1/4 DONE: FP16 perplexity re-measured = {ppl0:.4f} (baseline file: {ref_gun1}, "
                f"diff {res['fp16_rerun_minus_gun1']}), {res['perplexity_fp16_rerun_seconds']:.1f} s")
            store.save("full baseline")
        else:
            log("FULL STEP 1/4 SKIPPED: --skip-baseline given, FP16 not re-measured")

        res["generation_before"] = generate_samples(model, tokenizer, GEN_PROMPTS, args.gen_tokens)

        # 2) Apply JQP in the same order as apply_jqp (pruning -> quantization); the steps are
        #    called explicitly so the quantization fallback and intermediate saves can run in between.
        t0 = time.time()
        cfg = model.config
        budget = estimate_compression_budget(
            plan, hidden_size=cfg.hidden_size, head_dim=compressor._head_dim(model),
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
        )
        res["budget_nominal"] = budget
        print(compressor.format_tier_report(scores, tiers), flush=True)
        res["pruning"] = apply_structural_pruning(model, plan, verbose=True)
        log(f"FULL: structural pruning applied {res['pruning']}")
        store.save("full pruning")
        res["quantization"] = quantize_with_fallback(model, plan, log, store, "full")
        res["jqp_seconds"] = time.time() - t0
        res["model_bytes_after"] = module_bytes(model)
        res["cuda_allocated_gb_after"] = cuda_gb()
        n_attn = sum(1 for n in tiers if parse_block_key(n).kind == "attn")
        n_mlp = sum(1 for n in tiers if parse_block_key(n).kind == "mlp")
        tc = tier_counts(tiers)
        res["ratios"] = {
            "nominal_block_prune_ratio": {
                "definition": "fraction of BLOCKS pruned by within-type percentile; not a parameter ratio",
                "attn_heads": tc["attn"].get("prune", 0) / n_attn if n_attn else None,
                "mlp_blocks": tc["mlp"].get("prune", 0) / n_mlp if n_mlp else None,
                "tier_fractions": store.data["reference"].get("tier_fractions"),
            },
            "actual_parameter_prune_ratio": budget["pruned_ratio"],
            "nominal_decoder_size_ratio": budget["size_ratio"],
            "nominal_avg_bits_per_param": budget["avg_bits_per_param"],
            "measured_model_bytes_ratio": res["model_bytes_after"]["bytes"] / res["model_bytes_before"]["bytes"],
        }
        log(f"FULL STEP 2/4 DONE: JQP applied ({res['jqp_seconds']:.1f} s); pruning {res['pruning']}, "
            f"quantization {res['quantization']} (path={res['quantization_path']}); model size "
            f"{res['model_bytes_before']['gb']:.2f} -> {res['model_bytes_after']['gb']:.2f} GB "
            f"(measured ratio {res['ratios']['measured_model_bytes_ratio']:.3f}, nominal decoder ratio "
            f"{budget['size_ratio']:.3f}); head prune ratio {res['ratios']['nominal_block_prune_ratio']['attn_heads']:.3f}, "
            f"parameter prune ratio {budget['pruned_ratio']:.3f}")
        store.save("full JQP")

        # 3) compressed perplexity
        t0 = time.time()
        ppl1 = compute_perplexity(model, tokenizer, text, device)
        res["perplexity_compressed"] = ppl1
        res["perplexity_compressed_seconds"] = time.time() - t0
        res["perplexity_delta_vs_gun1"] = (ppl1 - ref_gun1) if ref_gun1 is not None else None
        res["perplexity_ratio_vs_gun1"] = (ppl1 / ref_gun1) if ref_gun1 else None
        if "perplexity_fp16_rerun" in res:
            res["perplexity_delta_vs_rerun"] = ppl1 - res["perplexity_fp16_rerun"]
        res["perplexity_finite"] = math.isfinite(ppl1)
        log(f"FULL STEP 3/4 DONE: compressed perplexity = {ppl1:.4f} (baseline FP16 {ref_gun1}, "
            f"delta={res['perplexity_delta_vs_gun1']}, ratio {res['perplexity_ratio_vs_gun1']}), "
            f"{res['perplexity_compressed_seconds']:.1f} s")
        store.save("full perplexity")

        # 4) generation samples (after compression)
        res["generation_after"] = generate_samples(model, tokenizer, GEN_PROMPTS, args.gen_tokens)
        for b, a in zip(res["generation_before"], res["generation_after"]):
            log(f"GENERATION '{b['prompt']}'\n    FP16 : {b['completion']!r}\n    JQP  : {a['completion']!r}", tag="GEN")
        log("FULL STEP 4/4 DONE: generation samples saved")
        res["status"] = "completed"
    finally:
        res["peak_vram_gb"] = cuda_gb("peak")
        res["stage_seconds"] = time.time() - t_stage
        store.save("full finished")
        free_model(model, log)
    log(f"FULL STAGE COMPLETED ({res['stage_seconds']:.1f} s, peak VRAM {res['peak_vram_gb']} GB)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log = Logger(args.log, to_file=should_log_to_file(args, LOG_FILE))  # preflight does not write to the tracked log
    store = ResultStore(args.output, log)
    t_start = time.time()
    log(f"===== End-to-end prototype starting: stage={args.stage} preflight={args.preflight} "
        f"skip_baseline={args.skip_baseline} =====")

    # --- inputs (no GPU needed) ---
    tiers_json = load_tiers_json(args.tiers)
    tiers: Dict[str, str] = tiers_json["tiers"]
    scores = load_scores(args.scores)
    baseline_gun1 = None
    if os.path.exists(args.baseline):
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline_gun1 = float(json.load(f)["perplexity"])
    plan = build_compression_plan(tiers)

    # Can the tier JSON be regenerated from the same scores? (informational; the JSON is authoritative)
    regenerated = allocate_compression_tiers(
        scores, tiers_json.get("n_tiers", 3),
        tier_fractions=tiers_json.get("tier_fractions"), mlp_min_tier=tiers_json.get("mlp_min_tier", "int4"),
    )
    n_diff = sum(1 for k in tiers if regenerated.get(k) != tiers[k]) + sum(1 for k in regenerated if k not in tiers)
    missing = [k for k in scores if k not in tiers]

    store.data = {
        "run": {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "args": vars(args),
            "model": MODEL_NAME,
            "env": env_info(),
        },
        "reference": {
            "fp16_baseline_gun1": baseline_gun1,
            "baseline_file": args.baseline,
            "tiers_file": args.tiers,
            "scores_file": args.scores,
            "n_tiers": tiers_json.get("n_tiers"),
            "tier_fractions": tiers_json.get("tier_fractions"),
            "mlp_min_tier": tiers_json.get("mlp_min_tier"),
            "tier_counts": tier_counts(tiers),
            "tiers_regenerated_diff": n_diff,
            "scores_missing_in_tiers": len(missing),
            "plan_summary": plan_summary(plan),
            "budget_nominal": estimate_compression_budget(plan),
        },
    }
    env = store.data["run"]["env"]
    bud = store.data["reference"]["budget_nominal"]
    log(f"Environment: torch {env['torch']}, transformers {env['transformers']}, bitsandbytes {env.get('bitsandbytes')}, "
        f"CUDA={env['cuda_available']} ({env.get('gpu')})")
    if env.get("bitsandbytes"):
        log(f"bnb Params4bit signature: {env['bnb_Params4bit_signature']}", tag="BNB")
        log(f"bnb Int8Params signature: {env['bnb_Int8Params_signature']}", tag="BNB")
    else:
        log(f"bitsandbytes could not be imported: {env.get('bitsandbytes_error')}", tag="BNB")
    log(f"Tiers: {tier_counts(tiers)}; JSON vs regenerated diff={n_diff}; scored blocks without a tier={len(missing)}; "
        f"plan {len(plan)} layers; nominal decoder size ratio {bud['size_ratio']:.3f}, "
        f"parameter prune ratio {bud['pruned_ratio']:.3f}; FP16 baseline={baseline_gun1}")
    if n_diff:
        log(f"WARNING: {n_diff} tiers in the JSON differ from those regenerated from the scores; using the JSON.")

    if args.preflight:
        store.data["run"]["status"] = "preflight_only"
        log("Preflight check completed (model not loaded). Result file not written.")
        return 0
    if not torch.cuda.is_available():
        log("WARNING: no CUDA; the 7B model + bitsandbytes cannot run on CPU. Run on a GPU machine.")
    store.save("start")

    exit_code = 0
    try:
        if args.stage in ("smoke", "both"):
            run_smoke(args, plan, store, log)
        if args.stage in ("full", "both"):
            run_full(args, scores, tiers, plan, store, log)
        store.data["run"]["status"] = "completed"
    except BaseException as e:  # includes KeyboardInterrupt / SystemExit so partial results are not lost
        store.data["run"]["status"] = "failed"
        store.data["run"]["error"] = repr(e)
        store.data["run"]["traceback"] = traceback.format_exc()
        log(f"ERROR: {e!r}; writing partial results to disk", tag="ERR")
        log(store.data["run"]["traceback"], tag="ERR")
        exit_code = 1
    finally:
        store.data["run"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        store.data["run"]["total_seconds"] = time.time() - t_start
        store.save("final")
        log(f"===== Done: status={store.data['run']['status']}, total {store.data['run']['total_seconds']:.1f} s =====")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
