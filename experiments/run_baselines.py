"""
Off-the-shelf quantization baselines (GPU required; CPU with --dry-run).

Library runs that place XAI-JQP relative to standard methods at comparable size, in
particular xAI-guided INT4 selection vs uniform NF4. For each configuration the model is
loaded cleanly; WikiText-2 perplexity (same harness as the FP16 baseline), the MMLU subset
(eval_mmlu.py), measured size, runtime and peak VRAM are recorded; the next configuration
is loaded only after the model has been fully released.

Configurations (select a subset with --configs):
  fp16         reference (5.2200)
  nf4_uniform  all decoder Linear layers in bitsandbytes NF4 (HF `load_in_4bit`,
               `bnb_4bit_quant_type="nf4"`, compute dtype fp16, lm_head excluded;
               double quantization ON, same setting as compressor._quantize_linear)
  gptq_4bit    auto-gptq 4-bit (g128)  } ONLY in the separate environment (.venv-quant, requirements-quant.txt);
  awq_4bit     autoawq 4-bit (g128)    } without the packages they are skipped and the reason is written to the JSON.
               Both are quantized by this script with the same calibration (WikiText-2 TRAIN) and
               cached under --quant-cache; measurement uses the model loaded via transformers.from_pretrained.

Peak VRAM: run_e2e_pipeline.free_model only deletes its own local name, so a model still
referenced by the caller stays resident while the next one loads. Here the model is kept in a
single dictionary that is cleared at the end of each configuration; allocated memory is logged
after gc + empty_cache, and torch.cuda.reset_peak_memory_stats() is called per configuration.

Usage:
    python experiments/run_baselines.py --preflight
    python experiments/run_baselines.py                           # fp16 + nf4_uniform (main environment)
    .venv-quant/bin/python run_baselines.py --resume  # GPTQ/AWQ, appended to the SAME JSON
    python experiments/run_baselines.py --configs nf4_uniform     # single configuration
    python experiments/run_baselines.py --dry-run                 # mini model on CPU, no GPU
    python experiments/run_baselines.py --resume                  # skip completed configurations

Outputs: results/baselines_gun7.json (dry-run: dryrun_out/baselines_gun7_dry.json), log_baselines_gun7.txt.
"""

# HF_HOME must be set BEFORE transformers/datasets are imported (see run_e2e_pipeline.py)
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import contextlib
import functools
import gc
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_mmlu
# Shared components of run_ablation_tests.py / run_e2e_pipeline.py, imported unchanged
from run_ablation_tests import DryRun, _fake_quantize_linear, set_seed
from run_e2e_pipeline import (
    BASELINE_FILE,
    MODEL_NAME,
    Logger,
    should_log_to_file,
    ResultStore,
    cuda_gb,
    env_info,
    load_model,
    load_wikitext_text,
    module_bytes,
    perplexity_tools,
)

OUTPUT_FILE = os.path.join("results", "baselines_gun7.json")
LOG_FILE = "log_baselines_gun7.txt"
CLEAN_GPU_TOL_GB = 0.5  # warn if more memory than this is still allocated before a new load

# `available`: static registry flag; a configuration with False is skipped and written to the JSON
# with status="skipped_unavailable" + reason.
# `requires`: package groups whose importability is also checked at run time (one member per group suffices).
BASELINES: Dict[str, Dict[str, Any]] = {
    "fp16": {
        "description": "FP16 reference (no compression)",
        "available": True, "requires": (),
    },
    "nf4_uniform": {
        "description": "uniform NF4: all decoder Linear layers in bitsandbytes 4-bit (load_in_4bit, nf4, compute fp16, "
                       "double quantization), lm_head excluded",
        "available": True, "requires": (("bitsandbytes",),),
    },
    # Tested with auto-gptq 0.7.1 / autoawq 0.2.5 + kernels 0.0.8, torch 2.4.1 + transformers 4.44.2
    # (see requirements-quant.txt). The packages live only in .venv-quant, so in the main environment
    # these two configurations are skipped as "required package not installed" and reported.
    "gptq_4bit": {
        "description": "GPTQ 4-bit (auto-gptq: g128, desc_act=False, sym; calibration WikiText-2 train 128×2048), lm_head excluded",
        "available": True, "requires": (("auto_gptq",), ("optimum",)), "separate_env": True,
    },
    "awq_4bit": {
        "description": "AWQ 4-bit (autoawq: g128, zero_point, GEMM; calibration WikiText-2 train paragraphs), lm_head excluded",
        "available": True, "requires": (("awq",),), "separate_env": True,
    },
}
# GPTQ/AWQ calibration uses the TRAIN split to avoid leakage, since perplexity is measured on the TEST split.
QUANT_CALIB = dict(dataset="wikitext-2-raw-v1", split="train", min_chars=200, group_size=128,
                   gptq_n_samples=128, gptq_seqlen=2048, awq_n_paragraphs=1024)
ALL_CONFIGS = list(BASELINES)
OPTIONAL_PACKAGES = ("bitsandbytes", "accelerate", "datasets", "optimum", "gptqmodel", "auto-gptq", "autoawq")


# --------------------------------------------------------------------------- #
# Arguments / environment
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Off-the-shelf quantization baselines (perplexity + MMLU + size + VRAM)")
    p.add_argument("--configs", default=",".join(ALL_CONFIGS),
                   help="comma-separated configuration list (default: all; available=False ones are skipped)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--log", default=LOG_FILE)
    p.add_argument("--baseline", default=BASELINE_FILE, help="FP16 baseline JSON (reference perplexity)")
    p.add_argument("--no-mmlu", action="store_true", help="do NOT measure the MMLU subset (default: measure)")
    p.add_argument("--mmlu-ids", default=eval_mmlu.SUBSET_IDS_FILE)
    p.add_argument("--mmlu-prompt-style", choices=eval_mmlu.PROMPT_STYLES, default="plain")
    p.add_argument("--mmlu-batch-size", type=int, default=8)
    p.add_argument("--nf4-no-double-quant", action="store_true",
                   help="disable double quantization for nf4_uniform (default on, same as compressor)")
    p.add_argument("--gptq-model", default=None, help="pre-quantized checkpoint for gptq_4bit (otherwise this script quantizes)")
    p.add_argument("--awq-model", default=None, help="pre-quantized checkpoint for awq_4bit (otherwise this script quantizes)")
    p.add_argument("--quant-cache", default="/workspace/quant_cache",
                   help="directory for the GPTQ/AWQ checkpoints produced by this script (~4 GB per configuration)")
    p.add_argument("--dry-run", action="store_true", help="mini Mistral + fake quantization/perplexity/MMLU, CPU")
    p.add_argument("--preflight", action="store_true", help="check packages/inputs/MMLU subset without loading the model, then exit")
    p.add_argument("--resume", action="store_true", help="skip configurations with status=completed in --output")
    args = p.parse_args(argv)
    if args.dry_run and args.output == OUTPUT_FILE:  # mini-model results must not overwrite results/baselines_gun7.json
        args.output = os.path.join("dryrun_out", "baselines_gun7_dry.json")
    return args


def package_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in OPTIONAL_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def availability(name: str, dry: bool) -> Tuple[bool, Optional[str]]:
    """(runnable?, reason if not). Registry flag + run-time import check."""
    spec = BASELINES[name]
    if not spec["available"]:
        return False, spec.get("unavailable_reason", "available=False")
    if dry:
        return True, None
    for group in spec["requires"]:
        if not any(importlib.util.find_spec(m) is not None for m in group):
            return False, f"required package not installed: {' | '.join(group)}"
    return True, None


# --------------------------------------------------------------------------- #
# Memory: no new load until the previous model is fully released
# --------------------------------------------------------------------------- #
def release_gpu() -> Optional[float]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cuda_gb()


def ensure_clean_gpu(log: Logger, where: str) -> Optional[float]:
    alloc = release_gpu()
    if alloc is not None and alloc > CLEAN_GPU_TOL_GB:
        log(f"WARNING: {alloc:.2f} GB still allocated on CUDA before {where} (previous model may not be fully "
            "released); peak VRAM of this configuration will read higher than it is", tag="MEM")
    return alloc


# --------------------------------------------------------------------------- #
# Loaders: (tokenizer, model, load time, load info)
# --------------------------------------------------------------------------- #
def _tokenizer(model_id: str = MODEL_NAME):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def linear_type_counts(model: nn.Module) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear) or "Linear" in type(m).__name__:
            counts[type(m).__name__] = counts.get(type(m).__name__, 0) + 1
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    return {"linear_module_types": counts, "lm_head_type": type(head).__name__ if head is not None else None}


def load_fp16(ctx: Dict[str, Any], log: Logger):
    tokenizer, model, load_s = load_model(log)
    return tokenizer, model, load_s, {"dtype": "float16", **linear_type_counts(model)}


def is_bnb_model(model) -> bool:
    """Is this a bitsandbytes-loaded (4/8-bit) model? At dispatch time inside from_pretrained, is_loaded_in_4bit is not
    set yet (it is set AFTER the weights are loaded); only is_quantized + quantization_method are known then."""
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        return True
    method = getattr(model, "quantization_method", None)
    return bool(getattr(model, "is_quantized", False)) and getattr(method, "value", method) == "bitsandbytes"


def _bnb_safe(dispatch: Callable) -> Callable:
    @functools.wraps(dispatch)
    def wrapper(model, *args, **kwargs):
        if is_bnb_model(model):
            kwargs["force_hooks"] = True
        return dispatch(model, *args, **kwargs)

    return wrapper


@contextlib.contextmanager
def bnb_safe_dispatch(modeling_utils=None):
    """
    transformers 4.44.2 + accelerate>=1.x on a single GPU: dispatch_model inside from_pretrained calls model.to(device)
    for a single-device device_map, and 4.44.2's PreTrainedModel.to raises ValueError on bnb models ("`.to` is not
    supported for `4-bit` or `8-bit` bitsandbytes models"). As older accelerate did, hooks are forced for bnb models
    (force_hooks=True) so .to is never called; the weights are already on the GPU via device_map and the hooks move
    any buffers left on CPU. No-op for transformers versions without dispatch_model.
    """
    if modeling_utils is None:
        import transformers.modeling_utils as modeling_utils
    orig = getattr(modeling_utils, "dispatch_model", None)
    if orig is None:
        yield
        return
    modeling_utils.dispatch_model = _bnb_safe(orig)
    try:
        yield
    finally:
        modeling_utils.dispatch_model = orig


def load_nf4_uniform(ctx: Dict[str, Any], log: Logger):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    double_quant = not ctx["args"].nf4_no_double_quant
    log(f"Loading model: {MODEL_NAME} (bitsandbytes NF4, compute fp16, double_quant={double_quant})")
    t0 = time.time()
    qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=double_quant)
    tokenizer = _tokenizer()
    with bnb_safe_dispatch():
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=qconfig,
                                                     torch_dtype=torch.float16, device_map="auto")
    model.eval()
    load_s = time.time() - t0
    info = {"quantization_config": qconfig.to_dict(), **linear_type_counts(model)}
    if type(model.get_output_embeddings()).__module__.startswith("bitsandbytes"):
        raise RuntimeError("nf4_uniform: lm_head was quantized; the 'lm_head excluded' condition is violated.")
    log(f"Model loaded ({load_s:.1f} s), modules={info['linear_module_types']}, lm_head={info['lm_head_type']}, "
        f"CUDA allocated={cuda_gb()} GB")
    return tokenizer, model, load_s, info


def _load_prequantized(kind: str, repo: Optional[str], log: Logger):
    """Pre-quantized 4-bit checkpoint (GPTQ/AWQ): transformers resolves the checkpoint's quantization_config itself."""
    from transformers import AutoModelForCausalLM

    if not repo:
        raise ValueError(f"{kind}: no checkpoint given (--{kind.split('_')[0]}-model)")
    log(f"Loading model: {repo} ({kind}, 4-bit checkpoint)")
    t0 = time.time()
    tokenizer = _tokenizer(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    load_s = time.time() - t0
    info = {"checkpoint": repo, "quantization_config": getattr(model.config, "quantization_config", None),
            **linear_type_counts(model)}
    log(f"Model loaded ({load_s:.1f} s), modules={info['linear_module_types']}, CUDA allocated={cuda_gb()} GB")
    return tokenizer, model, load_s, info


def _calibration_paragraphs() -> List[str]:
    from datasets import load_dataset

    ds = load_dataset("wikitext", QUANT_CALIB["dataset"], split=QUANT_CALIB["split"])
    return [s for s in (t.strip() for t in ds["text"]) if s and not s.startswith("=") and len(s) >= QUANT_CALIB["min_chars"]]


def quantize_gptq(model_id: str, out_dir: str, seed: int, log: Logger) -> Dict[str, Any]:
    """Quantizes to 4-bit with auto-gptq and writes to out_dir (as `model.safetensors`, the name transformers reads)."""
    import random

    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    tokenizer = _tokenizer(model_id)
    ids = tokenizer("\n\n".join(_calibration_paragraphs()), return_tensors="pt").input_ids
    seqlen = min(QUANT_CALIB["gptq_seqlen"], ids.shape[1] - 1)
    rng = random.Random(seed)
    examples = []
    for _ in range(QUANT_CALIB["gptq_n_samples"]):  # as in the GPTQ paper: random windows from the concatenated text
        i = rng.randint(0, ids.shape[1] - seqlen - 1)
        window = ids[:, i:i + seqlen]
        examples.append({"input_ids": window, "attention_mask": torch.ones_like(window)})
    qcfg = BaseQuantizeConfig(bits=4, group_size=QUANT_CALIB["group_size"], desc_act=False, sym=True,
                              model_file_base_name="model")
    model = AutoGPTQForCausalLM.from_pretrained(model_id, qcfg, torch_dtype=torch.float16)
    t0 = time.time()
    model.quantize(examples)
    model.save_quantized(out_dir, use_safetensors=True)
    tokenizer.save_pretrained(out_dir)
    info = {"library": "auto-gptq", "bits": 4, "group_size": QUANT_CALIB["group_size"], "desc_act": False, "sym": True,
            "calibration": {**QUANT_CALIB, "n_samples": len(examples), "seqlen": seqlen, "seed": seed},
            "quantize_seconds": time.time() - t0}
    log(f"GPTQ quantization done ({info['quantize_seconds']:.0f} s) -> {out_dir}")
    return info


def quantize_awq(model_id: str, out_dir: str, seed: int, log: Logger) -> Dict[str, Any]:
    """Quantizes to 4-bit (GEMM) with autoawq and writes to out_dir. autoawq splits paragraphs into 512-token blocks itself."""
    from awq import AutoAWQForCausalLM

    tokenizer = _tokenizer(model_id)
    paragraphs = _calibration_paragraphs()[: QUANT_CALIB["awq_n_paragraphs"]]
    qcfg = {"zero_point": True, "q_group_size": QUANT_CALIB["group_size"], "w_bit": 4, "version": "GEMM"}
    model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True, torch_dtype=torch.float16)
    t0 = time.time()
    model.quantize(tokenizer, quant_config=qcfg, calib_data=paragraphs)
    model.save_quantized(out_dir)
    tokenizer.save_pretrained(out_dir)
    info = {"library": "autoawq", **qcfg,
            "calibration": {**QUANT_CALIB, "n_paragraphs": len(paragraphs), "block_size": 512, "seed": None},
            "quantize_seconds": time.time() - t0}
    log(f"AWQ quantization done ({info['quantize_seconds']:.0f} s) -> {out_dir}")
    return info


QUANTIZERS: Dict[str, Callable] = {"gptq_4bit": quantize_gptq, "awq_4bit": quantize_awq}


def _load_quantized(kind: str, ctx: Dict[str, Any], log: Logger):
    """
    Uses the checkpoint given by --gptq-model/--awq-model; otherwise MODEL_NAME is quantized by this script
    (on the first run; the result is cached under --quant-cache and later runs load it from there). The
    quantization model is fully released before measurement; the measured model is always loaded with
    transformers.from_pretrained.
    """
    args = ctx["args"]
    repo = args.gptq_model if kind == "gptq_4bit" else args.awq_model
    quant_info: Dict[str, Any] = {"source": "pre-quantized checkpoint (given on the command line)"}
    if not repo:
        repo = os.path.join(args.quant_cache, f"{MODEL_NAME.split('/')[-1]}-{kind}-g{QUANT_CALIB['group_size']}")
        if os.path.exists(os.path.join(repo, "config.json")):
            quant_info = {}
            info_path = os.path.join(repo, "xai_jqp_quant_info.json")
            if os.path.exists(info_path):
                with open(info_path, "r", encoding="utf-8") as f:
                    quant_info = json.load(f)
            quant_info["source"] = "cache (quantized by this script in an earlier run)"
        else:
            log(f"{kind}: not in cache, quantizing -> {repo}")
            os.makedirs(repo, exist_ok=True)
            quant_info = {"source": "quantized in this run", **QUANTIZERS[kind](MODEL_NAME, repo, args.seed, log)}
            with open(os.path.join(repo, "xai_jqp_quant_info.json"), "w", encoding="utf-8") as f:
                json.dump(quant_info, f, indent=2, ensure_ascii=False)
            freed = release_gpu()
            log(f"{kind}: quantization model released; CUDA allocated={freed} GB")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # peak VRAM = inference peak; the quantization peak is a separate job
    tokenizer, model, load_s, info = _load_prequantized(kind, repo, log)
    return tokenizer, model, load_s, {**info, "quantization": quant_info}


def load_gptq(ctx: Dict[str, Any], log: Logger):
    return _load_quantized("gptq_4bit", ctx, log)


def load_awq(ctx: Dict[str, Any], log: Logger):
    return _load_quantized("awq_4bit", ctx, log)


LOADERS: Dict[str, Callable] = {"fp16": load_fp16, "nf4_uniform": load_nf4_uniform,
                                "gptq_4bit": load_gptq, "awq_4bit": load_awq}


def load_dry(name: str, ctx: Dict[str, Any], log: Logger):
    """Dry-run: mini model; for configurations other than fp16 every Linear except lm_head becomes fake 4-bit."""
    _, model, load_s = ctx["dry"].load_model(log)
    if name != "fp16":
        n = 0
        for parent_name, parent in list(model.named_modules()):
            for child_name, child in list(parent.named_children()):
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                if isinstance(child, nn.Linear) and full != "lm_head":
                    setattr(parent, child_name, _fake_quantize_linear(child, "int4"))
                    n += 1
        log(f"[DRY] {name}: {n} Linear layers converted to fake 4-bit (lm_head excluded)")
    return None, model, load_s, {"dry_run": True, **linear_type_counts(model)}


# --------------------------------------------------------------------------- #
# Single configuration: load -> measure -> fully release
# --------------------------------------------------------------------------- #
def measure(model: nn.Module, tokenizer, ctx: Dict[str, Any], rep: Dict[str, Any], store: ResultStore, log: Logger) -> None:
    """Separate function so the model reference dies with this frame (run_config keeps no local `model` name)."""
    dry: Optional[DryRun] = ctx["dry"]
    args = ctx["args"]
    rep["model_bytes"] = module_bytes(model)
    rep["cuda_allocated_after_load_gb"] = cuda_gb()
    log(f"size (param+buffer, excluding quant_state) {rep['model_bytes']['gb']:.2f} GB")

    t0 = time.time()
    if dry:
        ppl = dry.perplexity(model)
    else:
        ppl = ctx["compute_perplexity"](model, tokenizer, ctx["text"], next(model.parameters()).device)
    rep["perplexity"] = ppl
    rep["perplexity_seconds"] = time.time() - t0
    rep["perplexity_finite"] = math.isfinite(ppl)
    log(f"perplexity = {ppl:.4f} ({rep['perplexity_seconds']:.1f} s)")
    store.save("perplexity")

    if not args.no_mmlu:
        if dry:
            rep["mmlu"] = eval_mmlu.evaluate_mmlu_dry(model, args.seed, log=log)
        else:
            rep["mmlu"] = eval_mmlu.evaluate_mmlu(model, tokenizer, subset=ctx["mmlu_subset"], ids_path=args.mmlu_ids,
                                                  prompt_style=args.mmlu_prompt_style,
                                                  batch_size=args.mmlu_batch_size, log=log)
        rep["mmlu_subset_acc"] = rep["mmlu"]["mmlu_subset_acc"]


def run_config(name: str, ctx: Dict[str, Any], rep: Dict[str, Any], store: ResultStore, log: Logger) -> None:
    rep.update({"status": "running", "seed": ctx["args"].seed,
                "started_at": datetime.now().isoformat(timespec="seconds")})
    store.save(f"{name} started")
    t_cfg = time.time()
    set_seed(ctx["args"].seed)
    rep["cuda_allocated_before_load_gb"] = ensure_clean_gpu(log, name)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    box: Dict[str, Any] = {}
    try:
        if ctx["dry"]:
            box["tokenizer"], box["model"], rep["model_load_seconds"], rep["load_info"] = load_dry(name, ctx, log)
        else:
            box["tokenizer"], box["model"], rep["model_load_seconds"], rep["load_info"] = LOADERS[name](ctx, log)
        measure(box["model"], box["tokenizer"], ctx, rep, store, log)
        rep["status"] = "completed"
    finally:
        rep["peak_vram_gb"] = cuda_gb("peak")
        box.clear()
        rep["cuda_allocated_after_free_gb"] = release_gpu()
        rep["seconds"] = time.time() - t_cfg
        log(f"{name}: model released from memory; CUDA allocated={rep['cuda_allocated_after_free_gb']} GB, "
            f"peak VRAM={rep['peak_vram_gb']} GB")
        store.save(f"{name} finished")


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarize(configs: Dict[str, Any], baseline_gun1: Optional[float]) -> Dict[str, Any]:
    fp = configs.get("fp16", {})
    done = fp.get("status") == "completed"
    ref_ppl = fp.get("perplexity") if done else baseline_gun1
    ref_bytes = fp["model_bytes"]["bytes"] if done else None
    ref_acc = fp.get("mmlu_subset_acc") if done else None
    out: Dict[str, Any] = {"reference_perplexity": ref_ppl,
                           "reference_source": "fp16 configuration (this run)" if done else "FP16 baseline file",
                           "reference_mmlu_subset_acc": ref_acc, "configs": {}}
    for name, c in configs.items():
        if c.get("status") != "completed":
            continue
        row: Dict[str, Any] = {"perplexity": c["perplexity"], "mmlu_subset_acc": c.get("mmlu_subset_acc"),
                               "model_gb": c["model_bytes"]["gb"], "peak_vram_gb": c.get("peak_vram_gb"),
                               "seconds": c.get("seconds")}
        if ref_ppl is not None:
            row["perplexity_delta_vs_fp16"] = c["perplexity"] - ref_ppl
            row["perplexity_ratio_vs_fp16"] = c["perplexity"] / ref_ppl
        if ref_bytes:
            row["measured_model_bytes_ratio"] = c["model_bytes"]["bytes"] / ref_bytes
        if ref_acc is not None and c.get("mmlu_subset_acc") is not None:
            row["mmlu_delta_vs_fp16"] = c["mmlu_subset_acc"] - ref_acc
        out["configs"][name] = row
    return out


def format_summary_table(configs: Dict[str, Any], summary: Dict[str, Any]) -> str:
    def f(v: Any, spec: str) -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    lines = [f"{'config':<13}{'ppl':>9}{'Δfp16':>9}{'mmlu':>8}{'GB':>7}{'ratio':>7}{'VRAM':>7}{'s':>7}  status"]
    for name, c in configs.items():
        r = summary["configs"].get(name, {})
        lines.append(f"{name:<13}{f(r.get('perplexity'), '.4f'):>9}{f(r.get('perplexity_delta_vs_fp16'), '+.4f'):>9}"
                     f"{f(r.get('mmlu_subset_acc'), '.4f'):>8}{f(r.get('model_gb'), '.2f'):>7}"
                     f"{f(r.get('measured_model_bytes_ratio'), '.3f'):>7}{f(r.get('peak_vram_gb'), '.1f'):>7}"
                     f"{f(r.get('seconds'), '.0f'):>7}  {c.get('status')}"
                     + (f" ({c['unavailable_reason']})" if c.get("status") == "skipped_unavailable" else ""))
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
    unknown = [c for c in config_names if c not in BASELINES]
    if unknown:
        raise SystemExit(f"unknown configuration(s): {unknown}; valid: {ALL_CONFIGS}")
    log(f"===== Off-the-shelf baselines starting: configs={config_names} seed={args.seed} "
        f"mmlu={not args.no_mmlu} dry_run={args.dry_run} =====", tag="BASE")

    dry: Optional[DryRun] = DryRun(args.seed) if args.dry_run else None
    baseline_gun1 = None
    if not dry and os.path.exists(args.baseline):
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline_gun1 = float(json.load(f)["perplexity"])

    previous = None
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            previous = json.load(f)
        log(f"--resume: read {args.output}; completed configurations will be skipped", tag="BASE")

    avail = {n: availability(n, bool(dry)) for n in config_names}
    store.data = {
        "run": {"started_at": datetime.now().isoformat(timespec="seconds"), "status": "running",
                "args": vars(args), "model": "mini (dry-run)" if dry else MODEL_NAME,
                "env": {**env_info(), "packages": package_versions()}, "config_order": config_names},
        "reference": {"fp16_baseline_gun1": baseline_gun1,
                      "size_note": "model_bytes = parameter + buffer bytes; excludes bnb quant_state (absmax) overhead (same measurement as run_e2e_pipeline.py)",
                      "peak_vram_note": "reset_peak_memory_stats per configuration; memory allocated before loading is in cuda_allocated_before_load_gb"},
        "configs": {n: {"description": BASELINES[n]["description"], "available": avail[n][0],
                        **({"unavailable_reason": avail[n][1]} if not avail[n][0] else {}),
                        "status": "pending" if avail[n][0] else "skipped_unavailable"} for n in config_names},
        "summary": {},
    }
    for n, c in (previous or {}).get("configs", {}).items():  # --resume: keep earlier configurations not requested in this run
        store.data["configs"].setdefault(n, c)
    env = store.data["run"]["env"]
    log(f"Environment: torch {env['torch']}, transformers {env['transformers']}, bitsandbytes {env.get('bitsandbytes')}, "
        f"CUDA={env['cuda_available']} ({env.get('gpu')}); packages={env['packages']}", tag="BASE")
    for n in config_names:
        log(f"config {n}: " + ("will run" if avail[n][0] else f"WILL BE SKIPPED — {avail[n][1]}"), tag="BASE")

    if args.preflight:
        problems: List[str] = []
        if not torch.cuda.is_available():
            problems.append("no CUDA")
        if baseline_gun1 is None:
            problems.append(f"FP16 baseline could not be read: {args.baseline}")
        # gptq/awq packages are expected to be missing in the main environment (separate .venv-quant); logged above for information only
        problems += [f"{n}: {avail[n][1]}" for n in config_names
                     if BASELINES[n]["available"] and not avail[n][0] and not BASELINES[n].get("separate_env")]
        if not args.no_mmlu:
            try:
                subset = eval_mmlu.load_or_build_subset(args.mmlu_ids, log=lambda m: log(m, tag="MMLU"))
                log(f"MMLU subset ready: {len(subset['records'])} questions ({args.mmlu_ids})", tag="MMLU")
            except Exception as e:
                problems.append(f"MMLU subset could not be loaded: {e!r}")
            try:  # are " A".." D" single tokens in the real tokenizer? (the model is gated, so only checkable on the GPU host)
                _, letters = eval_mmlu.letter_token_ids(_tokenizer())
                log(f"MMLU letter tokens: {letters}", tag="MMLU")
                if not letters["context_stable"]:
                    problems.append(f"MMLU letter tokens are not single tokens in the 'Answer: X' context: {letters}")
            except Exception as e:
                problems.append(f"tokenizer / letter token check failed: {e!r}")
        store.data["run"]["status"] = "preflight_only"
        if problems:
            log("Preflight check FAILED (model not loaded): " + "; ".join(problems), tag="ERR")
            return 1
        log("Preflight check completed (model not loaded). Result file not written.", tag="BASE")
        return 0
    if not torch.cuda.is_available() and not dry:
        log("WARNING: no CUDA; the 7B model + bitsandbytes cannot run on CPU. Run on a GPU machine (or use --dry-run).", tag="BASE")
    store.save("start")

    ctx: Dict[str, Any] = {"args": args, "dry": dry}
    exit_code = 0
    try:
        if not dry:
            log("Loading WikiText-2 test set...", tag="BASE")
            ctx["text"] = load_wikitext_text()
            ctx["compute_perplexity"], max_len, stride = perplexity_tools()
            store.data["reference"]["perplexity_settings"] = {"max_length": max_len, "stride": stride,
                                                              "dataset": "wikitext-2-raw-v1 (test split)"}
            if not args.no_mmlu:
                ctx["mmlu_subset"] = eval_mmlu.load_or_build_subset(args.mmlu_ids, log=lambda m: log(m, tag="MMLU"))
                store.data["reference"]["mmlu_n_questions"] = len(ctx["mmlu_subset"]["records"])

        for ci, name in enumerate(config_names, 1):
            cfg = store.data["configs"][name]
            if not cfg["available"]:
                log(f"CONFIG {ci}/{len(config_names)} {name}: SKIPPED — {cfg['unavailable_reason']}", tag="BASE")
                continue
            prev_cfg = (previous or {}).get("configs", {}).get(name)
            if prev_cfg and prev_cfg.get("status") == "completed":
                store.data["configs"][name] = prev_cfg
                log(f"CONFIG {ci}/{len(config_names)} {name}: --resume, taken from the previous run "
                    f"(ppl={prev_cfg.get('perplexity')})", tag="BASE")
                continue
            log(f"===== CONFIG {ci}/{len(config_names)}: {name} — {cfg['description']} =====", tag="BASE")
            try:
                run_config(name, ctx, cfg, store, log)
                log(f"CONFIG {ci}/{len(config_names)} DONE: {name} ppl={cfg['perplexity']:.4f} "
                    f"mmlu={cfg.get('mmlu_subset_acc')} size={cfg['model_bytes']['gb']:.2f} GB "
                    f"peak VRAM={cfg['peak_vram_gb']} GB, {cfg['seconds']:.0f} s", tag="BASE")
            except Exception as e:  # one failing configuration must not abort the others (partial result + traceback in JSON)
                cfg.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
                log(f"CONFIG {name} ERROR: {e!r}; moving on to the next configuration", tag="ERR")
                log(cfg["traceback"], tag="ERR")
                exit_code = 1
            store.data["summary"] = summarize(store.data["configs"], baseline_gun1)
            store.save(f"{name} summary")
        store.data["run"]["status"] = "completed" if exit_code == 0 else "completed_with_failures"
    except BaseException as e:  # includes KeyboardInterrupt / SystemExit so partial results are not lost
        store.data["run"]["status"] = "failed"
        store.data["run"]["error"] = repr(e)
        store.data["run"]["traceback"] = traceback.format_exc()
        log(f"ERROR: {e!r}; writing partial results to disk", tag="ERR")
        log(store.data["run"]["traceback"], tag="ERR")
        exit_code = 1
    finally:
        store.data["summary"] = summarize(store.data["configs"], baseline_gun1)
        store.data["run"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        store.data["run"]["total_seconds"] = time.time() - t_start
        store.save("final")
        log("SUMMARY TABLE:\n" + format_summary_table(store.data["configs"], store.data["summary"]), tag="BASE")
        log(f"===== Done: status={store.data['run']['status']}, total {store.data['run']['total_seconds']:.1f} s =====",
            tag="BASE")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
