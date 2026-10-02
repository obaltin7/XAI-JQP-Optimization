"""
Speed / size measurement: FP16 vs masked pruning vs physical head removal vs removal + INT4 (GPU; --dry-run on CPU).

Masking alone gives no size or speed gain; physical removal (compressor.physically_prune_heads) gives a small
but real one. For the same head set, each variant is measured after a FRESH model load:
  fp16           reference
  masked         apply_structural_pruning (q rows / o columns zeroed; shapes unchanged)
  physical       physically_prune_heads (q/o matrices shrink, k/v unchanged, PrunedHeadAttention)
  physical_int4  physical + the plan's INT4 modules (bitsandbytes; removal happens BEFORE quantization)
  masked_int4    (optional, via --variants) the `both` path: masked + INT4, for comparison
Protocol: 8 fixed prompts (batch 1), greedy decoding of 64 new tokens, KV cache on; 1 untimed warm-up generation;
ms/token = total generation time / generated tokens (prefill included; prompts are short); peak VRAM
(reset_peak_memory_stats per variant), measured size (module_bytes: parameters + buffers, excluding
quant_state) and parameter count; the completion of the first prompt is stored as a sanity check.

The head set and INT4 plan are read from a result JSON (`--plan --heads-from [--fraction]`):
  results/ablation_gun6.json      configs[<config>].repeats[0].{pruned_heads, plan_summary}
  results/iterative_gun7.json     fractions[<ratio>].configs[<config>].repeats[0].{...}

Cost on one L40S: ~13 s load + 9 generations × 64 tokens ≈ 30-60 s per variant -> ~5 min for 4 variants;
peak VRAM ~15 GB (fp16).

Options beyond the defaults (the defaults reproduce the original single-repeat protocol):
  --n-repeats N            repeat the timed prompt pass N times after warm-up; per-repeat ms/token and ±std in the JSON (use >= 3 for speed claims)
  --attn-implementation X  eager|sdpa for the fp16/masked loads (None = transformers 4.44.2 default: sdpa, MistralSdpaAttention).
                           PrunedHeadAttention (physical*) always computes eager matmul+softmax; the config path and the layer-0
                           attention class are recorded per variant (attn_implementation / attention_class).
  --perplexity             WikiText-2 perplexity per variant (baseline_eval harness): numerical masked-vs-physical equivalence on 7B;
                           perplexity_minus_masked in the summary. ~70 s per variant.

Usage:
    python tools/evaluation/measure_speed.py --plan results/ablation_gun6.json --heads-from both
    python tools/evaluation/measure_speed.py --plan results/iterative_gun7.json --fraction 0.2 --heads-from xai_iter
    python tools/evaluation/measure_speed.py --dry-run                      # mini model (dryrun_out/)
    python tools/evaluation/measure_speed.py --plan results/ablation_gun6.json --heads-from both --n-repeats 3 --output results/speed_gun8_n3.json --log log_speed_gun8.txt
    python tools/evaluation/measure_speed.py --plan results/ablation_gun6.json --heads-from both --variants fp16,masked,physical --attn-implementation eager --n-repeats 3 --output results/speed_gun8_eager.json --log log_speed_gun8.txt
    python tools/evaluation/measure_speed.py --plan results/iterative_gun7.json --fraction 0.2 --heads-from xai_iter_fixedq --n-repeats 3 --perplexity --output results/speed_gun8_fixedq.json --log log_speed_gun8.txt

Output: results/speed_gun7.json by default (run metadata, source head set, per-variant measurements, summary).
"""

# HF_HOME must be set BEFORE transformers is imported (see run_e2e_pipeline.py)
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compressor import LayerPlan, apply_structural_pruning, count_parameters, physically_prune_heads
from run_ablation_tests import DryRun, _StoreView, set_seed
from run_baselines import ensure_clean_gpu, release_gpu
from run_e2e_pipeline import MODEL_NAME, Logger, ResultStore, should_log_to_file, cuda_gb, env_info, load_model, module_bytes, quantize_with_fallback

OUTPUT_FILE = os.path.join("results", "speed_gun7.json")
LOG_FILE = "log_speed_gun7.txt"
DRYRUN_DIR = "dryrun_out"
DEFAULT_PLAN = os.path.join("results", "ablation_gun6.json")
ALL_VARIANTS = ["fp16", "masked", "physical", "physical_int4", "masked_int4"]
DEFAULT_VARIANTS = ["fp16", "masked", "physical", "physical_int4"]
PROMPTS = [
    "The main causes of the French Revolution were",
    "In computer science, a hash table is",
    "The capital of Australia is",
    "Photosynthesis is the process by which plants",
    "Write a short description of the water cycle:",
    "The theory of relativity, proposed by Albert Einstein,",
    "A balanced diet should include",
    "The Roman Empire fell because",
]


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}  # choices for --dtype (default fp16)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Speed/size: fp16 vs masked vs physically pruned vs physically pruned + INT4")
    p.add_argument("--plan", default=DEFAULT_PLAN, help="result JSON providing the head set and the INT4 plan")
    p.add_argument("--heads-from", default="both", help="config name in the JSON (e.g. both, xai_single, xai_iter)")
    p.add_argument("--fraction", default=None, help="ratio key for iterative_gun7.json-style files (e.g. 0.2)")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help="comma-separated subset of: " + ",".join(ALL_VARIANTS))
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--n-repeats", type=int, default=1,
                   help="number of timed prompt passes after warm-up; ms/token over all repeats, ±std across repeats")
    p.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa"],
                   help="attn_implementation for the fp16/masked loads; default None keeps the library default "
                        "(transformers 4.44.2 -> sdpa). PrunedHeadAttention always computes eager attention")
    p.add_argument("--perplexity", action="store_true",
                   help="also measure WikiText-2 perplexity per variant (masked-vs-physical numerical equivalence; ~70 s per variant)")
    p.add_argument("--model", default=MODEL_NAME,
                   help="HF model id; default Mistral. E.g. Qwen/Qwen2.5-7B-Instruct together with "
                        "--plan results/model2_qwen_gun9.json --heads-from both")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="fp16",
                   help="load precision; default fp16. Use bf16 for Qwen2.5 (its main runs use bf16)")
    p.add_argument("--attn-kernel", choices=["eager", "sdpa", "auto"], default="eager",
                   help="PrunedHeadAttention kernel; default eager. auto = sdpa if the wrapped module is sdpa, so the numerical "
                        "comparison with the masked (sdpa) model uses the same kernel (on Qwen2.5 the sdpa-eager rounding gap grows in fp16/bf16)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--log", default=LOG_FILE)
    p.add_argument("--preflight", action="store_true", help="check the plan/head set without loading the model")
    p.add_argument("--dry-run", action="store_true", help="mini model, CPU, fake INT4; no plan JSON required")
    args = p.parse_args(argv)
    if args.dry_run:
        if args.output == OUTPUT_FILE:
            args.output = os.path.join(DRYRUN_DIR, "speed_dry.json")
        if args.log == LOG_FILE:
            args.log = os.path.join(DRYRUN_DIR, "log_speed_dry.txt")
    return args


# --------------------------------------------------------------------------- #
# Plan loading
# --------------------------------------------------------------------------- #
def load_heads_and_plan(path: str, config: str, fraction: Optional[str]) -> Tuple[List[Tuple[int, int]], Dict[int, LayerPlan], Dict[str, Any]]:
    """Pruned head list and per-layer plan (plan_summary) from a result JSON (ablation_gun6.json or iterative_gun7.json schema)."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "fractions" in d:
        if fraction is None:
            raise ValueError(f"{path} is a ratio-sweep JSON: --fraction is required ({list(d['fractions'])})")
        cfgs = d["fractions"][fraction]["configs"]
    else:
        cfgs = d["configs"]
    if config not in cfgs:
        raise ValueError(f"config {config!r} not found in the JSON; available: {list(cfgs)}")
    reps = [r for r in cfgs[config].get("repeats", []) if r.get("status") == "completed"]
    if not reps:
        raise ValueError(f"config {config!r} has no completed repeat")
    rep = reps[0]
    heads = sorted((int(l), int(h)) for l, h in rep["pruned_heads"])
    plan = {int(p["layer"]): LayerPlan(int(p["layer"]), pruned_heads=[int(h) for h in p["pruned_heads"]],
                                       attn_quant=p["attn_quant"], mlp_pruned=bool(p.get("mlp_pruned", False)),
                                       mlp_quant=p["mlp_quant"]) for p in rep["plan_summary"]}
    meta = {"plan_file": path, "config": config, "fraction": fraction, "seed": rep.get("seed"),
            "perplexity_in_source": rep.get("perplexity"), "n_heads": len(heads),
            "int4_layers": {"attn": sum(p.attn_quant == "int4" for p in plan.values()),
                            "mlp": sum(p.mlp_quant == "int4" for p in plan.values())}}
    return heads, plan, meta


def quant_only(plan: Dict[int, LayerPlan]) -> Dict[int, LayerPlan]:
    return {i: LayerPlan(i, pruned_heads=[], attn_quant=p.attn_quant, mlp_pruned=False, mlp_quant=p.mlp_quant) for i, p in plan.items()}


def prune_only(plan: Dict[int, LayerPlan]) -> Dict[int, LayerPlan]:
    return {i: LayerPlan(i, pruned_heads=list(p.pruned_heads), attn_quant="fp16", mlp_pruned=p.mlp_pruned, mlp_quant="fp16")
            for i, p in plan.items()}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def generate_timed(model: nn.Module, input_ids: torch.Tensor, max_new_tokens: int, pad_id: int) -> Tuple[torch.Tensor, float]:
    _sync()
    t0 = time.perf_counter()
    out = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_new_tokens,
                         min_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id, use_cache=True)
    _sync()
    return out, time.perf_counter() - t0


def load_model_attn(log: Logger, attn_implementation: str, dtype: torch.dtype = torch.float16, model_name: str = MODEL_NAME):
    """Same loading as run_e2e_pipeline.load_model (tokenizer, fp16, device_map=auto) plus an explicit attn_implementation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading model: {model_name} (dtype={dtype}, attn_implementation={attn_implementation})")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", attn_implementation=attn_implementation)
    model.eval()
    load_s = time.time() - t0
    log(f"Model loaded ({load_s:.1f} s), device={next(model.parameters()).device}, CUDA allocated={cuda_gb()} GB")
    return tokenizer, model, load_s


def attention_info(model: nn.Module) -> Dict[str, Any]:
    """Record the configured attention path and the actual attention class of layer 0 (PrunedHeadAttention after physical pruning)."""
    attn = model.model.layers[0].self_attn
    return {"attn_implementation": getattr(model.config, "_attn_implementation", None), "attention_class": type(attn).__name__,
            "wrapped_attention_class": type(getattr(attn, "_inner", None)).__name__ if hasattr(attn, "_inner") else None}


def measure_variant(model: nn.Module, encode, decode, prompts: List[str], max_new_tokens: int, pad_id: int,
                    log: Logger, n_repeats: int = 1) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    rep: Dict[str, Any] = {"n_params": count_parameters(model), "model_bytes": module_bytes(model),
                           "max_new_tokens": max_new_tokens, "n_prompts": len(prompts), "n_repeats": n_repeats,
                           "per_prompt": [], "repeats": []}
    warm_ids = encode(prompts[0]).to(device)
    _, warm_s = generate_timed(model, warm_ids, max_new_tokens, pad_id)  # one untimed warm-up
    rep["warmup_seconds"] = warm_s
    total_s, total_new = 0.0, 0
    for r in range(max(1, n_repeats)):
        rep_s, rep_new = 0.0, 0
        for i, prompt in enumerate(prompts):
            ids = encode(prompt).to(device)
            out, s = generate_timed(model, ids, max_new_tokens, pad_id)
            n_new = int(out.shape[1] - ids.shape[1])
            rep_s += s
            rep_new += n_new
            if r == 0:
                rep["per_prompt"].append({"prompt_tokens": int(ids.shape[1]), "new_tokens": n_new, "seconds": s,
                                          "ms_per_token": 1000.0 * s / max(n_new, 1)})
                if i == 0:
                    rep["sample_completion"] = decode(out[0, ids.shape[1]:])
        rep["repeats"].append({"repeat": r, "generated_tokens": rep_new, "generation_seconds": rep_s,
                               "ms_per_token": 1000.0 * rep_s / max(rep_new, 1)})
        total_s += rep_s
        total_new += rep_new
    rep["generated_tokens"] = total_new
    rep["generation_seconds"] = total_s
    rep["ms_per_token"] = 1000.0 * total_s / max(total_new, 1)
    rep["tokens_per_second"] = total_new / total_s if total_s > 0 else None
    per_rep = [x["ms_per_token"] for x in rep["repeats"]]
    rep["ms_per_token_std"] = float(torch.tensor(per_rep).std(unbiased=True)) if len(per_rep) > 1 else 0.0
    log(f"  {total_new} tokens / {total_s:.2f} s -> {rep['ms_per_token']:.2f} ms/token (±{rep['ms_per_token_std']:.2f}, "
        f"{len(per_rep)} repeats); size {rep['model_bytes']['gb']:.3f} GB; params {rep['n_params']:,}")
    return rep


def run_variant(name: str, heads, plan, ctx: Dict[str, Any], rep: Dict[str, Any], store: ResultStore, log: Logger) -> None:
    args, dry = ctx["args"], ctx["dry"]
    rep.update({"status": "running", "started_at": datetime.now().isoformat(timespec="seconds")})
    t0 = time.time()
    set_seed(args.seed)
    rep["cuda_allocated_before_load_gb"] = ensure_clean_gpu(log, name)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    box: Dict[str, Any] = {}
    try:
        if dry:
            box["tokenizer"], box["model"], rep["model_load_seconds"] = dry.load_model(log)
        elif args.attn_implementation:
            box["tokenizer"], box["model"], rep["model_load_seconds"] = load_model_attn(log, args.attn_implementation,
                                                                                         dtype=DTYPES[args.dtype], model_name=args.model)
        else:
            box["tokenizer"], box["model"], rep["model_load_seconds"] = load_model(log, DTYPES[args.dtype], model_name=args.model)
        model = box["model"]
        rep["params_before"] = count_parameters(model)
        if name in ("masked", "masked_int4"):
            rep["pruning"] = apply_structural_pruning(model, prune_only(plan), verbose=False)
        elif name in ("physical", "physical_int4"):
            rep["pruning"] = physically_prune_heads(model, heads, verbose=False, attn_kernel=args.attn_kernel)
            rep["attn_kernel"] = next((m.attn_kernel for m in model.modules() if hasattr(m, "attn_kernel")), None)  # resolved kernel
        if name.endswith("_int4"):
            rep["quantization"] = quantize_with_fallback(model, quant_only(plan), log, _StoreView(store, rep), "cfg")
        tok = box["tokenizer"]
        if dry:
            from eval_mmlu import FakeTokenizer

            ftok = FakeTokenizer(model.config.vocab_size)
            encode = lambda s: torch.tensor([ftok(s)["input_ids"]])  # noqa: E731
            decode = lambda ids: " ".join(str(int(i)) for i in ids)  # noqa: E731
            pad_id = 0
        else:
            encode = lambda s: tok(s, return_tensors="pt").input_ids  # noqa: E731
            decode = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
            pad_id = tok.pad_token_id
        if torch.cuda.is_available():  # separate load/apply peak and GENERATION peak (peak_vram_gb = max of both, the per-variant peak)
            torch.cuda.synchronize()
            rep["peak_vram_load_apply_gb"] = cuda_gb("peak")
            rep["cuda_allocated_after_apply_gb"] = cuda_gb()
            torch.cuda.reset_peak_memory_stats()
        rep.update(attention_info(model))
        log(f"  attention: config={rep['attn_implementation']} class={rep['attention_class']}"
            + (f" (wrapped: {rep['wrapped_attention_class']})" if rep.get("wrapped_attention_class") else ""))
        rep.update(measure_variant(model, encode, decode, ctx["prompts"], args.max_new_tokens, pad_id, log, args.n_repeats))
        if args.perplexity:  # masked-vs-physical numerical equivalence (same head set, same INT4 plan)
            t_ppl = time.time()
            if dry:
                rep["perplexity"] = dry.perplexity(model)
            else:
                rep["perplexity"] = ctx["compute_perplexity"](model, tok, ctx["text"], next(model.parameters()).device)
            rep["perplexity_seconds"] = time.time() - t_ppl
            log(f"  perplexity = {rep['perplexity']:.4f} ({rep['perplexity_seconds']:.1f} s)")
        rep["status"] = "completed"
    finally:
        gen_peak = cuda_gb("peak")
        if rep.get("peak_vram_load_apply_gb") is not None and gen_peak is not None:
            rep["peak_vram_generation_gb"] = gen_peak
            rep["peak_vram_gb"] = max(rep["peak_vram_load_apply_gb"], gen_peak)  # per-variant peak
        else:
            rep["peak_vram_gb"] = gen_peak
        box.clear()
        rep["cuda_allocated_after_free_gb"] = release_gpu()
        rep["seconds"] = time.time() - t0
        store.save(f"{name} done")


def summarize(variants: Dict[str, Any]) -> Dict[str, Any]:
    ref = variants.get("fp16")
    out: Dict[str, Any] = {}
    for name, v in variants.items():
        if v.get("status") != "completed":
            continue
        row = {"ms_per_token": v["ms_per_token"], "ms_per_token_std": v.get("ms_per_token_std", 0.0),
               "tokens_per_second": v["tokens_per_second"], "model_gb": v["model_bytes"]["gb"], "n_params": v["n_params"],
               "peak_vram_gb": v.get("peak_vram_gb"), "attn_implementation": v.get("attn_implementation"),
               "attention_class": v.get("attention_class"), "perplexity": v.get("perplexity")}
        masked = variants.get("masked")
        if v.get("perplexity") is not None and masked and masked.get("perplexity") is not None:
            row["perplexity_minus_masked"] = v["perplexity"] - masked["perplexity"]
        if ref and ref.get("status") == "completed":
            row["speedup_vs_fp16"] = ref["ms_per_token"] / v["ms_per_token"] if v["ms_per_token"] else None
            row["size_ratio_vs_fp16"] = v["model_bytes"]["bytes"] / ref["model_bytes"]["bytes"]
            row["param_ratio_vs_fp16"] = v["n_params"] / ref["n_params"]
        out[name] = row
    return out


def format_table(summary: Dict[str, Any]) -> str:
    def f(v: Any, s: str) -> str:
        return format(v, s) if isinstance(v, (int, float)) else "-"

    lines = [f"{'variant':<15}{'ms/token':>10}{'±std':>7}{'tok/s':>8}{'speed×':>7}{'GB':>8}{'size×':>8}{'params':>14}{'VRAM':>7}{'ppl':>10}{'attn':>22}"]
    for n, r in summary.items():
        lines.append(f"{n:<15}{f(r['ms_per_token'], '.2f'):>10}{f(r.get('ms_per_token_std'), '.2f'):>7}{f(r['tokens_per_second'], '.1f'):>8}"
                     f"{f(r.get('speedup_vs_fp16'), '.3f'):>7}{f(r['model_gb'], '.3f'):>8}{f(r.get('size_ratio_vs_fp16'), '.3f'):>8}"
                     f"{r['n_params']:>14,}{f(r.get('peak_vram_gb'), '.1f'):>7}{f(r.get('perplexity'), '.4f'):>10}"
                     f"{str(r.get('attention_class') or '-'):>22}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log = Logger(args.log, to_file=should_log_to_file(args, LOG_FILE))  # dry-run/preflight do not write to the default (tracked) log
    store = ResultStore(args.output, log)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in ALL_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {unknown}; valid: {ALL_VARIANTS}")
    if args.dtype != "fp16" and any(v.endswith("_int4") for v in variants):
        raise SystemExit("--dtype bf16 is only supported with the fp16/masked/physical variants (the INT4 path assumes fp16 compute)")
    prompts = PROMPTS[: args.n_prompts]
    dry: Optional[DryRun] = DryRun(args.seed) if args.dry_run else None
    log(f"===== Starting speed/size measurement: variants={variants} plan={args.plan} heads_from={args.heads_from} "
        f"fraction={args.fraction} max_new_tokens={args.max_new_tokens} n_prompts={len(prompts)} dry_run={args.dry_run} =====", tag="SPD")
    if dry:  # mini model: plan from fake scores (same as the run_ablation_tests dry run)
        from compressor import build_compression_plan
        from run_ablation_tests import heads_from_tiers

        plan = build_compression_plan(dry.tiers)
        heads = heads_from_tiers(dry.tiers)
        meta = {"plan_file": None, "config": "dry", "n_heads": len(heads)}
    else:
        heads, plan, meta = load_heads_and_plan(args.plan, args.heads_from, args.fraction)
    log(f"Head set: {len(heads)} heads; INT4 layers attn/mlp={meta.get('int4_layers')}; source {meta}", tag="SPD")
    store.data = {"run": {"started_at": datetime.now().isoformat(timespec="seconds"), "status": "running", "args": vars(args),
                          "model": "mini (dry-run)" if dry else args.model, "env": env_info(), "prompts": prompts},
                  "source": {**meta, "pruned_heads": [list(h) for h in heads]},
                  "variants": {v: {"status": "pending"} for v in variants}, "summary": {}}
    if args.preflight:
        log("Preflight check completed (model not loaded).", tag="SPD")
        return 0
    if not torch.cuda.is_available() and not dry:
        log("WARNING: no CUDA device; a 7B model is impractical on CPU. Run on a GPU (or use --dry-run).", tag="SPD")
    store.save("start")
    ctx = {"args": args, "dry": dry, "prompts": prompts}
    if args.perplexity and not dry:  # baseline_eval.compute_perplexity harness, same as run_iterative_pruning
        from run_e2e_pipeline import load_wikitext_text, perplexity_tools

        log("Loading the WikiText-2 test set (--perplexity)...", tag="SPD")
        ctx["text"] = load_wikitext_text()
        ctx["compute_perplexity"], max_len, stride = perplexity_tools()
        store.data["run"]["perplexity_settings"] = {"max_length": max_len, "stride": stride, "dataset": "wikitext-2-raw-v1 (test split)"}
    if dry and args.attn_implementation:
        log(f"[DRY] --attn-implementation={args.attn_implementation} is ignored on the mini model (constructed with eager attention)", tag="SPD")
    exit_code = 0
    try:
        for i, name in enumerate(variants, 1):
            log(f"===== VARIANT {i}/{len(variants)}: {name} =====", tag="SPD")
            try:
                run_variant(name, heads, plan, ctx, store.data["variants"][name], store, log)
            except Exception as e:  # a failing variant must not abort the others
                store.data["variants"][name].update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
                log(f"VARIANT {name} ERROR: {e!r}", tag="ERR")
                log(store.data["variants"][name]["traceback"], tag="ERR")
                exit_code = 1
            store.data["summary"] = summarize(store.data["variants"])
            store.save(f"{name} summary")
        store.data["run"]["status"] = "completed" if exit_code == 0 else "completed_with_failures"
    except BaseException as e:
        store.data["run"].update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
        log(f"ERROR: {e!r}", tag="ERR")
        exit_code = 1
    finally:
        store.data["summary"] = summarize(store.data["variants"])
        store.data["run"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        store.save("final")
        log("SUMMARY:\n" + format_table(store.data["summary"]), tag="SPD")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
