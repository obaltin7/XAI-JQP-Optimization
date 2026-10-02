"""
Diagnosis of the masked-vs-physical logit gap on Qwen: logic error or attention-kernel (sdpa vs eager) mismatch?

On the 7B smoke run (run_qwen_experiments.py --smoke, only 3 heads) the masked vs physically pruned |Δlogit|max was 9.37 in fp16
and 17.9 in bf16, versus ~0 on Mistral and on the mini Qwen2 test. This script measures three differences over a dtype × scenario
grid on a SMALL real model of the same architecture family (default Qwen/Qwen2.5-0.5B-Instruct: 14 q / 2 kv heads = GQA group 7,
q/k/v biases; ~1 GB):
  native_sdpa_vs_eager        same MASKED weights, native sdpa model vs native eager model, no wrapper (noise floor)
  masked_vs_physical_eager    masked (sdpa-loaded) vs physically pruned, PrunedHeadAttention attn_kernel="eager" (historical default)
  masked_vs_physical_matched  same, attn_kernel="auto" (wrapped module sdpa -> sdpa): matched kernel
It also records, in fp32, the per-layer |max| and median |x| of the residual stream (hidden_states) to expose massive activations.
Interpretation: if all three are ~1e-4 in fp32 the logic is correct; if at low precision masked_vs_physical_eager ≈ native_sdpa_vs_eager,
the gap comes from the kernel mismatch rather than the wrapper; matched << eager means the attn_kernel fix is effective.

Usage (no GPU required; fp16 runs on CUDA if available and is skipped otherwise; the model is downloaded if not cached):
    python tools/diagnostics/diagnose_qwen_kernel.py
    python tools/diagnostics/diagnose_qwen_kernel.py --model Qwen/Qwen2.5-0.5B-Instruct --output results/diagnose_qwen_kernel_gun12.json

Output: results/diagnose_qwen_kernel_gun12.json (run metadata, residual_profile_fp32, cases).
"""
import argparse
import importlib.util
import json
import os
import sys
import types
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

if "captum" not in sys.modules and importlib.util.find_spec("captum") is None:  # stub captum when absent so compressor -> xai_engine imports (attribution is not used here)
    _captum = types.ModuleType("captum")
    _captum.attr = types.ModuleType("captum.attr")
    _captum.attr.LayerIntegratedGradients = None
    sys.modules["captum"], sys.modules["captum.attr"] = _captum, _captum.attr

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compressor import LayerPlan, apply_structural_pruning, physically_prune_heads  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_FILE = os.path.join("results", "diagnose_qwen_kernel_gun12.json")
TEXT = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave "
        "Eiffel, whose company designed and built the tower from 1887 to 1889. Locally nicknamed the Iron Lady, it was constructed as the "
        "centrepiece of the 1889 World's Fair. ")


def _plan(heads, n_layers: int) -> Dict[int, LayerPlan]:
    plan = {i: LayerPlan(i) for i in range(n_layers)}
    for layer, h in heads:
        plan[layer].pruned_heads.append(h)
    return plan


@torch.no_grad()
def _logits(model, enc) -> torch.Tensor:
    return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.float().cpu()


def _stats(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    d = (a - b).abs()[valid]
    return {"max": float(d.max()), "mean": float(d.mean()), "top1": float((a.argmax(-1) == b.argmax(-1))[valid].float().mean())}


def scenario(tok, cfg, name: str, device: str):
    """smoke3_single: the 3 heads of the run_model2 smoke run + one short sequence; many_padded: 3 heads per layer (group boundary 6|7)
    plus an ENTIRE kv group in one layer, batch 3, right padding, longest sequence >= 256 tokens."""
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    group = n_heads // cfg.num_key_value_heads
    if name == "smoke3_single":
        heads = [(0, 1), (n_layers // 2, n_heads - 1), (n_layers - 1, 0)]
        enc = tok(TEXT, return_tensors="pt")
    else:
        heads = [(l, h) for l in range(n_layers) for h in (0, group - 1, group)] + [(3, h) for h in range(group, 2 * group)]
        enc = tok([TEXT * 5, TEXT * 2, "Short one."], return_tensors="pt", padding=True)
    return sorted(set(heads)), {k: v.to(device) for k, v in enc.items()}


def run_case(args, tok, dtype: torch.dtype, device: str, scen: str) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM

    def load(impl: str):
        return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation=impl).to(device).eval()

    out: Dict[str, Any] = {"dtype": str(dtype).split(".")[-1], "device": device, "scenario": scen}
    results: Dict[str, Dict[str, float]] = {}
    for kernel, key in (("eager", "masked_vs_physical_eager"), ("auto", "masked_vs_physical_matched")):
        model = load("sdpa")
        heads, enc = scenario(tok, model.config, scen, device)
        valid = enc["attention_mask"].bool().cpu()
        ref = _logits(model, enc)
        apply_structural_pruning(model, _plan(heads, model.config.num_hidden_layers), verbose=False)
        masked = _logits(model, enc)
        if kernel == "eager":  # noise floor: same masked weights, native eager model (no wrapper)
            other = load("eager")
            other.load_state_dict(model.state_dict())
            results["native_sdpa_vs_eager"] = _stats(masked, _logits(other, enc), valid)
            del other
            out.update({"n_heads_pruned": len(heads), "input_shape": list(enc["input_ids"].shape), "n_valid_tokens": int(valid.sum()),
                        "abs_logit_max": float(ref.abs().max()), "masked_vs_unpruned": _stats(ref, masked, valid)})
        physically_prune_heads(model, heads, verbose=False, attn_kernel=kernel)
        results[key] = _stats(masked, _logits(model, enc), valid)
        results[key]["attn_kernel"] = next(m.attn_kernel for m in model.modules() if hasattr(m, "attn_kernel"))
        del model
    out.update(results)
    return out


@torch.no_grad()
def residual_profile(args, tok) -> List[Dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    hs = model(**tok(TEXT, return_tensors="pt"), output_hidden_states=True).hidden_states
    rows = []
    for i, h in enumerate(hs):
        x = h[0].abs()
        pos, dim = divmod(int(x.argmax()), x.shape[1])
        top = float(x.max())
        exp = torch.floor(torch.log2(torch.tensor(top)))
        rows.append({"hidden_state": i, "abs_max": top, "token": pos, "dim": dim, "median_abs": float(x.median()),
                     "bf16_step": float(2.0 ** (exp - 7)), "fp16_step": float(2.0 ** (exp - 10))})  # bf16: 7-bit mantissa, fp16: 10-bit
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Diagnosis of the Qwen masked-vs-physical gap (sdpa vs eager kernel mismatch)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default=OUTPUT_FILE)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    plan = [(torch.float32, "cpu")] + ([(torch.float16, "cuda")] if cuda else []) + [(torch.bfloat16, "cpu")]
    data: Dict[str, Any] = {
        "run": {"created_at": datetime.now().isoformat(timespec="seconds"), "model": args.model,
                "env": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": cuda,
                        "gpu": torch.cuda.get_device_name(0) if cuda else None},
                "note": "fp16 only on CUDA (fp16 attention on CPU is unsupported/very slow); bf16 and fp32 on CPU. Difference = |Δlogit| "
                        "over valid (non-pad) tokens; top1 = argmax agreement."},
        "residual_profile_fp32": residual_profile(args, tok), "cases": []}
    for dtype, device in plan:
        for scen in ("smoke3_single", "many_padded"):
            case = run_case(args, tok, dtype, device, scen)
            data["cases"].append(case)
            print(f"{case['dtype']:>9} {device:>4} {scen:>14} T={case['input_shape']}: native sdpa-eager max {case['native_sdpa_vs_eager']['max']:.3e} | "
                  f"masked-physical(eager) max {case['masked_vs_physical_eager']['max']:.3e} | "
                  f"masked-physical(matched) max {case['masked_vs_physical_matched']['max']:.3e}")
    big = max(data["residual_profile_fp32"], key=lambda r: r["abs_max"])
    print(f"residual stream: largest |x| = {big['abs_max']:.1f} (hidden_state {big['hidden_state']}, token {big['token']}, dim {big['dim']}); "
          f"median |x| = {big['median_abs']:.3f}; bf16 step {big['bf16_step']:g}, fp16 step {big['fp16_step']:g}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, args.output)
    print(f"Written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
