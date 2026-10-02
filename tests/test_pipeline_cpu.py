# test_pipeline_cpu.py
"""
XAI-JQP thin orchestration: xai_engine -> allocate_compression_tiers -> apply_jqp

Uses the small randomly initialised MistralForCausalLM and the fake calibration
batches from tests/test_xai_engine_mini.py; no GPU is required. The full
Mistral-7B pipeline is run by the GPU scripts (run_xai_on_mistral.py,
run_e2e_pipeline.py, run_ablation_tests.py); this file is intentionally the
shortest chain showing that the three stages connect.

Usage:
    python tests/test_pipeline_cpu.py                  # mini model, n_tiers=3, pruning only (CPU)
    python tests/test_pipeline_cpu.py --n-tiers 4
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments"), os.path.join(_REPO_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compressor import allocate_compression_tiers, apply_jqp, format_tier_report  # noqa: E402
from xai_engine import calculate_importance_scores, format_importance_report  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="XAI-JQP mini orchestration (CPU only)")
    p.add_argument("--n-tiers", type=int, default=3, choices=(3, 4))
    p.add_argument("--n-steps", type=int, default=8, help="number of IG steps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from test_xai_engine_mini import build_dummy_dataloader, build_mini_model

    print("=== XAI-JQP (Explainable AI-Guided Joint Quantization and Pruning) — mini model ===")

    print("\nStep 1: building the mini MistralForCausalLM (random weights) and fake calibration batches...")
    torch.manual_seed(args.seed)
    model = build_mini_model(seed=args.seed)
    dataloader = build_dummy_dataloader(seed=args.seed + 1)
    cfg = model.config
    print(f"  {cfg.num_hidden_layers} layers × {cfg.num_attention_heads} heads, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters, device=cpu")

    # Stage 1: structural explainability analysis
    print("\nStep 2: running the XAI engine (Captum LayerIntegratedGradients)...")
    scores = calculate_importance_scores(model, dataloader, n_steps=args.n_steps, verbose=False)
    print(format_importance_report(scores))

    # Stage 2: dynamic budget allocation + JQP (quantization needs bitsandbytes/GPU, so it is disabled here)
    print(f"\nStep 3: tier assignment (n_tiers={args.n_tiers}, percentile) and structural pruning...")
    tiers = allocate_compression_tiers(scores, n_tiers=args.n_tiers)
    print(format_tier_report(scores, tiers))
    result = apply_jqp(model, scores, tiers=tiers, quantize=False, verbose=False)
    print(f"  pruning: {result.pruning}, quantization (disabled): {result.quantization}, "
          f"nominal decoder size ratio {result.budget['size_ratio']:.3f}")

    # Pruned heads must score exactly 0 on re-scoring
    new_scores = calculate_importance_scores(model, dataloader, n_steps=args.n_steps, verbose=False)
    pruned = [k for k, t in tiers.items() if t == "prune"]
    zeroed = all(new_scores[k] == 0.0 for k in pruned)
    print(f"\nStep 4: are the re-computed scores of the {len(pruned)} pruned blocks exactly 0? {zeroed}")

    print("\n=== XAI-JQP mini pipeline finished ===")
    return 0 if zeroed else 1


if __name__ == "__main__":
    sys.exit(main())
