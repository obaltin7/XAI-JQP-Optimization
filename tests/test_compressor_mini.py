"""
CPU-only mini tests for compressor.py.

Instead of real Mistral-7B scores, small random score dicts with the same key
schema (layer_{i}.attn.head_{h}, layer_{i}.mlp) are generated, and the tier
assignment / plan / budget / cluster analysis functions are checked on CPU in
seconds. Structural pruning is applied to the mini MistralForCausalLM
(tests/test_xai_engine_mini.build_mini_model) and closed with xai_engine as
"the importance score of a pruned head is exactly 0". Quantization
(bitsandbytes, GPU) is not run here.

Usage:
    python tests/test_compressor_mini.py      # prints a report
    pytest tests/test_compressor_mini.py -q   # runs the asserts
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compressor import (  # noqa: E402
    DEFAULT_TIER_FRACTIONS,
    TIER_LABELS,
    allocate_compression_tiers,
    analyze_natural_clusters,
    apply_jqp,
    build_compression_plan,
    estimate_compression_budget,
    format_cluster_report,
    format_tier_report,
    head_magnitude_scores,
    tier_thresholds,
)

N_LAYERS, N_HEADS = 4, 8


def build_random_scores(seed: int = 0, n_layers: int = N_LAYERS, n_heads: int = N_HEADS):
    """Mimic the scales of the Mistral-7B importance scores: heads ~0.01-0.7 (log-normal), MLPs ~7-20."""
    rng = random.Random(seed)
    scores = {}
    for i in range(n_layers):
        for h in range(n_heads):
            scores[f"layer_{i}.attn.head_{h}"] = 10 ** rng.uniform(-2.0, -0.15)
        scores[f"layer_{i}.mlp"] = rng.uniform(7.0, 20.0)
    return scores


def _heads(scores):
    return {k: v for k, v in scores.items() if ".attn." in k}


def _mlps(scores):
    return {k: v for k, v in scores.items() if k.endswith(".mlp")}


# --------------------------------------------------------------------------- #
# Tier assignment
# --------------------------------------------------------------------------- #
def test_three_tiers_have_exactly_three_labels():
    scores = build_random_scores()
    tiers = allocate_compression_tiers(scores, n_tiers=3)
    assert set(tiers.values()) == {"prune", "int4", "fp16"}
    assert list(tiers.keys()) == list(scores.keys())  # order preserved, no block skipped


def test_four_tiers_have_exactly_four_labels():
    scores = build_random_scores()
    tiers = allocate_compression_tiers(scores, n_tiers=4)
    assert set(tiers.values()) == {"prune", "int4", "int8", "fp16"}
    assert list(tiers.keys()) == list(scores.keys())


def test_lowest_head_pruned_highest_fp16():
    scores = build_random_scores()
    heads = _heads(scores)
    for n_tiers in (3, 4):
        tiers = allocate_compression_tiers(scores, n_tiers=n_tiers)
        assert tiers[min(heads, key=heads.get)] == "prune"
        assert tiers[max(heads, key=heads.get)] == "fp16"
        # tiers are monotone in score: every head in prune scores below every head in int4, etc.
        labels = TIER_LABELS[n_tiers]
        ranges = tier_thresholds(scores, tiers)["attn"]
        for lo_label, hi_label in zip(labels[:-1], labels[1:]):
            assert ranges[lo_label][1] <= ranges[hi_label][0]


def test_mlp_min_tier_default_protects_mlp_blocks():
    scores = build_random_scores()
    mlps = _mlps(scores)
    tiers = allocate_compression_tiers(scores, n_tiers=3)  # mlp_min_tier="int4"
    assert tiers[min(mlps, key=mlps.get)] == "int4"
    assert tiers[max(mlps, key=mlps.get)] == "fp16"
    assert all(tiers[k] != "prune" for k in mlps)
    # when explicitly allowed, the lowest MLP is pruned
    tiers_p = allocate_compression_tiers(scores, n_tiers=3, mlp_min_tier="prune")
    assert tiers_p[min(mlps, key=mlps.get)] == "prune"


def test_fraction_counts_are_exact_within_kind():
    scores = build_random_scores(seed=3, n_layers=10, n_heads=10)  # 100 head, 10 MLP
    tiers = allocate_compression_tiers(scores, n_tiers=3, tier_fractions=(0.2, 0.4, 0.4), mlp_min_tier=None)
    heads = _heads(scores)
    counts = {l: sum(1 for k in heads if tiers[k] == l) for l in TIER_LABELS[3]}
    assert counts == {"prune": 20, "int4": 40, "fp16": 40}
    mcounts = {l: sum(1 for k in _mlps(scores) if tiers[k] == l) for l in TIER_LABELS[3]}
    assert mcounts == {"prune": 2, "int4": 4, "fp16": 4}


def test_scale_invariance_within_kind():
    """Percentile is rank based: scaling head scores by 1000x must not change the assignment."""
    scores = build_random_scores(seed=5)
    scaled = {k: (v * 1000 if ".attn." in k else v) for k, v in scores.items()}
    assert allocate_compression_tiers(scores, 3) == allocate_compression_tiers(scaled, 3)


def test_invalid_arguments_raise():
    scores = build_random_scores()
    for bad in (dict(n_tiers=2), dict(n_tiers=5), dict(n_tiers=3, tier_fractions=(0.5, 0.5)),
                dict(n_tiers=3, tier_fractions=(0.2, 0.2, 0.2)), dict(n_tiers=3, mlp_min_tier="int2")):
        try:
            allocate_compression_tiers(scores, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {bad}")
    try:
        allocate_compression_tiers({"layer_0.attn.head_0": 0.1, "unknown_block": 0.2}, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unrecognized key")


def test_exclude_heads_removes_blocks_and_keeps_default_behaviour():
    """exclude_heads (iterative pruning support): excluded blocks get no tier, the remaining pool is re-split."""
    scores = build_random_scores(seed=11, n_layers=10, n_heads=10)  # 100 head, 10 MLP
    base = allocate_compression_tiers(scores, 3, tier_fractions=(0.2, 0.4, 0.4))
    # None / empty -> identical result
    assert allocate_compression_tiers(scores, 3, tier_fractions=(0.2, 0.4, 0.4), exclude_heads=None) == base
    assert allocate_compression_tiers(scores, 3, tier_fractions=(0.2, 0.4, 0.4), exclude_heads=[]) == base

    # exclude the round-1 pruned heads: 20 heads leave the ranking, the remaining 80 are split 16/32/32
    round1 = [k for k, t in base.items() if t == "prune" and ".attn." in k]
    assert len(round1) == 20
    tiers = allocate_compression_tiers(scores, 3, tier_fractions=(0.2, 0.4, 0.4), exclude_heads=round1)
    assert not any(k in tiers for k in round1)
    assert list(tiers.keys()) == [k for k in scores if k not in round1]  # order preserved
    heads = {k: v for k, v in _heads(scores).items() if k not in round1}
    counts = {l: sum(1 for k in heads if tiers[k] == l) for l in TIER_LABELS[3]}
    assert counts == {"prune": 16, "int4": 32, "fp16": 32}
    # round-2 prunes must be the 16 lowest-scoring heads among those not pruned in round 1
    round2 = sorted((k for k, t in tiers.items() if t == "prune"), key=heads.get)
    lowest16 = sorted(sorted(heads, key=heads.get)[:16])
    assert sorted(round2) == lowest16
    # the MLP pool is unaffected by the exclusion
    assert {k: tiers[k] for k in _mlps(scores)} == {k: base[k] for k in _mlps(scores)}
    # unknown name -> error; excluding everything -> error
    for bad in (["layer_99.attn.head_0"], list(scores)):
        try:
            allocate_compression_tiers(scores, 3, exclude_heads=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: exclude_heads={bad[:1]}...")


def test_head_magnitude_scores_on_mini_model():
    """L2 norm of the o_proj column block; a zeroed head has norm 0, and the scores can be fed to tier assignment."""
    from test_xai_engine_mini import HEAD_DIM, build_mini_model

    model = build_mini_model()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    mag = head_magnitude_scores(model)
    assert list(mag) == [f"layer_{i}.attn.head_{h}" for i in range(n_layers) for h in range(n_heads)]
    assert all(v > 0 for v in mag.values())
    # matches a manual computation
    w = model.model.layers[1].self_attn.o_proj.weight
    manual = float(w.detach()[:, 2 * HEAD_DIM:3 * HEAD_DIM].float().norm())
    assert abs(mag["layer_1.attn.head_2"] - manual) < 1e-5
    # a head-only dict can be passed to allocate (no MLP pool); the lowest-norm head is pruned
    tiers = allocate_compression_tiers(mag, 3, tier_fractions=(0.25, 0.25, 0.5))
    assert set(tiers) == set(mag) and tiers[min(mag, key=mag.get)] == "prune"
    # after masking, the pruned head's norm is exactly 0
    with torch.no_grad():
        model.model.layers[0].self_attn.o_proj.weight[:, 0:HEAD_DIM] = 0
    assert head_magnitude_scores(model)["layer_0.attn.head_0"] == 0.0


# --------------------------------------------------------------------------- #
# Plan and budget
# --------------------------------------------------------------------------- #
def test_plan_aggregates_heads_and_budget_is_monotone():
    scores = build_random_scores()
    plan3 = build_compression_plan(allocate_compression_tiers(scores, 3))
    assert sorted(plan3) == list(range(N_LAYERS))
    total_pruned = sum(len(p.pruned_heads) for p in plan3.values())
    assert total_pruned == round(0.2 * N_LAYERS * N_HEADS)
    for p in plan3.values():
        assert p.attn_quant in ("int4", "fp16", "prune")
        assert p.mlp_quant in ("int4", "fp16") and not p.mlp_pruned

    # if all heads are pruned the block is "prune"
    all_prune = {f"layer_0.attn.head_{h}": "prune" for h in range(N_HEADS)}
    assert build_compression_plan(all_prune)[0].attn_quant == "prune"
    # median rule: 4 int4 + 4 fp16 out of 8 heads -> lower median = int4; "max" -> fp16
    mixed = {f"layer_0.attn.head_{h}": ("int4" if h < 4 else "fp16") for h in range(N_HEADS)}
    assert build_compression_plan(mixed)[0].attn_quant == "int4"
    assert build_compression_plan(mixed, attn_rule="max")[0].attn_quant == "fp16"

    # budget with deterministic plans: no compression 1.0 > all int4 0.25 > heads pruned + MLP int4
    none = estimate_compression_budget(build_compression_plan({k: "fp16" for k in scores}))
    all4 = estimate_compression_budget(build_compression_plan({k: "int4" for k in scores}))
    heads_gone = estimate_compression_budget(
        build_compression_plan({k: ("prune" if ".attn." in k else "int4") for k in scores})
    )
    assert abs(none["size_ratio"] - 1.0) < 1e-9 and none["pruned_ratio"] == 0
    assert abs(all4["size_ratio"] - 0.25) < 1e-9
    assert 0 < heads_gone["size_ratio"] < all4["size_ratio"]
    # the pruned ratio increases monotonically with tier_fractions (the size ratio need not be
    # monotone for small dicts because of the median rule; see the build_compression_plan docstring)
    b20 = estimate_compression_budget(build_compression_plan(allocate_compression_tiers(scores, 3, tier_fractions=(0.2, 0.4, 0.4))))
    b60 = estimate_compression_budget(build_compression_plan(allocate_compression_tiers(scores, 3, tier_fractions=(0.6, 0.2, 0.2))))
    assert b60["pruned_ratio"] > b20["pruned_ratio"] > 0
    assert 0 < b20["size_ratio"] < 1.0 and 0 < b60["size_ratio"] < 1.0


# --------------------------------------------------------------------------- #
# Natural-cluster analysis
# --------------------------------------------------------------------------- #
def test_cluster_analysis_recovers_planted_clusters():
    rng = np.random.default_rng(0)
    scores = {}
    i = 0
    for center in (0.01, 0.1, 1.0):  # 3 well-separated clusters on the log scale
        for _ in range(60):
            scores[f"layer_{i // 32}.attn.head_{i % 32}"] = float(center * 10 ** rng.normal(0, 0.08))
            i += 1
    for l in range(4):
        scores[f"layer_{l}.mlp"] = float(rng.uniform(7, 10))
    res = analyze_natural_clusters(scores, ks=(2, 3, 4, 5), log_scale=True)
    assert res["attn"]["recommended_k"] == 3
    assert res["attn"]["by_k"][3]["silhouette"] > 0.8
    assert sorted(res["attn"]["by_k"][3]["sizes"]) == [60, 60, 60]
    b = res["attn"]["by_k"][3]["boundaries"]
    assert 0.01 < b[0] < 0.1 < b[1] < 1.0
    assert "mlp" in res and res["mlp"]["n"] == 4  # small pool: only k < n is tried
    assert set(res["mlp"]["by_k"]) == {2, 3}
    assert isinstance(format_cluster_report(res), str)


# --------------------------------------------------------------------------- #
# Structural pruning (mini Mistral, CPU), cross-checked with xai_engine
# --------------------------------------------------------------------------- #
def test_structural_pruning_on_mini_model_zeroes_importance():
    from test_xai_engine_mini import HEAD_DIM, build_dummy_dataloader, build_mini_model
    from xai_engine import calculate_importance_scores

    model = build_mini_model()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    scores = build_random_scores(seed=7, n_layers=n_layers, n_heads=n_heads)
    tiers = allocate_compression_tiers(scores, 3, tier_fractions=(0.25, 0.25, 0.5))
    pruned = [k for k, t in tiers.items() if t == "prune"]
    assert len(pruned) == 2 and all(".attn." in k for k in pruned)

    result = apply_jqp(model, scores, tiers=tiers, quantize=False, verbose=False)
    assert result.model is model
    assert result.pruning == {"pruned_heads": 2, "pruned_mlp_blocks": 0}
    assert result.quantization == {"int4_modules": 0, "int8_modules": 0}

    # the weight slices are really zero; neighbouring heads are untouched
    for k in pruned:
        layer = int(k.split(".")[0].split("_")[1])
        head = int(k.split("head_")[1])
        sl = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
        attn = model.model.layers[layer].self_attn
        assert torch.all(attn.o_proj.weight[:, sl] == 0) and torch.all(attn.q_proj.weight[sl, :] == 0)
        assert attn.o_proj.weight.abs().sum() > 0

    # cross-check with Stage 1: the pruned head's importance score is exactly 0, the others > 0
    new_scores = calculate_importance_scores(model, build_dummy_dataloader(n_batches=1), n_steps=4, verbose=False)
    for k in pruned:
        assert new_scores[k] == 0.0
    assert all(v > 0 for k, v in new_scores.items() if k not in pruned)

    # with MLP pruning enabled, down_proj is zeroed
    model2 = build_mini_model()
    tiers2 = dict(tiers)
    tiers2["layer_1.mlp"] = "prune"
    r2 = apply_jqp(model2, scores, tiers=tiers2, quantize=False, verbose=False)
    assert r2.pruning["pruned_mlp_blocks"] == 1
    assert torch.all(model2.model.layers[1].mlp.down_proj.weight == 0)


TESTS = [
    test_three_tiers_have_exactly_three_labels,
    test_four_tiers_have_exactly_four_labels,
    test_lowest_head_pruned_highest_fp16,
    test_mlp_min_tier_default_protects_mlp_blocks,
    test_fraction_counts_are_exact_within_kind,
    test_scale_invariance_within_kind,
    test_invalid_arguments_raise,
    test_exclude_heads_removes_blocks_and_keeps_default_behaviour,
    test_head_magnitude_scores_on_mini_model,
    test_plan_aggregates_heads_and_budget_is_monotone,
    test_cluster_analysis_recovers_planted_clusters,
    test_structural_pruning_on_mini_model_zeroes_importance,
]

if __name__ == "__main__":
    import time

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    scores = build_random_scores()
    print(f"[TEST] Random score dict: {N_LAYERS} layers x {N_HEADS} heads + {N_LAYERS} MLPs = {len(scores)} blocks")
    for n_tiers in (3, 4):
        tiers = allocate_compression_tiers(scores, n_tiers=n_tiers)
        print(f"\n[TEST] n_tiers={n_tiers}, fractions={DEFAULT_TIER_FRACTIONS[n_tiers]}:")
        print(format_tier_report(scores, tiers))

    print("\n[TEST] Running asserts...")
    t0 = time.time()
    for fn in TESTS:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"[TEST] {len(TESTS)}/{len(TESTS)} tests passed ({time.time() - t0:.1f} s).")
