"""
Mini tests for the 4-tier setting with an INT8 tier (prune / INT4 / INT8 / FP16; 20% ratio only).

  * assignment: heads (0.2, 0.3, 0.3, 0.2), MLP (int4, int8, fp16) = (0.4, 0.3, 0.3); MLPs are not pruned;
    mlp_tier_fractions=None keeps the previous behaviour
  * with the real Mistral-7B scores (results/importance_scores_gun3.json): 205/307/307/205 heads, 13/10/9 MLPs; the pruned set
    is IDENTICAL to the 3-tier one; 91 INT4 + 102 INT8 modules under the median rule;
    the default 3-tier plan is unchanged (205 heads / 129 INT4, regression check)
  * bnb int8 dispatch: _quantize_linear "int8" -> Linear8bitLt(has_fp16_weights=False, threshold=6.0) + Int8Params (fake bnb module)
  * run_iterative_pruning: *_4tier configs are opt-in, 20% only, the int8_modules field appears only in these configs, pruned sets match

Usage:
    pytest tests/test_four_tier_mini.py -q
"""
import json
import os
import sys
import types
from collections import Counter

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compressor  # noqa: E402
from compressor import (  # noqa: E402
    LayerPlan,
    allocate_compression_tiers,
    apply_quantization,
    build_compression_plan,
    estimate_compression_budget,
    load_scores,
)
from run_iterative_pruning import (  # noqa: E402
    ALL_CONFIGS,
    EXTRA_CONFIGS,
    FOUR_TIER_FRACTIONS,
    FOUR_TIER_MLP_FRACTIONS,
    four_tier_plan,
    int4_module_count,
    quant_module_count,
)
from run_iterative_pruning import main as gun7_main  # noqa: E402
from test_compressor_mini import build_random_scores  # noqa: E402
from test_xai_engine_mini import build_dummy_dataloader, build_mini_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_QUANTIZE_LINEAR = compressor._quantize_linear  # DryRun permanently replaces it with a fake; still the real one at import time


def _kind_counts(tiers, kind):
    return Counter(v for k, v in tiers.items() if (".attn." in k) == (kind == "attn"))


def test_four_tier_allocation_with_separate_mlp_fractions():
    scores = build_random_scores(seed=5, n_layers=10, n_heads=10)  # 100 head, 10 MLP
    tiers = allocate_compression_tiers(scores, 4, tier_fractions=FOUR_TIER_FRACTIONS, mlp_tier_fractions=FOUR_TIER_MLP_FRACTIONS)
    assert _kind_counts(tiers, "attn") == {"prune": 20, "int4": 30, "int8": 30, "fp16": 20}
    assert _kind_counts(tiers, "mlp") == {"int4": 4, "int8": 3, "fp16": 3}  # MLPs are not pruned
    heads = {k: v for k, v in scores.items() if ".attn." in k}
    order = sorted(heads, key=heads.get)
    assert all(tiers[k] == "prune" for k in order[:20]) and all(tiers[k] == "fp16" for k in order[-20:])  # rank based
    full = allocate_compression_tiers(scores, 4, tier_fractions=FOUR_TIER_FRACTIONS, mlp_tier_fractions=(0.0, 0.4, 0.3, 0.3))
    assert full == tiers  # the 3-value form (int4/int8/fp16) equals the 4-value form with a leading 0
    # None = previous behaviour: MLPs use the head fractions, and the "prune" share is raised to int4 by mlp_min_tier
    old = allocate_compression_tiers(scores, 4, tier_fractions=FOUR_TIER_FRACTIONS)
    assert old == allocate_compression_tiers(scores, 4, tier_fractions=FOUR_TIER_FRACTIONS, mlp_tier_fractions=None)
    assert _kind_counts(old, "mlp") == {"int4": 5, "int8": 3, "fp16": 2}
    for bad in (dict(mlp_tier_fractions=(0.5, 0.5)), dict(mlp_tier_fractions=(0.4, 0.3, 0.2)), dict(mlp_tier_fractions=(0.4, 0.3, 0.3), group_by_kind=False)):
        try:
            allocate_compression_tiers(scores, 4, tier_fractions=FOUR_TIER_FRACTIONS, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {bad}")


def test_median_rule_extends_to_four_labels():
    tiers = {"layer_0.attn.head_0": "prune", "layer_0.attn.head_1": "int4", "layer_0.attn.head_2": "int8", "layer_0.attn.head_3": "int8",
             "layer_0.mlp": "int8",
             "layer_1.attn.head_0": "int4", "layer_1.attn.head_1": "int8", "layer_1.attn.head_2": "fp16", "layer_1.attn.head_3": "fp16",
             "layer_1.mlp": "fp16"}
    plan = build_compression_plan(tiers)
    assert plan[0].pruned_heads == [0] and plan[0].attn_quant == "int8" and plan[0].mlp_quant == "int8"  # median of remaining [int4, int8, int8]
    assert plan[1].attn_quant == "int8"  # [int4, int8, fp16, fp16]: LOWER median for an even count (conservative)
    assert quant_module_count(plan, "int8") == 4 + 3 + 4 and quant_module_count(plan, "int4") == 0 == int4_module_count(plan)
    b = estimate_compression_budget(plan, hidden_size=32, head_dim=8, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64)
    assert 0.5 < b["size_ratio"] < 1.0  # int8 = 8 bit, fp16 MLP = 16 bit


def test_real_gun3_scores_four_tier_plan_and_three_tier_regression():
    path = os.path.join(ROOT, "results", "importance_scores_gun3.json")
    if not os.path.exists(path):
        return
    scores = load_scores(path)
    tiers4, plan4 = four_tier_plan(scores)
    assert _kind_counts(tiers4, "attn") == {"prune": 205, "int4": 307, "int8": 307, "fp16": 205}
    assert _kind_counts(tiers4, "mlp") == {"int4": 13, "int8": 10, "fp16": 9}
    tiers3 = allocate_compression_tiers(scores, 3, tier_fractions=(0.2, 0.4, 0.4))
    plan3 = build_compression_plan(tiers3)
    assert {k for k, v in tiers3.items() if v == "prune"} == {k for k, v in tiers4.items() if v == "prune"}  # same 205 heads
    assert int4_module_count(plan3) == 129 and sum(len(p.pruned_heads) for p in plan3.values()) == 205  # 3-tier plan unchanged
    assert (int4_module_count(plan4), quant_module_count(plan4, "int8")) == (91, 102)
    assert Counter(p.attn_quant for p in plan4.values()) == {"int4": 13, "int8": 18, "fp16": 1}
    b3, b4 = estimate_compression_budget(plan3), estimate_compression_budget(plan4)
    assert round(b3["size_ratio"], 3) == 0.549 and round(b4["size_ratio"], 3) == 0.506 and b4["params_pruned"] == b3["params_pruned"]


class _FakeLinear4bit(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type="nf4"):
        super().__init__(in_features, out_features, bias=bias)


class _FakeLinear8bitLt(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, has_fp16_weights=True, threshold=0.0):
        super().__init__(in_features, out_features, bias=bias)
        self.has_fp16_weights, self.threshold = has_fp16_weights, threshold


def test_bnb_int8_dispatch_uses_llm_int8_with_threshold_6():
    bnb = types.ModuleType("bitsandbytes")
    bnb.nn = types.ModuleType("bitsandbytes.nn")
    bnb.nn.Linear4bit, bnb.nn.Linear8bitLt = _FakeLinear4bit, _FakeLinear8bitLt
    bnb.nn.Params4bit = lambda data, requires_grad=False, quant_type="nf4": nn.Parameter(data.float(), requires_grad=requires_grad)
    bnb.nn.Int8Params = lambda data, requires_grad=False, has_fp16_weights=False: nn.Parameter(data.float(), requires_grad=requires_grad)
    model = build_mini_model()
    batch = build_dummy_dataloader(n_batches=1)[0]
    with torch.no_grad():
        before = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    saved = {k: sys.modules.get(k) for k in ("bitsandbytes", "bitsandbytes.nn")}
    sys.modules["bitsandbytes"], sys.modules["bitsandbytes.nn"] = bnb, bnb.nn
    patched = compressor._quantize_linear
    compressor._quantize_linear = _REAL_QUANTIZE_LINEAR
    try:
        counts = apply_quantization(model, {0: LayerPlan(0, attn_quant="int8", mlp_quant="int4"), 1: LayerPlan(1, attn_quant="fp16", mlp_quant="int8")},
                                    compute_dtype=torch.float32, verbose=False)
    finally:
        compressor._quantize_linear = patched
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert counts == {"int4_modules": 3, "int8_modules": 4 + 3}
    attn0, l1 = model.model.layers[0].self_attn, model.model.layers[1]
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        m = getattr(attn0, n)
        assert isinstance(m, _FakeLinear8bitLt) and m.threshold == 6.0 and m.has_fp16_weights is False
    assert isinstance(model.model.layers[0].mlp.down_proj, _FakeLinear4bit) and isinstance(l1.mlp.gate_proj, _FakeLinear8bitLt)
    assert type(l1.self_attn.q_proj) is nn.Linear  # the fp16 tier is left untouched
    with torch.no_grad():  # the fake modules are lossless: only the dispatch is tested
        assert torch.allclose(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits, before, atol=1e-6, rtol=0)


def test_four_tier_configs_are_opt_in_only_at_20_percent(tmp_path):
    assert all(n in EXTRA_CONFIGS and n not in ALL_CONFIGS for n in ("xai_single_4tier", "xai_iter_fixedq_4tier")) and len(ALL_CONFIGS) == 7
    out = tmp_path / "t4.json"
    assert gun7_main(["--dry-run", "--fractions", "0.2,0.4", "--configs", "xai_single,xai_iter_fixedq,xai_single_4tier,xai_iter_fixedq_4tier",
                      "--no-mmlu", "--output", str(out), "--log", str(tmp_path / "log.txt"), "--scores-dir", str(tmp_path / "scores")]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    c20, c40 = d["fractions"]["0.2"]["configs"], d["fractions"]["0.4"]["configs"]
    assert list(c40) == ["xai_single", "xai_iter_fixedq"]  # 4 tiers at 20% only
    assert list(c20) == ["xai_single", "xai_iter_fixedq", "xai_single_4tier", "xai_iter_fixedq_4tier"]
    assert all(c["status"] == "completed" for c in c20.values())
    assert "int8_modules" not in c20["xai_single"] and "int8_modules" not in c20["xai_single"]["repeats"][0]  # previous schema unchanged
    s4, i4 = c20["xai_single_4tier"], c20["xai_iter_fixedq_4tier"]
    assert s4["pruned_heads"] == c20["xai_single"]["pruned_heads"] and i4["pruned_heads"] == c20["xai_iter_fixedq"]["pruned_heads"]
    for c in (s4, i4):
        r = c["repeats"][0]
        assert c["int8_modules"] == r["int8_modules"] == r["int8_modules_planned"] and r["int4_modules"] == r["int4_modules_planned"]
        assert r["quantization"] == {"int4_modules": r["int4_modules"], "int8_modules": r["int8_modules"]}
        assert {p["attn_quant"] for p in r["plan_summary"]} | {p["mlp_quant"] for p in r["plan_summary"]} <= {"prune", "int4", "int8", "fp16"}
    assert s4["attribution_calls"] == 0 and "rescore_after" not in s4["repeats"][0]  # pruned set = xai_single: no wasted rescoring
    assert i4["attribution_calls"] == len(i4["repeats"][0]["rounds"]) and i4["repeats"][0]["int4_plan_source"] == "xai_single_4tier"
    assert "xai_single_4tier_minus_xai_single_ppl" in d["fractions"]["0.2"]["comparison"]
    files = os.listdir(tmp_path / "scores")
    assert any(f.startswith("f0.2_xai_iter_fixedq_4tier_round") for f in files) and not any("xai_single_4tier" in f for f in files)
