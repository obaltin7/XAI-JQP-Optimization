"""
Model independence: the mini tests repeated on a mini Qwen2 (q/k/v_proj with bias, GQA group 7) + Mistral regression.

The structural differences of Qwen2.5-7B-Instruct at mini scale: 14 q heads / 2 kv heads (group 7; 28/4 in the 7B model),
head_dim 4, q/k/v_proj bias=True (randomised; HF initialisation zeroes biases), o_proj without bias. CPU only, random weights.
  * xai_engine: keys from the config (14 heads/layer); a head whose o_proj columns are zeroed scores exactly 0
  * apply_structural_pruning: q_proj rows + q_proj BIAS slice + o_proj columns zeroed; k/v (weight + bias) unchanged
  * masked vs physical (PrunedHeadAttention wrapping Qwen2Attention): |Δlogit| <= 1e-6 (fp32), exact parameter count incl. bias
  * generate / save -> load_physically_pruned round trip
  * bnb dispatch: compressor._quantize_linear turns a Linear with bias into Linear4bit(bias=True) and keeps the bias (fake bnb module)
  * criteria (magnitude / Taylor / Wanda) and budget estimate (model_budget_dims, attention_bias) from the config
  * Mistral regression: results/importance_scores_gun3.json -> (0.2, 0.4, 0.4) plan = `both` in results/ablation_gun6.json
    (205 heads / 129 INT4 modules), exactly

Usage:
    pytest tests/test_qwen2_mini.py -q
    .venv-quant\\Scripts\\python.exe tests/test_qwen2_mini.py    # transformers 4.44.2 (without pytest; captum-dependent tests are skipped)
"""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HAS_CAPTUM = importlib.util.find_spec("captum") is not None
if not HAS_CAPTUM:  # .venv-quant (4.44.2) has no captum: stub it so xai_engine imports (attribution tests are skipped)
    _captum = types.ModuleType("captum")
    _captum.attr = types.ModuleType("captum.attr")
    _captum.attr.LayerIntegratedGradients = None
    sys.modules["captum"], sys.modules["captum.attr"] = _captum, _captum.attr

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import compressor  # noqa: E402
from compressor import (  # noqa: E402
    LayerPlan,
    PrunedHeadAttention,
    allocate_compression_tiers,
    apply_quantization,
    apply_structural_pruning,
    build_compression_plan,
    count_parameters,
    estimate_compression_budget,
    head_magnitude_scores,
    head_taylor_scores,
    head_wanda_scores,
    load_physically_pruned,
    load_scores,
    model_budget_dims,
    physically_prune_heads,
)
from test_xai_engine_mini import VOCAB, build_dummy_dataloader, build_mini_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_QUANTIZE_LINEAR = compressor._quantize_linear  # DryRun (other tests) permanently replaces it with a fake; captured at import time
Q_LAYERS, Q_HEADS, Q_KV_HEADS, Q_HEAD_DIM, Q_INTER = 2, 14, 2, 4, 64
Q_GROUP = Q_HEADS // Q_KV_HEADS  # 7 (Qwen2.5-7B: 28 / 4)
HEADS = [(0, 1), (0, 8), (1, 0), (1, 6), (1, 13)]  # from both kv groups; in layer 1 the group boundary (6 | 7) and the last head


def build_mini_qwen2(seed: int = 0, *, attn_implementation: str = "eager", n_layers: int = Q_LAYERS,
                     max_position_embeddings: int = 64) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=Q_HEADS * Q_HEAD_DIM,
        intermediate_size=Q_INTER,
        num_hidden_layers=n_layers,
        num_attention_heads=Q_HEADS,
        num_key_value_heads=Q_KV_HEADS,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=0,
        use_sliding_window=False,
        tie_word_embeddings=False,
        attn_implementation=attn_implementation,
        torch_dtype=torch.float32,
    )
    model = Qwen2ForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():  # HF initialisation zeroes Linear biases; randomise them so the bias path is exercised
        for layer in model.model.layers:
            for name in ("q_proj", "k_proj", "v_proj"):
                b = getattr(layer.self_attn, name).bias
                b.copy_(torch.randn(b.shape, generator=g) * 0.5)
    return model


def _plan(heads, n_layers):
    plan = {i: LayerPlan(i) for i in range(n_layers)}
    for l, h in heads:
        plan[l].pruned_heads.append(h)
    return plan


@torch.no_grad()
def _logits(model, batch):
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits.float()


def _pair():
    base = build_mini_qwen2()
    masked, physical = copy.deepcopy(base), copy.deepcopy(base)
    apply_structural_pruning(masked, _plan(HEADS, Q_LAYERS), verbose=False)
    info = physically_prune_heads(physical, HEADS, verbose=False)
    return base, masked, physical, info


# --------------------------------------------------------------------------- #
# Architecture / xai_engine
# --------------------------------------------------------------------------- #
def test_mini_qwen2_has_qkv_bias_and_gqa_group_7():
    model = build_mini_qwen2()
    attn = model.model.layers[0].self_attn
    assert type(attn).__name__.startswith("Qwen2")
    assert all(getattr(attn, n).bias is not None and float(getattr(attn, n).bias.detach().abs().sum()) > 0
               for n in ("q_proj", "k_proj", "v_proj"))
    assert attn.o_proj.bias is None
    assert compressor._head_dim(model) == Q_HEAD_DIM and Q_GROUP == 7
    assert attn.k_proj.out_features == Q_KV_HEADS * Q_HEAD_DIM and attn.q_proj.out_features == Q_HEADS * Q_HEAD_DIM


def test_xai_scores_keys_from_config_and_zeroed_head_scores_zero():
    if not HAS_CAPTUM:
        return
    from xai_engine import calculate_importance_scores, discover_structural_targets

    model = build_mini_qwen2()
    targets = discover_structural_targets(model)
    assert [(t.kind, t.n_groups, t.group_size) for t in targets[:2]] == [("attn", Q_HEADS, Q_HEAD_DIM), ("mlp", 1, Q_INTER)]
    head = 9
    with torch.no_grad():
        model.model.layers[0].self_attn.o_proj.weight[:, head * Q_HEAD_DIM:(head + 1) * Q_HEAD_DIM] = 0.0
    scores = calculate_importance_scores(model, build_dummy_dataloader(n_batches=1), n_steps=4, verbose=False)
    assert len(scores) == Q_LAYERS * (Q_HEADS + 1)
    assert list(scores)[:Q_HEADS] == [f"layer_0.attn.head_{h}" for h in range(Q_HEADS)] and "layer_1.mlp" in scores
    assert scores[f"layer_0.attn.head_{head}"] == 0.0
    assert all(v > 0 for k, v in scores.items() if k != f"layer_0.attn.head_{head}")


def test_structural_pruning_masks_q_bias_and_gives_zero_score():
    base = build_mini_qwen2()
    model = copy.deepcopy(base)
    res = apply_structural_pruning(model, _plan(HEADS, Q_LAYERS), verbose=False)
    assert res == {"pruned_heads": len(HEADS), "pruned_mlp_blocks": 0}
    d = Q_HEAD_DIM
    for l, h in HEADS:
        attn = model.model.layers[l].self_attn
        sl = slice(h * d, (h + 1) * d)
        assert float(attn.q_proj.weight.detach()[sl].abs().sum()) == 0.0 and float(attn.q_proj.bias.detach()[sl].abs().sum()) == 0.0
        assert float(attn.o_proj.weight.detach()[:, sl].abs().sum()) == 0.0
    for l in range(Q_LAYERS):  # k/v (weight + bias) and the q bias slices of unpruned heads are unchanged
        a, b = model.model.layers[l].self_attn, base.model.layers[l].self_attn
        assert torch.equal(a.k_proj.weight, b.k_proj.weight) and torch.equal(a.k_proj.bias, b.k_proj.bias)
        assert torch.equal(a.v_proj.weight, b.v_proj.weight) and torch.equal(a.v_proj.bias, b.v_proj.bias)
        kept = [h for h in range(Q_HEADS) if (l, h) not in HEADS]
        assert all(torch.equal(a.q_proj.bias[h * d:(h + 1) * d], b.q_proj.bias[h * d:(h + 1) * d]) for h in kept)
    if HAS_CAPTUM:
        from xai_engine import calculate_importance_scores

        scores = calculate_importance_scores(model, build_dummy_dataloader(n_batches=1), n_steps=4, verbose=False)
        pruned = {f"layer_{l}.attn.head_{h}" for l, h in HEADS}
        assert all(scores[k] == 0.0 for k in pruned)
        assert all(v > 0 for k, v in scores.items() if k not in pruned)


# --------------------------------------------------------------------------- #
# Physical pruning (PrunedHeadAttention wrapping Qwen2Attention)
# --------------------------------------------------------------------------- #
def test_masked_vs_physical_logits_within_1e6_and_param_count_with_bias():
    base, masked, physical, info = _pair()
    batches = build_dummy_dataloader(seed=7, n_batches=3, batch_size=3, seq_len=16)  # padded samples: exercises the mask path
    max_abs = max(float((_logits(masked, b) - _logits(physical, b)).abs().max()) for b in batches)
    assert max_abs <= 1e-6, max_abs
    assert max(float((_logits(base, b) - _logits(physical, b)).abs().max()) for b in batches) > 1e-3  # pruning has an effect
    hidden = base.config.hidden_size
    expected = len(HEADS) * (2 * hidden * Q_HEAD_DIM + Q_HEAD_DIM)  # q rows + o columns + q BIAS slice
    assert info["params_removed"] == info["params_removed_expected"] == expected
    assert count_parameters(physical) == count_parameters(base) - expected
    assert info["num_heads_per_layer"] == {0: Q_HEADS - 2, 1: Q_HEADS - 3}
    for i, layer in enumerate(physical.model.layers):
        attn = layer.self_attn
        assert isinstance(attn, PrunedHeadAttention) and type(attn._inner).__name__.startswith("Qwen2")
        assert attn.q_proj.out_features == attn.num_heads * Q_HEAD_DIM == attn.q_proj.bias.shape[0]
        assert attn.o_proj.in_features == attn.num_heads * Q_HEAD_DIM
        ref = base.model.layers[i].self_attn
        assert torch.equal(attn.k_proj.weight, ref.k_proj.weight) and torch.equal(attn.k_proj.bias, ref.k_proj.bias)
        assert attn.kv_index.tolist() == [h // Q_GROUP for h in attn.kept_heads]
        assert attn._apply_rotary is sys.modules[type(attn._inner).__module__].apply_rotary_pos_emb
    kept0 = physical.model.layers[0].self_attn.kept_heads
    d = Q_HEAD_DIM
    assert torch.equal(physical.model.layers[0].self_attn.q_proj.bias,
                       torch.cat([base.model.layers[0].self_attn.q_proj.bias[h * d:(h + 1) * d] for h in kept0]))
    print(f"[REPORT] mini Qwen2 masked vs physical ({len(HEADS)} head, fp32): max |dlogit| = {max_abs:.2e}")  # ASCII only: cp1254 console


def test_generate_and_save_load_roundtrip(tmp_path=None):
    _, masked, physical, _ = _pair()
    prompt = torch.tensor([[5, 9, 17, 23]])
    kw = dict(max_new_tokens=8, do_sample=False, pad_token_id=0)
    out_m, out_p = masked.generate(prompt, **kw), physical.generate(prompt, **kw)  # KV cache path (4.44.2: legacy rotary + cache)
    assert out_p.shape == (1, 12) and torch.equal(out_m, out_p)
    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    physical.save_pretrained(tmp_dir, safe_serialization=True)
    loaded = load_physically_pruned(tmp_dir, dtype=torch.float32)
    assert count_parameters(loaded) == count_parameters(physical)
    batch = build_dummy_dataloader(n_batches=1)[0]
    assert torch.allclose(_logits(loaded, batch), _logits(physical, batch), atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# bnb dispatch: Linear with bias -> Linear4bit(bias=True), bias preserved (fake bitsandbytes module)
# --------------------------------------------------------------------------- #
class _FakeLinear4bit(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type="nf4"):
        super().__init__(in_features, out_features, bias=bias)
        self.compute_dtype, self.quant_type = compute_dtype, quant_type


def _fake_bnb():
    bnb = types.ModuleType("bitsandbytes")
    bnb.nn = types.ModuleType("bitsandbytes.nn")
    bnb.nn.Linear4bit = _FakeLinear4bit
    bnb.nn.Params4bit = lambda data, requires_grad=False, quant_type="nf4": nn.Parameter(data.float(), requires_grad=requires_grad)
    return bnb


def test_bnb_dispatch_keeps_qkv_bias_on_qwen2():
    model = build_mini_qwen2()
    physically_prune_heads(model, HEADS, verbose=False)  # pruning BEFORE quantisation
    batch = build_dummy_dataloader(n_batches=1)[0]
    before = _logits(model, batch)
    ref_bias = {n: getattr(model.model.layers[0].self_attn, n).bias.detach().clone() for n in ("q_proj", "k_proj", "v_proj")}
    saved = {k: sys.modules.get(k) for k in ("bitsandbytes", "bitsandbytes.nn")}
    fake = _fake_bnb()
    sys.modules["bitsandbytes"], sys.modules["bitsandbytes.nn"] = fake, fake.nn
    patched = compressor._quantize_linear
    compressor._quantize_linear = _REAL_QUANTIZE_LINEAR  # real bnb code path (with the fake bnb module)
    try:
        counts = apply_quantization(model, {0: LayerPlan(0, attn_quant="int4", mlp_quant="int4")},
                                    compute_dtype=torch.float32, verbose=False)
    finally:
        compressor._quantize_linear = patched
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert counts == {"int4_modules": 7, "int8_modules": 0}
    attn = model.model.layers[0].self_attn
    for n in ("q_proj", "k_proj", "v_proj"):
        mod = getattr(attn, n)
        assert isinstance(mod, _FakeLinear4bit) and mod.bias is not None and torch.equal(mod.bias, ref_bias[n])
    assert isinstance(attn.o_proj, _FakeLinear4bit) and attn.o_proj.bias is None
    assert attn.q_proj.out_features == attn.num_heads * Q_HEAD_DIM  # pruned shape preserved
    assert not isinstance(model.model.layers[1].self_attn.q_proj, _FakeLinear4bit)  # layer outside the plan stays fp
    assert torch.allclose(_logits(model, batch), before, atol=1e-6, rtol=0)  # fake 4-bit is lossless: only dispatch is tested


# --------------------------------------------------------------------------- #
# Criteria and budget (from the config)
# --------------------------------------------------------------------------- #
def test_head_criteria_and_budget_dims_from_config():
    model = build_mini_qwen2()
    batches = build_dummy_dataloader(n_batches=1)
    for scores in (head_magnitude_scores(model), head_taylor_scores(model, batches), head_wanda_scores(model, batches)):
        assert len(scores) == Q_LAYERS * Q_HEADS and all(v >= 0 and v == v for v in scores.values())
    dims = model_budget_dims(model)
    assert dims == {"hidden_size": Q_HEADS * Q_HEAD_DIM, "head_dim": Q_HEAD_DIM, "num_attention_heads": Q_HEADS,
                    "num_key_value_heads": Q_KV_HEADS, "intermediate_size": Q_INTER, "attention_bias": True}
    plan = _plan(HEADS, Q_LAYERS)
    budget = estimate_compression_budget(plan, **dims)
    linear_params = sum(p.numel() for layer in model.model.layers for blk in (layer.self_attn, layer.mlp) for p in blk.parameters())
    assert budget["params_total"] == linear_params  # including bias, from the config
    assert budget["params_pruned"] == len(HEADS) * (2 * dims["hidden_size"] * Q_HEAD_DIM + Q_HEAD_DIM)
    mistral = build_mini_model()
    assert model_budget_dims(mistral)["attention_bias"] is False


# --------------------------------------------------------------------------- #
# Mistral regression: the 20% `both` plan of results/ablation_gun6.json (205 heads / 129 INT4 modules), exactly
# --------------------------------------------------------------------------- #
def test_mistral_gun6_both_plan_regression():
    scores_path = os.path.join(ROOT, "results", "importance_scores_gun3.json")
    gun6_path = os.path.join(ROOT, "results", "ablation_gun6.json")
    if not (os.path.exists(scores_path) and os.path.exists(gun6_path)):
        return  # meaningless without the result files (they are part of the repository)
    from run_ablation_tests import heads_from_tiers
    from run_iterative_pruning import int4_module_count

    scores = load_scores(scores_path)
    tiers = allocate_compression_tiers(scores, 3, tier_fractions=(0.2, 0.4, 0.4), mlp_min_tier="int4")
    plan = build_compression_plan(tiers)
    heads = heads_from_tiers(tiers)
    assert len(heads) == 205 and int4_module_count(plan) == 129 and len(plan) == 32
    both = json.load(open(gun6_path, encoding="utf-8"))["configs"]["both"]["repeats"][0]
    assert [list(h) for h in heads] == both["pruned_heads"]
    assert [(p["layer"], p["pruned_heads"], p["attn_quant"], p["mlp_quant"]) for p in both["plan_summary"]] == \
           [(i, p.pruned_heads, p.attn_quant, p.mlp_quant) for i, p in sorted(plan.items())]
    gun7_path = os.path.join(ROOT, "results", "iterative_gun7.json")
    if os.path.exists(gun7_path):
        single = json.load(open(gun7_path, encoding="utf-8"))["fractions"]["0.2"]["configs"]["xai_single"]["repeats"][0]
        assert single["pruned_heads"] == both["pruned_heads"] and single["budget_nominal"] == both["budget_nominal"]
    # budget: defaults = Mistral-7B dimensions, attention_bias=False -> bit-identical to the stored value
    explicit = estimate_compression_budget(plan, hidden_size=4096, head_dim=128, num_attention_heads=32, num_key_value_heads=8,
                                           intermediate_size=14336, attention_bias=False)
    assert estimate_compression_budget(plan) == explicit == both["budget_nominal"]
    assert round(explicit["size_ratio"], 3) == 0.549 and round(explicit["pruned_ratio"], 3) == 0.031


def test_mistral_wrapper_rotary_path_unchanged():
    from transformers.models.mistral import modeling_mistral

    model = build_mini_model()
    physically_prune_heads(model, [(0, 1)], verbose=False)
    attn = model.model.layers[0].self_attn
    assert attn._apply_rotary is modeling_mistral.apply_rotary_pos_emb and attn._rotary_seq_len is False


# --------------------------------------------------------------------------- #
# run_qwen_experiments --dry-run (mini Qwen2) end to end + --resume + --smoke
# --------------------------------------------------------------------------- #
def test_run_model2_dry_run_end_to_end_resume_and_smoke(tmp_path=None):
    if not HAS_CAPTUM:
        return  # the script uses xai_engine (captum)
    from tools.evaluation.measure_speed import load_heads_and_plan
    from run_qwen_experiments import ALL_CONFIGS, QUANTIZED_CONFIGS, expected_prune_count
    from run_qwen_experiments import main as model2_main

    assert expected_prune_count(28 * 28, 0.2) == 157 and expected_prune_count(32 * 32, 0.2) == 205  # Qwen2.5-7B / Mistral-7B
    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    out, scores_dir = os.path.join(tmp_dir, "m2.json"), os.path.join(tmp_dir, "scores")
    common = ["--dry-run", "--output", out, "--log", os.path.join(tmp_dir, "log.txt"), "--scores-dir", scores_dir, "--n-repeats", "2"]
    assert model2_main(common) == 0
    d = json.load(open(out, encoding="utf-8"))
    assert d["run"]["status"] == "completed" and list(d["configs"]) == ALL_CONFIGS
    info = d["model_info"]
    assert info["n_heads_total"] == Q_LAYERS * Q_HEADS and info["gqa_group"] == 7 and info["attention_bias"]["q_proj"] is True
    ref = d["reference"]
    assert ref["budget_heads"]["n_prune_heads"] == 6 and ref["tier_fractions"] == [0.2, 0.4, 0.4] and ref["round_shares"] == [2, 2, 2]
    assert d["baseline"]["status"] == "completed" and len(d["baseline"]["mmlu"]["per_question"]) == d["baseline"]["mmlu"]["n_questions"]
    scores = load_scores(d["attribution"]["scores_file"])  # run_xai_on_mistral schema: readable by compare_scores / compressor
    assert len(scores) == Q_LAYERS * (Q_HEADS + 1) and d["attribution"]["source"] == "computed"
    for name, c in d["configs"].items():
        assert c["status"] == "completed" and c["quantized"] == (name in QUANTIZED_CONFIGS), name
        assert c["n_pruned_heads"] == (0 if name == "quant_only" else 6), name
        assert c["int4_modules"] == (ref["int4_modules_in_plan"] if name in QUANTIZED_CONFIGS else 0), name
        r0 = c["repeats"][0]
        assert "per_question" in r0["mmlu"] and r0["budget_nominal"]["params_total"] > 0
    cfgs = d["configs"]
    assert cfgs["both"]["pruned_heads"] == cfgs["prune_only"]["pruned_heads"]
    assert [r["seed"] for r in cfgs["prune_random"]["repeats"]] == [42, 43]
    fq = cfgs["xai_iter_fixedq"]["repeats"][0]
    assert len(fq["rounds"]) == 3 and fq["int4_module_set_equals_single"] and fq["attribution_calls"] == 3
    assert all(os.path.exists(r["scores_file"]) and r["pruned_heads_zero_score"] == r["n_pruned_cumulative"] for r in fq["rounds"])
    assert "interaction" in d["comparison"] and "xai_iter_fixedq_minus_both" in d["comparison"]
    heads, plan, meta = load_heads_and_plan(out, "both", None)  # compatible with run_eval_from_plans / measure_speed --plan
    assert len(heads) == 6 and len(plan) == Q_LAYERS
    assert model2_main(common + ["--resume"]) == 0
    d2 = json.load(open(out, encoding="utf-8"))
    assert all(c.get("resumed") for c in d2["configs"].values()) and d2["attribution"]["source"] == "loaded"
    assert d2["configs"]["both"]["perplexity_mean"] == cfgs["both"]["perplexity_mean"]
    assert model2_main(common + ["--smoke"]) == 0
    smoke = json.load(open(os.path.join(tmp_dir, "m2_smoke.json"), encoding="utf-8"))["smoke"]
    assert smoke["status"] == "completed" and smoke["problems"] == [] and smoke["masked_vs_physical"]["max_abs_logit_diff"] <= 1e-6
    assert smoke["int4_layer0"]["counts"]["int4_modules"] == 7 and smoke["attribution"]["n_zero"] == 0


# --------------------------------------------------------------------------- #
# Diagnosis of the masked-vs-physical gap in the 7B smoke test (fp16 9.37 / bf16 17.9) under 7B-like conditions
# (see tools/diagnostics/diagnose_qwen_kernel.py)
# --------------------------------------------------------------------------- #
LONG_HEADS = (HEADS + [(2, h) for h in range(Q_GROUP, Q_HEADS)]   # layer 2: ALL q heads of one kv group (group 1)
              + [(3, h) for h in range(Q_HEADS - 1)])              # layer 3: 13 of 14 heads (layer almost fully pruned)


def _long_padded_batch(seq_len=300):
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(1, VOCAB, (3, seq_len), generator=g)
    mask = torch.ones_like(ids)
    mask[1, 200:] = 0  # right padding (tokenizer.padding_side = "right")
    mask[2, 17:] = 0
    return ids * mask, mask


def _masked_vs_physical(attn_implementation, dtype, attn_kernel):
    ids, mask = _long_padded_batch()
    model = build_mini_qwen2(attn_implementation=attn_implementation, n_layers=4, max_position_embeddings=512).to(dtype)
    apply_structural_pruning(model, _plan(LONG_HEADS, 4), verbose=False)
    with torch.no_grad():
        masked = model(input_ids=ids, attention_mask=mask).logits.float()
        res = physically_prune_heads(model, LONG_HEADS, verbose=False, attn_kernel=attn_kernel)
        physical = model(input_ids=ids, attention_mask=mask).logits.float()
    assert res["params_removed"] == res["params_removed_expected"]
    kernels = {m.attn_kernel for m in model.modules() if isinstance(m, PrunedHeadAttention)}
    return float((masked - physical).abs()[mask.bool()].max()), kernels, model


def test_masked_vs_physical_long_batched_padded_with_matched_attention_kernel():
    """Long sequence (300 > 256) + batch 3 + padding + GQA group 7 + loss of a whole kv group + bias: with matched kernels the gap is fp32 <= 1e-5, bf16 <= 1e-3."""
    for impl in ("eager", "sdpa"):
        for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-3)):
            diff, kernels, model = _masked_vs_physical(impl, dtype, "auto")
            assert kernels == {impl}  # auto: sdpa if the wrapped module is sdpa, otherwise eager
            assert diff <= tol, f"{impl} {dtype}: masked vs physical {diff:.3e} > {tol}"
            l2, l3 = model.model.layers[2].self_attn, model.model.layers[3].self_attn
            assert l2.kept_heads == list(range(Q_GROUP)) and l2.kv_index.tolist() == [0] * Q_GROUP  # kv group 1 has no q head left
            assert l3.kept_heads == [Q_HEADS - 1] and l3.kv_index.tolist() == [1] and l3.q_proj.bias.shape == (Q_HEAD_DIM,)
    # generation with KV cache and the sdpa kernel also yields the same tokens as the masked model
    prompt = torch.randint(1, VOCAB, (1, 9), generator=torch.Generator().manual_seed(3))
    m = build_mini_qwen2(attn_implementation="sdpa", n_layers=4, max_position_embeddings=512)
    apply_structural_pruning(m, _plan(LONG_HEADS, 4), verbose=False)
    a = m.generate(prompt, max_new_tokens=12, do_sample=False, pad_token_id=0)
    physically_prune_heads(m, LONG_HEADS, verbose=False, attn_kernel="sdpa")
    assert torch.equal(a, m.generate(prompt, max_new_tokens=12, do_sample=False, pad_token_id=0))


def test_kernel_mismatch_not_wrapper_logic_explains_low_precision_gap():
    """
    Root cause: masked model loaded with sdpa vs an EAGER wrapper. In fp32 the gap is negligible (same logic); in bf16 the kernel
    mismatch produces a measurable gap that vanishes when the kernels are matched (auto). The default kernel stays "eager"
    (unchanged behaviour of the Mistral path).
    """
    mismatch32, kernels, _ = _masked_vs_physical("sdpa", torch.float32, "eager")
    assert kernels == {"eager"} and mismatch32 <= 1e-5
    mismatch16, _, _ = _masked_vs_physical("sdpa", torch.bfloat16, "eager")
    matched16, _, _ = _masked_vs_physical("sdpa", torch.bfloat16, "auto")
    assert matched16 <= 1e-3 < mismatch16, (matched16, mismatch16)  # mini model: 0.0 vs ~4e-3; grows to 1-12 logits on the real Qwen2.5
    default, kernels, _ = _masked_vs_physical("sdpa", torch.bfloat16, None)
    assert kernels == {"eager"} and default == mismatch16  # without attn_kernel the previous behaviour is bit-identical
    assert compressor.resolve_attn_kernel("auto", build_mini_qwen2().model.layers[0].self_attn) == "eager"
    try:
        compressor.resolve_attn_kernel("flash", None)
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid attn_kernel must raise ValueError")


def test_diagnose_qwen_kernel_helpers_and_result_schema():
    """Helpers of tools/diagnostics/diagnose_qwen_kernel.py (no model download) + schema and main finding of its stored result JSON."""
    from tools.diagnostics import diagnose_qwen_kernel as dq

    cfg = types.SimpleNamespace(num_hidden_layers=4, num_attention_heads=Q_HEADS, num_key_value_heads=Q_KV_HEADS)

    class Tok:
        def __call__(self, text, return_tensors=None, padding=False):
            n = 3 if isinstance(text, list) else 1
            return {"input_ids": torch.ones(n, 5, dtype=torch.long), "attention_mask": torch.ones(n, 5, dtype=torch.long)}

    heads, enc = dq.scenario(Tok(), cfg, "smoke3_single", "cpu")
    assert heads == [(0, 1), (2, Q_HEADS - 1), (3, 0)] and enc["input_ids"].shape == (1, 5)  # the 3 heads of the run_qwen_experiments smoke test
    heads, enc = dq.scenario(Tok(), cfg, "many_padded", "cpu")
    assert {(3, h) for h in range(Q_GROUP, 2 * Q_GROUP)} <= set(heads) and (0, Q_GROUP - 1) in heads and enc["input_ids"].shape[0] == 3
    a = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    st = dq._stats(a, a + torch.tensor([0.5, 0.0]), torch.tensor([[True, False]]))
    assert st["max"] == 0.5 and st["top1"] == 1.0  # only valid tokens are counted
    path = os.path.join(ROOT, dq.OUTPUT_FILE)
    if not os.path.exists(path):
        return
    cases = {(c["dtype"], c["scenario"]): c for c in json.load(open(path, encoding="utf-8"))["cases"]}
    for (dtype, _), c in cases.items():
        if dtype == "float32":  # same logic: all three gaps are negligible in fp32
            assert max(c[k]["max"] for k in ("native_sdpa_vs_eager", "masked_vs_physical_eager", "masked_vs_physical_matched")) < 1e-3
        else:  # low precision: the eager-wrapper gap is of the order of the native sdpa-vs-eager floor and shrinks clearly with matched kernels
            assert c["masked_vs_physical_eager"]["max"] < 2 * c["native_sdpa_vs_eager"]["max"] + 1e-3
            assert c["masked_vs_physical_matched"]["max"] < 0.5 * c["masked_vs_physical_eager"]["max"]
            assert c["masked_vs_physical_matched"]["attn_kernel"] == "sdpa"


def test_fraction_010_plan_is_subset_of_020_and_writes_separate_output(tmp_path=None):
    """E-2 (Qwen sensitivity): --fraction 0.1 is supported; default 0.2 unchanged; the 10% head set is a subset of the 20% set."""
    from run_qwen_experiments import OUTPUT_FILE, build_plan, expected_prune_count, parse_args
    from run_qwen_experiments import main as model2_main

    assert parse_args([]).fraction == 0.2 and parse_args([]).output == OUTPUT_FILE  # defaults unchanged
    assert expected_prune_count(784, 0.1) == 78 and expected_prune_count(784, 0.2) == 157
    real = os.path.join(ROOT, "results", "gun9_scores", "importance_scores_qwen_qwen2_5_7b_instruct.json")
    if os.path.exists(real):
        scores = load_scores(real)
        p10, p20 = build_plan(scores, 0.1), build_plan(scores, 0.2)
        assert (p10["n_prune_heads"], p10["int4_modules"]) == (78, 104) and (p20["n_prune_heads"], p20["int4_modules"]) == (157, 119)
        assert set(p10["xai_heads"]) <= set(p20["xai_heads"]) and p10["tier_fractions"] == [0.1, 0.45, 0.45]
    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    out = os.path.join(tmp_dir, "m2_f010.json")
    assert model2_main(["--dry-run", "--fraction", "0.1", "--configs", "prune_only,both,prune_random,xai_iter_fixedq", "--n-repeats", "1",
                        "--no-mmlu", "--output", out, "--log", os.path.join(tmp_dir, "log.txt"), "--scores-dir", os.path.join(tmp_dir, "scores")]) == 0
    d = json.load(open(out, encoding="utf-8"))
    assert d["reference"]["fraction"] == 0.1 and list(d["configs"]) == ["prune_only", "both", "prune_random", "xai_iter_fixedq"]
    assert all(c["status"] == "completed" for c in d["configs"].values()) and len(d["configs"]["prune_random"]["repeats"]) == 1
    n = expected_prune_count(Q_LAYERS * Q_HEADS, 0.1)
    assert all(c["n_pruned_heads"] == n for c in d["configs"].values())
    assert any(f.startswith("f0.1_xai_iter_fixedq_round") for f in os.listdir(os.path.join(tmp_dir, "scores")))  # does not collide with the 20% round files


TESTS = [test_mini_qwen2_has_qkv_bias_and_gqa_group_7, test_xai_scores_keys_from_config_and_zeroed_head_scores_zero,
         test_structural_pruning_masks_q_bias_and_gives_zero_score,
         test_masked_vs_physical_logits_within_1e6_and_param_count_with_bias, test_generate_and_save_load_roundtrip,
         test_bnb_dispatch_keeps_qkv_bias_on_qwen2, test_head_criteria_and_budget_dims_from_config,
         test_mistral_gun6_both_plan_regression, test_mistral_wrapper_rotary_path_unchanged,
         test_run_model2_dry_run_end_to_end_resume_and_smoke,
         test_masked_vs_physical_long_batched_padded_with_matched_attention_kernel,
         test_kernel_mismatch_not_wrapper_logic_explains_low_precision_gap,
         test_diagnose_qwen_kernel_helpers_and_result_schema,
         test_fraction_010_plan_is_subset_of_020_and_writes_separate_output]

if __name__ == "__main__":  # environment without pytest (.venv-quant, transformers 4.44.2)
    import transformers

    print(f"[TEST] transformers {transformers.__version__}, torch {torch.__version__}, captum={'yes' if HAS_CAPTUM else 'no (attribution tests skipped)'}")
    for fn in TESTS:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"[TEST] {len(TESTS)}/{len(TESTS)} tests passed.")
