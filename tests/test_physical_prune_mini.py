"""
CPU-only mini tests for compressor.physically_prune_heads (transformers 5.x and 4.44.2).

Three heads (in different layers, one in layer 0) are physically removed from a random mini Mistral:
  * logits match the masked version (apply_structural_pruning) to near bit level (atol 1e-4, fp32)
  * the parameter count drops by exactly the expected amount (q rows + o columns), k/v unchanged
  * generate (greedy, with cache) runs and produces the same tokens as the masked version
  * the save_pretrained -> load_physically_pruned round trip (per-layer head count in the config) works
  * successive pruning calls are cumulative; a quantized model raises; (fake) INT4 AFTER pruning works
  * the kv_index buffer follows the module device (cpu / meta stand-in / cuda if available) + forward safety net
  * masked (eager AND sdpa) vs physical perplexity/logit equivalence: |dlogit| < 1e-5, relative ppl < 1e-6, report line;
    masked == physical perplexity in a measure_speed --n-repeats/--perplexity/--attn-implementation dry run

Usage:
    pytest tests/test_physical_prune_mini.py -q
    .venv-quant\\Scripts\\python.exe tests/test_physical_prune_mini.py    # transformers 4.44.2 (without pytest)
"""
import copy
import importlib.util
import os
import sys
import tempfile
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if importlib.util.find_spec("captum") is None:  # .venv-quant (4.44.2) has no captum, but the mini-model builder imports xai_engine
    _captum = types.ModuleType("captum")
    _captum.attr = types.ModuleType("captum.attr")
    _captum.attr.LayerIntegratedGradients = None
    sys.modules["captum"], sys.modules["captum.attr"] = _captum, _captum.attr

import compressor  # noqa: E402
from compressor import (  # noqa: E402
    LayerPlan,
    PrunedHeadAttention,
    apply_quantization,
    apply_structural_pruning,
    count_parameters,
    load_physically_pruned,
    physically_prune_heads,
    pruned_heads_from_config,
)
from test_xai_engine_mini import HEAD_DIM, N_HEADS, build_dummy_dataloader, build_mini_model  # noqa: E402

HEADS = [(0, 1), (1, 2), (1, 3)]  # 3 heads over two layers, one in layer 0


def _plan(heads, n_layers):
    plan = {i: LayerPlan(i) for i in range(n_layers)}
    for l, h in heads:
        plan[l].pruned_heads.append(h)
    return plan


@torch.no_grad()
def _logits(model, batch):
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits.float()


def _pair():
    base = build_mini_model()
    masked, physical = copy.deepcopy(base), copy.deepcopy(base)
    apply_structural_pruning(masked, _plan(HEADS, base.config.num_hidden_layers), verbose=False)
    info = physically_prune_heads(physical, HEADS, verbose=False)
    return base, masked, physical, info


def test_logits_match_masked_and_param_count_exact():
    base, masked, physical, info = _pair()
    batch = build_dummy_dataloader(n_batches=1)[0]  # the last sample is padded, so the mask path is exercised too
    a, b = _logits(masked, batch), _logits(physical, batch)
    assert a.shape == b.shape and torch.allclose(a, b, atol=1e-4, rtol=0)
    assert not torch.allclose(_logits(base, batch), b, atol=1e-4, rtol=0)  # pruning actually has an effect
    hidden = base.config.hidden_size
    expected = len(HEADS) * 2 * hidden * HEAD_DIM
    assert info["params_removed"] == info["params_removed_expected"] == expected
    assert count_parameters(physical) == count_parameters(base) - expected
    assert info["pruned_heads"] == 3 and info["num_heads_per_layer"] == {0: N_HEADS - 1, 1: N_HEADS - 2}
    for i, layer in enumerate(physical.model.layers):
        attn = layer.self_attn
        assert isinstance(attn, PrunedHeadAttention)
        assert attn.q_proj.out_features == attn.num_heads * HEAD_DIM and attn.o_proj.in_features == attn.num_heads * HEAD_DIM
        assert attn.k_proj.out_features == base.model.layers[i].self_attn.k_proj.out_features  # k/v untouched
        assert torch.equal(attn.k_proj.weight, base.model.layers[i].self_attn.k_proj.weight)
    assert physical.model.layers[0].self_attn.kept_heads == [0, 2, 3]
    assert physical.model.layers[0].self_attn.kv_index.tolist() == [0, 1, 1]  # n_rep = 4/2 = 2


def _perplexity(model, batches):
    """exp of the mean cross-entropy over valid target tokens (mini equivalent of the perplexity harness)."""
    total, n = 0.0, 0
    with torch.no_grad():
        for b in batches:
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits.float()
            valid = b["attention_mask"][:, 1:].bool()
            loss = nn.functional.cross_entropy(logits[:, :-1][valid], b["input_ids"][:, 1:][valid], reduction="sum")
            total += float(loss)
            n += int(valid.sum())
    return float(torch.exp(torch.tensor(total / n)))


def _reload_with_attn(model, attn_implementation: str):
    """Rebuild the same weights with a different attention path (eager/sdpa); save->load because 4.44.2 picks the class at construction."""
    from transformers import AutoModelForCausalLM

    tmp_dir = tempfile.mkdtemp()
    model.save_pretrained(tmp_dir, safe_serialization=True)
    return AutoModelForCausalLM.from_pretrained(tmp_dir, torch_dtype=torch.float32, attn_implementation=attn_implementation).eval()


def test_masked_vs_physical_perplexity_equivalence_report():
    """
    Masked (eager and sdpa) vs physical (PrunedHeadAttention, eager) logit and perplexity equivalence with heavy head pruning
    (3/4 heads in one layer) on padded batches; tolerances are tight for fp32 (logit 1e-5, relative ppl 1e-6).
    Prints a report line (the 7B counterpart is measure_speed.py --perplexity, summary.perplexity_minus_masked).
    """
    heads = [(0, 3), (1, 0), (1, 1), (1, 2)]  # 4/8 heads; a single head remains in layer 1 (hardest shape)
    base = build_mini_model()
    masked, physical = copy.deepcopy(base), copy.deepcopy(base)
    apply_structural_pruning(masked, _plan(heads, base.config.num_hidden_layers), verbose=False)
    info = physically_prune_heads(physical, heads, verbose=False)
    assert info["num_heads_per_layer"] == {0: N_HEADS - 1, 1: 1}
    batches = build_dummy_dataloader(seed=7, n_batches=3, batch_size=3, seq_len=16)
    variants = {"masked_eager": masked}
    try:
        variants["masked_sdpa"] = _reload_with_attn(masked, "sdpa")
    except Exception as e:  # if the sdpa path is unavailable here, only eager is compared
        print(f"[REPORT] could not build the sdpa path, skipped: {e!r}")
    ppl_phys = _perplexity(physical, batches)
    report = {"physical_ppl": ppl_phys, "params_removed": info["params_removed"]}
    for name, m in variants.items():
        max_abs = max(float((_logits(m, b) - _logits(physical, b)).abs().max()) for b in batches)
        ppl_m = _perplexity(m, batches)
        rel = abs(ppl_m - ppl_phys) / ppl_phys
        report[name] = {"max_abs_logit_diff": max_abs, "ppl": ppl_m, "ppl_rel_diff": rel}
        assert max_abs < 1e-5, (name, max_abs)
        assert rel < 1e-6, (name, ppl_m, ppl_phys)
    base_ppl = _perplexity(base, batches)
    assert abs(base_ppl - ppl_phys) / ppl_phys > 1e-4  # pruning actually has an effect (we are not measuring the same model)
    print(f"[REPORT] masked vs physical (mini, fp32, {len(heads)} heads): " +
          ", ".join(f"{k}: |Δlogit|max {v['max_abs_logit_diff']:.2e}, ppl {v['ppl']:.6f} (relative {v['ppl_rel_diff']:.1e})"
                    for k, v in report.items() if isinstance(v, dict)) + f"; physical ppl {ppl_phys:.6f}, base ppl {base_ppl:.6f}")


def test_generate_runs_and_matches_masked():
    _, masked, physical, _ = _pair()
    prompt = torch.tensor([[5, 9, 17, 23]])
    kw = dict(max_new_tokens=8, do_sample=False, pad_token_id=0)
    out_m = masked.generate(prompt, **kw)
    out_p = physical.generate(prompt, **kw)
    assert out_p.shape == (1, 12) and torch.equal(out_m, out_p)


def test_save_and_load_roundtrip(tmp_path=None):
    _, _, physical, _ = _pair()
    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    physical.save_pretrained(tmp_dir, safe_serialization=True)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(tmp_dir)
    assert getattr(cfg, compressor.PRUNED_HEADS_CONFIG_KEY) == {"0": [1], "1": [2, 3]}
    assert getattr(cfg, compressor.NUM_HEADS_CONFIG_KEY) == {"0": N_HEADS - 1, "1": N_HEADS - 2}
    assert pruned_heads_from_config(cfg) == HEADS
    loaded = load_physically_pruned(tmp_dir, dtype=torch.float32)
    assert count_parameters(loaded) == count_parameters(physical)
    batch = build_dummy_dataloader(n_batches=1)[0]
    assert torch.allclose(_logits(loaded, batch), _logits(physical, batch), atol=1e-5, rtol=0)


def test_incremental_pruning_equals_single_call():
    base = build_mini_model()
    one, two = copy.deepcopy(base), copy.deepcopy(base)
    physically_prune_heads(one, [(0, 1), (0, 2), (1, 3)], verbose=False)
    physically_prune_heads(two, [(0, 1)], verbose=False)
    physically_prune_heads(two, ["layer_0.attn.head_2", "layer_1.attn.head_3"], verbose=False)
    physically_prune_heads(two, [(0, 1)], verbose=False)  # repeated: no-op
    batch = build_dummy_dataloader(n_batches=1)[0]
    assert torch.allclose(_logits(one, batch), _logits(two, batch), atol=1e-6, rtol=0)
    assert count_parameters(one) == count_parameters(two)
    assert two.model.layers[0].self_attn.kept_heads == [0, 3]
    try:
        physically_prune_heads(copy.deepcopy(base), [(0, h) for h in range(N_HEADS)], verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("removing all heads should raise ValueError")


def test_quantized_model_raises_and_quantize_after_prune_works():
    from run_ablation_tests import _fake_quantize_linear

    model = build_mini_model()
    q = model.model.layers[0].self_attn.q_proj
    q.weight = nn.Parameter(q.weight.detach().to(torch.int8), requires_grad=False)  # stand-in for a quantized module
    try:
        physically_prune_heads(model, [(0, 1)], verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a quantized q_proj")

    model = build_mini_model()
    physically_prune_heads(model, HEADS, verbose=False)
    orig = compressor._quantize_linear
    compressor._quantize_linear = _fake_quantize_linear
    try:
        counts = apply_quantization(model, {0: LayerPlan(0, attn_quant="int4", mlp_quant="int4")}, verbose=False)
    finally:
        compressor._quantize_linear = orig
    assert counts == {"int4_modules": 7, "int8_modules": 0}
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.out_features == attn.num_heads * HEAD_DIM  # pruned shape preserved
    batch = build_dummy_dataloader(n_batches=1)[0]
    assert torch.isfinite(_logits(model, batch)).all()


def test_kv_index_follows_module_device():
    """
    Regression for an index_select device mismatch (cuda:0 vs cpu): when the wrapper was built while the model was
    on GPU, kv_index stayed on CPU. Devices: cpu, cuda if available, and meta as a "different device" stand-in without CUDA.
    """
    devices = ["cpu", "meta"] + (["cuda"] if torch.cuda.is_available() else [])
    batch = build_dummy_dataloader(n_batches=1)[0]
    for dev in devices:
        # (a) pruning while the model is ALREADY on the target device: kv_index is built on q_proj's device (incl. successive pruning)
        model = build_mini_model().to(dev)
        physically_prune_heads(model, HEADS, verbose=False)
        physically_prune_heads(model, [(0, 2)], verbose=False)
        for layer in model.model.layers:
            attn = layer.self_attn
            assert attn.kv_index.device == attn.q_proj.weight.device and attn.kv_index.device.type == dev, dev
        if dev != "meta":
            assert model.model.layers[0].self_attn.kv_index.tolist() == [0, 1]  # remaining [0, 3], n_rep = 2
            moved = {k: v.to(dev) for k, v in batch.items()}
            assert torch.isfinite(_logits(model, moved)).all()
        # (b) model pruned on CPU and moved AFTERWARDS: register_buffer -> moves with the module on .to(), not in state_dict
        model = build_mini_model()
        physically_prune_heads(model, HEADS, verbose=False)
        attn = model.model.layers[0].self_attn
        assert "kv_index" in dict(attn.named_buffers(recurse=False)) and "kv_index" not in attn.state_dict()
        model.to(dev)
        assert all(l.self_attn.kv_index.device.type == dev for l in model.model.layers), dev
    # (c) forward safety net: a buffer left on the wrong device (here meta) is rebuilt on the input device
    _, _, physical, _ = _pair()
    expected = _logits(physical, batch)
    for layer in physical.model.layers:
        layer.self_attn.kv_index = layer.self_attn.kv_index.to("meta")
    assert torch.equal(_logits(physical, batch), expected)
    attn = physical.model.layers[0].self_attn
    assert attn.kv_index.device.type == "cpu" and attn.kv_index.tolist() == [0, 1, 1]


def test_measure_speed_dry_run_and_plan_loading(tmp_path=None):
    from tools.evaluation.measure_speed import load_heads_and_plan
    from tools.evaluation.measure_speed import main as speed_main

    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    out = os.path.join(tmp_dir, "speed.json")
    assert speed_main(["--dry-run", "--max-new-tokens", "6", "--n-prompts", "3", "--output", out,
                       "--log", os.path.join(tmp_dir, "log.txt"),
                       "--variants", "fp16,masked,physical,physical_int4"]) == 0
    import json

    d = json.load(open(out, encoding="utf-8"))
    v = d["variants"]
    assert all(v[n]["status"] == "completed" for n in ("fp16", "masked", "physical", "physical_int4"))
    assert v["fp16"]["generated_tokens"] == 3 * 6 and len(v["fp16"]["per_prompt"]) == 3
    assert v["masked"]["n_params"] == v["fp16"]["n_params"]  # masking removes no parameters
    assert v["physical"]["n_params"] < v["fp16"]["n_params"] and v["physical_int4"]["n_params"] == v["physical"]["n_params"]
    assert v["physical"]["pruning"]["params_removed"] == v["physical"]["pruning"]["params_removed_expected"]
    assert d["summary"]["physical"]["param_ratio_vs_fp16"] < 1.0 and "speedup_vs_fp16" in d["summary"]["physical"]
    # defaults: a single repeat, no perplexity, attention path recorded
    assert v["fp16"]["n_repeats"] == 1 and len(v["fp16"]["repeats"]) == 1 and v["fp16"]["ms_per_token_std"] == 0.0
    assert "perplexity" not in v["fp16"] and d["run"]["args"]["attn_implementation"] is None
    assert v["physical"]["attention_class"] == "PrunedHeadAttention" and v["fp16"]["attention_class"] != "PrunedHeadAttention"
    # --n-repeats 2 --perplexity -> ms/token per repeat, ±std, masked == physical perplexity (same head set)
    out2 = os.path.join(tmp_dir, "speed2.json")
    assert speed_main(["--dry-run", "--max-new-tokens", "4", "--n-prompts", "2", "--n-repeats", "2", "--perplexity",
                       "--attn-implementation", "eager", "--output", out2, "--log", os.path.join(tmp_dir, "log2.txt"),
                       "--variants", "fp16,masked,physical"]) == 0
    d2 = json.load(open(out2, encoding="utf-8"))
    v2 = d2["variants"]
    assert all(x["n_repeats"] == 2 and len(x["repeats"]) == 2 and x["generated_tokens"] == 2 * 2 * 4 for x in v2.values())
    assert all("ms_per_token_std" in x and x["perplexity"] > 0 for x in v2.values())
    assert abs(v2["masked"]["perplexity"] - v2["physical"]["perplexity"]) <= 1e-4 * v2["masked"]["perplexity"]
    assert abs(d2["summary"]["physical"]["perplexity_minus_masked"]) <= 1e-4 * v2["masked"]["perplexity"]
    assert v2["fp16"]["perplexity"] != v2["masked"]["perplexity"]
    # head set + plan from the real ablation JSON (results/ablation_gun6.json, if present)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gun6 = os.path.join(root, "results", "ablation_gun6.json")
    if os.path.exists(gun6):
        heads, plan, meta = load_heads_and_plan(gun6, "both", None)
        assert len(heads) == 205 and meta["n_heads"] == 205 and len(plan) == 32
        assert sum(len(p.pruned_heads) for p in plan.values()) == 205
        assert 4 * meta["int4_layers"]["attn"] + 3 * meta["int4_layers"]["mlp"] == 129


TESTS = [test_logits_match_masked_and_param_count_exact, test_masked_vs_physical_perplexity_equivalence_report,
         test_generate_runs_and_matches_masked,
         test_save_and_load_roundtrip, test_incremental_pruning_equals_single_call,
         test_quantized_model_raises_and_quantize_after_prune_works, test_kv_index_follows_module_device,
         test_measure_speed_dry_run_and_plan_loading]

if __name__ == "__main__":  # environment without pytest (.venv-quant, transformers 4.44.2)
    import transformers

    print(f"[TEST] transformers {transformers.__version__}, torch {torch.__version__}")
    for fn in TESTS:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"[TEST] {len(TESTS)}/{len(TESTS)} tests passed.")
