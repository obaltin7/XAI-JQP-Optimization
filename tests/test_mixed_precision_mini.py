"""
Mini tests for D-1: xAI-guided mixed precision (NO pruning, FP16/NF4; mixed_* configs of run_iterative_pruning).

  * module score: attn = LOWER median of the head scores (same index as the median rule), MLP = MLP score
  * k=0 -> all decoder Linears NF4 (with the real scores 224 = number of Linear4bit modules of nf4_uniform in baselines_gun7.json);
    k=100 -> no NF4
  * FP16 count per kind = round_half_up(k/100·N); selections are deterministic; the random control uses the same count, seeded
  * magnitude / MLP Wanda module scores; the run_iterative_pruning mixed_* configs are opt-in, separate output, no attribution
  * Mistral 205 heads / 129 INT4 regression (allocate_compression_tiers bit-identical)

Run:
    pytest tests/test_mixed_precision_mini.py -q
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_iterative_pruning as g7
from compressor import (  # noqa: E402
    allocate_compression_tiers,
    allocate_mixed_precision,
    build_compression_plan,
    load_scores,
    mixed_precision_plan,
    mlp_wanda_scores,
    module_importance_scores,
    module_magnitude_scores,
    round_half_up,
)
from test_compressor_mini import build_random_scores  # noqa: E402
from test_xai_engine_mini import build_dummy_dataloader, build_mini_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUN3 = os.path.join(ROOT, "results", "importance_scores_gun3.json")


def _fp16(tiers, kind=None):
    return [k for k, t in tiers.items() if t == "fp16" and (kind is None or k.endswith("." + kind))]


def test_module_score_is_lower_median_of_heads_and_mlp_score():
    scores = {"layer_0.attn.head_0": 0.4, "layer_0.attn.head_1": 0.1, "layer_0.attn.head_2": 0.3, "layer_0.attn.head_3": 0.2,
              "layer_0.mlp": 9.0, "layer_1.attn.head_0": 0.5, "layer_1.attn.head_1": 0.7, "layer_1.attn.head_2": 0.6, "layer_1.mlp": 7.0}
    m = module_importance_scores(scores)
    assert list(m) == ["layer_0.attn", "layer_0.mlp", "layer_1.attn", "layer_1.mlp"]
    assert m == {"layer_0.attn": 0.2, "layer_0.mlp": 9.0, "layer_1.attn": 0.6, "layer_1.mlp": 7.0}  # even count: LOWER median (0.2, 0.3 -> 0.2)
    try:
        module_importance_scores({"layer_0.foo": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("an unrecognised key must raise ValueError")


def test_fp16_count_is_round_k_percent_of_n_per_kind_and_top_ranked():
    assert [round_half_up(x) for x in (0.4, 0.5, 1.5, 1.6, 2.5, 3.2, 6.4)] == [0, 1, 2, 2, 3, 3, 6]  # NOT banker's rounding
    scores = build_random_scores(seed=3, n_layers=32, n_heads=8)
    m = module_importance_scores(scores)
    for k, expected in ((0, 0), (5, 2), (10, 3), (20, 6), (50, 16), (100, 32)):
        tiers = allocate_mixed_precision(m, k)
        assert list(tiers) == list(m) and set(tiers.values()) <= {"fp16", "int4"}
        for kind in ("attn", "mlp"):
            chosen = _fp16(tiers, kind)
            assert len(chosen) == expected == round_half_up(k / 100 * 32)
            pool = sorted((n for n in m if n.endswith("." + kind)), key=lambda n: -m[n])
            assert set(chosen) == set(pool[:expected])  # highest-scoring within the kind
    assert allocate_mixed_precision(m, 10) == allocate_mixed_precision(dict(m), 10)  # deterministic
    for bad in (-1, 100.5):
        try:
            allocate_mixed_precision(m, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("an out-of-range k must raise ValueError")


def test_random_control_same_counts_seeded_and_reproducible():
    m = module_importance_scores(build_random_scores(seed=4, n_layers=32, n_heads=4))
    a, b, c = (allocate_mixed_precision(m, 10, random_seed=s) for s in (42, 42, 43))
    assert a == b and a != c
    for t in (a, c):
        assert len(_fp16(t, "attn")) == len(_fp16(t, "mlp")) == 3  # same budget as the xAI selection (same count per kind)
    assert _fp16(a) != _fp16(allocate_mixed_precision(m, 10))


def test_plan_has_no_pruning_and_k0_k100_extremes():
    m = module_importance_scores(build_random_scores(seed=5, n_layers=6, n_heads=4))
    p0, p100 = mixed_precision_plan(allocate_mixed_precision(m, 0)), mixed_precision_plan(allocate_mixed_precision(m, 100))
    assert g7.int4_module_count(p0) == 6 * 7 and g7.fp16_module_set(p0) == []  # k=0: all decoder Linears NF4
    assert g7.int4_module_count(p100) == 0 and len(g7.fp16_module_set(p100)) == 6 * 7  # k=100: no NF4
    p50 = mixed_precision_plan(allocate_mixed_precision(m, 50))
    assert all(not p.pruned_heads and not p.mlp_pruned for p in p50.values())
    assert sorted(g7.int4_module_set(p50) + g7.fp16_module_set(p50)) == sorted(g7.int4_module_set(p0))  # disjoint and complete cover
    assert g7.int4_module_count(p50) == 3 * 4 + 3 * 3
    try:
        mixed_precision_plan({"layer_0.attn": "prune"})
    except ValueError:
        pass
    else:
        raise AssertionError("'prune' in a pruning-free plan must raise ValueError")


def test_real_gun3_scores_k0_matches_nf4_uniform_and_205_129_regression():
    if not os.path.exists(GUN3):
        return
    scores = load_scores(GUN3)
    m = module_importance_scores(scores)
    assert len(m) == 64 and sum(k.endswith(".attn") for k in m) == 32
    n_int4 = g7.int4_module_count(mixed_precision_plan(allocate_mixed_precision(m, 0)))
    assert n_int4 == 224
    base = os.path.join(ROOT, "results", "baselines_gun7.json")
    if os.path.exists(base):
        with open(base, "r", encoding="utf-8") as f:
            info = json.load(f)["configs"]["nf4_uniform"]["load_info"]
        assert info["linear_module_types"]["Linear4bit"] == n_int4 and info["lm_head_type"] == "Linear"  # same module count, lm_head FP16
    for k, n_blocks in ((5, 2), (10, 3), (20, 6)):
        plan = mixed_precision_plan(allocate_mixed_precision(m, k))
        assert len(g7.fp16_module_set(plan)) == n_blocks * 7 and g7.int4_module_count(plan) == 224 - n_blocks * 7
    top10 = allocate_mixed_precision(m, 10)
    assert _fp16(top10, "attn") == ["layer_8.attn", "layer_10.attn", "layer_13.attn"]
    assert _fp16(top10, "mlp") == ["layer_0.mlp", "layer_1.mlp", "layer_6.mlp"]
    assert set(_fp16(allocate_mixed_precision(m, 5))) <= set(_fp16(top10)) <= set(_fp16(allocate_mixed_precision(m, 20)))  # nested
    # regression: the existing 3-tier plan is bit-identical (205 heads / 129 INT4)
    plan3 = build_compression_plan(allocate_compression_tiers(scores, 3, tier_fractions=(0.2, 0.4, 0.4)))
    assert sum(len(p.pruned_heads) for p in plan3.values()) == 205 and g7.int4_module_count(plan3) == 129


def test_magnitude_and_mlp_wanda_module_scores_on_mini_model():
    model = build_mini_model()
    mag = module_magnitude_scores(model)
    assert list(mag) == ["layer_0.attn", "layer_0.mlp", "layer_1.attn", "layer_1.mlp"]
    attn0 = model.model.layers[0].self_attn
    expected = math.sqrt(sum(float(getattr(attn0, n).weight.detach().float().pow(2).sum()) for n in ("q_proj", "k_proj", "v_proj", "o_proj")))
    assert math.isclose(mag["layer_0.attn"], expected, rel_tol=1e-6)
    batches = build_dummy_dataloader()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    w1, w2 = mlp_wanda_scores(model, batches), mlp_wanda_scores(model, batches)
    assert w1 == w2 and list(w1) == ["layer_0.mlp", "layer_1.mlp"] and all(v > 0 for v in w1.values())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())  # weights unchanged
    assert not model.model.layers[0].mlp.down_proj._forward_pre_hooks  # hooks removed
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight.mul_(3.0)
    assert mlp_wanda_scores(model, batches)["layer_1.mlp"] > 2.5 * w1["layer_1.mlp"]  # scales with |W|


def test_mixed_configs_are_opt_in_with_separate_outputs():
    assert all(n in g7.EXTRA_CONFIGS and n not in g7.ALL_CONFIGS for n in g7.MIXED_CONFIGS) and len(g7.ALL_CONFIGS) == 7
    assert "mixed_random_k10" in g7.STOCHASTIC_CONFIGS and all(n in g7.ONLY_20_CONFIGS for n in g7.MIXED_CONFIGS)
    a = g7.parse_args(["--configs", "mixed_xai_k10,mixed_random_k10"])
    assert a.output == g7.MIXED_OUTPUT_FILE and a.log == g7.MIXED_LOG_FILE and a.fractions == "0.2"
    assert g7.default_log_file(a) == g7.MIXED_LOG_FILE
    d = g7.parse_args([])  # the default run is unchanged
    assert d.output == g7.OUTPUT_FILE and d.log == g7.LOG_FILE and d.fractions == "0.2,0.4,0.6" and g7.default_log_file(d) == g7.LOG_FILE
    dry = g7.parse_args(["--dry-run", "--configs", "mixed_xai_k5"])
    assert dry.output.startswith(g7.DRYRUN_DIR) and dry.log.startswith(g7.DRYRUN_DIR)
    est = g7.estimate_minutes([0.2], ["mixed_xai_k10", "mixed_random_k10"], 3, 3, True, True, True)
    assert est["total_minutes"] == (2.5 + 3.0) * 4  # NO attribution: 1 + 3 repeats × (load/ppl + MMLU)


def test_dry_run_mixed_configs_end_to_end(tmp_path):
    out = tmp_path / "mixed.json"
    names = "mixed_xai_k0,mixed_xai_k10,mixed_random_k10,mixed_magnitude_k10,mixed_wanda_ln_k10"
    assert g7.main(["--dry-run", "--configs", names, "--n-repeats", "2", "--output", str(out), "--log", str(tmp_path / "log.txt"),
                    "--scores-dir", str(tmp_path / "scores")]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert list(d["fractions"]) == ["0.2"] and d["run"]["status"] == "completed"
    cfgs = d["fractions"]["0.2"]["configs"]
    assert list(cfgs) == names.split(",") and all(c["status"] == "completed" for c in cfgs.values())
    ref = d["reference"]["mixed_precision"]
    assert ref["n_modules"] == {"attn": 2, "mlp": 2} and set(ref["configs"]) == set(cfgs)
    k0 = cfgs["mixed_xai_k0"]["repeats"][0]
    assert k0["int4_modules"] == 14 and k0["mixed_precision"]["fp16_modules"] == [] and k0["n_pruned_heads"] == 0
    for name, c in cfgs.items():
        assert c["attribution_calls"] == 0 and c["n_pruned_heads"] == 0 and len(c["repeats"]) == (2 if name == "mixed_random_k10" else 1)
        for r in c["repeats"]:
            mp = r["mixed_precision"]
            assert "rescore_after" not in r and r["pruning"] == {"pruned_heads": 0, "pruned_mlp_blocks": 0}
            if name != "mixed_xai_k0":  # dry run: effective k = 50 -> 1 FP16 block per kind (2-layer mini model)
                assert mp["k_percent"] == 10 and mp["k_percent_effective"] == 50 and mp["n_fp16_blocks"] == {"attn": 1, "mlp": 1}
                assert r["int4_modules"] == r["int4_modules_planned"] == 7 and len(mp["fp16_modules"]) == 7
            assert r["int4_modules"] + len(mp["fp16_modules"]) == 14
    rnd = cfgs["mixed_random_k10"]
    assert [r["seed"] for r in rnd["repeats"]] == [42, 43] and len(rnd["mixed_precision"]["fp16_blocks_per_repeat"]) == 2
    assert len(set(rnd["model_bytes_after_gb_values"])) == 1  # same number of blocks -> same size
    assert cfgs["mixed_xai_k10"]["repeats"][0]["mixed_precision"]["overlap_with_xai_fp16_blocks"] == 2
    assert "mixed_random_k10_minus_mixed_xai_k10_ppl" in d["fractions"]["0.2"]["comparison"]
    assert not os.path.exists(tmp_path / "scores") or not any("mixed" in f for f in os.listdir(tmp_path / "scores"))  # no attribution file


def test_pareto_gun11_figure_and_table_from_dry_run_json(tmp_path, monkeypatch):
    from tools.visualization import make_figures as mf

    monkeypatch.chdir(ROOT)  # results/ relative paths (baselines, final model)
    paths = {"figures": str(tmp_path / "figures"), "tables": str(tmp_path / "tables")}
    monkeypatch.setattr(mf, "MIXED_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setattr(mf, "HQQ_FILE", str(tmp_path / "yok_hqq.json"))
    assert mf.fig_pareto_gun11(paths) == []  # no figure without the D-1/D-2 result files (baseline_pareto is not duplicated)
    out = tmp_path / "mixed.json"
    names = "mixed_xai_k5,mixed_xai_k10,mixed_xai_k20,mixed_random_k10,mixed_magnitude_k10"
    assert g7.main(["--dry-run", "--configs", names, "--n-repeats", "3", "--no-mmlu", "--output", str(out),
                    "--log", str(tmp_path / "log.txt"), "--scores-dir", str(tmp_path / "scores")]) == 0
    monkeypatch.setattr(mf, "MIXED_FILE", str(out))
    outs = mf.fig_pareto_gun11(paths)
    assert [os.path.basename(o) for o in outs] == ["pareto_gun11.png", "pareto_gun11.md", "pareto_gun11.tex"]
    table = open(outs[1], encoding="utf-8").read()
    for needle in ("mixed precision xai k=5", "mixed precision xai k=20", "mixed precision random k=10", " ± ", "FP16 blocks: 1 attn + 1 MLP"):
        assert needle in table
    if os.path.exists(mf.BASELINES_FILE):
        assert "uniform NF4" in table and "5.3065" in table  # k=0 reference taken from baselines_gun7.json, not re-run
    rows = {r["name"]: r for r in mf.pareto_gun11_rows()}
    assert rows["mixed_random_k10"]["std"] is not None and rows["mixed_xai_k10"]["std"] is None


def test_k0_check_against_nf4_uniform_and_wanda_ln_dropped_from_pod_command():
    """mixed_xai_k0 is in the full-run command + nf4_uniform check (<= 0.01; only logged if violated); wanda_ln is not in the run list."""
    ok, bad = g7.nf4_uniform_check(5.3100, 5.3065), g7.nf4_uniform_check(5.3500, 5.3065)
    assert ok["within_tolerance"] and ok["tolerance"] == g7.K0_TOL == 0.01 and abs(ok["abs_diff"] - 0.0035) < 1e-9
    assert not bad["within_tolerance"] and g7.nf4_uniform_check(5.31, None) is None and g7.nf4_uniform_check(None, 5.3) is None
    pod_lines = [ln for ln in g7.__doc__.splitlines() if "mixed_xai_k5" in ln and "--configs" in ln]
    assert pod_lines and all("mixed_xai_k0" in ln and "mixed_wanda_ln" not in ln for ln in pod_lines)
    assert "mixed_wanda_ln_k10" in g7.MIXED_CONFIGS and "mixed_wanda_ln_k10" in g7.EXTRA_CONFIGS  # the code stays opt-in
