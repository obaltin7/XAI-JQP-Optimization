"""
Mini tests for D-2: sub-4-bit, xAI-guided bit allocation (HQQ, experiments/run_mixed_precision.py).

  * budget: the template shares meet the budget; with 32 blocks B=3.0 -> 0/11/10/11 (no 8-bit tier, equal 4/3/2 shares),
    B=3.5 -> 2/12/12/6 (average EXACTLY B); the average never exceeds B
  * allocation: within-kind importance order 8 > 4 > 3 > 2 bits; the random control uses the same counts, seeded; deterministic;
    parameter-weighted average = B
  * hqq dispatch (mini model; when hqq is installed, e.g. .venv-quant): correct nbits per module, lm_head untouched, size includes
    W_q + scale + zero, two quantizations are bit-identical (deterministic); without hqq the fake path (FakeHQQLinear) exercises the
    same dispatch
  * dry run end to end + --resume + default paths redirected to dryrun_out/

Run:
    pytest tests/test_mixed_bits_mini.py -q                      # .venv (no hqq: hqq tests are skipped)
    .venv-quant/Scripts/python.exe tests/test_mixed_bits_mini.py # hqq 0.2.8.post1 + torch 2.4.1 + transformers 4.44.2 (without pytest)
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if importlib.util.find_spec("captum") is None:  # .venv-quant has no captum; the mini model builder imports xai_engine
    _captum = types.ModuleType("captum")
    _captum.attr = types.ModuleType("captum.attr")
    _captum.attr.LayerIntegratedGradients = None
    sys.modules["captum"], sys.modules["captum.attr"] = _captum, _captum.attr

import run_mixed_precision as mb
from compressor import module_importance_scores  # noqa: E402
from test_compressor_mini import build_random_scores  # noqa: E402
from test_xai_engine_mini import build_dummy_dataloader, build_mini_model  # noqa: E402

HAS_HQQ = mb.hqq_available()


def test_budget_template_and_integer_solution():
    for b in (3.0, 3.5):
        s = mb.template_shares(b)
        assert math.isclose(sum(s.values()), 1.0) and math.isclose(sum(k * v for k, v in s.items()), b) and s[4] == s[3]
    assert mb.template_shares(3.5)[8] == 0.05  # B=3.5 template unchanged
    s3 = mb.template_shares(3.0)
    assert s3[8] == 0.0 and s3[4] == s3[3] == s3[2]  # B=3.0: no 8-bit tier, equal 4/3/2 shares
    assert mb.solve_tier_counts(32, 3.0) == {8: 0, 4: 11, 3: 10, 2: 11}  # B=3.0: closest to equal 4/3/2 shares (34% at 2 bits)
    assert mb.solve_tier_counts(32, 3.5) == {8: 2, 4: 12, 3: 12, 2: 6}
    for n in (1, 2, 5, 28, 32):
        for b in (2.3, 3.0, 3.5, 3.7):
            c = mb.solve_tier_counts(n, b)
            total = sum(k * v for k, v in c.items())
            assert sum(c.values()) == n and total <= b * n + 1e-9 and b * n - total < 1.0  # budget never exceeded, at most 1 bit-block unused
    assert mb.solve_tier_counts(2, 3.0) == {8: 0, 4: 1, 3: 0, 2: 1} and mb.solve_tier_counts(2, 3.5) == {8: 0, 4: 1, 3: 1, 2: 0}  # mini model
    for bad in (2.0, 4.0, 9.0):  # outside the template (p8 = 0.05) range: 2.3–3.725
        try:
            mb.solve_tier_counts(8, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("an out-of-range budget must raise ValueError")


def test_assignment_follows_importance_and_random_is_seeded():
    m = module_importance_scores(build_random_scores(seed=7, n_layers=32, n_heads=4))
    for b, counts in ((3.0, {"4": 11, "3": 10, "2": 11}), (3.5, {"8": 2, "4": 12, "3": 12, "2": 6})):
        plan = mb.assign_bits(m, b)
        assert list(plan) == list(m) and mb.bit_counts(plan) == {"attn": counts, "mlp": counts}
        for kind in ("attn", "mlp"):
            names = [n for n in m if n.endswith("." + kind)]
            ranked = sorted(names, key=lambda n: -m[n])
            assert [plan[n] for n in ranked] == sorted((plan[n] for n in names), reverse=True)  # more important module >= bits
        params = {n: (4 if n.endswith(".attn") else 17) * 1000 for n in m}  # kinds of different size: weighted average is still B
        assert math.isclose(mb.average_bits(plan, params), b)
        assert plan == mb.assign_bits(dict(m), b)  # deterministic
        r1, r2, r3 = (mb.assign_bits(m, b, random_seed=s) for s in (42, 42, 43))
        assert r1 == r2 and r1 != r3 and r1 != plan and mb.bit_counts(r1) == mb.bit_counts(plan)  # same budget, different placement
        assert math.isclose(mb.average_bits(r1, params), b)
    assert set(mb.uniform_bits(m, 3).values()) == {3}


def test_config_registry_and_cli_defaults():
    assert mb.ALL_CONFIGS == ["hqq_uniform_3bit", "hqq_uniform_4bit", "hqq_xai_b3.0", "hqq_random_b3.0", "hqq_magnitude_b3.0",
                              "hqq_xai_b3.5", "hqq_random_b3.5", "hqq_magnitude_b3.5"]
    assert (mb.GROUP_SIZE, mb.AXIS, mb.BIT_TIERS) == (64, 1, (8, 4, 3, 2))
    a = mb.parse_args([])
    assert a.output == mb.OUTPUT_FILE and a.log == mb.LOG_FILE and a.n_repeats == 3 and a.mmlu and a.seed == 42
    d = mb.parse_args(["--dry-run"])
    assert d.output.startswith(mb.DRYRUN_DIR) and d.log.startswith(mb.DRYRUN_DIR)  # real results/ and tracked log are untouched
    req = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements-quant.txt"), encoding="utf-8").read()
    assert f"hqq=={mb.HQQ_PIN}" in req
    main_req = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.txt"), encoding="utf-8").read()
    assert "hqq" not in main_req  # only in requirements-quant.txt


def _dispatch(fake):
    torch.manual_seed(0)
    model = build_mini_model()
    batch = build_dummy_dataloader(n_batches=1)[0]
    plan = {"layer_0.attn": 8, "layer_0.mlp": 2, "layer_1.attn": 3, "layer_1.mlp": 4}
    params = mb.module_param_counts(model)
    before = mb.module_bytes(model)["bytes"]
    with torch.no_grad():
        ref = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    res = mb.apply_hqq(model, plan, compute_dtype=torch.float32, fake=fake)
    assert res["modules_by_bits"] == {"8": 4, "2": 3, "3": 4, "4": 3} and res["n_modules"] == 14
    with torch.no_grad():
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    assert torch.isfinite(out).all() and not torch.equal(out, ref) and type(model.lm_head) is nn.Linear  # lm_head stays FP16/FP32
    size = mb.hqq_model_bytes(model)
    assert 0 < size["quant_bytes"] and size["bytes"] < before  # compressed
    assert 8.0 * size["quant_bytes"] / sum(params.values()) > mb.average_bits(plan, params)  # effective bits > nominal (scale + zero included)
    return model, plan, size, out


def test_fake_dispatch_without_hqq_package():
    model, plan, size, _ = _dispatch(fake=True)
    l0, l1 = model.model.layers
    assert all(isinstance(getattr(l0.self_attn, n), mb.FakeHQQLinear) and getattr(l0.self_attn, n).nbits == 8 for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert l0.mlp.down_proj.nbits == 2 and l1.self_attn.o_proj.nbits == 3 and l1.mlp.gate_proj.nbits == 4
    w = l0.mlp.down_proj.weight.detach().reshape(-1, 64)
    assert all(len(torch.unique(row)) <= 4 for row in w)  # 2-bit: at most 4 levels per group
    try:
        mb.apply_hqq(model, plan, fake=True)  # already quantized
    except ValueError:
        pass
    else:
        raise AssertionError("applying to an already quantized model must raise ValueError")
    try:
        mb.apply_hqq(build_mini_model(), {"layer_0.attn": 4}, fake=True)
    except ValueError:
        pass
    else:
        raise AssertionError("an incomplete plan must raise ValueError")


def test_real_hqq_dispatch_bits_size_and_determinism():
    if not HAS_HQQ:
        return
    model, plan, size, out1 = _dispatch(fake=False)
    l0, l1 = model.model.layers
    for mod, bits in ((l0.self_attn.q_proj, 8), (l0.self_attn.k_proj, 8), (l0.mlp.up_proj, 2), (l1.self_attn.o_proj, 3), (l1.mlp.down_proj, 4)):
        assert mb.is_hqq_module(mod) and mod.meta["nbits"] == bits and mod.meta["group_size"] == 64 and mod.meta["axis"] == 1
    meta_bytes = sum(v.numel() * v.element_size() for m in model.modules() if mb.is_hqq_module(m) for v in m.meta.values() if torch.is_tensor(v))
    assert size["hqq_meta_bytes"] == meta_bytes > 0 and size["bytes"] == size["param_buffer_bytes"] + meta_bytes  # scale + zero are counted
    _, _, size2, out2 = _dispatch(fake=False)
    assert torch.equal(out1, out2) and size2 == size  # HQQ is deterministic (calibration-free, fixed solver)


def test_dry_run_end_to_end_and_resume(tmp_path=None):
    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    out = os.path.join(tmp_dir, "bits.json")
    common = ["--dry-run", "--n-repeats", "2", "--output", out, "--log", os.path.join(tmp_dir, "log.txt")]
    assert mb.main(common + ["--preflight"]) == 0 and not os.path.exists(out)
    assert mb.main(common) == 0
    d = json.load(open(out, encoding="utf-8"))
    assert d["run"]["status"] == "completed" and list(d["configs"]) == mb.ALL_CONFIGS
    assert d["reference"]["hqq"]["group_size"] == 64 and d["reference"]["hqq"]["axis"] == 1 and set(d["reference"]["budgets"]) == {"3.0", "3.5"}
    assert ("fake" in d["run"]["hqq_backend"]) == (not HAS_HQQ)
    for name, c in d["configs"].items():
        assert c["status"] == "completed" and len(c["repeats"]) == (2 if c["stochastic"] else 1) and c["perplexity_mean"] > 0
        assert c["avg_bits_nominal"] <= c["budget_bits"] + 1e-9 and c["avg_bits_effective"] > c["avg_bits_nominal"] and c["model_gb"] > 0
        r = c["repeats"][0]
        assert r["quantization"]["n_modules"] == 14 and len(r["mmlu"]["per_question"]) == r["mmlu"]["n_questions"]
        assert sum(sum(v.values()) for v in r["bit_counts"].values()) == 4
    assert d["configs"]["hqq_uniform_3bit"]["avg_bits_nominal"] == 3.0 and d["configs"]["hqq_uniform_4bit"]["avg_bits_nominal"] == 4.0
    assert d["configs"]["hqq_xai_b3.0"]["same_bits_as_xai"] == [4] and len(set(d["configs"]["hqq_random_b3.0"]["model_gb_values"])) == 1
    assert "b3.0_hqq_uniform_3bit_minus_xai_ppl" in d["comparison"] and "b3.5_hqq_uniform_4bit_minus_xai_ppl" in d["comparison"]
    assert all("tasks" not in r for c in d["configs"].values() for r in c["repeats"])  # --with-tasks is OFF by default
    assert mb.main(common + ["--resume"]) == 0
    assert all(c.get("resumed") is True for c in json.load(open(out, encoding="utf-8"))["configs"].values())


def test_with_tasks_only_for_uniform_xai_and_first_random_seed(tmp_path=None):
    """--with-tasks -> HellaSwag + ARC (per question) only for uniform, xai and the seed-42 repeat of the random control."""
    a = mb.parse_args(["--with-tasks"])
    assert mb.parse_args([]).with_tasks is False and a.with_tasks is True
    expect = {"hqq_uniform_3bit": True, "hqq_uniform_4bit": True, "hqq_xai_b3.0": True, "hqq_xai_b3.5": True,
              "hqq_magnitude_b3.0": False, "hqq_magnitude_b3.5": False}
    assert all(mb.tasks_apply(a, n, 42) is v for n, v in expect.items())
    assert mb.tasks_apply(a, "hqq_random_b3.0", 42) and not mb.tasks_apply(a, "hqq_random_b3.0", 43)
    assert not mb.tasks_apply(mb.parse_args([]), "hqq_xai_b3.0", 42)
    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    out = os.path.join(tmp_dir, "bits_tasks.json")
    assert mb.main(["--dry-run", "--with-tasks", "--no-mmlu", "--n-repeats", "2", "--configs", "hqq_xai_b3.0,hqq_random_b3.0,hqq_magnitude_b3.0",
                    "--output", out, "--log", os.path.join(tmp_dir, "log.txt")]) == 0
    c = json.load(open(out, encoding="utf-8"))["configs"]
    assert set(c["hqq_xai_b3.0"]["tasks_summary"]) == {"hellaswag", "arc_challenge"} and c["hqq_xai_b3.0"]["tasks_seed"] == 42
    r42, r43 = c["hqq_random_b3.0"]["repeats"]
    assert "tasks" in r42 and "tasks" not in r43 and c["hqq_random_b3.0"]["tasks_seed"] == 42
    assert "tasks_summary" not in c["hqq_magnitude_b3.0"] and "tasks" not in c["hqq_magnitude_b3.0"]["repeats"][0]
    for t, res in r42["tasks"].items():
        per_q = next(v for v in res.values() if isinstance(v, list) and v and isinstance(v[0], dict))  # per-question records
        assert len(per_q) == res["n"] and r42["tasks_summary"][t]["acc_norm"] == res["acc_norm"]


def test_pareto_gun11_includes_hqq_rows(tmp_path=None):
    if importlib.util.find_spec("matplotlib") is None:
        return
    from tools.visualization import make_figures as mf

    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    out = os.path.join(tmp_dir, "bits.json")
    assert mb.main(["--dry-run", "--no-mmlu", "--n-repeats", "2", "--output", out, "--log", os.path.join(tmp_dir, "log.txt")]) == 0
    saved = (mf.MIXED_FILE, mf.HQQ_FILE)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # results/ relative paths
    try:
        mf.MIXED_FILE, mf.HQQ_FILE = os.path.join(tmp_dir, "missing.json"), out
        outs = mf.fig_pareto_gun11({"figures": os.path.join(tmp_dir, "figures"), "tables": os.path.join(tmp_dir, "tables")})
        rows = {r["name"]: r for r in mf.pareto_gun11_rows()}
    finally:
        mf.MIXED_FILE, mf.HQQ_FILE = saved
        os.chdir(cwd)
    assert [os.path.basename(o) for o in outs] == ["pareto_gun11.png", "pareto_gun11.md", "pareto_gun11.tex"]
    table = open(outs[1], encoding="utf-8").read()
    assert "HQQ uniform 3-bit" in table and "HQQ xAI B=3.5" in table and "bits nominal" in table and "W_q + scale + zero-point" in table
    assert rows["hqq_random_b3.0"]["std"] is not None and rows["hqq_xai_b3.0"]["std"] is None and rows["hqq_xai_b3.0"]["group"] == "hqq"


TESTS = [test_budget_template_and_integer_solution, test_assignment_follows_importance_and_random_is_seeded, test_config_registry_and_cli_defaults,
         test_fake_dispatch_without_hqq_package, test_real_hqq_dispatch_bits_size_and_determinism, test_dry_run_end_to_end_and_resume,
         test_with_tasks_only_for_uniform_xai_and_first_random_seed,
         test_pareto_gun11_includes_hqq_rows]

if __name__ == "__main__":  # environment without pytest (.venv-quant: hqq + transformers 4.44.2)
    import transformers

    print(f"[TEST] transformers {transformers.__version__}, torch {torch.__version__}, hqq={mb.hqq_version() or 'MISSING (real hqq test skipped)'}")
    for fn in TESTS:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"[TEST] {len(TESTS)}/{len(TESTS)} tests passed.")
