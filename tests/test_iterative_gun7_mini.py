"""
GPU-free mini tests for run_iterative_pruning / drift_metrics / compute_drift.

  * round-share split (total equals the budget, remainder to the first rounds)
  * round selection with exclude_heads: pruned heads score exactly 0 after rescoring but never
    enter the ranking; the selected heads are the lowest of the remaining pool, count = share
  * equal-budget guarantee: xai_single / wanda / taylor / random plans have the same head count
    and the same INT4 module SET
  * xai_iter_fixedq: pruning identical to xai_iter, INT4 module set identical to xai_single
  * drift metrics: hand-computed on small known vectors (Spearman, Jaccard)
  * end-to-end dry run (3 ratios × all configs, mini model, tmp dir) + compute_drift + --resume

Run:
    pytest tests/test_iterative_gun7_mini.py -q
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compressor import apply_structural_pruning, head_taylor_scores, head_wanda_scores  # noqa: E402
from compute_drift import collect_score_files, compute_drift  # noqa: E402
from drift_metrics import compare_score_dicts, rankdata, spearman, topk_jaccard  # noqa: E402
from run_ablation_tests import merge_plan, prune_plan_from_heads, select_heads_by_score, select_heads_random  # noqa: E402
from run_iterative_pruning import (  # noqa: E402
    ALL_CONFIGS,
    fraction_budget,
    head_names,
    int4_module_set,
    select_round_heads,
    split_budget,
    tier_fractions_for,
)
from run_iterative_pruning import main as gun7_main  # noqa: E402
from test_compressor_mini import build_random_scores  # noqa: E402
from test_xai_engine_mini import build_dummy_dataloader, build_mini_model  # noqa: E402
from xai_engine import calculate_importance_scores  # noqa: E402


# --------------------------------------------------------------------------- #
# Round shares / ratios
# --------------------------------------------------------------------------- #
def test_split_budget_sums_exactly():
    assert split_budget(205, 3) == [69, 68, 68]
    assert split_budget(410, 3) == [137, 137, 136]
    assert split_budget(614, 3) == [205, 205, 204]
    assert split_budget(2, 3) == [1, 1, 0]
    assert split_budget(7, 1) == [7]
    for n, r in ((205, 3), (1024, 5), (13, 4), (0, 2)):
        shares = split_budget(n, r)
        assert sum(shares) == n and len(shares) == r and max(shares) - min(shares) <= 1
    for bad in ((5, 0), (-1, 2)):
        try:
            split_budget(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {bad}")


def test_tier_fractions_for_matches_gun4_convention():
    assert tier_fractions_for(0.2) == (0.2, 0.4, 0.4)
    for f in (0.2, 0.4, 0.6):
        tf = tier_fractions_for(f)
        assert math.isclose(sum(tf), 1.0) and tf[0] == f and tf[1] == tf[2]


# --------------------------------------------------------------------------- #
# Round selection with exclude_heads (rescored pruned heads score 0)
# --------------------------------------------------------------------------- #
def test_select_round_heads_excludes_pruned_and_takes_lowest():
    scores = build_random_scores(seed=11, n_layers=10, n_heads=10)  # 100 head
    shares = split_budget(20, 3)  # [7, 7, 6]
    pruned = []
    for share in shares:
        chosen = select_round_heads(scores, share, pruned)
        names = head_names(chosen)
        assert len(names) == share and not (set(names) & set(pruned))
        remaining = {k: v for k, v in scores.items() if ".attn." in k and k not in pruned}
        lowest = sorted(sorted(remaining, key=remaining.get)[:share])
        assert sorted(names) == lowest
        pruned += names
    assert len(pruned) == 20
    # union of the three rounds == lowest 20 in one shot (with unchanged scores, iterative = one shot)
    heads = {k: v for k, v in scores.items() if ".attn." in k}
    assert sorted(pruned) == sorted(sorted(heads, key=heads.get)[:20])
    assert select_round_heads(scores, 0, pruned) == []
    try:
        select_round_heads(scores, 81, pruned)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a share larger than the remaining pool")


def test_round_rescoring_gives_zero_to_pruned_and_exclusion_keeps_them_out():
    """Round 1 is pruned on the mini model and rescored with xai_engine: pruned heads score exactly 0 and round 2 skips them."""
    model = build_mini_model()
    batches = build_dummy_dataloader(n_batches=1)
    n_layers = model.config.num_hidden_layers
    scores0 = calculate_importance_scores(model, batches, n_steps=2, verbose=False)
    round1 = select_round_heads(scores0, 2, [])
    apply_structural_pruning(model, prune_plan_from_heads(round1, n_layers), verbose=False)
    pruned = head_names(round1)
    scores1 = calculate_importance_scores(model, batches, n_steps=2, verbose=False)
    assert all(scores1[h] == 0.0 for h in pruned)
    assert all(v > 0 for k, v in scores1.items() if k not in pruned)
    # without exclusion the zero-score heads would again be the "lowest"
    naive = head_names(select_heads_by_score(scores1, 2, lowest=True))
    assert set(naive) == set(pruned)
    round2 = select_round_heads(scores1, 2, pruned)
    assert not (set(head_names(round2)) & set(pruned)) and len(round2) == 2
    remaining = {k: v for k, v in scores1.items() if ".attn." in k and k not in pruned}
    assert sorted(head_names(round2)) == sorted(sorted(remaining, key=remaining.get)[:2])


# --------------------------------------------------------------------------- #
# Equal-budget guarantee
# --------------------------------------------------------------------------- #
def test_same_budget_across_selectors():
    model = build_mini_model()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    scores = build_random_scores(seed=5, n_layers=n_layers, n_heads=n_heads)
    batches = build_dummy_dataloader()
    for fraction in (0.2, 0.4, 0.6):
        fb = fraction_budget(scores, fraction, dry=True)
        n = fb["n_prune_heads"]
        assert n == len(fb["xai_heads"]) > 0
        selections = {
            "xai_single": fb["xai_heads"],
            "prune_wanda": select_heads_by_score(head_wanda_scores(model, batches), n, lowest=True),
            "prune_taylor": select_heads_by_score(head_taylor_scores(model, batches), n, lowest=True),
            "prune_random": select_heads_random(sorted((l, h) for l in range(n_layers) for h in range(n_heads)), n, 42),
        }
        for name, heads in selections.items():
            plan = merge_plan(prune_plan_from_heads(heads, n_layers), fb["single_plan"])
            assert sum(len(p.pruned_heads) for p in plan.values()) == n, name
            assert int4_module_set(plan) == fb["int4_module_set_single"], name
            assert sorted(sum(([(i, h) for h in p.pruned_heads] for i, p in plan.items()), [])) == sorted(heads), name


# --------------------------------------------------------------------------- #
# Drift metrics (hand-computed)
# --------------------------------------------------------------------------- #
def test_drift_metrics_by_hand():
    assert rankdata([10, 30, 20]).tolist() == [1.0, 3.0, 2.0]
    assert rankdata([5, 5, 1]).tolist() == [2.5, 2.5, 1.0]  # ties get the average rank
    a = [1, 2, 3, 4, 5]
    assert math.isclose(spearman(a, a), 1.0)
    assert math.isclose(spearman(a, [5, 4, 3, 2, 1]), -1.0)
    # one adjacent swap: 1 - 6*(1+1)/(5*24) = 0.9
    assert math.isclose(spearman(a, [1, 2, 3, 5, 4]), 0.9)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))  # constant vector
    s0 = {"layer_0.attn.head_0": 0.1, "layer_0.attn.head_1": 0.5, "layer_0.attn.head_2": 0.3, "layer_0.attn.head_3": 0.9}
    s1 = {"layer_0.attn.head_0": 0.2, "layer_0.attn.head_1": 0.9, "layer_0.attn.head_2": 0.3, "layer_0.attn.head_3": 0.5}
    assert math.isclose(topk_jaccard(s0, s1, 2), 1.0)  # {1,3} vs {1,3}
    assert math.isclose(topk_jaccard(s0, s1, 1), 0.0)  # {3} vs {1}
    assert math.isclose(topk_jaccard(s0, s1, 3), 1.0)  # {1,2,3} vs {1,2,3}
    s0["layer_0.mlp"], s1["layer_0.mlp"] = 9.0, 8.0
    s0["layer_1.mlp"], s1["layer_1.mlp"] = 7.0, 9.5
    res = compare_score_dicts(s0, s1, exclude=["layer_0.attn.head_2"], ks=(1, 2))
    assert res["n_heads"] == 4 and res["n_mlp"] == 2 and res["n_excluded"] == 1
    assert math.isclose(res["spearman"]["heads_all"], spearman([0.1, 0.5, 0.3, 0.9], [0.2, 0.9, 0.3, 0.5]))
    assert math.isclose(res["spearman"]["heads_surviving"], spearman([0.1, 0.5, 0.9], [0.2, 0.9, 0.5]))
    assert math.isclose(res["spearman"]["mlp"], -1.0)
    assert res["jaccard"]["heads_surviving"] == {"top1": 0.0, "top2": 1.0}
    for bad in ({**s0, "layer_9.mlp": 1.0}, {k: v for k, v in s0.items() if k != "layer_1.mlp"}):
        try:
            compare_score_dicts(s0, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for differing key sets")


# --------------------------------------------------------------------------- #
# End-to-end dry run + compute_drift + --resume
# --------------------------------------------------------------------------- #
def test_dry_run_end_to_end_and_resume(tmp_path):
    out = tmp_path / "iterative_dry.json"
    log = tmp_path / "log_dry.txt"
    sdir = tmp_path / "gun7_scores"
    common = ["--dry-run", "--n-repeats", "2", "--output", str(out), "--log", str(log), "--scores-dir", str(sdir)]
    assert gun7_main(common + ["--fractions", "0.2,0.4,0.6"]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["run"]["status"] == "completed" and d["fp16_rerun"]["status"] == "completed"
    assert "mmlu_subset_acc" in d["fp16_rerun"]
    assert list(d["fractions"]) == ["0.2", "0.4", "0.6"]
    for fk, fr in d["fractions"].items():
        b = fr["budget"]
        assert sum(b["round_shares"]) == b["n_prune_heads"] > 0
        expected = ALL_CONFIGS if fk == "0.2" else [c for c in ALL_CONFIGS if c != "az_buda_cok_kuantize"]
        assert list(fr["configs"]) == expected
        for name, c in fr["configs"].items():
            assert c["status"] == "completed", (fk, name)
            assert c["perplexity_mean"] is not None and "mmlu_subset_acc_mean" in c
            if name != "az_buda_cok_kuantize":
                assert c["n_pruned_heads"] == b["n_prune_heads"], (fk, name)
            if name in ("xai_single", "xai_iter_fixedq", "prune_wanda", "prune_taylor", "prune_random"):
                assert c["int4_module_set_equals_single"] and c["int4_modules"] == b["int4_modules_single"], (fk, name)
        it = fr["configs"]["xai_iter"]["repeats"][0]
        rounds = it["rounds"]
        assert [r["share"] for r in rounds] == [s for s in b["round_shares"] if s > 0]
        assert rounds[-1]["n_pruned_cumulative"] == b["n_prune_heads"]
        assert all(r["pruned_heads_zero_score"] == r["n_pruned_cumulative"] for r in rounds)  # pruned heads score 0
        assert it["attribution_calls"] == len(rounds) and it["final_prune_tier_matches_pruned"]
        assert fr["configs"]["prune_random"]["n_completed_repeats"] == 2
        assert "xai_iter_minus_xai_single_ppl" in fr["comparison"]
        assert fr["configs"]["xai_single"]["repeats"][0]["rescore_after"]["pruned_heads_zero_score"] == b["n_prune_heads"]
    assert "reproduction_check" not in d["fractions"]["0.2"]["configs"]["xai_single"]  # dry run: no ablation-run reference
    files = collect_score_files(str(sdir))
    n_rounds_written = sum(len(fr["configs"]["xai_iter"]["repeats"][0]["rounds"]) for fr in d["fractions"].values())
    assert len(files) == n_rounds_written + 3 * 4 + 1 + 3 * 1  # rounds + 3 ratios × 4 one-shot after-files + az_buda + second random seed
    drift = compute_drift(d["reference"]["scores_file"], str(sdir))
    assert drift["n_entries"] == len(files)
    for e in drift["entries"]:
        assert e["metrics"]["excluded_zero_in_s1"] == e["n_pruned"]

    # --resume: completed configs are taken from the previous file, not re-run (same repeats, resumed flag)
    before = json.loads(out.read_text(encoding="utf-8"))
    assert gun7_main(common + ["--fractions", "0.2", "--configs", "xai_single,prune_random", "--resume"]) == 0
    after = json.loads(out.read_text(encoding="utf-8"))
    for name in ("xai_single", "prune_random"):
        c = after["fractions"]["0.2"]["configs"][name]
        assert c.get("resumed") is True
        assert c["repeats"] == before["fractions"]["0.2"]["configs"][name]["repeats"]
    assert after["fp16_rerun"]["perplexity"] == before["fp16_rerun"]["perplexity"]
    assert after["run"]["total_seconds"] < before["run"]["total_seconds"]


def test_fixedq_int4_set_equals_single(tmp_path):
    """xai_iter_fixedq: pruning identical to xai_iter (iterative), INT4 module SET identical to xai_single."""
    out = tmp_path / "fixedq.json"
    assert gun7_main(["--dry-run", "--fractions", "0.2", "--configs", "xai_single,xai_iter,xai_iter_fixedq",
                      "--output", str(out), "--log", str(tmp_path / "log.txt"), "--scores-dir", str(tmp_path / "s")]) == 0
    fr = json.loads(out.read_text(encoding="utf-8"))["fractions"]["0.2"]
    assert list(fr["configs"]) == ["xai_single", "xai_iter", "xai_iter_fixedq"]
    reps = {n: c["repeats"][0] for n, c in fr["configs"].items()}
    assert all(c["status"] == "completed" for c in fr["configs"].values())

    def quant(rep):  # per-layer quantization decisions from the plan summary == INT4 module set
        return [(p["layer"], p["attn_quant"], p["mlp_quant"]) for p in rep["plan_summary"]]

    single, it, fixed = reps["xai_single"], reps["xai_iter"], reps["xai_iter_fixedq"]
    assert quant(fixed) == quant(single)
    assert fixed["int4_module_set_equals_single"] and fixed["int4_plan_source"] == "xai_single"
    assert fixed["int4_modules"] == single["int4_modules"] == fr["budget"]["int4_modules_single"]
    assert it["int4_plan_source"] == "last_round_scores"
    # same pruning procedure as xai_iter: round shares, head count and (on the deterministic mini model) the selected heads
    assert fixed["round_shares"] == it["round_shares"] and fixed["n_pruned_heads"] == fr["budget"]["n_prune_heads"]
    assert fixed["pruned_heads"] == it["pruned_heads"]
    assert "rescore_after" not in fixed and fixed["attribution_calls"] == len(fixed["rounds"])
    # round files use a distinct name: xai_iter's are not overwritten and compute_drift does not double-count them
    assert all(os.path.basename(r["scores_file"]) == f"f0.2_xai_iter_fixedq_round{r['round']}.json" for r in fixed["rounds"])
    assert all(os.path.exists(r["scores_file"]) for r in it["rounds"] + fixed["rounds"])
    assert {e["config"] for e in collect_score_files(str(tmp_path / "s"))} == {"xai_single", "xai_iter"}
    cmp = fr["comparison"]
    assert "xai_iter_fixedq_minus_xai_single_ppl" in cmp and "xai_iter_fixedq_minus_xai_iter_ppl" in cmp


def test_rescore_nan_fallback_recomputes_round_and_restores_weights(tmp_path, monkeypatch):
    """E-3: NaN in a round rescore -> default (off) keeps the original error; with the opt-in flag the round is recomputed and weights are restored bit-identically."""
    import torch

    import run_iterative_pruning as g7

    assert g7.parse_args([]).rescore_nan_fallback == "off"  # default behavior unchanged
    # 1) unit: scoring runs at the target precision and fp16 weights come back BIT-IDENTICAL (fp16 -> bf16 -> fp16 alone is lossy)
    model = build_mini_model().half()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    seen = []
    with monkeypatch.context() as mp:
        mp.setattr(g7, "rescore", lambda m, b, d: (seen.append(next(m.parameters()).dtype) or {"layer_0.mlp": 1.0}, 0.0))
        assert g7.rescore_high_precision(model, [], True, "bf16")[0] == {"layer_0.mlp": 1.0} and seen == [torch.bfloat16]
    assert all(p.dtype == torch.float16 and torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert any(not torch.equal(t.to(torch.bfloat16).to(torch.float16), t) for t in before.values())
    assert g7.nonfinite_score_keys({"a": 1.0, "b": float("nan"), "c": float("inf")}) == ["b", "c"]

    # 2) end to end: in the first rescore one surviving head and one MLP become NaN
    real, calls = g7.rescore, []

    def flaky(m, b, d):
        scores, s = real(m, b, d)
        calls.append(1)
        if len(calls) == 1:
            alive = next(k for k, v in scores.items() if ".attn.head_" in k and v != 0.0)
            scores = {**scores, alive: float("nan"), "layer_0.mlp": float("nan")}
        return scores, s

    def run(tag, *extra):
        calls.clear()
        out = tmp_path / f"{tag}.json"
        try:
            rc = gun7_main(["--dry-run", "--fractions", "0.2", "--configs", "xai_iter_fixedq", "--output", str(out),
                            "--log", str(tmp_path / f"{tag}.txt"), "--scores-dir", str(tmp_path / tag), *extra])
        except ValueError as e:
            return None, str(e)
        return json.loads(out.read_text(encoding="utf-8"))["fractions"]["0.2"]["configs"]["xai_iter_fixedq"], rc

    clean, _ = run("clean")
    monkeypatch.setattr(g7, "rescore", flaky)
    broken, info = run("off")
    assert broken is None or broken["status"] != "completed", info  # off: original behavior (NaN raises, never passes silently)
    fixed, _ = run("fb", "--rescore-nan-fallback", "fp32")
    assert fixed["status"] == "completed"
    r_clean, r_fixed = clean["repeats"][0], fixed["repeats"][0]
    fb = r_fixed["rounds"][0]["rescore_fallback"]
    assert fb["dtype"] == "fp32" and fb["n_nonfinite_before"] == 2 and fb["n_nonfinite_after"] == 0 and "layer_0.mlp" in fb["nonfinite_before"]
    assert all("rescore_fallback" not in r for r in r_fixed["rounds"][1:] + r_clean["rounds"])  # only in the round where NaN appeared
    assert r_fixed["pruned_heads"] == r_clean["pruned_heads"] and r_fixed["perplexity"] == r_clean["perplexity"]
    saved = json.loads(open(r_fixed["rounds"][0]["scores_file"], encoding="utf-8").read())
    assert saved["rescore_fallback"]["dtype"] == "fp32" and all(math.isfinite(v) for v in saved["scores"].values())
    assert "rescore_fallback" not in json.loads(open(r_clean["rounds"][0]["scores_file"], encoding="utf-8").read())


def test_preflight_dry_does_not_write_output(tmp_path):
    out = tmp_path / "x.json"
    assert gun7_main(["--dry-run", "--preflight", "--output", str(out), "--log", str(tmp_path / "l.txt"),
                      "--scores-dir", str(tmp_path / "s")]) == 0
    assert not out.exists()
