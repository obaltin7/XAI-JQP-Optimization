"""
Mini tests for windowed perplexity (equality with the tools/evaluation/baseline_eval.py formula), determinism of the C4
evaluation subset and its disjointness from the calibration data, run_eval_from_plans --ppl-datasets / --save-window-nll
recording, the paired bootstrap, the calibration-domain diagnosis and layer-concentration metrics.
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_ppl_windows as epw  # noqa: E402
from test_xai_engine_mini import VOCAB, build_mini_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_window_nll_matches_gun1_harness_formula():
    model = build_mini_model()
    ids = torch.randint(1, VOCAB, (1, 75), generator=torch.Generator().manual_seed(0))
    w = epw.window_nlls_from_ids(model, ids, "cpu", max_length=32, stride=16)
    assert sum(w["tokens"]) == 75 == w["n_tokens"] and len(w["nll"]) == len(w["tokens"]) == 4
    # same algorithm as baseline_eval.compute_perplexity (sum of loss * trg_len / sequence length)
    nlls, prev = [], 0
    for begin in range(0, 75, 16):
        end = min(begin + 32, 75)
        trg = end - prev
        x = ids[:, begin:end]
        t = x.clone()
        t[:, :-trg] = -100
        with torch.no_grad():
            nlls.append(model(x, labels=t).loss * trg)
        prev = end
        if end == 75:
            break
    assert math.isclose(w["perplexity"], float(torch.exp(torch.stack(nlls).sum() / 75)), rel_tol=1e-6)


def test_c4_eval_subset_is_deterministic_disjoint_and_verified(tmp_path):
    stream = [{"text": f"document {i} " + "word " * (20 if i % 3 else 200), "url": f"u{i}"} for i in range(40)]
    ids = str(tmp_path / "ids.json")
    text, meta = epw.load_c4_eval_text(ids, stream=iter(stream), skip_docs=10, n_docs=5, min_chars=500)
    stored = json.load(open(ids, encoding="utf-8"))
    idx = [d["stream_index"] for d in stored["docs"]]
    assert idx == [12, 15, 18, 21, 24] and min(idx) >= 10 and meta["first_stream_index"] == 12  # the first 10 documents (calibration region) are skipped
    assert epw.load_c4_eval_text(ids, stream=iter(stream), skip_docs=10, n_docs=5, min_chars=500)[0] == text  # deterministic + sha1 verified
    changed = [dict(d, text=d["text"] + "!") for d in stream]
    for bad_stream, kw in ((iter(changed), {}), (iter(stream[:20]), {})):
        try:
            epw.load_c4_eval_text(ids, stream=bad_stream, skip_docs=10, n_docs=5, min_chars=500, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("a changed / truncated stream must raise ValueError")
    assert epw.C4_EVAL["skip_docs"] >= 1000  # real setting: cannot overlap calibration stream indices 0–24


def test_paired_bootstrap_ci_and_cli(tmp_path):
    rng = torch.Generator().manual_seed(1)
    base = (torch.rand(60, generator=rng) * 100 + 900).tolist()
    tokens = [512] * 60
    a = {"nll": [x + 40 for x in base], "tokens": tokens}   # A is worse in every window -> Δ > 0, CI excludes zero
    b = {"nll": base, "tokens": tokens}
    r = epw.paired_bootstrap(a, b)
    assert r["delta"] > 0 and r["ci95"][0] > 0 and r["excludes_zero"] and r["n_boot"] == 10000 and r["seed"] == 42
    assert epw.paired_bootstrap(a, b) == r  # seeded: reproducible
    same = epw.paired_bootstrap(b, b)
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0] and not same["excludes_zero"]
    try:
        epw.paired_bootstrap(a, {"nll": base[:-1], "tokens": tokens[:-1]})
    except ValueError:
        pass
    else:
        raise AssertionError("a different window structure must raise ValueError")
    e = lambda w: {"ppl": {"wikitext2": {"windows": w}, "c4": {"windows": w}}}  # noqa: E731
    (tmp_path / "w.json").write_text(json.dumps({"entries": {"f0.2/xai_single": e(b)}}), encoding="utf-8")
    (tmp_path / "c.json").write_text(json.dumps({"entries": {"f0.2/xai_single": e(a)}}), encoding="utf-8")
    out = tmp_path / "boot.json"
    assert epw.main(["--tasks", str(tmp_path / "w.json"), "--merge", f"{tmp_path / 'c.json'}:c4", "--pair", "c4:f0.2/xai_single", "f0.2/xai_single",
                     "--output", str(out)]) == 0
    pairs = json.loads(out.read_text(encoding="utf-8"))["pairs"]
    assert [p["dataset"] for p in pairs] == ["wikitext2", "c4"] and all(p["excludes_zero"] and p["a"].startswith("c4:") for p in pairs)


def test_run_eval_records_window_nll_only_when_requested(tmp_path):
    import run_eval_from_plans as rp

    a = rp.parse_args([])
    assert a.ppl_datasets is None and a.save_window_nll is False  # default behaviour unchanged
    out = tmp_path / "e5.json"
    common = ["--dry-run", "--fractions", "0.2", "--configs", "xai_single", "--output", str(out), "--log", str(tmp_path / "log.txt")]
    assert rp.main(common + ["--ppl-datasets", "wikitext2,c4", "--save-window-nll"]) == 0
    e = json.loads(out.read_text(encoding="utf-8"))["entries"]
    for entry in e.values():
        for ds in ("wikitext2", "c4"):
            p = entry["ppl"][ds]
            assert p["perplexity"] > 0 and len(p["windows"]["nll"]) == p["n_windows"] == len(p["windows"]["tokens"])
    assert e["fp16"]["ppl"]["c4"]["windows"]["tokens"] == e["f0.2/xai_single"]["ppl"]["c4"]["windows"]["tokens"]  # pairable
    assert epw.paired_bootstrap(e["f0.2/xai_single"]["ppl"]["c4"]["windows"], e["fp16"]["ppl"]["c4"]["windows"], n_boot=200)["n_windows"] > 0
    assert rp.main(common + ["--ppl-datasets", "wikitext2"]) == 0
    e = json.loads(out.read_text(encoding="utf-8"))["entries"]
    assert "windows" not in e["fp16"]["ppl"]["wikitext2"] and "c4" not in e["fp16"]["ppl"]
    assert rp.main(common) == 0 and "ppl" not in json.loads(out.read_text(encoding="utf-8"))["entries"]["fp16"]


def test_summary_table_shows_ppl_datasets_only_when_present():
    import run_eval_from_plans as rp

    base = {"fp16": {"n_pruned_heads": 0, "int4_modules": 0, "summary": {"hellaswag": {"acc": 0.5, "acc_norm": 0.6}}, "seconds": 3.0, "status": "completed"}}
    old = rp.format_summary_table(base, ["hellaswag"])
    assert "ppl[" not in old and old.splitlines()[0].endswith(f"{'mmlu':>8}{'ppl':>10}{'s':>7}  status")  # without --ppl-datasets the original layout is byte-identical
    with_ppl = {"fp16": dict(base["fp16"], ppl={"wikitext2": {"perplexity": 5.22}, "c4": {"perplexity": 7.8264}}),
                "f0.2/x": dict(base["fp16"], ppl={"wikitext2": {"perplexity": 6.3171}})}
    head, r1, r2 = rp.format_summary_table(with_ppl, ["hellaswag"]).splitlines()
    assert "ppl[wiki]" in head and "ppl[c4]" in head and "5.2200" in r1 and "7.8264" in r1 and "6.3171" in r2
    assert len(head.split("status")[0]) == len(r1.split("completed")[0]) == len(r2.split("completed")[0])  # columns aligned; missing domain shown as "-"


def test_d1_d2_window_nll_is_opt_in_and_bootstrappable(tmp_path):
    import run_iterative_pruning as g7
    import run_mixed_precision as mb

    import pytest

    for mod in (g7, mb):
        a = mod.parse_args([])
        assert a.save_window_nll is False and a.window_nll_datasets == "wikitext2,c4"  # OFF by default; when enabled, BOTH domains
    assert epw.parse_ppl_datasets("c4, wikitext2") == ["c4", "wikitext2"]
    with pytest.raises(SystemExit):
        epw.parse_ppl_datasets("wikitext2,unknown")
    assert epw.load_ppl_texts(["wikitext2"], "text") == ({"wikitext2": "text"}, {})  # WikiText is not reloaded
    with pytest.raises(ValueError):
        epw.load_ppl_texts(["wikitext2"])
    runs = [(g7, tmp_path / "d1.json", ["--skip-baseline", "--configs", "mixed_xai_k10,mixed_random_k10"], "f0.2/mixed_xai_k10", "f0.2/mixed_random_k10"),
            (mb, tmp_path / "d2.json", ["--configs", "hqq_xai_b3.0,hqq_random_b3.0"], "hqq_xai_b3.0", "hqq_random_b3.0")]
    for mod, out, extra, xai, rnd in runs:
        common = ["--dry-run", "--n-repeats", "2", "--no-mmlu", "--output", str(out), "--log", str(tmp_path / "log.txt"), *extra]
        assert mod.main(common) == 0
        off = epw.entries_from_json(json.loads(out.read_text(encoding="utf-8")))
        assert off and all("ppl" not in r and "perplexity_minus_windows" not in r for r in off.values())  # output schema unchanged without the flag
        assert mod.main(common + ["--save-window-nll"]) == 0
        on = epw.entries_from_json(json.loads(out.read_text(encoding="utf-8")))
        assert {xai, f"{xai}@s42", f"{rnd}@s42", f"{rnd}@s43"} <= set(on) and on[rnd] is on[f"{rnd}@s42"]
        assert all(r["perplexity"] == off[k]["perplexity"] for k, r in on.items())  # existing perplexity measurement unchanged
        w, c4 = on[xai]["ppl"]["wikitext2"], on[xai]["ppl"]["c4"]
        assert len(w["windows"]["nll"]) == w["n_windows"] and w["windows"]["nll"] != c4["windows"]["nll"]  # both domains measured separately
        for ds, rec in (("wikitext2", w), ("c4", c4)):  # window structure pairable across runs
            assert rec["windows"]["tokens"] == on[f"{rnd}@s43"]["ppl"][ds]["windows"]["tokens"]
        boot = tmp_path / "boot.json"
        assert epw.main(["--tasks", str(out), "--pair", xai, f"{rnd}@s43", "--output", str(boot)]) == 0
        pairs = json.loads(boot.read_text(encoding="utf-8"))["pairs"]
        assert [p["dataset"] for p in pairs] == ["wikitext2", "c4"] and pairs[0]["n_windows"] == w["n_windows"]
        assert mod.main(common + ["--save-window-nll", "--window-nll-datasets", "wikitext2"]) == 0  # can be narrowed when offline
        narrow = epw.entries_from_json(json.loads(out.read_text(encoding="utf-8")))
        assert set(narrow[xai]["ppl"]) == {"wikitext2"}


def test_layer_concentration_metrics_reference_and_plan_formats(tmp_path):
    from tools.diagnostics import diagnose_layer_concentration as lc

    assert lc.gini([2, 2, 2, 2]) == 0.0 and math.isclose(lc.gini([0, 0, 0, 8]), 0.75) and lc.gini([0, 0]) == 0.0  # by hand: (n-1)/n
    m = lc.concentration(lc.layer_counts([(0, 0), (0, 1), (0, 2), (3, 1)], 4), n_heads=4)
    assert m == {"gini": lc.gini([3, 0, 0, 1]), "max_layer_fraction": 0.75, "n_layers_ge_half": 1, "top3_layer_share": 1.0}
    ref = lc.random_reference(4, 4, 4, n_draws=300)
    assert ref == lc.random_reference(4, 4, 4, n_draws=300) and ref["gini"]["p2_5"] <= ref["gini"]["mean"] <= ref["gini"]["p97_5"] < 0.75
    rep = lambda seed, heads, status="completed": {"seed": seed, "status": status, "pruned_heads": heads}  # noqa: E731
    stacked, spread = [[0, 0], [0, 1], [0, 2], [0, 3]], [[0, 0], [1, 1], [2, 2], [3, 3]]
    model2 = {"model_info": {"n_layers": 4, "n_heads": 4},
              "configs": {"prune_only": {"status": "completed", "repeats": [rep(42, stacked)]},
                          "both": {"status": "completed", "repeats": [rep(42, stacked)]},  # same set as the one-shot config: not counted separately
                          "prune_random": {"status": "completed", "repeats": [rep(42, spread), rep(43, spread, "failed"), rep(44, spread)]}}}
    r = lc.diagnose_source(model2, None, (9, 9), n_draws=300)
    assert list(r["sets"]) == ["prune_only", "prune_random_s42", "prune_random_s44"] and r["n_layers"] == 4
    assert r["sets"]["prune_only"]["per_layer"] == [4, 0, 0, 0] and r["sets"]["prune_only"]["gini_above_random_p97_5"] is True
    assert r["sets"]["prune_random_s42"]["gini"] == 0.0 and r["sets"]["prune_random_s42"]["gini_above_random_p97_5"] is False
    gun7 = {"fractions": {"0.2": {"configs": {"xai_single": {"status": "completed", "repeats": [rep(42, stacked)]}}}}}
    assert list(lc.diagnose_source(gun7, "0.2", (4, 4), n_draws=50)["sets"]) == ["xai_single"]  # fallback dimensions when model_info is absent
    out = tmp_path / "lc.json"
    assert lc.main(["--output", str(out), "--no-figure", "--n-draws", "50"]) == 0 and "sources" in json.loads(out.read_text(encoding="utf-8"))


def test_calibration_domain_diagnosis_sets_ranks_and_groups():
    from tools.diagnostics import diagnose_calib_domain as dc

    wiki, c4 = [(0, 0), (0, 1), (30, 2)], [(0, 1), (22, 5), (30, 2)]
    scores = {f"layer_{l}.attn.head_{h}": 0.1 * (l + h + 1) for l in (0, 22, 30) for h in range(6)}
    scores["layer_0.mlp"] = 9.0
    r = dc.diagnose(wiki, c4, dc.head_ranks(scores), dc.head_ranks(scores), {"ref": [(22, 5)]}, n_layers=32)
    assert r["sets"] == {"only_c4": [[22, 5]], "only_wikitext": [[0, 0]], "common": [[0, 1], [30, 2]]} and math.isclose(r["jaccard"], 2 / 4)
    assert r["per_layer"]["only_c4"][22] == 1 and sum(r["per_layer"]["common"]) == 2 and r["overlaps_with_gun6"]["ref"]["only_c4"] == 1
    assert dc.head_ranks(scores)[(0, 0)] == 0 and len(dc.head_ranks(scores)) == 18  # MLP keys are not ranked
    assert r["groups_only_c4"] == {"L21-26": [[22, 5]]} and len(dc.LAYER_BLOCKS) <= 6
    real = os.path.join(ROOT, dc.OUTPUT_FILE)
    if os.path.exists(real):
        d = json.load(open(real, encoding="utf-8"))
        assert d["n_pruned"] == {"wikitext": 205, "c4": 205} and len(d["sets"]["only_c4"]) == len(d["sets"]["only_wikitext"]) == 47
        assert d["overlaps_with_gun6"]["prune_reverse"] == {"only_c4": 0, "only_wikitext": 0, "common": 0}  # none is among the "top 205 most important"
