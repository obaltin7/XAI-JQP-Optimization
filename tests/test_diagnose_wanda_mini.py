"""
CPU-only mini tests for the Wanda diagnosis (tools/diagnostics/diagnose_wanda.py).

  * compressor.head_wanda_scores (raw) == reference implementation (Sun et al. 2023 Eq. 1, explicit X matrix) to ~1e-6
  * the normalize=None default is unchanged; layer_zscore (within-layer mean 0 / std 1) and layer_percentile ([0, 1])
    preserve the WITHIN-layer ranking; an invalid option raises ValueError
  * synthetic scale test: scaling one layer by k moves the raw selection onto the other layer, the z-score selection spreads
  * diagnose_wanda.main (mini + 7B evidence if present) writes JSON; on 7B the prune_wanda share in L0-5 is 181/205
  * run_iterative_pruning: prune_wanda_ln is NOT in the default config list; the dry run writes the criterion file and
    the INT4 module set equals that of xai_single

Run:
    pytest tests/test_diagnose_wanda_mini.py -q
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compressor import WANDA_NORMALIZE_OPTIONS, _normalize_per_layer, head_wanda_scores, parse_block_key  # noqa: E402
from tools.diagnostics.diagnose_wanda import GUN7_FILE, lowest_n, per_layer_counts, reference_scores, scale_test  # noqa: E402
from tools.diagnostics.diagnose_wanda import main as diag_main  # noqa: E402
from drift_metrics import spearman  # noqa: E402
from run_iterative_pruning import ALL_CONFIGS, CONFIG_DESCRIPTIONS, EXTRA_CONFIGS, SELECTOR_CONFIGS, parse_args  # noqa: E402
from run_iterative_pruning import main as gun7_main  # noqa: E402
from test_xai_engine_mini import build_dummy_dataloader, build_mini_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _by_layer(scores):
    out = {}
    for k, v in scores.items():
        out.setdefault(parse_block_key(k).layer, []).append((k, v))
    return out


def test_raw_matches_reference_and_default_unchanged():
    model, batches = build_mini_model(), build_dummy_dataloader()
    raw = head_wanda_scores(model, batches)
    assert raw == head_wanda_scores(model, batches, normalize=None)  # default = None
    ref = reference_scores(model, batches)["reference"]
    assert list(ref) == list(raw)
    assert max(abs(raw[k] - ref[k]) / abs(ref[k]) for k in raw) < 1e-5
    assert all(p["n_tokens"] == sum(int(b["attention_mask"].sum()) for b in batches) for p in reference_scores(model, batches)["profile"])


def test_layer_normalization_variants():
    model, batches = build_mini_model(), build_dummy_dataloader()
    raw = head_wanda_scores(model, batches)
    z = head_wanda_scores(model, batches, normalize="layer_zscore")
    p = head_wanda_scores(model, batches, normalize="layer_percentile")
    assert list(z) == list(raw) == list(p)
    for layer, items in _by_layer(z).items():
        vals = torch.tensor([v for _, v in items], dtype=torch.float64)
        assert abs(float(vals.mean())) < 1e-9 and abs(float(vals.std(unbiased=False)) - 1.0) < 1e-9, layer
        keys = [k for k, _ in items]
        assert spearman([z[k] for k in keys], [raw[k] for k in keys]) == 1.0  # within-layer ranking is preserved
        assert spearman([p[k] for k in keys], [raw[k] for k in keys]) == 1.0
        pv = [p[k] for k in keys]
        assert min(pv) == 0.0 and max(pv) == 1.0 and all(0.0 <= v <= 1.0 for v in pv)
    # ties get the average rank; a layer with std=0 gets z-score 0
    assert _normalize_per_layer(torch.tensor([3.0, 1.0, 1.0, 5.0]), "layer_percentile").tolist() == [2 / 3, 1 / 6, 1 / 6, 1.0]
    assert _normalize_per_layer(torch.tensor([2.0, 2.0]), "layer_zscore").tolist() == [0.0, 0.0]
    assert None in WANDA_NORMALIZE_OPTIONS and "layer_zscore" in WANDA_NORMALIZE_OPTIONS
    try:
        head_wanda_scores(model, batches, normalize="global")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an invalid normalize option")


def test_scale_dominates_raw_selection_but_not_zscore():
    """v_proj of the last layer ×k: raw scores of that layer scale by k (same within-layer order) -> the global lowest-N comes
    entirely from the other layer; the layer_zscore selection takes heads from every layer."""
    model, batches = build_mini_model(), build_dummy_dataloader()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    st = scale_test(model, batches, scale=20.0, layer=n_layers - 1, n_select=n_heads)
    assert abs(st["raw_score_ratio_scaled_layer"] - 20.0) < 1e-3 and abs(st["raw_score_ratio_other_layers"] - 1.0) < 1e-6
    assert st["within_layer_spearman_raw_vs_base"] == 1.0
    assert st["selection_per_layer"]["None"] == [n_heads, 0]  # raw: ALL heads of the unscaled layer
    assert 0 < st["selection_per_layer"]["layer_zscore"][1] < n_heads and 0 < st["selection_per_layer"]["layer_zscore"][0]
    assert sum(st["selection_per_layer"]["layer_percentile"]) == n_heads
    # helpers
    sc = {"layer_0.attn.head_0": 5.0, "layer_1.attn.head_0": 1.0, "layer_1.attn.head_1": 3.0}
    assert lowest_n(sc, 2) == ["layer_1.attn.head_0", "layer_1.attn.head_1"] and per_layer_counts(lowest_n(sc, 2), 2) == [0, 2]


def test_diagnose_main_writes_json_and_reads_gun7_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "diag.json"
    assert diag_main(["--output", str(out), "--no-gun7", "--scale", "10"]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["mini"]["definition_check"]["max_rel_diff_vs_reference"] < 1e-5 and "No computation error" in d["verdict"]
    assert d["mini"]["within_layer_rank_agreement"]["layer_zscore"] == 1.0
    assert d["gun7"]["status"] == "not found"
    if os.path.exists(GUN7_FILE):
        out2 = tmp_path / "diag2.json"
        assert diag_main(["--output", str(out2), "--scores-dir", str(tmp_path)]) == 0
        g = json.loads(out2.read_text(encoding="utf-8"))["gun7"]["fractions"]
        w = g["0.2"]["prune_wanda"]
        assert w["n"] == 205 and w["layers_0_5"] == 181 and w["median_layer"] == 3 and w["spearman_count_vs_layer"] < -0.5
        assert g["0.2"]["xai_single"]["layers_20_31"] == 173 and g["0.2"]["xai_single"]["median_layer"] == 24
        assert w["criterion_score_range"][1] / w["criterion_score_range"][0] > 1000
        assert isinstance(g["0.2"]["wanda_raw_profile_7b"], str) and "not found" in g["0.2"]["wanda_raw_profile_7b"]
        # once a prune_wanda_ln run has written the criterion file, the profile is computed (synthetic file)
        raw = {f"layer_{i}.attn.head_{h}": float(10 ** (i / 8) * (1 + h)) for i in range(32) for h in range(32)}
        (tmp_path / "f0.2_prune_wanda_ln_criterion.json").write_text(json.dumps({"raw": raw, "normalized": raw}), encoding="utf-8")
        out3 = tmp_path / "diag3.json"
        assert diag_main(["--output", str(out3), "--scores-dir", str(tmp_path)]) == 0
        prof = json.loads(out3.read_text(encoding="utf-8"))["gun7"]["fractions"]["0.2"]["wanda_raw_profile_7b"]
        assert prof["spearman_score_vs_layer"] > 0.5 and len(prof["per_layer_mean"]) == 32


def test_prune_wanda_ln_is_opt_in_and_dry_run_writes_criterion(tmp_path):
    assert "prune_wanda_ln" not in ALL_CONFIGS and "prune_wanda_ln" in CONFIG_DESCRIPTIONS and EXTRA_CONFIGS[0] == "prune_wanda_ln"  # EXTRA_CONFIGS also holds later opt-in configs
    assert len(ALL_CONFIGS) == 7 and "prune_wanda_ln" in SELECTOR_CONFIGS
    assert parse_args([]).configs.split(",") == ALL_CONFIGS  # default run unchanged
    out = tmp_path / "wln.json"
    sdir = tmp_path / "scores"
    assert gun7_main(["--dry-run", "--fractions", "0.2", "--configs", "xai_single,prune_wanda,prune_wanda_ln", "--n-repeats", "1",
                      "--output", str(out), "--log", str(tmp_path / "log.txt"), "--scores-dir", str(sdir)]) == 0
    fr = json.loads(out.read_text(encoding="utf-8"))["fractions"]["0.2"]
    assert list(fr["configs"]) == ["xai_single", "prune_wanda", "prune_wanda_ln"]
    c = fr["configs"]["prune_wanda_ln"]
    assert c["status"] == "completed" and c["int4_module_set_equals_single"] and c["n_pruned_heads"] == fr["budget"]["n_prune_heads"]
    crit = c["repeats"][0]["criterion"]
    assert crit["normalize"] == "layer_zscore" and os.path.exists(crit["scores_file"]) and "raw_score_range" in crit
    f = json.loads(open(crit["scores_file"], encoding="utf-8").read())
    assert set(f["raw"]) == set(f["normalized"]) and f["normalize"] == "layer_zscore" and f["config"] == "prune_wanda_ln"
    assert all(math.isfinite(v) for v in f["normalized"].values())
    assert "prune_wanda_ln_minus_xai_single_ppl" in fr["comparison"]
    assert (sdir / "f0.2_prune_wanda_ln_after.json").exists()  # rescore-after (drift) file is written for this config too
    from compute_drift import collect_score_files

    assert {e["config"] for e in collect_score_files(str(sdir))} == {"xai_single", "prune_wanda", "prune_wanda_ln"}
