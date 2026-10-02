"""
Mini tests for compressor.head_attention_confidence_scores (Voita et al. 2019 confidence) and the
run_iterative_pruning `prune_attnconf` config.

  * shape/range: same keys as head_magnitude_scores; 0 < confidence <= 1, entropy >= 0, first-key share in [0, 1]
  * definition: matches a manual computation from output_attentions (valid query tokens, t = 0 excluded)
  * pruned head (zeroed q rows) -> uniform attention: score is NOT 0 but mean 1/(t+1); entropy mean ln(t+1) (defined, lowest end)
  * model state unchanged, deterministic, empty calibration raises
  * prune_attnconf is opt-in (not in ALL_CONFIGS), writes a criterion file in dry-run, INT4 set equals xai_single, recognized by compute_drift

Run:
    pytest tests/test_attnconf_mini.py -q
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compressor import LayerPlan, apply_structural_pruning, head_attention_confidence_scores, head_magnitude_scores  # noqa: E402
from compute_drift import collect_score_files  # noqa: E402
from run_iterative_pruning import ALL_CONFIGS, CONFIG_DESCRIPTIONS, EXTRA_CONFIGS, SELECTOR_CONFIGS  # noqa: E402
from run_iterative_pruning import main as gun7_main  # noqa: E402
from test_xai_engine_mini import N_HEADS, N_LAYERS, build_dummy_dataloader, build_mini_model  # noqa: E402


def test_shape_range_and_details():
    model = build_mini_model()
    batches = build_dummy_dataloader()
    scores, det = head_attention_confidence_scores(model, batches, return_details=True)
    assert list(scores) == list(head_magnitude_scores(model)) and len(scores) == N_LAYERS * N_HEADS
    assert all(0.0 < v <= 1.0 for v in scores.values())
    assert all(v >= 0.0 for v in det["entropy"].values()) and all(0.0 <= v <= 1.0 for v in det["first_key_share"].values())
    assert det["n_query_tokens"] == sum(int(b["attention_mask"].sum()) - b["attention_mask"].shape[0] for b in batches)
    assert head_attention_confidence_scores(model, batches) == scores  # return_details=False: score dict only; deterministic


def test_matches_manual_computation_from_attention_weights():
    model = build_mini_model()
    batch = build_dummy_dataloader(n_batches=1)[0]  # the last sample has 3 pad tokens
    scores, det = head_attention_confidence_scores(model, [batch], return_details=True)
    with torch.no_grad():
        att = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_attentions=True).attentions
    valid = batch["attention_mask"].bool().clone()
    valid[:, 0] = False
    for layer, head in ((0, 0), (1, 3)):
        p = att[layer][:, head].double()  # [B, T, T]
        conf = p.max(dim=-1).values[valid].mean()
        ent = (-(p * torch.log(p.clamp_min(1e-30))).sum(-1))[valid].mean()
        assert math.isclose(scores[f"layer_{layer}.attn.head_{head}"], float(conf), rel_tol=1e-9)
        assert math.isclose(det["entropy"][f"layer_{layer}.attn.head_{head}"], float(ent), rel_tol=1e-9)
        assert math.isclose(det["first_key_share"][f"layer_{layer}.attn.head_{head}"], float(p[..., 0][valid].mean()), rel_tol=1e-9)


def test_pruned_head_score_is_defined_uniform_value_not_zero():
    model = build_mini_model()
    batch = build_dummy_dataloader(n_batches=1)[0]
    before = head_attention_confidence_scores(model, [batch])
    plan = {i: LayerPlan(i) for i in range(N_LAYERS)}
    plan[0].pruned_heads = [2]
    apply_structural_pruning(model, plan, verbose=False)  # zero q rows -> all attention logits 0 -> uniform softmax
    scores, det = head_attention_confidence_scores(model, [batch], return_details=True)
    mask = batch["attention_mask"]
    ts = [t for row in mask for t in range(1, int(row.sum()))]  # valid query positions (t > 0); causal: t+1 keys
    expected_conf = sum(1.0 / (t + 1) for t in ts) / len(ts)
    expected_ent = sum(math.log(t + 1) for t in ts) / len(ts)
    key = "layer_0.attn.head_2"
    assert scores[key] > 0 and math.isclose(scores[key], expected_conf, rel_tol=1e-6)
    assert math.isclose(det["entropy"][key], expected_ent, rel_tol=1e-6)
    assert scores[key] == min(v for k, v in scores.items() if k.startswith("layer_0."))  # uniform = lowest possible confidence
    assert all(math.isclose(scores[k], before[k], rel_tol=1e-9) for k in scores if k.startswith("layer_0.") and k != key)  # other heads in the same layer are unchanged


def test_model_state_unchanged_and_empty_calibration_raises():
    model = build_mini_model().train()
    flags = [p.requires_grad for p in model.parameters()]
    weights = [p.detach().clone() for p in model.parameters()]
    head_attention_confidence_scores(model, build_dummy_dataloader(n_batches=1))
    assert model.training and flags == [p.requires_grad for p in model.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(weights, model.parameters()))
    assert getattr(model.config, "output_attentions", False) is False
    try:
        head_attention_confidence_scores(model, [])
    except ValueError:
        pass
    else:
        raise AssertionError("empty calibration should raise ValueError")


def test_prune_attnconf_is_opt_in_and_dry_run_writes_criterion(tmp_path):
    assert "prune_attnconf" not in ALL_CONFIGS and "prune_attnconf" in CONFIG_DESCRIPTIONS and "prune_attnconf" in EXTRA_CONFIGS
    assert len(ALL_CONFIGS) == 7 and "prune_attnconf" in SELECTOR_CONFIGS  # default run unchanged
    out, scores_dir = tmp_path / "attn.json", tmp_path / "scores"
    assert gun7_main(["--dry-run", "--fractions", "0.2,0.4", "--configs", "xai_single,prune_attnconf", "--n-repeats", "1",
                      "--output", str(out), "--log", str(tmp_path / "log.txt"), "--scores-dir", str(scores_dir)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    for fk in ("0.2", "0.4"):
        cfgs = d["fractions"][fk]["configs"]
        c, single = cfgs["prune_attnconf"], cfgs["xai_single"]
        assert c["status"] == "completed" and c["n_pruned_heads"] == single["n_pruned_heads"] and c["int4_module_set_equals_single"] is True
        crit = c["repeats"][0]["criterion"]
        f = json.loads(open(crit["scores_file"], encoding="utf-8").read())
        assert set(f["confidence"]) == set(f["entropy"]) == set(f["first_key_share"]) and f["config"] == "prune_attnconf"
        assert 0 < crit["score_range"][0] <= crit["score_range"][1] <= 1.0 and crit["mean_entropy_kept"] >= 0
        pruned = {f"layer_{l}.attn.head_{h}" for l, h in c["repeats"][0]["pruned_heads"]}
        lowest = set(sorted(f["confidence"], key=lambda k: (f["confidence"][k], list(f["confidence"]).index(k)))[: len(pruned)])
        assert pruned == lowest  # the lowest-confidence heads are pruned
        assert "prune_attnconf_minus_xai_single_ppl" in d["fractions"][fk]["comparison"]
    found = {(e["fraction"], e["config"]) for e in collect_score_files(str(scores_dir))}  # the rescore-after file enters the drift analysis
    assert ("0.2", "prune_attnconf") in found and ("0.4", "prune_attnconf") in found
