"""
CPU-only tests for compare_scores (two score runs) and the run_xai_on_mistral arguments.

Run:
    pytest tests/test_compare_scores_mini.py -q
"""
import json
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.evaluation.compare_scores import compare_files, tier_change_summary  # noqa: E402
from tools.evaluation.compare_scores import main as compare_main  # noqa: E402
from test_compressor_mini import build_random_scores  # noqa: E402


def test_tier_change_summary_by_hand():
    # 10 head + 1 MLP; (0.2, 0.4, 0.4) -> 2 prune / 4 int4 / 4 fp16
    s0 = {f"layer_0.attn.head_{h}": float(h + 1) for h in range(10)}
    s0["layer_0.mlp"] = 50.0
    s1 = dict(s0)
    s1["layer_0.attn.head_0"], s1["layer_0.attn.head_9"] = 100.0, 0.5  # lowest head becomes highest, highest becomes lowest
    r = tier_change_summary(s0, s1)
    # s0: prune {0,1}, int4 {2..5}, fp16 {6..9}; s1 order: 9,1,2,3,4,5,6,7,8,0 -> prune {9,1}, int4 {2,3,4,5}, fp16 {6,7,8,0}
    assert r["heads"]["n"] == 10 and r["heads"]["n_changed"] == 2
    assert r["heads"]["transitions"] == {"fp16->prune": 1, "prune->fp16": 1}
    assert r["heads"]["tier_counts"] == {"prune": 2, "int4": 4, "fp16": 4}
    assert r["mlp"]["n_changed"] == 0
    assert r["plan"]["pruned_heads_common"] == 1 and math.isclose(r["plan"]["pruned_set_jaccard"], 1 / 3)
    assert r["plan"]["int4_modules_a"] == r["plan"]["int4_modules_b"]
    same = tier_change_summary(s0, s0)
    assert same["heads"]["n_changed"] == 0 and same["plan"]["pruned_set_jaccard"] == 1.0


def _write(path, scores, n_steps):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": "mini", "precision": "fp32", "attribution": {"n_steps": n_steps}, "seed": None,
                   "n_scores": len(scores), "scores": scores}, f)


def test_compare_files_and_cli(tmp_path):
    s0 = build_random_scores(seed=1, n_layers=8, n_heads=16)  # 128 head + 8 MLP
    rng = torch.Generator().manual_seed(3)
    noise = torch.rand(len(s0), generator=rng).tolist()
    s1 = {k: v * (1 + 0.05 * n) for (k, v), n in zip(s0.items(), noise)}  # small noise: high correlation
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, s0, 8)
    _write(b, s1, 16)
    r = compare_files(str(a), str(b), ks=(10, 20))
    assert r["meta_b"]["attribution"]["n_steps"] == 16
    assert r["metrics"]["spearman"]["heads_all"] > 0.95 and r["metrics"]["pearson"]["heads_all"] > 0.95
    assert set(r["metrics"]["jaccard"]["heads_all"]) == {"top10", "top20"}
    assert 0 <= r["tiers"]["heads"]["n_changed"] < 30
    # same file on both sides: exact match
    same = compare_files(str(a), str(a))
    assert math.isclose(same["metrics"]["spearman"]["heads_all"], 1.0) and same["tiers"]["heads"]["n_changed"] == 0
    out = tmp_path / "cmp.json"
    assert compare_main([str(a), str(b), "--output", str(out), "--ks", "10,20"]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["a"] == str(a) and "tiers" in d and "metrics" in d


def test_run_xai_on_mistral_args_default_to_gun3():
    """--n-steps/--output/--seed: defaults match the reference run (n_steps 8, 16 passages, T=256, B=4, 4 batches, no seed)."""
    import importlib.util

    if importlib.util.find_spec("datasets") is None:  # the module imports datasets at top level (not installed in the local .venv)
        sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))
    import run_xai_on_mistral as rx

    a = rx.parse_args([])
    assert (a.n_steps, a.n_passages, a.max_length, a.batch_size, a.max_batches, a.internal_batch_size, a.seed) == (8, 16, 256, 4, 4, 2, None)
    assert a.output == os.path.join("results", "importance_scores_gun3.json")
    b = rx.parse_args(["--n-steps", "16", "--output", "results/importance_scores_gun3_n16.json", "--seed", "42"])
    assert (b.n_steps, b.seed, b.output) == (16, 42, "results/importance_scores_gun3_n16.json")
    rx.set_seed(42)
    x = torch.rand(3)
    rx.set_seed(42)
    assert torch.equal(x, torch.rand(3))
