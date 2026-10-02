"""
Compare two importance-score runs (CPU only), e.g. the default attribution run vs. an n_steps=16 run.

Between two score JSONs (output of run_xai_on_mistral.py; the "scores" dict):
  * Spearman, Pearson (heads: all; MLPs; combined) -- drift_metrics.compare_score_dicts
  * top-k Jaccard (k=100/200; heads)
  * within-kind percentile plan difference: allocate_compression_tiers(tier_fractions, default (0.2,0.4,0.4))
    is built from both score sets; number of heads/MLPs whose tier changes and the transition matrix
    (prune->int4 etc.); layer-plan (build_compression_plan) difference: Jaccard of the pruned-head sets,
    number of layers whose attn/MLP INT4 decision changes
Output: JSON (--output) + a short table on the console.

Usage:
    python tools/evaluation/compare_scores.py results/importance_scores_gun3.json results/importance_scores_gun3_n16.json \
        --output results/compare_gun3_vs_n16.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compressor import TIER_LABELS, allocate_compression_tiers, build_compression_plan, load_scores, parse_block_key
from drift_metrics import compare_score_dicts


def tier_change_summary(s0: Dict[str, float], s1: Dict[str, float], tier_fractions: Sequence[float] = (0.2, 0.4, 0.4),
                        mlp_min_tier: Optional[str] = "int4") -> Dict[str, Any]:
    """How many blocks change tier when the same percentile plan is built from two score sets (per kind + transition matrix + layer-plan difference)."""
    t0 = allocate_compression_tiers(s0, 3, tier_fractions=tuple(tier_fractions), mlp_min_tier=mlp_min_tier)
    t1 = allocate_compression_tiers(s1, 3, tier_fractions=tuple(tier_fractions), mlp_min_tier=mlp_min_tier)
    out: Dict[str, Any] = {"tier_fractions": list(tier_fractions), "mlp_min_tier": mlp_min_tier}
    for kind in ("attn", "mlp"):
        names = [k for k in t0 if parse_block_key(k).kind == kind]
        trans: Dict[str, int] = {}
        changed = 0
        for k in names:
            if t0[k] != t1[k]:
                changed += 1
                key = f"{t0[k]}->{t1[k]}"
                trans[key] = trans.get(key, 0) + 1
        counts0 = {l: sum(1 for k in names if t0[k] == l) for l in TIER_LABELS[3]}
        out["heads" if kind == "attn" else "mlp"] = {"n": len(names), "n_changed": changed,
                                                     "changed_ratio": changed / len(names) if names else None,
                                                     "transitions": dict(sorted(trans.items())), "tier_counts": counts0}
    p0, p1 = build_compression_plan(t0), build_compression_plan(t1)
    pr0 = {(i, h) for i, p in p0.items() for h in p.pruned_heads}
    pr1 = {(i, h) for i, p in p1.items() for h in p.pruned_heads}
    out["plan"] = {
        "pruned_heads_a": len(pr0), "pruned_heads_b": len(pr1), "pruned_heads_common": len(pr0 & pr1),
        "pruned_set_jaccard": len(pr0 & pr1) / len(pr0 | pr1) if pr0 | pr1 else None,
        "layers_attn_quant_changed": sum(1 for i in p0 if p0[i].attn_quant != p1[i].attn_quant),
        "layers_mlp_quant_changed": sum(1 for i in p0 if p0[i].mlp_quant != p1[i].mlp_quant),
        "int4_modules_a": sum(4 for p in p0.values() if p.attn_quant == "int4") + sum(3 for p in p0.values() if p.mlp_quant == "int4"),
        "int4_modules_b": sum(4 for p in p1.values() if p.attn_quant == "int4") + sum(3 for p in p1.values() if p.mlp_quant == "int4"),
    }
    return out


def _meta(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {k: d.get(k) for k in ("model", "precision", "calibration", "attribution", "seed", "timing", "n_scores")}


def compare_files(path_a: str, path_b: str, tier_fractions: Sequence[float] = (0.2, 0.4, 0.4),
                  ks: Sequence[int] = (100, 200)) -> Dict[str, Any]:
    s0, s1 = load_scores(path_a), load_scores(path_b)
    return {"a": path_a, "b": path_b, "meta_a": _meta(path_a), "meta_b": _meta(path_b),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": compare_score_dicts(s0, s1, ks=ks),
            "tiers": tier_change_summary(s0, s1, tier_fractions)}


def format_table(r: Dict[str, Any]) -> str:
    m, t = r["metrics"], r["tiers"]
    sp, pe, jc = m["spearman"], m["pearson"], m["jaccard"]["heads_all"]
    rows = [
        ("Spearman ρ  head / MLP / all", f"{sp['heads_all']:.4f} / {sp['mlp']:.4f} / {sp['all_blocks']:.4f}"),
        ("Pearson r   head", f"{pe['heads_all']:.4f}"),
        ("Jaccard     " + " / ".join(jc.keys()), " / ".join(f"{v:.3f}" for v in jc.values())),
        (f"heads with changed tier ({t['tier_fractions']})", f"{t['heads']['n_changed']}/{t['heads']['n']} ({100 * t['heads']['changed_ratio']:.1f} %)  {t['heads']['transitions']}"),
        ("MLPs with changed tier", f"{t['mlp']['n_changed']}/{t['mlp']['n']}  {t['mlp']['transitions']}"),
        ("pruned set Jaccard / common", f"{t['plan']['pruned_set_jaccard']:.3f} / {t['plan']['pruned_heads_common']}"),
        ("layers with changed INT4 decision attn / MLP", f"{t['plan']['layers_attn_quant_changed']} / {t['plan']['layers_mlp_quant_changed']}"),
        ("INT4 modules a / b", f"{t['plan']['int4_modules_a']} / {t['plan']['int4_modules_b']}"),
    ]
    w = max(len(a) for a, _ in rows)
    return "\n".join(f"{a:<{w}}  {b}" for a, b in rows)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare two importance-score JSONs (Spearman, Pearson, Jaccard, tier differences)")
    p.add_argument("a", help="reference score JSON (e.g. results/importance_scores_gun3.json)")
    p.add_argument("b", help="compared score JSON (e.g. an n_steps=16 run)")
    p.add_argument("--output", default=None, help="output JSON (default: results/compare_<a>_vs_<b>.json)")
    p.add_argument("--fractions", type=float, nargs=3, default=(0.2, 0.4, 0.4), help="percentile plan fractions")
    p.add_argument("--ks", default="100,200")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    r = compare_files(args.a, args.b, args.fractions, ks)
    out = args.output or os.path.join("results", f"compare_{os.path.splitext(os.path.basename(args.a))[0]}_vs_"
                                                 f"{os.path.splitext(os.path.basename(args.b))[0]}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out)
    print(f"{args.a}  vs  {args.b}")
    print(format_table(r))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
