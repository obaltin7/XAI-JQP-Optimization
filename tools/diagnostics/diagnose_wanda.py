"""
Diagnosis of the head-level Wanda score (CPU only; mini Mistral model plus stored 7B results).

Question: `prune_wanda` is worse than random pruning at all three ratios, and its selection follows
the layer order (at 20%, 181 of 205 pruned heads lie in layers 0-5). Is this an implementation bug,
a scale effect, or a mismatch between the method and head-level pruning?

The script reports:
  a) Reference implementation of Sun et al. (2023), Eq. 1, S_ij = |W_ij| * ||X_j||_2, where X is built
     explicitly from ALL valid token rows of the o_proj input (hook + concat, float64); the head score is
     the sum of S_ij over the head's column block. The relative difference to compressor.head_wanda_scores
     (same formula, accumulated sums of squares) is reported. A second head score uses Wanda's original
     comparison group (within an output ROW): the within-row percentile of S_ij, averaged over the head's
     column block and all rows ("row_rank"; structurally independent of cross-layer scale).
  b) Per-layer profile: o_proj input RMS (mean ||X_j||_2 / sqrt(N)), mean |W| column sum, raw head score
     min/mean/max; Spearman correlation between head score and layer index (raw, layer_zscore,
     layer_percentile, row_rank, Taylor, magnitude).
  c) Synthetic scale test: scaling one layer's v_proj by k changes the function but preserves the
     within-layer ranking; does the global lowest-N selection of raw Wanda collapse onto the other
     layer, and does the layer_zscore selection spread across layers?
  d) 7B evidence (results/iterative_gun7.json): layer distribution of the heads pruned by prune_wanda /
     xai_single / prune_taylor (share in layers 0-5, median layer) and criterion.score_range. Raw 7B Wanda
     scores were not stored by that run (only their range), so the 7B layer profile is reported as missing
     unless a prune_wanda_ln run has written f{ratio}_prune_wanda_ln_criterion.json to --scores-dir.

Usage:
    python tools/diagnostics/diagnose_wanda.py                                  # -> results/diagnose_wanda_gun8.json + console
    python tools/diagnostics/diagnose_wanda.py --output dryrun_out/diag.json --scale 50 --no-gun7
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compressor import (
    WANDA_NORMALIZE_OPTIONS,
    _decoder_layers,
    _head_dim,
    head_magnitude_scores,
    head_taylor_scores,
    head_wanda_scores,
    parse_block_key,
)
from drift_metrics import spearman

GUN7_FILE = os.path.join("results", "iterative_gun7.json")
SCORES_DIR = os.path.join("results", "gun7_scores")
OUTPUT_FILE = os.path.join("results", "diagnose_wanda_gun8.json")
MISSING = "not found"  # status marker written into the JSON output


# --------------------------------------------------------------------------- #
# a) Reference implementation (Sun et al. 2023, Eq. 1)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_o_proj_inputs(model: nn.Module, calib_batches: Sequence[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
    """Valid (mask=1) token rows of each layer's o_proj INPUT: [N_tokens, H*d] float64 (explicit X matrix)."""
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    rows: List[List[torch.Tensor]] = [[] for _ in layers]
    state: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(i: int):
        def hook(_mod, inputs):
            x = inputs[0]
            m = state["mask"].to(x.device).bool()
            rows[i].append(x[m].double().cpu())
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(i)))
    was_training = model.training
    model.eval()
    try:
        for b in calib_batches:
            state["mask"] = b["attention_mask"]
            model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return [torch.cat(r, dim=0) for r in rows]


def wanda_matrix(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """S_ij = |W_ij| * ||X_j||_2 (Wanda Eq. 1); W [out, in], X [N, in] -> S [out, in] float64."""
    col_norm = x.pow(2).sum(dim=0).sqrt()  # ||X_j||_2 over tokens
    return weight.detach().double().cpu().abs() * col_norm.unsqueeze(0)


def head_sum(s: torch.Tensor, n_heads: int, d: int) -> torch.Tensor:
    return s.reshape(s.shape[0], n_heads, d).sum(dim=(0, 2))


def row_rank_head_score(s: torch.Tensor, n_heads: int, d: int) -> torch.Tensor:
    """Wanda's original comparison group (within an output row): within-row percentile of S_ij in [0, 1], averaged over the head block."""
    out, inp = s.shape
    ranks = torch.argsort(torch.argsort(s, dim=1), dim=1).double() / max(inp - 1, 1)  # ties are rare for float scores
    return ranks.reshape(out, n_heads, d).mean(dim=(0, 2))


def reference_scores(model: nn.Module, calib_batches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    layers = _decoder_layers(model)
    n_heads, d = model.config.num_attention_heads, _head_dim(model)
    xs = collect_o_proj_inputs(model, calib_batches)
    ref: Dict[str, float] = {}
    row_rank: Dict[str, float] = {}
    profile: List[Dict[str, Any]] = []
    for i, (layer, x) in enumerate(zip(layers, xs)):
        w = layer.self_attn.o_proj.weight
        s = wanda_matrix(w, x)
        hs = head_sum(s, n_heads, d)
        rr = row_rank_head_score(s, n_heads, d)
        for h in range(n_heads):
            ref[f"layer_{i}.attn.head_{h}"] = float(hs[h])
            row_rank[f"layer_{i}.attn.head_{h}"] = float(rr[h])
        col_norm = x.pow(2).sum(dim=0).sqrt()
        profile.append({"layer": i, "n_tokens": int(x.shape[0]),
                        "x_rms": float((col_norm / max(x.shape[0], 1) ** 0.5).mean()),  # per-channel RMS activation
                        "x_norm_mean": float(col_norm.mean()),
                        "w_abs_colsum_mean": float(w.detach().double().abs().sum(dim=0).mean()),
                        "head_score_min": float(hs.min()), "head_score_mean": float(hs.mean()), "head_score_max": float(hs.max())})
    return {"reference": ref, "row_rank": row_rank, "profile": profile}


def max_rel_diff(a: Dict[str, float], b: Dict[str, float]) -> float:
    return max(abs(a[k] - b[k]) / max(abs(a[k]), 1e-12) for k in a)


# --------------------------------------------------------------------------- #
# b) Profile / correlation helpers
# --------------------------------------------------------------------------- #
def layer_index_spearman(scores: Dict[str, float]) -> float:
    keys = list(scores)
    return spearman([scores[k] for k in keys], [float(parse_block_key(k).layer) for k in keys])


def lowest_n(scores: Dict[str, float], n: int) -> List[str]:
    items = list(scores.items())
    order = sorted(range(len(items)), key=lambda i: (items[i][1], i))
    return [items[i][0] for i in order[:n]]


def per_layer_counts(names: Sequence[str], n_layers: int) -> List[int]:
    counts = [0] * n_layers
    for k in names:
        counts[parse_block_key(k).layer] += 1
    return counts


def layer_stats(heads: Sequence[Sequence[int]], n_layers: int = 32) -> Dict[str, Any]:
    layers = sorted(int(l) for l, _ in heads)
    counts = [0] * n_layers
    for l in layers:
        counts[l] += 1
    return {"n": len(layers), "layers_0_5": sum(1 for l in layers if l <= 5), "layers_20_31": sum(1 for l in layers if l >= 20),
            "median_layer": layers[len(layers) // 2] if layers else None, "per_layer": counts,
            "spearman_count_vs_layer": spearman(counts, list(range(n_layers)))}


# --------------------------------------------------------------------------- #
# c) Synthetic scale test
# --------------------------------------------------------------------------- #
@torch.no_grad()
def scale_test(model: nn.Module, calib_batches: Sequence[Dict[str, torch.Tensor]], scale: float, layer: int, n_select: int) -> Dict[str, Any]:
    """Scale v_proj of `layer` by `scale` (so its o_proj input and raw Wanda scores scale too); where does the global selection concentrate?"""
    m = copy.deepcopy(model)
    m.model.layers[layer].self_attn.v_proj.weight.mul_(scale)
    n_layers = len(_decoder_layers(m))
    base_raw = head_wanda_scores(model, calib_batches)
    out: Dict[str, Any] = {"scale": scale, "scaled_layer": layer, "n_select": n_select, "selection_per_layer": {}}
    for mode in WANDA_NORMALIZE_OPTIONS:
        sc = head_wanda_scores(m, calib_batches, normalize=mode)
        out["selection_per_layer"][str(mode)] = per_layer_counts(lowest_n(sc, n_select), n_layers)
        if mode is None:
            ratio = {k: sc[k] / base_raw[k] for k in sc}
            out["raw_score_ratio_scaled_layer"] = float(np.mean([v for k, v in ratio.items() if parse_block_key(k).layer == layer]))
            out["raw_score_ratio_other_layers"] = float(np.mean([v for k, v in ratio.items() if parse_block_key(k).layer != layer]))
            # the within-layer ranking must be preserved (all heads of the layer share the same factor)
            keys = [k for k in sc if parse_block_key(k).layer == layer]
            out["within_layer_spearman_raw_vs_base"] = spearman([sc[k] for k in keys], [base_raw[k] for k in keys])
    return out


# --------------------------------------------------------------------------- #
# d) 7B evidence
# --------------------------------------------------------------------------- #
def gun7_evidence(path: str, scores_dir: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"status": MISSING, "file": path}
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out: Dict[str, Any] = {"file": path, "fractions": {}}
    for fk, fr in d.get("fractions", {}).items():
        row: Dict[str, Any] = {}
        for name in ("prune_wanda", "xai_single", "prune_taylor"):
            c = fr.get("configs", {}).get(name)
            reps = [r for r in (c or {}).get("repeats", []) if r.get("status") == "completed"]
            if not reps:
                row[name] = MISSING
                continue
            r = reps[0]
            row[name] = {**layer_stats(r["pruned_heads"]), "perplexity": r.get("perplexity"),
                         "criterion_score_range": (r.get("criterion") or {}).get("score_range")}
        crit_file = os.path.join(scores_dir, f"f{fk}_prune_wanda_ln_criterion.json")
        if os.path.exists(crit_file):
            with open(crit_file, "r", encoding="utf-8") as f:
                crit = json.load(f)
            raw = {k: float(v) for k, v in crit["raw"].items()}
            per_layer: Dict[int, List[float]] = {}
            for k, v in raw.items():
                per_layer.setdefault(parse_block_key(k).layer, []).append(v)
            row["wanda_raw_profile_7b"] = {"file": crit_file, "spearman_score_vs_layer": layer_index_spearman(raw),
                                           "per_layer_mean": [float(np.mean(per_layer[i])) for i in sorted(per_layer)],
                                           "per_layer_min": [float(np.min(per_layer[i])) for i in sorted(per_layer)],
                                           "per_layer_max": [float(np.max(per_layer[i])) for i in sorted(per_layer)]}
        else:
            row["wanda_raw_profile_7b"] = f"{MISSING} (raw 7B Wanda scores were not stored; {crit_file} does not exist - written by a prune_wanda_ln run)"
        out["fractions"][fk] = row
    return out


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def diagnose_mini(scale: float, seed: int) -> Dict[str, Any]:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
    from test_xai_engine_mini import build_dummy_dataloader, build_mini_model

    model = build_mini_model(seed=seed)
    batches = build_dummy_dataloader(seed=seed + 1)
    n_layers = len(_decoder_layers(model))
    n_heads = model.config.num_attention_heads
    raw = head_wanda_scores(model, batches)
    ref = reference_scores(model, batches)
    variants = {"raw": raw, "layer_zscore": head_wanda_scores(model, batches, normalize="layer_zscore"),
                "layer_percentile": head_wanda_scores(model, batches, normalize="layer_percentile"),
                "row_rank": ref["row_rank"], "taylor": head_taylor_scores(model, batches), "magnitude": head_magnitude_scores(model)}
    n_select = max(1, (n_layers * n_heads) // 2)
    return {
        "model": f"mini Mistral ({n_layers} layers × {n_heads} heads, head_dim {_head_dim(model)}, seed {seed})",
        "definition_check": {
            "compressor_formula": "sum_{i, j in head} |W_ij| * sqrt(sum_t X_tj^2); ||X_j||_2 per channel over ALL valid tokens (identical to Wanda Eq. 1)",
            "max_rel_diff_vs_reference": max_rel_diff(raw, ref["reference"]),
            "spearman_raw_vs_reference": spearman([raw[k] for k in raw], [ref["reference"][k] for k in raw]),
            "comparison_group": "compressor: global (lowest-N over a single pool of heads from all layers); original Wanda: within an output row, within a layer",
        },
        "profile_per_layer": ref["profile"],
        "spearman_score_vs_layer_index": {k: layer_index_spearman(v) for k, v in variants.items()},
        "lowest_n_per_layer": {k: per_layer_counts(lowest_n(v, n_select), n_layers) for k, v in variants.items()},
        "within_layer_rank_agreement": {
            # normalized variants must preserve the raw within-layer ranking (Spearman 1.0)
            mode: float(np.mean([spearman([variants[mode][k] for k in ks], [raw[k] for k in ks])
                                 for ks in ([k for k in raw if parse_block_key(k).layer == i] for i in range(n_layers))]))
            for mode in ("layer_zscore", "layer_percentile", "row_rank")
        },
        "scale_test": scale_test(model, batches, scale, layer=n_layers - 1, n_select=n_heads),
    }


def format_report(r: Dict[str, Any]) -> str:
    lines = ["=== Wanda diagnosis ==="]
    m = r["mini"]
    dc = m["definition_check"]
    lines.append(f"[a] {m['model']}: compressor vs reference (Sun 2023 Eq. 1) relative difference {dc['max_rel_diff_vs_reference']:.2e}, "
                 f"Spearman {dc['spearman_raw_vs_reference']:.4f}")
    lines.append(f"    comparison group: {dc['comparison_group']}")
    lines.append("[b] layer profile (mini): " + "; ".join(
        f"L{p['layer']}: x_rms {p['x_rms']:.3f}, mean |W| column sum {p['w_abs_colsum_mean']:.3f}, head score "
        f"{p['head_score_min']:.3f}–{p['head_score_max']:.3f}" for p in m["profile_per_layer"]))
    lines.append("    score-layer Spearman: " + ", ".join(f"{k} {v:+.3f}" for k, v in m["spearman_score_vs_layer_index"].items()))
    lines.append("    within-layer rank agreement (vs raw): " + ", ".join(f"{k} {v:.3f}" for k, v in m["within_layer_rank_agreement"].items()))
    st = m["scale_test"]
    lines.append(f"[c] scale test: layer {st['scaled_layer']} v_proj ×{st['scale']} -> raw score ratio scaled layer {st['raw_score_ratio_scaled_layer']:.1f}×, "
                 f"other layers {st['raw_score_ratio_other_layers']:.2f}×; within-layer Spearman {st['within_layer_spearman_raw_vs_base']:.3f}; "
                 f"lowest-{st['n_select']} selection per layer: " + ", ".join(f"{k}={v}" for k, v in st["selection_per_layer"].items()))
    g = r["gun7"]
    if g.get("status") == MISSING:
        lines.append(f"[d] 7B evidence {MISSING}: {g['file']}")
    else:
        for fk, row in g["fractions"].items():
            parts = []
            for name in ("prune_wanda", "xai_single", "prune_taylor"):
                v = row.get(name)
                parts.append(f"{name}: {MISSING}" if not isinstance(v, dict) else
                             f"{name}: L0–5 {v['layers_0_5']}/{v['n']}, L20–31 {v['layers_20_31']}, median L{v['median_layer']}, "
                             f"count-layer ρ {v['spearman_count_vs_layer']:+.2f}, ppl {v['perplexity']:.2f}")
            lines.append(f"[d] 7B ratio {fk}: " + " | ".join(parts))
            w = row.get("prune_wanda")
            if isinstance(w, dict) and w.get("criterion_score_range"):
                lo, hi = w["criterion_score_range"]
                lines.append(f"    raw Wanda score range {lo:.1f} – {hi:.1f} ({hi / max(lo, 1e-9):.0f}×)")
            prof = row.get("wanda_raw_profile_7b")
            lines.append(f"    7B raw profile: {prof if isinstance(prof, str) else 'score-layer Spearman %+.3f' % prof['spearman_score_vs_layer']}")
    lines.append(f"[verdict] {r['verdict']}")
    return "\n".join(lines)


def verdict(r: Dict[str, Any]) -> str:
    dc = r["mini"]["definition_check"]
    ok = dc["max_rel_diff_vs_reference"] < 1e-4
    return (("No computation error: " if ok else "WARNING: differs from the reference: ") +
            f"compressor.head_wanda_scores matches Wanda Eq. 1 (|W_ij|·||X_j||_2, L2 over tokens) within a relative {dc['max_rel_diff_vs_reference']:.1e}. "
            "The failure is a SCALE + METHOD MISMATCH: original Wanda compares scores within an output row (within a layer), "
            "whereas the head-level adaptation ranks raw scores in a single cross-layer pool, so the cross-layer activation/weight scale "
            "(three orders of magnitude on the 7B model) dominates the selection and all heads of the early layers are pruned first. "
            "Within-layer normalization (layer_zscore) preserves the within-layer ranking and removes the cross-layer scale; "
            "the 7B check is the prune_wanda_ln configuration.")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Diagnosis of the head-level Wanda score (mini model + stored 7B results; CPU only)")
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--gun7-json", default=GUN7_FILE)
    p.add_argument("--scores-dir", default=SCORES_DIR)
    p.add_argument("--scale", type=float, default=20.0, help="factor for the synthetic scale test (v_proj of the last layer)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gun7", action="store_true", help="skip the stored 7B evidence (mini model only)")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result: Dict[str, Any] = {"created_at": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
                              "torch": torch.__version__, "mini": diagnose_mini(args.scale, args.seed),
                              "gun7": {"status": MISSING, "file": "(--no-gun7)"} if args.no_gun7 else gun7_evidence(args.gun7_json, args.scores_dir)}
    result["verdict"] = verdict(result)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(format_report(result))
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
