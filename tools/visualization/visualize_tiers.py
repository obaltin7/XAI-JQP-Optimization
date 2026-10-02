"""
Visualization of the tier assignment (n_tiers=3) + sanity checks.

Reads the raw scores in results/importance_scores_gun3.json and the tier assignment
in results/tier_analysis_gun4.json (compressor.py output, n_tiers=3: prune / int4 / fp16)
and draws a two-panel figure:
    left panel : attention heads (32 layers x 32 heads), each cell colored by its tier
                 (prune=red, int4=orange, fp16=green)
    right panel: MLP blocks (32 layers), single column with the same color coding.
                 Because of the mlp_min_tier="int4" constraint there must be NO red
                 (prune) cell here -- the sanity check verifies this separately.

Also runs numerical sanity checks (same pattern as visualize_importance.py):
is every block in exactly one tier, are the labels valid, are the counts and the
per-tier score ranges (thresholds) consistent with tier_analysis_gun4.json, are the
tiers monotone in the score, is there no pruned MLP, and can the assignment be
reproduced by compressor.allocate_compression_tiers with the same parameters.
Exit code 1 if any check fails.

No GPU required; CPU + matplotlib.

Usage:
    python tools/visualization/visualize_tiers.py
    python tools/visualization/visualize_tiers.py --analysis results/x.json --output assets/x.png
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless environments (remote SSH, CI)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compressor import TIER_LABELS, allocate_compression_tiers, parse_block_key, tier_thresholds

DEFAULT_SCORES = os.path.join(_REPO_ROOT, "results", "importance_scores_gun3.json")
DEFAULT_ANALYSIS = os.path.join(_REPO_ROOT, "results", "tier_analysis_gun4.json")
DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "assets", "tier_assignment.png")

# Tier -> color (order of the method description: Blind Spot / Low Impact / [INT8] / Critical Core)
TIER_COLORS = {"prune": "#d62728", "int4": "#ff7f0e", "int8": "#ffd92f", "fp16": "#2ca02c"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_inputs(scores_path: str, analysis_path: str) -> Tuple[Dict[str, float], dict, Dict[str, str]]:
    """Raw scores, analysis metadata and the tier dict. Raises if a field is missing."""
    scores_doc = load_json(scores_path)
    if "scores" not in scores_doc or not isinstance(scores_doc["scores"], dict):
        raise ValueError(f"{scores_path}: 'scores' dict not found")
    analysis = load_json(analysis_path)
    for field in ("n_tiers", "tier_fractions", "mlp_min_tier", "tier_counts", "tier_score_ranges", "tiers"):
        if field not in analysis:
            raise ValueError(f"{analysis_path}: field '{field}' missing (expected compressor.py --output output)")
    return {k: float(v) for k, v in scores_doc["scores"].items()}, analysis, dict(analysis["tiers"])


def to_tier_matrices(tiers: Dict[str, str], labels: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    tiers -> (heads[L, H], mlp[L]); cell value = tier index (order of labels),
    missing cell -1. Unrecognized keys/labels raise an error.
    """
    parsed = [(parse_block_key(k), lbl) for k, lbl in tiers.items()]
    for bk, lbl in parsed:
        if lbl not in labels:
            raise ValueError(f"{bk.name}: unknown tier {lbl!r} (expected {labels})")
    n_layers = max(bk.layer for bk, _ in parsed) + 1
    n_heads = max((bk.index for bk, _ in parsed if bk.kind == "attn"), default=-1) + 1
    heads = np.full((n_layers, n_heads), -1, dtype=int)
    mlp = np.full(n_layers, -1, dtype=int)
    for bk, lbl in parsed:
        idx = labels.index(lbl)
        if bk.kind == "attn":
            heads[bk.layer, bk.index] = idx
        else:
            mlp[bk.layer] = idx
    return heads, mlp


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
def sanity_check(scores: Dict[str, float], analysis: dict, tiers: Dict[str, str],
                 heads: np.ndarray, mlp: np.ndarray) -> Tuple[int, int]:
    """Print the checks as a table; return (passed, total)."""
    labels = TIER_LABELS[int(analysis["n_tiers"])]
    rows: List[Tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        rows.append((name, bool(passed), detail))

    # 1) Every block in exactly one tier: key set identical to the scores, no empty
    #    matrix cell (duplicates are impossible in a dict; missing ones are caught here)
    missing = set(scores) - set(tiers)
    extra = set(tiers) - set(scores)
    n_empty = int((heads < 0).sum() + (mlp < 0).sum())
    check("Every block in exactly one tier (none missing/extra)",
          not missing and not extra and n_empty == 0,
          f"{len(tiers)} assignments / {len(scores)} scores, missing {len(missing)}, extra {len(extra)}, empty cells {n_empty}")

    # 2) Labels only from the n_tiers tiers
    used = set(tiers.values())
    check(f"Labels valid (n_tiers={analysis['n_tiers']})", used <= set(labels), f"used: {sorted(used)}")

    # 3) Counts equal to tier_counts in the JSON
    counts = {
        kind: {l: sum(1 for k, t in tiers.items() if t == l and parse_block_key(k).kind == kind) for l in labels}
        for kind in ("attn", "mlp")
    }
    declared = analysis["tier_counts"]
    same = all(counts[kind].get(l, 0) == declared.get(kind, {}).get(l, 0) for kind in counts for l in labels)
    check("Counts consistent with tier_counts", same,
          f"attn {counts['attn']} | mlp {counts['mlp']}")

    # 4) No pruned MLP (evidence of the mlp_min_tier constraint)
    n_mlp_prune = int((mlp == labels.index("prune")).sum())
    check("No prune among MLP blocks (mlp_min_tier constraint)",
          n_mlp_prune == 0 and analysis["mlp_min_tier"] == "int4",
          f"{n_mlp_prune} pruned MLPs, mlp_min_tier={analysis['mlp_min_tier']!r}")

    # 5) Thresholds: tier ranges recomputed from the scores equal those in the JSON
    recomputed = tier_thresholds(scores, tiers)
    declared_ranges = analysis["tier_score_ranges"]
    diffs = []
    for kind, per_label in recomputed.items():
        for l, (lo, hi) in per_label.items():
            d_lo, d_hi = declared_ranges.get(kind, {}).get(l, (math.nan, math.nan))
            diffs.append(max(abs(lo - d_lo), abs(hi - d_hi)))
    max_diff = max(diffs) if diffs else math.inf
    check("Tier thresholds consistent with tier_score_ranges", max_diff < 1e-9, f"max diff {max_diff:.2e}")

    # 6) Tiers monotone in the score (ranges do not overlap): max(prune) < min(int4) < ...
    monotone = True
    detail = []
    for kind, per_label in recomputed.items():
        present = [l for l in labels if l in per_label]
        for lo_l, hi_l in zip(present[:-1], present[1:]):
            ok = per_label[lo_l][1] <= per_label[hi_l][0]
            monotone &= ok
            detail.append(f"{kind}: {lo_l}≤{per_label[lo_l][1]:.4f} < {hi_l}≥{per_label[hi_l][0]:.4f}")
    check("Tier ranges do not overlap (monotone in score)", monotone, "; ".join(detail))

    # 7) Reproducibility: allocate_compression_tiers gives the same result with the same parameters
    regenerated = allocate_compression_tiers(
        scores, int(analysis["n_tiers"]),
        tier_fractions=analysis["tier_fractions"], mlp_min_tier=analysis["mlp_min_tier"],
    )
    n_diff = sum(1 for k in tiers if regenerated.get(k) != tiers[k])
    check("Assignment reproduced with compressor.py", n_diff == 0,
          f"{n_diff} differing blocks (fractions {tuple(analysis['tier_fractions'])})")

    # 8) Fractions: attn prune count = round(fraction * n_heads) (±1, largest-remainder method)
    n_attn = int(heads.size)
    expected = analysis["tier_fractions"][0] * n_attn
    check("attn prune fraction matches tier_fractions[0]",
          abs(counts["attn"]["prune"] - expected) <= 1,
          f"{counts['attn']['prune']} / {n_attn} = {counts['attn']['prune'] / n_attn:.3f}, expected ≈ {expected:.0f}")

    # --- table ---
    print("\n=== SANITY CHECKS ===")
    width = max(len(r[0]) for r in rows)
    print(f"  {'#':>2}  {'state':<5} {'check'.ljust(width)}  details")
    for i, (name, passed, detail) in enumerate(rows, 1):
        print(f"  {i:>2}  {'OK' if passed else 'FAIL':<5} {name.ljust(width)}  {detail}")
    n_pass = sum(1 for r in rows if r[1])
    print(f"\n  RESULT: {n_pass}/{len(rows)} checks OK" + ("" if n_pass == len(rows) else "  -- SOME CHECKS FAILED"))
    return n_pass, len(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def summary_line(tiers: Dict[str, str], labels: Tuple[str, ...]) -> str:
    """Subtitle: 'Head: 205 prune, 410 int4, 409 fp16 | MLP: 0 prune, 19 int4, 13 fp16'."""
    parts = []
    for kind, title in (("attn", "Head"), ("mlp", "MLP")):
        names = [k for k in tiers if parse_block_key(k).kind == kind]
        parts.append(f"{title}: " + ", ".join(f"{sum(1 for k in names if tiers[k] == l)} {l}" for l in labels))
    return " | ".join(parts)


def plot_tiers(heads: np.ndarray, mlp: np.ndarray, labels: Tuple[str, ...], analysis: dict,
               subtitle: str, output: str) -> None:
    """
    Left: [L, H] categorical heatmap; right: [L, 1] strip with the tier name written in each cell.
    Shared y axis (layer); imshow puts row 0 at the top -> layer_0 at the top.
    """
    n_layers, n_heads = heads.shape
    cmap = ListedColormap([TIER_COLORS[l] for l in labels])
    vmin, vmax = -0.5, len(labels) - 0.5

    fig_h = max(9.0, 0.30 * n_layers + 2.2)
    fig, (ax_h, ax_m) = plt.subplots(
        1, 2, figsize=(11.0, fig_h), sharey=True,
        gridspec_kw={"width_ratios": [n_heads, 2.2], "wspace": 0.08},
    )

    ax_h.imshow(heads, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax_h.set_title("Attention heads (layer x head)", fontsize=10, pad=8)
    ax_h.set_xlabel("head", fontsize=9)
    ax_h.set_ylabel("Layer", fontsize=9)
    ax_h.set_xticks(range(0, n_heads, 2))
    ax_h.set_xticklabels([str(h) for h in range(0, n_heads, 2)], fontsize=7)
    ax_h.set_yticks(range(n_layers))
    ax_h.set_yticklabels([f"layer_{i}" for i in range(n_layers)], fontsize=7.5)
    # cell grid (thin white lines) for readability
    ax_h.set_xticks(np.arange(-0.5, n_heads, 1), minor=True)
    ax_h.set_yticks(np.arange(-0.5, n_layers, 1), minor=True)
    ax_h.grid(which="minor", color="white", linewidth=0.4)
    ax_h.tick_params(which="minor", length=0)
    ax_h.tick_params(which="major", length=0)

    ax_m.imshow(mlp.reshape(-1, 1), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax_m.set_title("MLP block", fontsize=10, pad=8)
    ax_m.set_xticks([])
    ax_m.set_yticks(np.arange(-0.5, n_layers, 1), minor=True)
    ax_m.grid(which="minor", axis="y", color="white", linewidth=0.4)
    ax_m.tick_params(which="both", length=0)
    for i, idx in enumerate(mlp):
        ax_m.text(0, i, labels[idx] if idx >= 0 else "?", ha="center", va="center", fontsize=7.5, color="white",
                  fontweight="bold")
    for ax in (ax_h, ax_m):
        for s in ax.spines.values():
            s.set_visible(False)

    handles = [Patch(facecolor=TIER_COLORS[l], label=l) for l in labels]
    fig.legend(handles=handles, loc="lower center", ncol=len(labels), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    fr = tuple(analysis["tier_fractions"])
    fig.suptitle(f"XAI-JQP tier assignment (n_tiers={analysis['n_tiers']}, percentile, within kind)",
                 fontsize=12, y=0.995)
    fig.text(0.5, 0.962,
             f"{subtitle}\nfractions {fr} | mlp_min_tier={analysis['mlp_min_tier']} "
             f"(MLPs are never pruned → no red in the right panel) | source: {os.path.basename(analysis.get('source', '?'))}",
             ha="center", va="top", fontsize=8, color="#555555")
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.09, right=0.98)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default=DEFAULT_SCORES)
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # UTF-8 output for non-ASCII characters on the Windows console

    print(f"Reading: {args.scores} + {args.analysis}")
    scores, analysis, tiers = load_inputs(args.scores, args.analysis)
    labels = TIER_LABELS[int(analysis["n_tiers"])]
    heads, mlp = to_tier_matrices(tiers, labels)
    n_layers, n_heads = heads.shape
    print(f"  {len(tiers)} assignments -> {n_layers} layers x {n_heads} heads, {n_layers} MLPs; tiers {labels}")

    n_pass, n_total = sanity_check(scores, analysis, tiers, heads, mlp)

    subtitle = summary_line(tiers, labels)
    plot_tiers(heads, mlp, labels, analysis, subtitle, args.output)
    print(f"\nSummary: {subtitle}")
    print(f"Figure saved: {args.output} ({os.path.getsize(args.output):,} bytes)")

    # Per-layer table: number of heads per tier in each layer + MLP tier
    print("\n=== Per-layer table ===")
    print(f"  {'layer':<9}" + "".join(f"{('head ' + l):>11}" for l in labels) + f"{'MLP':>8}")
    for i in range(n_layers):
        row = "".join(f"{int((heads[i] == labels.index(l)).sum()):>11}" for l in labels)
        print(f"  layer_{i:<3}{row}{labels[mlp[i]]:>8}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
