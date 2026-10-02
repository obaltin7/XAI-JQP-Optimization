"""
Per-layer visualization of the importance scores + sanity checks.

Reads the 1056 scores (32 layers x 32 heads + 32 MLPs) in results/importance_scores_gun3.json
and draws a two-column heatmap per layer:
    column 1: MLP block score
    column 2: mean (or sum) score of the attention heads in that layer

MLP and attention scores differ in scale by roughly 100x, so the two columns are
drawn with SEPARATE color scales; otherwise the attention column would be uniformly
pale and the differences between layers would not be visible.

Also runs numerical sanity checks: score count/key schema, NaN/inf, positivity,
magnitude of the MLP >> attention observation. Exit code 1 if any check fails.

No GPU required; CPU + matplotlib.

Usage:
    python tools/visualization/visualize_importance.py
    python tools/visualization/visualize_importance.py --attn-agg sum
    python tools/visualization/visualize_importance.py --input results/x.json --output assets/x.png
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless environments (remote SSH, CI)
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_INPUT = os.path.join(_REPO_ROOT, "results", "importance_scores_gun3.json")
DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "assets", "importance_heatmap.png")

KEY_HEAD = re.compile(r"^layer_(\d+)\.attn\.head_(\d+)$")
KEY_MLP = re.compile(r"^layer_(\d+)\.mlp$")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_scores(path: str) -> Tuple[dict, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "scores" not in data or not isinstance(data["scores"], dict):
        raise ValueError(f"{path}: 'scores' dict not found")
    return data, data["scores"]


def to_matrices(scores: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    scores -> (heads[L, H], mlp[L])
    Key schema: layer_{l}.attn.head_{h} and layer_{l}.mlp
    Unrecognized keys raise an error (silently skipping them would defeat the sanity check).
    Missing keys stay NaN and are reported by the sanity check.
    """
    head_entries: List[Tuple[int, int, float]] = []
    mlp_entries: List[Tuple[int, float]] = []
    unknown: List[str] = []
    for k, v in scores.items():
        m = KEY_HEAD.match(k)
        if m:
            head_entries.append((int(m.group(1)), int(m.group(2)), float(v)))
            continue
        m = KEY_MLP.match(k)
        if m:
            mlp_entries.append((int(m.group(1)), float(v)))
            continue
        unknown.append(k)
    if unknown:
        raise ValueError(f"Unrecognized key(s): {unknown[:5]} ... (total {len(unknown)})")
    if not head_entries or not mlp_entries:
        raise ValueError("No head or MLP scores found")

    n_layers = max(max(l for l, _, _ in head_entries), max(l for l, _ in mlp_entries)) + 1
    n_heads = max(h for _, h, _ in head_entries) + 1

    heads = np.full((n_layers, n_heads), np.nan)
    mlp = np.full(n_layers, np.nan)
    for l, h, v in head_entries:
        heads[l, h] = v
    for l, v in mlp_entries:
        mlp[l] = v
    return heads, mlp


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
def sanity_check(heads: np.ndarray, mlp: np.ndarray, attn_col: np.ndarray,
                 n_scores_declared: int) -> bool:
    n_layers, n_heads = heads.shape
    ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        tag = "OK  " if passed else "FAIL"
        print(f"  [{tag}] {name}" + (f": {detail}" if detail else ""))

    print("\n=== SANITY CHECKS ===")

    # 1) Shape / missing keys
    missing_heads = int(np.isnan(heads).sum())
    missing_mlp = int(np.isnan(mlp).sum())
    report("Shape", missing_heads == 0 and missing_mlp == 0,
           f"{n_layers} layers x {n_heads} heads + {n_layers} MLPs = {n_layers * (n_heads + 1)} "
           f"(missing heads: {missing_heads}, missing MLPs: {missing_mlp})")
    report("Consistent with declared n_scores",
           n_scores_declared == n_layers * (n_heads + 1),
           f"JSON n_scores={n_scores_declared}, expected={n_layers * (n_heads + 1)}")

    # 2) NaN / inf. NaN/Infinity may appear as literals in the JSON (json.load accepts them);
    #    NaNs from missing keys were reported above and are subtracted here.
    all_vals = np.concatenate([heads.ravel(), mlp])
    n_nan = int(np.isnan(all_vals).sum()) - missing_heads - missing_mlp
    n_inf = int(np.isinf(all_vals).sum())
    report("No NaN", n_nan == 0, f"{n_nan} NaN")
    report("No inf", n_inf == 0, f"{n_inf} inf")

    finite = all_vals[np.isfinite(all_vals)]

    # 3) Positivity. The score is a mean |attribution|, so negatives should be impossible;
    #    an exact 0 means the block contributes nothing (ablation candidate, but suspicious).
    n_neg = int((finite < 0).sum())
    n_zero = int((finite == 0).sum())
    report("No negative scores", n_neg == 0, f"{n_neg} negative")
    report("No exactly-zero scores", n_zero == 0,
           f"{n_zero} zero" + ("" if n_zero == 0 else " (warning: dead block?)"))

    # 4) MLP >> attention observation (per-head scale)
    head_mean = float(np.nanmean(heads))
    head_max = float(np.nanmax(heads))
    mlp_mean = float(np.nanmean(mlp))
    mlp_min = float(np.nanmin(mlp))
    ratio_mean = mlp_mean / head_mean if head_mean > 0 else math.inf
    ratio_worst = mlp_min / head_max if head_max > 0 else math.inf
    per_layer_ratio = mlp / np.nanmean(heads, axis=1)  # within layer: MLP / mean head score of that layer

    print("\n  --- MLP vs attention (per head) ---")
    print(f"  mean MLP score               : {mlp_mean:10.4f}")
    print(f"  mean head score              : {head_mean:10.4f}")
    print(f"  RATIO (mean MLP / mean head) : {ratio_mean:10.1f}x")
    print(f"  lowest MLP / highest head    : {mlp_min:.4f} / {head_max:.4f} = {ratio_worst:.1f}x")
    print(f"  within-layer ratio (min/median/max): "
          f"{np.nanmin(per_layer_ratio):.1f}x / {np.nanmedian(per_layer_ratio):.1f}x / "
          f"{np.nanmax(per_layer_ratio):.1f}x")
    report("Mean MLP > 10x mean head", ratio_mean > 10, f"{ratio_mean:.1f}x")
    report("Every MLP > every head (worst case)", ratio_worst > 1, f"{ratio_worst:.1f}x")

    # 5) Also compare attention as a block (SUM of the 32 heads vs the MLP block).
    #    This shows whether the "MLP dominates" claim is an artefact of the per-head scale.
    attn_sum = np.nansum(heads, axis=1)
    block_ratio = mlp / attn_sum
    print("\n  --- MLP vs attention (layer SUM, block vs block) ---")
    print(f"  mean(MLP / attn sum): {float(np.nanmean(block_ratio)):.2f}x "
          f"(layer min {float(np.nanmin(block_ratio)):.2f}x, max {float(np.nanmax(block_ratio)):.2f}x)")
    report("MLP block > attention block (sum) in every layer",
           bool(np.all(mlp > attn_sum)),
           f"{int((mlp > attn_sum).sum())}/{n_layers} layers")

    # 6) Extreme values (for manual inspection)
    print("\n  --- Extreme values ---")
    print(f"  highest MLP     : layer_{int(np.nanargmax(mlp))} = {float(np.nanmax(mlp)):.3f}")
    print(f"  lowest MLP      : layer_{int(np.nanargmin(mlp))} = {float(np.nanmin(mlp)):.3f}")
    hi = np.unravel_index(int(np.nanargmax(heads)), heads.shape)
    lo = np.unravel_index(int(np.nanargmin(heads)), heads.shape)
    print(f"  highest head    : layer_{hi[0]}.attn.head_{hi[1]} = {heads[hi]:.4f}")
    print(f"  lowest head     : layer_{lo[0]}.attn.head_{lo[1]} = {heads[lo]:.4f}")
    print(f"  attn column max layer: layer_{int(np.nanargmax(attn_col))}, "
          f"min layer: layer_{int(np.nanargmin(attn_col))}")

    print(f"\n  RESULT: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_heatmap(mlp: np.ndarray, attn_col: np.ndarray, attn_label: str,
                 meta: dict, output: str) -> None:
    """
    Two side-by-side panels with a shared y axis (layer). Each panel is a single-column
    heatmap with its OWN color scale (single hue, light->dark). Values are written in the cells.
    """
    n_layers = mlp.shape[0]
    fig_h = max(8.0, 0.32 * n_layers + 1.8)
    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, fig_h), sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.6},
    )

    panels = [
        (axes[0], mlp, "MLP block score", "Blues"),
        (axes[1], attn_col, f"Attention ({attn_label})", "Oranges"),
    ]
    for ax, col, title, cmap in panels:
        mat = col.reshape(-1, 1)
        vmin, vmax = float(np.nanmin(col)), float(np.nanmax(col))
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_xticks([])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"layer_{i}" for i in range(n_layers)], fontsize=7.5)
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        # cell values: light text on dark cells, dark text on light cells
        span = (vmax - vmin) or 1.0
        for i, v in enumerate(col):
            frac = (v - vmin) / span
            ax.text(0, i, f"{v:.3f}" if v < 1 else f"{v:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if frac > 0.6 else "#1b1b1b")
        # aspect=45 so the colorbar spans the full height of the narrow panel
        cbar = fig.colorbar(im, ax=ax, fraction=0.10, pad=0.06, aspect=45)
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_visible(False)

    # imshow puts row 0 at the top -> layer_0 at the top, layer_31 at the bottom
    axes[0].set_ylabel("Layer", fontsize=9)

    model = meta.get("model", "?")
    cal = meta.get("calibration", {})
    att = meta.get("attribution", {})
    subtitle = (
        f"{model} | {meta.get('precision', '?')} | {cal.get('n_valid_tokens', '?')} token, "
        f"T={cal.get('max_length', '?')} | LIG n_steps={att.get('n_steps', '?')}\n"
        "Color scales are per column (MLP and attention differ in scale by ~100x)."
    )
    fig.suptitle("Per-layer structural importance scores", fontsize=12, y=0.995)
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=7.5, color="#555555")
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.13, right=0.90)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--attn-agg", choices=["mean", "sum"], default="mean",
                        help="attention column: mean or sum of the heads in the layer")
    args = parser.parse_args()

    # UTF-8 output for non-ASCII characters on the Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Reading: {args.input}")
    meta, scores = load_scores(args.input)
    heads, mlp = to_matrices(scores)
    n_layers, n_heads = heads.shape
    print(f"  {len(scores)} scores -> {n_layers} layers x {n_heads} heads, {n_layers} MLPs")

    if args.attn_agg == "mean":
        attn_col = np.nanmean(heads, axis=1)
        attn_label = "head mean"
    else:
        attn_col = np.nansum(heads, axis=1)
        attn_label = "head sum"

    ok = sanity_check(heads, mlp, attn_col, int(meta.get("n_scores", -1)))

    plot_heatmap(mlp, attn_col, attn_label, meta, args.output)
    print(f"\nHeatmap saved: {args.output}")

    # Per-layer table (also kept as text)
    print("\n=== Per-layer table ===")
    print(f"  {'layer':<9}{'MLP':>10}{'attn mean':>11}{'attn sum':>11}{'MLP/mean':>9}")
    attn_mean = np.nanmean(heads, axis=1)
    attn_sum = np.nansum(heads, axis=1)
    for i in range(n_layers):
        print(f"  layer_{i:<3}{mlp[i]:>10.3f}{attn_mean[i]:>11.4f}{attn_sum[i]:>11.3f}"
              f"{mlp[i] / attn_mean[i]:>8.0f}x")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
