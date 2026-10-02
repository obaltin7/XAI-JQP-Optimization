# compressor.py
"""
XAI-JQP, Stage 2: dynamic budget allocation (tier assignment + JQP application).

This module takes the structural importance scores produced by Stage 1
(xai_engine.calculate_importance_scores), assigns a compression tier to every
structural block (attention head / MLP block), and applies those tiers to the
model via structural pruning (pure torch masking, no bitsandbytes) and
bitsandbytes quantization.

Tier structure (`n_tiers`):
  * n_tiers=3 (DEFAULT), the base design:
        Blind spot      -> "prune"  (block removed entirely)
        Low impact      -> "int4"   (bitsandbytes NF4)
        Critical core   -> "fp16"   (original precision)
  * n_tiers=4 (OPTIONAL), with an additional INT8 intermediate tier:
        "prune" / "int4" / "int8" / "fp16"
    This option should only be used when there is EMPIRICAL evidence that the
    xAI importance distribution actually splits into 4 natural clusters, not as
    an arbitrary increase in complexity. `analyze_natural_clusters` (1-D k-means
    + silhouette, numpy only, no GPU) provides the data for this decision.

Method (thresholding: percentile / rank based, within kind):
  Thresholds are set by PERCENTILE: blocks are ranked by score within their
  kind (heads among heads, MLPs among MLPs) and distributed over the tiers from
  lowest to highest according to `tier_fractions`. Percentile thresholds are
  preferred over equal-width bins (splitting min-max into n_tiers) because:
    1. The ablation ratio (20/40/60%) maps directly to a tier boundary:
       tier_fractions=(0.2, 0.4, 0.4) => the lowest 20% is pruned. With
       equal-width bins this ratio is not controllable and depends on the
       shape of the distribution.
    2. On the Mistral-7B importance scores (results/importance_scores_gun3.json)
       head scores are log-normal-like and right-skewed (0.010-0.712, median
       0.086). Equal-width bins would put ~97% of the heads into the lowest
       tier because of a single outlier (layer_0.head_29).
    3. Within-kind normalization (MLP/head scale gap ~100x) comes for free:
       being rank based, the scale gap does not affect the result.
  Cost: tier boundaries may not coincide with "natural gaps" in the data, so
  `analyze_natural_clusters` also reports cluster boundaries (natural
  thresholds) for comparison with the percentile thresholds.

MLP blocks and pruning:
  An MLP block in Mistral-7B has ~176M parameters (~2.4% of the model), and in
  the Mistral-7B importance scores no MLP block looks like a "blind spot"
  (min 7.17, median ~9; even the lowest MLP is 10x the highest head). With the
  default `mlp_min_tier="int4"` MLP blocks are therefore reduced to INT4 at
  most and never pruned. Whole-MLP pruning is enabled with
  `mlp_min_tier="prune"`.

Granularity note (head tier vs. Linear-module quantization):
  Tiers are assigned per head, but bitsandbytes quantizes an nn.Linear module
  as a whole (q/k/v/o_proj); a single head cannot be made INT4. Hence:
    * PRUNING is at head granularity: the head's ROWS of q_proj and COLUMNS of
      o_proj are zeroed (k/v_proj are shared across heads under GQA and left
      untouched).
    * QUANTIZATION is at the granularity of the layer's attention block: the
      block tier is the MEDIAN of the tiers of its unpruned heads
      (`build_compression_plan`). If all heads are pruned the block is "prune".
  The MLP block is a single unit, so both granularities coincide.

Usage:
    tiers = allocate_compression_tiers(scores, n_tiers=3)   # {"layer_0.attn.head_0": "int4", ...}
    print(format_tier_report(scores, tiers))
    report = analyze_natural_clusters(scores)                 # data for the n_tiers decision
    result = apply_jqp(model, scores, n_tiers=3)              # GPU (bitsandbytes)

    python src/compressor.py --scores results/importance_scores_gun3.json --n-tiers 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TIER_LABELS: Dict[int, Tuple[str, ...]] = {
    3: ("prune", "int4", "fp16"),
    4: ("prune", "int4", "int8", "fp16"),
}
# Default fractions: prune the lowest 20% (the first ablation ratio); the
# remaining blocks are split evenly between low impact and critical core.
DEFAULT_TIER_FRACTIONS: Dict[int, Tuple[float, ...]] = {
    3: (0.20, 0.40, 0.40),
    4: (0.20, 0.30, 0.20, 0.30),
}
# Nominal bits per weight for budget estimation; the bitsandbytes block-absmax
# overhead (~0.5 bit for NF4) is deliberately excluded.
BITS_PER_TIER: Dict[str, float] = {"prune": 0.0, "int4": 4.0, "int8": 8.0, "fp16": 16.0}

KEY_HEAD = re.compile(r"^layer_(\d+)\.attn\.head_(\d+)$")
KEY_MLP = re.compile(r"^layer_(\d+)\.mlp(?:\.group_(\d+))?$")


# --------------------------------------------------------------------------- #
# Key parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BlockKey:
    """Parsed score key. kind: "attn" | "mlp"."""

    name: str
    kind: str
    layer: int
    index: int  # attn: head index, mlp: group index (0 if there are no groups)


def parse_block_key(name: str) -> BlockKey:
    """
    Parse a `layer_{i}.attn.head_{h}` / `layer_{i}.mlp[.group_{g}]` key.
    Unrecognized keys raise (silently skipping them would make thresholding
    meaningless; same policy as visualize_importance.py).
    """
    m = KEY_HEAD.match(name)
    if m:
        return BlockKey(name, "attn", int(m.group(1)), int(m.group(2)))
    m = KEY_MLP.match(name)
    if m:
        return BlockKey(name, "mlp", int(m.group(1)), int(m.group(2) or 0))
    raise ValueError(
        f"Unrecognized score key: {name!r} "
        "(expected: layer_{i}.attn.head_{h} or layer_{i}.mlp[.group_{g}])"
    )


def split_by_kind(importance_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Split scores by kind into {"attn": {...}, "mlp": {...}} (key order preserved)."""
    out: Dict[str, Dict[str, float]] = {"attn": {}, "mlp": {}}
    for name, v in importance_scores.items():
        out[parse_block_key(name).kind][name] = float(v)
    return {k: v for k, v in out.items() if v}


def _labels_for(tiers: Dict[str, str]) -> Tuple[str, ...]:
    """Infer from the labels whether a tier dict uses 3 or 4 tiers."""
    return TIER_LABELS[4] if "int8" in set(tiers.values()) else TIER_LABELS[3]


# --------------------------------------------------------------------------- #
# Tier assignment
# --------------------------------------------------------------------------- #
def _resolve_fractions(n_tiers: int, tier_fractions: Optional[Sequence[float]]) -> Tuple[float, ...]:
    if n_tiers not in TIER_LABELS:
        raise ValueError(f"n_tiers={n_tiers} is not supported; must be one of {sorted(TIER_LABELS)}.")
    src = tier_fractions if tier_fractions is not None else DEFAULT_TIER_FRACTIONS[n_tiers]
    fr = tuple(float(f) for f in src)
    if len(fr) != n_tiers:
        raise ValueError(f"tier_fractions has length {len(fr)}, which does not match n_tiers={n_tiers}.")
    if any(f < 0 for f in fr) or not math.isclose(sum(fr), 1.0, abs_tol=1e-6):
        raise ValueError(f"tier_fractions must be non-negative and sum to 1: {fr}")
    return fr


def _tier_counts(n: int, fractions: Sequence[float]) -> List[int]:
    """
    Split n blocks into integer tier sizes according to `fractions`
    (largest-remainder method). Every tier with a fraction > 0 receives at
    least one block when n is large enough, so that even in small dicts the
    lowest block is "prune" and the highest is "fp16".
    """
    raw = [f * n for f in fractions]
    counts = [int(math.floor(r)) for r in raw]
    remainder = n - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    nonzero = [i for i, f in enumerate(fractions) if f > 0]
    if n >= len(nonzero):
        for i in nonzero:
            if counts[i] == 0:
                donor = max(range(len(counts)), key=lambda j: counts[j])
                counts[donor] -= 1
                counts[i] += 1
    return counts


def _assign_ranked(scores: Dict[str, float], labels: Sequence[str], fractions: Sequence[float]) -> Dict[str, str]:
    """Rank a single pool by score (ties broken by key order) and split it into tiers."""
    items = list(scores.items())
    order = sorted(range(len(items)), key=lambda i: (items[i][1], i))
    counts = _tier_counts(len(items), fractions)
    out: Dict[str, str] = {}
    pos = 0
    for label, c in zip(labels, counts):
        for i in order[pos : pos + c]:
            out[items[i][0]] = label
        pos += c
    return out


def allocate_compression_tiers(
    importance_scores: Dict[str, float],
    n_tiers: int = 3,
    *,
    tier_fractions: Optional[Sequence[float]] = None,
    group_by_kind: bool = True,
    mlp_min_tier: Optional[str] = "int4",
    exclude_heads: Optional[Iterable[str]] = None,
    mlp_tier_fractions: Optional[Sequence[float]] = None,
) -> Dict[str, str]:
    """
    Assign each structural block a compression tier according to the
    within-kind percentile position of its score.

    Method:
        Blocks are sorted by ascending score within their kind (attn / mlp);
        the first tier_fractions[0] share gets the lowest tier ("prune"), the
        next share the next tier, ..., and the highest-scoring blocks get
        "fp16". Being rank based, the MLP/head scale gap does not affect the
        result (within-kind normalization comes for free). See the module
        docstring for the rationale.

    Args:
        importance_scores: xai_engine output, {"layer_{i}.attn.head_{h}": s,
            "layer_{i}.mlp": s, ...}. Unrecognized keys raise.
        n_tiers: 3 -> prune/int4/fp16 (default);
            4 -> prune/int4/int8/fp16 (only with empirical justification).
        tier_fractions: Tier fractions from lowest to highest, summing to 1.
            None = DEFAULT_TIER_FRACTIONS[n_tiers]. E.g. (0.4, 0.3, 0.3) for
            a 40% ablation.
        group_by_kind: True = heads and MLPs are thresholded in separate pools
            (recommended). False = a single pool (only meaningful if the scales
            have been made comparable).
        mlp_min_tier: Lowest tier an MLP block may receive. "int4" (default) =
            MLPs are never pruned; "prune" = whole-MLP pruning allowed;
            None = no constraint (same as "prune").
        exclude_heads: Block names that are left out of the ranking and receive
            NO tier (default None = none). Used by iterative XAI-JQP: heads
            pruned in earlier rounds are passed here, the fractions are applied
            to the REMAINING pool, and excluded blocks are absent from the
            returned dict. A name missing from importance_scores raises
            (silently skipping it would produce a wrong budget). None/empty
            reproduces the non-iterative behaviour exactly.
        mlp_tier_fractions: (4-tier setting) SEPARATE fractions for the MLP pool; None (default) = MLPs also use
            tier_fractions (unchanged behaviour). If the length equals n_tiers the fractions are used as given; if it
            equals the number of tiers at or above mlp_min_tier (e.g. (int4, int8, fp16) = (0.4, 0.3, 0.3) for
            n_tiers=4, mlp_min_tier="int4"), the lower tiers get 0. Only meaningful with group_by_kind=True.

    Returns:
        {block_name: label}; key order matches importance_scores (excluding
        exclude_heads). Labels are a subset of TIER_LABELS[n_tiers].
    """
    if not importance_scores:
        raise ValueError("importance_scores is empty.")
    fractions = _resolve_fractions(n_tiers, tier_fractions)
    labels = TIER_LABELS[n_tiers]
    if mlp_min_tier is not None and mlp_min_tier not in labels:
        raise ValueError(f"mlp_min_tier={mlp_min_tier!r} is invalid; must be one of {labels}.")
    for v in importance_scores.values():
        if not math.isfinite(float(v)):
            raise ValueError("importance_scores contains NaN/inf; clean it first.")

    excluded = set(exclude_heads or ())
    if excluded:
        unknown = sorted(excluded - set(importance_scores))
        if unknown:
            raise ValueError(f"exclude_heads contains block(s) without a score: {unknown[:5]}{'...' if len(unknown) > 5 else ''}")
        importance_scores = {k: v for k, v in importance_scores.items() if k not in excluded}
        if not importance_scores:
            raise ValueError("exclude_heads excluded every block; no blocks left to assign tiers to.")

    if group_by_kind:
        pools = split_by_kind(importance_scores)
    else:
        for name in importance_scores:
            parse_block_key(name)  # key validation
        pools = {"all": {k: float(v) for k, v in importance_scores.items()}}

    mlp_fractions: Optional[Tuple[float, ...]] = None
    if mlp_tier_fractions is not None:
        if not group_by_kind:
            raise ValueError("mlp_tier_fractions can only be used with group_by_kind=True.")
        mf = [float(f) for f in mlp_tier_fractions]
        n_upper = n_tiers - (labels.index(mlp_min_tier) if mlp_min_tier is not None else 0)
        if len(mf) == n_upper and n_upper != n_tiers:
            mf = [0.0] * (n_tiers - n_upper) + mf  # tiers below mlp_min_tier get no share
        mlp_fractions = _resolve_fractions(n_tiers, mf)

    assigned: Dict[str, str] = {}
    for kind, pool in pools.items():
        use = mlp_fractions if (kind == "mlp" and mlp_fractions is not None) else fractions
        assigned.update(_assign_ranked(pool, labels, use))

    # MLP floor: raise any MLP whose tier index is below mlp_min_tier
    if mlp_min_tier is not None:
        min_idx = labels.index(mlp_min_tier)
        for name in assigned:
            if parse_block_key(name).kind == "mlp" and labels.index(assigned[name]) < min_idx:
                assigned[name] = labels[min_idx]

    return {name: assigned[name] for name in importance_scores}


def tier_thresholds(importance_scores: Dict[str, float], tiers: Dict[str, str]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Return the per-kind score range of each assigned tier:
    {"attn": {"prune": (min, max), "int4": (min, max), ...}, "mlp": {...}}.
    Used to compare the percentile thresholds with the cluster boundaries from
    `analyze_natural_clusters`.
    """
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name, label in tiers.items():
        kind = parse_block_key(name).kind
        v = float(importance_scores[name])
        lo, hi = out.setdefault(kind, {}).get(label, (math.inf, -math.inf))
        out[kind][label] = (min(lo, v), max(hi, v))
    return out


def format_tier_report(importance_scores: Dict[str, float], tiers: Dict[str, str]) -> str:
    """Return the tier distribution per kind (count, fraction, score range, bar) as a readable table."""
    if not tiers:
        return "(empty tier dict)"
    labels = _labels_for(tiers)
    ranges = tier_thresholds(importance_scores, tiers)
    lines: List[str] = []
    for kind in ("attn", "mlp"):
        if kind not in ranges:
            continue
        names = [n for n in tiers if parse_block_key(n).kind == kind]
        total = len(names)
        lines.append(f"[{kind}] {total} blocks")
        lines.append(f"  {'tier':<7}{'count':>6}{'frac':>8}  {'score min':>10}  {'score max':>10}  bar")
        for label in labels:
            cnt = sum(1 for n in names if tiers[n] == label)
            lo, hi = ranges[kind].get(label, (math.nan, math.nan))
            bar = "#" * int(round(30 * cnt / total)) if total else ""
            lines.append(f"  {label:<7}{cnt:>6}{cnt / total:>8.3f}  {lo:>10.4f}  {hi:>10.4f}  {bar}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Natural-cluster analysis (n_tiers=3 or 4? CPU only, numpy)
# --------------------------------------------------------------------------- #
def _kmeans_1d(
    x: np.ndarray, k: int, n_init: int = 10, max_iter: int = 200, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    One-dimensional Lloyd k-means with k-means++ initialization and n_init
    restarts (lowest inertia wins). Returned centers are sorted ascending and
    labels are renumbered accordingly. No scikit-learn dependency
    (milliseconds for ~1000 points).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    for _ in range(n_init):
        centers = [x[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min((x[:, None] - np.asarray(centers)[None, :]) ** 2, axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else None
            centers.append(x[rng.choice(n, p=p)])
        c = np.asarray(centers, dtype=float)
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            labels = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
            new_c = np.array([x[labels == j].mean() if np.any(labels == j) else c[j] for j in range(k)])
            if np.allclose(new_c, c):
                break
            c = new_c
        inertia = float(((x - c[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, c, labels)
    assert best is not None
    inertia, c, labels = best
    order = np.argsort(c)
    remap = np.empty_like(order)
    remap[order] = np.arange(k)
    return c[order], remap[labels], inertia


def _silhouette_1d(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score (distance = |xi - xj|); s=0 for singleton clusters."""
    k = int(labels.max()) + 1
    n = len(x)
    if k < 2 or n < 2:
        return float("nan")
    D = np.abs(x[:, None] - x[None, :])
    sizes = np.bincount(labels, minlength=k)
    # M[i, j] = sum of distances from point i to the points of cluster j
    M = np.stack([D[:, labels == j].sum(axis=1) for j in range(k)], axis=1)
    own = sizes[labels]
    a = np.where(own > 1, M[np.arange(n), labels] / np.maximum(own - 1, 1), 0.0)
    M_other = M / np.maximum(sizes, 1)[None, :]
    M_other[np.arange(n), labels] = np.inf
    b = M_other.min(axis=1)
    s = np.where(own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    return float(s.mean())


def analyze_natural_clusters(
    importance_scores: Dict[str, float],
    *,
    ks: Sequence[int] = (2, 3, 4, 5),
    log_scale: bool = True,
    n_init: int = 10,
    seed: int = 0,
) -> Dict[str, Dict]:
    """
    Test, per kind (attn / mlp separately), how many natural clusters the score
    distribution splits into; provides DATA for the n_tiers=3 vs. n_tiers=4
    decision without making it.

    Method:
        For each kind the scores (optionally on a log10 scale) are clustered
        with 1-D k-means for k in ks; for each k the mean silhouette score,
        inertia, cluster sizes, centers, and midpoints between consecutive
        centers ("natural thresholds", on the original scale) are computed.
        The k with the highest silhouette is marked as "recommended k". A
        silhouette below ~0.5 indicates weak clustering (a continuous /
        unimodal distribution); in that case the number of tiers is a design
        choice rather than a property of the data.
        Log scale rationale: head scores are log-normal-like and right-skewed;
        on the raw scale k-means puts outliers into their own clusters.
        Since silhouette naturally tends to decrease as k grows, a small
        difference between k=3 and k=4 (< ~0.02) should be read as "no clear
        separation".

    Args:
        importance_scores: xai_engine output.
        ks: Cluster counts to try.
        log_scale: True = cluster on log10(score) (a small epsilon is added if
            any score is zero).
        n_init, seed: Number of k-means restarts and the seed (deterministic).

    Returns:
        {"attn": {"n": int, "log_scale": bool, "by_k": {k: {"silhouette", "inertia",
          "sizes", "centers", "boundaries"}}, "recommended_k": int}, "mlp": {...}}
    """
    result: Dict[str, Dict] = {}
    for kind, pool in split_by_kind(importance_scores).items():
        vals = np.asarray(list(pool.values()), dtype=float)
        if log_scale:
            positive = vals[vals > 0]
            eps = (positive.min() if positive.size else 1.0) * 1e-3
            x = np.log10(vals + eps)
        else:
            x = vals
        by_k: Dict[int, Dict] = {}
        for k in ks:
            if k >= len(x):
                continue
            centers, labels, inertia = _kmeans_1d(x, k, n_init=n_init, seed=seed)
            mids = (centers[:-1] + centers[1:]) / 2
            by_k[k] = {
                "silhouette": _silhouette_1d(x, labels),
                "inertia": inertia,
                "sizes": np.bincount(labels, minlength=k).tolist(),
                "centers": (10 ** centers if log_scale else centers).tolist(),
                "boundaries": (10 ** mids if log_scale else mids).tolist(),
            }
        rec = max(by_k, key=lambda k: by_k[k]["silhouette"]) if by_k else None
        result[kind] = {"n": int(len(x)), "log_scale": log_scale, "by_k": by_k, "recommended_k": rec}
    return result


def format_cluster_report(analysis: Dict[str, Dict]) -> str:
    """Return the output of analyze_natural_clusters as a readable table."""
    lines: List[str] = []
    for kind, res in analysis.items():
        scale = "log10" if res["log_scale"] else "raw"
        lines.append(f"[{kind}] n={res['n']}, scale={scale}, recommended k={res['recommended_k']}")
        lines.append(f"  {'k':>2}  {'silhouette':>10}  {'inertia':>10}  cluster sizes / natural thresholds (original scale)")
        for k, r in res["by_k"].items():
            bounds = ", ".join(f"{b:.4f}" for b in r["boundaries"])
            lines.append(f"  {k:>2}  {r['silhouette']:>10.4f}  {r['inertia']:>10.4f}  {r['sizes']} / [{bounds}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Compression plan (tiers -> per-layer pruning/quantization decisions)
# --------------------------------------------------------------------------- #
@dataclass
class LayerPlan:
    """Operations to apply to one decoder layer."""

    layer: int
    pruned_heads: List[int] = field(default_factory=list)
    attn_quant: str = "fp16"  # "prune" (all heads pruned) | "int4" | "int8" | "fp16"
    mlp_pruned: bool = False
    mlp_quant: str = "fp16"


def build_compression_plan(tiers: Dict[str, str], *, attn_rule: str = "median") -> Dict[int, LayerPlan]:
    """
    Turn head/MLP tiers into a per-layer application plan (no model needed).

    Method:
        * pruned_heads: heads whose tier is "prune".
        * attn_quant: median of the tier indices of the unpruned heads
          (attn_rule="median"; for an even count the lower median, i.e. the
          conservative lower precision), or the lowest ("min") / highest
          ("max"). "prune" if all heads are pruned.
        * MLP: if group-level keys (layer_i.mlp.group_g) are given, the block
          tier is the median over groups; group pruning is not supported, so
          "prune" groups count as "int4" and the block is not pruned.

    Args:
        tiers: Output of allocate_compression_tiers.
        attn_rule: "median" | "min" | "max".

    Returns:
        {layer_index: LayerPlan}, in layer order.
    """
    if attn_rule not in ("median", "min", "max"):
        raise ValueError(f"attn_rule={attn_rule!r} is invalid.")
    labels = _labels_for(tiers)
    heads: Dict[int, Dict[int, str]] = {}
    mlps: Dict[int, List[str]] = {}
    for name, label in tiers.items():
        bk = parse_block_key(name)
        if bk.kind == "attn":
            heads.setdefault(bk.layer, {})[bk.index] = label
        else:
            mlps.setdefault(bk.layer, []).append(label)

    def aggregate(lbls: List[str]) -> str:
        idx = sorted(labels.index(l) for l in lbls)
        if attn_rule == "min":
            return labels[idx[0]]
        if attn_rule == "max":
            return labels[idx[-1]]
        return labels[idx[(len(idx) - 1) // 2]]

    plans: Dict[int, LayerPlan] = {}
    for layer in sorted(set(heads) | set(mlps)):
        p = LayerPlan(layer)
        if layer in heads:
            p.pruned_heads = sorted(h for h, l in heads[layer].items() if l == "prune")
            kept = [l for h, l in heads[layer].items() if l != "prune"]
            p.attn_quant = aggregate(kept) if kept else "prune"
        if layer in mlps:
            lbls = mlps[layer]
            if len(lbls) == 1:
                p.mlp_pruned = lbls[0] == "prune"
                p.mlp_quant = "prune" if p.mlp_pruned else lbls[0]
            else:
                p.mlp_quant = aggregate([l if l != "prune" else "int4" for l in lbls])
        plans[layer] = p
    return plans


def estimate_compression_budget(
    plan: Dict[int, LayerPlan],
    *,
    hidden_size: int = 4096,
    head_dim: int = 128,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    intermediate_size: int = 14336,
    bits_per_tier: Optional[Dict[str, float]] = None,
    attention_bias: bool = False,
) -> Dict[str, float]:
    """
    Estimate the parameter/bit budget of the plan over the decoder layers
    (embeddings and lm_head excluded; default dimensions are Mistral-7B). For other
    models take the dimensions from the config via `model_budget_dims(model)`;
    `attention_bias=True` (e.g. Qwen2, whose q/k/v_proj have biases) also counts the
    bias parameters, including the q-bias slice of each pruned head. The default False
    matches models without attention biases.

    Method:
        Parameter counts of q/k/v/o_proj and gate/up/down_proj are computed
        per layer; q rows + o columns of pruned heads and pruned MLP blocks
        cost 0 bits, everything else is multiplied by the tier's nominal bits.

    Returns:
        {"params_total", "params_pruned", "pruned_ratio", "bits_fp16", "bits_compressed",
         "size_ratio" (compressed/fp16), "avg_bits_per_param"}
    """
    bits = dict(BITS_PER_TIER, **(bits_per_tier or {}))
    q = o = hidden_size * num_attention_heads * head_dim
    kv = 2 * hidden_size * num_key_value_heads * head_dim
    per_head = 2 * hidden_size * head_dim  # q rows + o columns
    mlp = 3 * hidden_size * intermediate_size
    attn_bias = (num_attention_heads + 2 * num_key_value_heads) * head_dim if attention_bias else 0
    if attention_bias:
        per_head += head_dim  # q-bias slice of the pruned head

    params_total = params_pruned = bits_total = 0.0
    for p in plan.values():
        attn_params = q + o + kv + attn_bias
        pruned = len(p.pruned_heads) * per_head
        params_total += attn_params + mlp
        params_pruned += pruned
        if p.attn_quant != "prune":
            bits_total += (attn_params - pruned) * bits[p.attn_quant]
        if p.mlp_pruned or p.mlp_quant == "prune":
            params_pruned += mlp
        else:
            bits_total += mlp * bits[p.mlp_quant]
    bits_fp16 = params_total * 16.0
    return {
        "params_total": params_total,
        "params_pruned": params_pruned,
        "pruned_ratio": params_pruned / params_total if params_total else 0.0,
        "bits_fp16": bits_fp16,
        "bits_compressed": bits_total,
        "size_ratio": bits_total / bits_fp16 if bits_fp16 else 0.0,
        "avg_bits_per_param": bits_total / params_total if params_total else 0.0,
    }


# --------------------------------------------------------------------------- #
# Application to the model: structural pruning (pure torch) + quantization (bitsandbytes)
# --------------------------------------------------------------------------- #
def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    from xai_engine import _get_decoder_layers  # same discovery logic (model.model.layers / model.layers)

    return _get_decoder_layers(model)


def _head_dim(model: nn.Module) -> int:
    cfg = model.config
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def model_budget_dims(model: nn.Module) -> Dict[str, Any]:
    """
    Dimension arguments for estimate_compression_budget, read from the model config and the first
    layer's q_proj (model-agnostic; for Mistral-7B identical to the defaults, attention_bias=False).
    Usage: estimate_compression_budget(plan, **model_budget_dims(model)).
    """
    cfg = model.config
    q_proj = _decoder_layers(model)[0].self_attn.q_proj
    return {
        "hidden_size": cfg.hidden_size,
        "head_dim": _head_dim(model),
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads,
        "intermediate_size": cfg.intermediate_size,
        "attention_bias": getattr(q_proj, "bias", None) is not None,
    }


@torch.no_grad()
def head_magnitude_scores(model: nn.Module) -> Dict[str, float]:
    """
    Magnitude-based head score: the classical baseline control against xAI
    selection. For each head, the L2 norm (computed in fp32) of that head's
    COLUMN block of `self_attn.o_proj.weight`. Only o_proj is considered: the
    q_proj rows also belong to the head, but the o_proj columns determine the
    pruning effect; k/v_proj are shared under GQA and excluded. The score scale
    is not comparable with the xAI score; it is used for RANKING only (it can be
    passed directly to the percentile-based allocate_compression_tiers; it has
    no MLP keys, so only a head pool is built).

    Returns:
        {"layer_{i}.attn.head_{h}": ||W_o[:, h*d:(h+1)*d]||_2}, in structural order.
    """
    d = _head_dim(model)
    n_heads = model.config.num_attention_heads
    out: Dict[str, float] = {}
    for i, layer in enumerate(_decoder_layers(model)):
        w = layer.self_attn.o_proj.weight
        if w.shape[1] != n_heads * d:
            raise ValueError(f"layer_{i}: o_proj.in_features={w.shape[1]} does not match n_heads*head_dim={n_heads * d}.")
        norms = w.detach().float().reshape(w.shape[0], n_heads, d).pow(2).sum(dim=(0, 2)).sqrt()
        for h, v in enumerate(norms.tolist()):
            out[f"layer_{i}.attn.head_{h}"] = float(v)
    return out


def _o_proj_weights(model: nn.Module) -> List[torch.Tensor]:
    """self_attn.o_proj weight of every layer (with shape and not-quantized checks); for calibrated head criteria."""
    d = _head_dim(model)
    n_heads = model.config.num_attention_heads
    weights: List[torch.Tensor] = []
    for i, layer in enumerate(_decoder_layers(model)):
        o = layer.self_attn.o_proj
        w = o.weight
        if type(o).__module__.startswith("bitsandbytes") or not w.is_floating_point():
            raise ValueError(f"layer_{i}: o_proj is quantized; head criteria must be computed on an uncompressed (fp16/fp32) model.")
        if w.shape[1] != n_heads * d:
            raise ValueError(f"layer_{i}: o_proj.in_features={w.shape[1]} does not match n_heads*head_dim={n_heads * d}.")
        weights.append(w)
    return weights


def _head_score_dict(per_layer: Sequence[torch.Tensor]) -> Dict[str, float]:
    """Convert a list of [n_heads] tensors to the key schema/order of head_magnitude_scores; NaN/Inf raises."""
    out: Dict[str, float] = {}
    for i, t in enumerate(per_layer):
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"layer_{i}: NaN/Inf in head score (possibly fp16 gradient/activation overflow; retry with bf16/fp32).")
        for h, v in enumerate(t.tolist()):
            out[f"layer_{i}.attn.head_{h}"] = float(v)
    return out


def head_taylor_scores(model: nn.Module, calib_batches: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """
    First-order Taylor (gradient x weight) head score, used by the `prune_taylor`
    control. The criterion of the Michel et al. 2019 / LLM-Pruner (Ma et al. 2023)
    family, implemented without installing the LLM-Pruner package, for a
    selector comparison at the same budget.

    Method:
        Loss = next-token cross-entropy on the calibration batches (SUM over valid
        target tokens; masked with attention_mask). Gradients accumulate in
        o_proj.weight.grad across all batches (in the model dtype, e.g. fp16).
        Score of head h = sum |G * W| (element-wise absolute value over h's COLUMN
        block of o_proj; product in fp32) / number of valid target tokens.
        Only o_proj is considered, for the same reason as head_magnitude_scores;
        k/v_proj are shared under GQA and excluded. The scale is not comparable
        with the xAI score; it is used for RANKING only.

    Model state: requires_grad is enabled only on the o_proj weights (disabled on
        all other parameters -> ~1 GB extra gradient memory for a 7B model); on
        exit all requires_grad flags, o_proj .grad fields and the train/eval mode
        are restored to their pre-call state. Weights are unchanged.

    Args:
        calib_batches: {"input_ids", "attention_mask"} dicts (right-padded [B, T]);
            e.g. run_ablation_tests.load_calibration_batches for the 16 calibration passages.

    Returns:
        {"layer_{i}.attn.head_{h}": score >= 0}, in structural order (head_magnitude_scores format).
    """
    weights = _o_proj_weights(model)
    d = _head_dim(model)
    n_heads = model.config.num_attention_heads
    device = next(model.parameters()).device
    saved_flags = [(p, p.requires_grad) for p in model.parameters()]
    saved_grads = [w.grad for w in weights]
    was_training = model.training
    n_tokens = 0
    try:
        for p, _ in saved_flags:
            p.requires_grad_(False)
        for w in weights:
            w.requires_grad_(True)
            w.grad = None
        model.eval()
        with torch.enable_grad():
            for b in calib_batches:
                ids = b["input_ids"].to(device)
                mask = b["attention_mask"].to(device)
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                valid = mask[:, 1:].bool()
                loss = nn.functional.cross_entropy(logits[:, :-1][valid].float(), ids[:, 1:][valid], reduction="sum")
                loss.backward()
                n_tokens += int(valid.sum().item())
        if n_tokens == 0:
            raise ValueError("No valid target tokens in the calibration batches.")
        per_layer = [
            (w.grad.float() * w.detach().float()).abs().reshape(w.shape[0], n_heads, d).sum(dim=(0, 2)).cpu() / n_tokens
            for w in weights
        ]
    finally:
        for w, g in zip(weights, saved_grads):
            w.grad = g
        for p, flag in saved_flags:
            p.requires_grad_(flag)
        model.train(was_training)
    return _head_score_dict(per_layer)


WANDA_NORMALIZE_OPTIONS = (None, "layer_zscore", "layer_percentile")


def _normalize_per_layer(t: torch.Tensor, mode: Optional[str]) -> torch.Tensor:
    """
    WITHIN-layer normalization (Wanda diagnosis): raw |W|*||X|| head scores are dominated by
    cross-layer activation/weight scale (early layers end up lowest as a whole), so each layer's
    scores are rescaled within the layer before the global ranking.
      None               -> unchanged (raw Wanda scores)
      "layer_zscore"     -> (s - mean) / std (population std; 0 if std=0)
      "layer_percentile" -> mean-rank percentile in [0, 1] (ties get the mean rank; 0 for a single head)
    """
    if mode is None:
        return t
    t = t.double()
    if mode == "layer_zscore":
        std = t.std(unbiased=False)
        return (t - t.mean()) / std if std > 0 else torch.zeros_like(t)
    if mode == "layer_percentile":
        n = t.numel()
        if n < 2:
            return torch.zeros_like(t)
        order = torch.argsort(t, stable=True)
        ranks = torch.empty(n, dtype=torch.double)
        sorted_t = t[order]
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_t[j + 1] == sorted_t[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return ranks / (n - 1)
    raise ValueError(f"normalize={mode!r} is invalid; options: {WANDA_NORMALIZE_OPTIONS}")


@torch.no_grad()
def head_wanda_scores(model: nn.Module, calib_batches: Iterable[Dict[str, torch.Tensor]], *,
                      normalize: Optional[str] = None) -> Dict[str, float]:
    """
    Wanda-style (Sun et al. 2024) head score, used by the `prune_wanda` control:
    |W| x L2 norm of the input activation, summed per head. Addresses the
    activation-agnostic naivety of the pure magnitude control (head_magnitude_scores).

    Method:
        A single forward pass accumulates, in every layer, the per-channel L2 norm
        ||X_j||_2 of the o_proj INPUT (head context vectors, [B, T, H*d]) over valid
        tokens (forward pre-hook with attention_mask; sum of squares in float64).
        Wanda element score S_ij = |W_ij| * ||X_j||_2; score of head h = sum of S_ij
        over h's column block. Per-head summation is used instead of Wanda's original
        per-output-row comparison (the structural pruning unit is the head). k/v_proj
        are untouched. Weights, flags and the train/eval mode are unchanged.

    normalize (optional; default None = raw scores): raw scores carry the cross-layer
        scale gap, so a global lowest-N selection prunes early layers wholesale.
        "layer_zscore" / "layer_percentile" rescale each layer's head scores within
        the layer (_normalize_per_layer); only the RANKING changes, raw scores are
        not returned.

    Args / Returns: same as head_taylor_scores.
    """
    if normalize not in WANDA_NORMALIZE_OPTIONS:
        raise ValueError(f"normalize={normalize!r} is invalid; options: {WANDA_NORMALIZE_OPTIONS}")
    weights = _o_proj_weights(model)
    layers = _decoder_layers(model)
    d = _head_dim(model)
    n_heads = model.config.num_attention_heads
    device = next(model.parameters()).device
    sq = [torch.zeros(w.shape[1], dtype=torch.float64) for w in weights]
    state: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(i: int):
        def hook(_mod, inputs):
            x = inputs[0]
            m = state["mask"].to(x.device).unsqueeze(-1).float()
            sq[i] += (x.float().pow(2) * m).sum(dim=(0, 1)).double().cpu()
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_hook(i)))
    n_tokens = 0
    was_training = model.training
    model.eval()
    try:
        for b in calib_batches:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            state["mask"] = mask
            n_tokens += int(mask.sum().item())
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    if n_tokens == 0:
        raise ValueError("No valid tokens in the calibration batches.")
    per_layer = []
    for w, s in zip(weights, sq):
        col = w.detach().float().abs().sum(dim=0).cpu() * s.sqrt().float()
        per_layer.append(_normalize_per_layer(col.reshape(n_heads, d).sum(dim=1), normalize))
    return _head_score_dict(per_layer)


@torch.no_grad()
def head_attention_confidence_scores(model: nn.Module, calib_batches: Iterable[Dict[str, torch.Tensor]], *,
                                     return_details: bool = False):
    """
    Attention-based head score, used by the `prune_attnconf` control (a fourth criterion family: attention pattern instead of
    gradient/attribution/weight). Voita et al. 2019 "confidence": the head's MAXIMUM attention probability averaged over the
    calibration tokens (high = focused/"confident" head; the LOWEST-scoring heads are pruned). Independent of xai_engine
    (no gradients, a single forward pass).

    Method:
        Per-layer [B, H, T, T] attention probabilities are obtained with model(..., output_attentions=True). For query
        position t, max_k p(t, k) and the entropy -sum_k p log p are computed and averaged over VALID query tokens
        (attention_mask), excluding t = 0 (under the causal mask the first token attends only to itself: max = 1,
        entropy = 0, carrying no information). Also recorded: the mean probability assigned to the first key (BOS /
        "attention sink"), to interpret where high confidence comes from. A pruned head (zero q rows) attends uniformly:
        its score is NOT 0 but the mean of 1/(t+1) (well defined, the lowest end).

    Notes: the sdpa/flash paths do not return attention weights; in transformers 4.44.2 the sdpa classes fall back to eager
        with output_attentions=True (with a warning). Missing weights raise explicitly. Weights, flags and the train/eval
        mode are unchanged.

    Args / Returns: same as head_taylor_scores ({"layer_{i}.attn.head_{h}": confidence in (0, 1]}, in structural order).
        return_details=True -> (scores, {"entropy": {...}, "first_key_share": {...}, "n_query_tokens": int}).
    """
    layers = _decoder_layers(model)
    n_heads = model.config.num_attention_heads
    device = next(model.parameters()).device
    conf = [torch.zeros(n_heads, dtype=torch.float64) for _ in layers]
    ent = [torch.zeros(n_heads, dtype=torch.float64) for _ in layers]
    first = [torch.zeros(n_heads, dtype=torch.float64) for _ in layers]
    n_queries = 0
    was_training = model.training
    model.eval()
    try:
        for b in calib_batches:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask, use_cache=False, output_attentions=True)
            attentions = getattr(out, "attentions", None)
            if not attentions or any(a is None for a in attentions):
                raise RuntimeError("the model did not return attention weights (sdpa/flash path?); load it with attn_implementation='eager'.")
            valid = mask.bool().clone()
            valid[:, 0] = False  # first query: a single key, max = 1 (uninformative)
            n_queries += int(valid.sum().item())
            w = valid.unsqueeze(1).double()  # [B, 1, T]
            for i, a in enumerate(attentions):
                if a.shape[1] != n_heads:
                    raise ValueError(f"layer_{i}: attention weights for {a.shape[1]} heads, config has {n_heads} (physically pruned model?)")
                p = a.double()
                conf[i] += (p.max(dim=-1).values * w).sum(dim=(0, 2)).cpu()
                ent[i] += (-(p * torch.log(p.clamp_min(1e-30))).sum(dim=-1) * w).sum(dim=(0, 2)).cpu()
                first[i] += (p[..., 0] * w).sum(dim=(0, 2)).cpu()
    finally:
        model.train(was_training)
    if n_queries == 0:
        raise ValueError("No valid query tokens in the calibration batches.")
    scores = _head_score_dict([c / n_queries for c in conf])
    if not return_details:
        return scores
    return scores, {"entropy": _head_score_dict([e / n_queries for e in ent]),
                    "first_key_share": _head_score_dict([f / n_queries for f in first]), "n_query_tokens": n_queries}


@torch.no_grad()
def apply_structural_pruning(model: nn.Module, plan: Dict[int, LayerPlan], verbose: bool = True) -> Dict[str, int]:
    """
    Apply the plan's pruning decisions by weight masking (in place, no GPU needed).

    Method:
        * head h: `self_attn.q_proj.weight[h*d:(h+1)*d, :] = 0` (and the bias, if any)
          + `self_attn.o_proj.weight[:, h*d:(h+1)*d] = 0`. Zeroing the o_proj
          columns removes the head's output exactly (the same operation as the
          head ablation test in xai_engine); the q rows are zeroed additionally.
          k/v_proj are shared under GQA and left untouched.
        * MLP block: the gate_proj / up_proj / down_proj weights are zeroed.
        Weights are not physically removed (shapes unchanged); the size gain is
        reported by `estimate_compression_budget`.

    Returns:
        {"pruned_heads": count, "pruned_mlp_blocks": count}
    """
    layers = _decoder_layers(model)
    d = _head_dim(model)
    n_heads = model.config.num_attention_heads
    n_h = n_m = 0
    for idx, p in plan.items():
        if idx >= len(layers):
            raise ValueError(f"The plan contains layer_{idx}, but the model has only {len(layers)} layers.")
        layer = layers[idx]
        for h in p.pruned_heads:
            if h >= n_heads:
                raise ValueError(f"layer_{idx}: head_{h} does not exist (n_heads={n_heads}).")
            sl = slice(h * d, (h + 1) * d)
            layer.self_attn.q_proj.weight[sl, :] = 0
            if getattr(layer.self_attn.q_proj, "bias", None) is not None:
                layer.self_attn.q_proj.bias[sl] = 0
            layer.self_attn.o_proj.weight[:, sl] = 0
            n_h += 1
        if p.mlp_pruned:
            for name in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(layer.mlp, name)
                mod.weight.zero_()
                if getattr(mod, "bias", None) is not None:
                    mod.bias.zero_()
            n_m += 1
    if verbose:
        print(f"[COMPRESSOR] Structural pruning: masked {n_h} heads, {n_m} MLP blocks.")
    return {"pruned_heads": n_h, "pruned_mlp_blocks": n_m}


def _quantize_linear(linear: nn.Linear, tier: str, compute_dtype: torch.dtype = torch.float16) -> nn.Module:
    """
    Replace an nn.Linear with a bitsandbytes module: "int4" -> Linear4bit (NF4,
    double quantization), "int8" -> Linear8bitLt (LLM.int8, outlier threshold 6.0).
    The actual quantization happens when the module is moved to CUDA (.cuda()/.to).
    """
    try:
        import bitsandbytes as bnb
    except ImportError as e:  # pragma: no cover
        raise ImportError("bitsandbytes is not installed; quantization requires a GPU environment.") from e

    has_bias = linear.bias is not None
    w = linear.weight.data
    if tier == "int4":
        new = bnb.nn.Linear4bit(
            linear.in_features, linear.out_features, bias=has_bias,
            compute_dtype=compute_dtype, compress_statistics=True, quant_type="nf4",
        )
        new.weight = bnb.nn.Params4bit(w.to(compute_dtype).cpu(), requires_grad=False, quant_type="nf4")
    elif tier == "int8":
        new = bnb.nn.Linear8bitLt(
            linear.in_features, linear.out_features, bias=has_bias, has_fp16_weights=False, threshold=6.0,
        )
        new.weight = bnb.nn.Int8Params(w.to(compute_dtype).cpu(), requires_grad=False, has_fp16_weights=False)
    else:
        raise ValueError(f"Tier cannot be quantized: {tier!r}")
    if has_bias:
        new.bias = nn.Parameter(linear.bias.data.to(compute_dtype), requires_grad=False)
    return new


def apply_quantization(
    model: nn.Module,
    plan: Dict[int, LayerPlan],
    *,
    device: Optional[torch.device] = None,
    compute_dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Apply the plan's int4/int8 tiers per layer with bitsandbytes.

    Method:
        For attn_quant the layer's q/k/v/o_proj, for mlp_quant its gate/up/down_proj
        are replaced together (bitsandbytes module granularity). The "fp16" and
        "prune" tiers are skipped (pruning is done in apply_structural_pruning).
        New modules are moved to `device`; quantization happens at that point.

    Returns:
        {"int4_modules": count, "int8_modules": count}
    """
    layers = _decoder_layers(model)
    if device is None:
        device = next(model.parameters()).device
    counts = {"int4_modules": 0, "int8_modules": 0}
    for idx, p in plan.items():
        layer = layers[idx]
        groups = [
            (p.attn_quant, layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
            (p.mlp_quant, layer.mlp, ("gate_proj", "up_proj", "down_proj")),
        ]
        for tier, parent, names in groups:
            if tier not in ("int4", "int8"):
                continue
            for name in names:
                lin = getattr(parent, name)
                # bitsandbytes' Linear4bit/Linear8bitLt subclass nn.Linear, so an
                # isinstance check is not enough; the module name is checked.
                if type(lin).__module__.startswith("bitsandbytes") or not isinstance(lin, nn.Linear):
                    continue  # already quantized (or not a Linear)
                setattr(parent, name, _quantize_linear(lin, tier, compute_dtype).to(device))
                counts[f"{tier}_modules"] += 1
    if verbose:
        print(f"[COMPRESSOR] Quantization: {counts['int4_modules']} INT4, {counts['int8_modules']} INT8 modules.")
    return counts


@dataclass
class JQPResult:
    """Output of apply_jqp: the model plus the decision trace (for the end-to-end pipeline and reporting)."""

    model: nn.Module
    tiers: Dict[str, str]
    plan: Dict[int, LayerPlan]
    budget: Dict[str, float]
    pruning: Dict[str, int]
    quantization: Dict[str, int]


def apply_jqp(
    model: nn.Module,
    importance_scores: Dict[str, float],
    *,
    n_tiers: int = 3,
    tier_fractions: Optional[Sequence[float]] = None,
    mlp_min_tier: Optional[str] = "int4",
    tiers: Optional[Dict[str, str]] = None,
    quantize: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> JQPResult:
    """
    Apply Joint Quantization and Pruning (JQP) to the model based on xAI scores.

    Method:
        1. allocate_compression_tiers (or the given `tiers`) -> block tiers
        2. build_compression_plan -> per-layer pruning/quantization plan
        3. apply_structural_pruning (pure torch, any device)
        4. apply_quantization (bitsandbytes; skipped with quantize=False, for
           CPU tests and the "pruning only" ablation)

    Args:
        model: HF causal LM (Mistral/Llama architecture).
        importance_scores: Output of xai_engine.calculate_importance_scores.
        n_tiers, tier_fractions, mlp_min_tier: see allocate_compression_tiers.
        tiers: If given, tier assignment is skipped and this dict is used.
        quantize: False = pruning only.
        device: Device for the quantized modules (None = the model's device).
        verbose: Print progress.

    Returns:
        JQPResult(model, tiers, plan, budget, pruning, quantization). The model
        is modified in place and the same object is returned.
    """
    if tiers is None:
        tiers = allocate_compression_tiers(
            importance_scores, n_tiers, tier_fractions=tier_fractions, mlp_min_tier=mlp_min_tier
        )
    plan = build_compression_plan(tiers)
    cfg = model.config
    budget = estimate_compression_budget(
        plan,
        hidden_size=cfg.hidden_size,
        head_dim=_head_dim(model),
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
    )
    if verbose:
        print(
            f"[COMPRESSOR] Starting XAI-guided JQP: {len(_labels_for(tiers))} tiers, "
            f"{len(plan)} layers, estimated size ratio {budget['size_ratio']:.3f}, "
            f"pruned parameter ratio {budget['pruned_ratio']:.3f}"
        )
        print(format_tier_report(importance_scores, tiers))
    pruning = apply_structural_pruning(model, plan, verbose=verbose)
    if quantize:
        quantization = apply_quantization(model, plan, device=device, verbose=verbose)
    else:
        quantization = {"int4_modules": 0, "int8_modules": 0}
    if verbose:
        print("[COMPRESSOR] Compression complete.")
    return JQPResult(model, tiers, plan, budget, pruning, quantization)


# --------------------------------------------------------------------------- #
# D-1/D-2: module-level importance score + pruning-free mixed-precision assignment
# (separate functions that do NOT touch allocate_compression_tiers, so existing plans stay bit-identical)
# --------------------------------------------------------------------------- #
MODULE_KEY = re.compile(r"^layer_(\d+)\.(attn|mlp)$")
ATTN_LINEARS: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_LINEARS: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


def _lower_median(values: Sequence[float]) -> float:
    """Lower median (the SMALLER of the two middle elements for an even count); same index as build_compression_plan's median rule."""
    v = sorted(float(x) for x in values)
    return v[(len(v) - 1) // 2]


def round_half_up(x: float) -> int:
    """Deterministic rounding for round(k/100*N) (Python's round() uses banker's rounding: round(2.5) = 2)."""
    return int(math.floor(x + 0.5))


def module_importance_scores(importance_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Reduce block (head / MLP) scores to MODULE scores at quantization granularity:
        "layer_{i}.attn" = LOWER median of the layer's head scores (consistent with the median rule; shared by q/k/v/o_proj),
        "layer_{i}.mlp"  = MLP score (lower median over groups if group-level keys are given; shared by gate/up/down_proj).
    Scales are not comparable across kinds (MLP ~100x) -> for WITHIN-kind ranking only. Order: by layer, attn before mlp.
    """
    pools: Dict[Tuple[int, str], List[float]] = {}
    for name, v in importance_scores.items():
        if not math.isfinite(float(v)):
            raise ValueError("importance_scores contains NaN/inf; clean it first.")
        bk = parse_block_key(name)
        pools.setdefault((bk.layer, bk.kind), []).append(float(v))
    return {f"layer_{layer}.{kind}": _lower_median(vals)
            for (layer, kind), vals in sorted(pools.items(), key=lambda kv: (kv[0][0], kv[0][1]))}


def _module_pools(module_scores: Dict[str, float]) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {}
    for name in module_scores:
        m = MODULE_KEY.match(name)
        if not m:
            raise ValueError(f"Unrecognized module key: {name!r} (expected: layer_{{i}}.attn | layer_{{i}}.mlp)")
        pools.setdefault(m.group(2), []).append(name)
    return pools


def rank_modules(module_scores: Dict[str, float], names: Sequence[str], *, random_seed: Optional[int] = None) -> List[str]:
    """Rank the modules of one kind from MOST to least important (ties by dict order); with random_seed, a seeded shuffle instead."""
    names = list(names)
    if random_seed is not None:
        import random as _random

        _random.Random(random_seed).shuffle(names)
        return names
    return [names[i] for i in sorted(range(len(names)), key=lambda i: (-float(module_scores[names[i]]), i))]


def allocate_mixed_precision(module_scores: Dict[str, float], k_percent: float, *,
                             random_seed: Optional[int] = None) -> Dict[str, str]:
    """
    Pruning-free mixed precision (D-1): the top k% of modules in the WITHIN-kind ranking get "fp16", the rest "int4" (NF4).
    FP16 count per kind = round_half_up(k/100 * N_kind) (Mistral-7B: N_attn = N_mlp = 32). k=0 -> all "int4"
    (the same module set as uniform NF4), k=100 -> all "fp16". With random_seed the scores are ignored and the same
    NUMBER of modules is drawn at random with that seed (mixed_random control: same budget, same count per kind).
    The returned dict has the same key order as module_scores.
    """
    if not 0.0 <= float(k_percent) <= 100.0:
        raise ValueError(f"k_percent must be in [0, 100]: {k_percent}")
    out: Dict[str, str] = {}
    for kind, names in _module_pools(module_scores).items():
        n_fp16 = round_half_up(float(k_percent) / 100.0 * len(names))
        seed = None if random_seed is None else random_seed + (0 if kind == "attn" else 1)
        keep = set(rank_modules(module_scores, names, random_seed=seed)[:n_fp16])
        out.update({n: ("fp16" if n in keep else "int4") for n in names})
    return {n: out[n] for n in module_scores}


def mixed_precision_plan(module_tiers: Dict[str, str]) -> Dict[int, LayerPlan]:
    """{"layer_{i}.attn"|"layer_{i}.mlp": tier} -> pruning-free LayerPlan dict (input to apply_quantization / quantize_with_fallback)."""
    plans: Dict[int, LayerPlan] = {}
    for name, tier in module_tiers.items():
        m = MODULE_KEY.match(name)
        if not m:
            raise ValueError(f"Unrecognized module key: {name!r}")
        if tier not in BITS_PER_TIER or tier == "prune":
            raise ValueError(f"{name}: invalid tier {tier!r} (pruning-free: int4 | int8 | fp16)")
        p = plans.setdefault(int(m.group(1)), LayerPlan(int(m.group(1))))
        if m.group(2) == "attn":
            p.attn_quant = tier
        else:
            p.mlp_quant = tier
    return dict(sorted(plans.items()))


def module_linears(model: nn.Module) -> Dict[str, List[Tuple[str, nn.Module]]]:
    """{"layer_{i}.attn": [("layer_{i}.self_attn.q_proj", module), ...], "layer_{i}.mlp": [...]}; same keys as the module scores."""
    out: Dict[str, List[Tuple[str, nn.Module]]] = {}
    for i, layer in enumerate(_decoder_layers(model)):
        out[f"layer_{i}.attn"] = [(f"layer_{i}.self_attn.{n}", getattr(layer.self_attn, n)) for n in ATTN_LINEARS]
        out[f"layer_{i}.mlp"] = [(f"layer_{i}.mlp.{n}", getattr(layer.mlp, n)) for n in MLP_LINEARS]
    return out


@torch.no_grad()
def module_magnitude_scores(model: nn.Module) -> Dict[str, float]:
    """
    Magnitude control (mixed_magnitude): joint Frobenius norm of the module's Linear weights
    sqrt(sum ||W||_F^2) (in fp32; attn = q/k/v/o_proj, mlp = gate/up/down_proj). For WITHIN-kind ranking only.
    Must be called on an uncompressed (fp16/fp32) model.
    """
    out: Dict[str, float] = {}
    for key, linears in module_linears(model).items():
        total = 0.0
        for name, lin in linears:
            if _is_quantized_linear(lin):
                raise ValueError(f"{name} is quantized; the magnitude score must be computed on an uncompressed model.")
            total += float(lin.weight.detach().float().pow(2).sum())
        out[key] = math.sqrt(total)
    return out


@torch.no_grad()
def mlp_wanda_scores(model: nn.Module, calib_batches: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """
    Wanda-style module score for the MLP block (the MLP half of mixed_wanda_ln; the saved wanda_ln file has head scores only):
    sum_ij |W_down_ij| * ||X_j||_2, with the channel norm of the down_proj INPUT over valid tokens (the MLP counterpart of
    head_wanda_scores' use of o_proj: the block's output projection). With a single MLP per layer, within-layer normalization
    is UNDEFINED; the raw score is ranked within kind. Weights, flags and the train/eval mode are unchanged.
    Returns: {"layer_{i}.mlp": score}.
    """
    layers = _decoder_layers(model)
    device = next(model.parameters()).device
    sq = [torch.zeros(layer.mlp.down_proj.in_features, dtype=torch.float64) for layer in layers]
    state: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(i: int):
        def hook(_mod, inputs):
            x = inputs[0]
            m = state["mask"].to(x.device).unsqueeze(-1).float()
            sq[i] += (x.float().pow(2) * m).sum(dim=(0, 1)).double().cpu()
        return hook

    for i, layer in enumerate(layers):
        if _is_quantized_linear(layer.mlp.down_proj):
            raise ValueError(f"layer_{i}: down_proj is quantized; the Wanda score must be computed on an uncompressed model.")
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(i)))
    n_tokens = 0
    was_training = model.training
    model.eval()
    try:
        for b in calib_batches:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            state["mask"] = mask
            n_tokens += int(mask.sum().item())
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    if n_tokens == 0:
        raise ValueError("No valid tokens in the calibration batches.")
    out: Dict[str, float] = {}
    for i, (layer, s) in enumerate(zip(layers, sq)):
        col = layer.mlp.down_proj.weight.detach().float().abs().sum(dim=0).cpu() * s.sqrt().float()
        v = float(col.sum())
        if not math.isfinite(v):
            raise ValueError(f"layer_{i}: NaN/Inf in MLP Wanda score (possibly fp16 activation overflow).")
        out[f"layer_{i}.mlp"] = v
    return out


# --------------------------------------------------------------------------- #
# Physical head pruning
# --------------------------------------------------------------------------- #
PRUNED_HEADS_CONFIG_KEY = "xai_jqp_pruned_heads"  # config.json: {"<layer>": [original head indices]}
NUM_HEADS_CONFIG_KEY = "xai_jqp_num_heads_per_layer"  # config.json: {"<layer>": number of remaining heads}


def _is_quantized_linear(mod: nn.Module) -> bool:
    w = getattr(mod, "weight", None)
    return type(mod).__module__.startswith("bitsandbytes") or (w is not None and not w.is_floating_point())


def _head_dim_from_config(cfg) -> int:
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


class PrunedHeadAttention(nn.Module):
    """
    THIN WRAPPER that allows a different number of heads per layer (Mistral/Llama attention).

    Why it is needed: transformers 4.44.2 MistralAttention (and the 5.x eager/sdpa path) maps k/v heads
    to q heads in order via `repeat_kv(n_rep = num_heads // num_key_value_heads)`. Under GQA
    (32 q / 8 kv), removing arbitrary q heads leaves a count not divisible by the kv count and breaks the
    h -> h//n_rep mapping; i.e. the stock module does NOT allow a per-layer num_heads. This class wraps
    the original attention module (the q/k/v/o_proj submodules become children of this module under the
    SAME names -> state_dict keys are unchanged, only the q_proj/o_proj shapes shrink), stores the original
    kv head index (`kv_index`) of every kept q head, and computes attention itself on the eager path.
    k_proj/v_proj are untouched (shared under GQA). The original module is kept as `_inner` (not
    registered); rotary embeddings (inside attention in 4.44.2) are taken from it, while 5.x passes the
    `position_embeddings` input. Return format depends on the version: 4.4x (singular `past_key_value`
    parameter) -> (output, weights, cache); 5.x -> (output, weights). No dropout (inference only).
    Attention mask: a 4-D additive (float) or bool mask from the model is used; if it is None and
    q_len > 1, a causal mask is built (sdpa's is_causal shortcut).

    Model-agnostic: the number of heads/kv heads, head_dim and bias are read from the wrapped module/config
    (Qwen2: 28 q / 4 kv, with q/k/v_proj biases; the bias is sliced together with q_proj, see _slice_linear).
    The rotary function is taken from the wrapped attention class's OWN module (`apply_rotary_pos_emb`; for
    Mistral the same object as before). In 4.44.2 the Qwen2 rotary is old-style (`rotary_emb(x, seq_len=...)` +
    `apply_rotary_pos_emb(q, k, cos, sin, position_ids)`): detected from the signature (`_rotary_seq_len`); the
    Mistral path is unchanged.
    """

    def __init__(self, inner: nn.Module, kept_heads: Sequence[int], orig_num_heads: int, attn_kernel: str = "eager"):
        super().__init__()
        import inspect

        self.attn_kernel = resolve_attn_kernel(attn_kernel, inner)  # "eager" (default) | "sdpa"

        object.__setattr__(self, "_inner", inner)  # not registered, so parameters are not counted twice
        self.config = inner.config
        self.layer_idx = inner.layer_idx
        self.head_dim = int(getattr(inner, "head_dim", None) or _head_dim_from_config(inner.config))
        self.num_key_value_heads = int(getattr(inner, "num_key_value_heads", None) or inner.config.num_key_value_heads)
        self.hidden_size = inner.config.hidden_size
        self.is_causal = True
        self.orig_num_heads = int(orig_num_heads)
        self.kept_heads: List[int] = [int(h) for h in kept_heads]
        self.num_heads = len(self.kept_heads)
        self.num_key_value_groups = None  # meaningless here (the mapping uses kv_index)
        # Buffer: moves with the module on .to(); built on the device of the wrapped module's weights so that
        # it does not stay on CPU when the wrapper is created while the model is already on GPU (pruning AFTER loading)
        self.register_buffer("kv_index", self._build_kv_index(inner.q_proj.weight.device), persistent=False)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = inner.q_proj, inner.k_proj, inner.v_proj, inner.o_proj
        if hasattr(inner, "rotary_emb"):  # 4.44.2: rotary lives inside attention (non-persistent buffer; not in state_dict)
            self.rotary_emb = inner.rotary_emb
        self._legacy = "past_key_value" in inspect.signature(inner.forward).parameters  # 4.4x: 3-tuple return
        # Rotary from the wrapped class's own module (Mistral -> modeling_mistral, Qwen2 -> modeling_qwen2)
        self._apply_rotary = getattr(sys.modules.get(type(inner).__module__), "apply_rotary_pos_emb", None)
        self._rotary_seq_len = (hasattr(inner, "rotary_emb")
                                and "seq_len" in inspect.signature(inner.rotary_emb.forward).parameters)  # 4.44.2 Qwen2

    def _build_kv_index(self, device) -> torch.Tensor:
        """Original kv head index (h // n_rep) for every kept q head; built from kept_heads (cannot be copied from meta)."""
        n_rep = self.orig_num_heads // self.num_key_value_heads
        return torch.tensor([h // n_rep for h in self.kept_heads], dtype=torch.long, device=device)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, past_key_value=None, past_key_values=None,
                output_attentions: bool = False, use_cache: bool = False, cache_position: Optional[torch.Tensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, **kwargs):
        apply_rotary_pos_emb = self._apply_rotary
        if apply_rotary_pos_emb is None:
            from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

        cache = past_key_value if past_key_value is not None else past_key_values
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif self._rotary_seq_len:  # 4.44.2 Qwen2: cos/sin cache up to kv length, position selection inside apply
            kv_seq_len = q_len + (cache.get_usable_length(q_len, self.layer_idx) if cache is not None else 0)
            cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        else:
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if cache is not None:
            if self._legacy:
                k, v = cache.update(k, v, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})
            else:
                k, v = cache.update(k, v, self.layer_idx)
        if self.kv_index.device != k.device:  # safety net (device_map / manually moved submodule): rebuilt once
            self.kv_index = self._build_kv_index(k.device)
        k = k.index_select(1, self.kv_index)  # [B, n_kept, kv_len, d]: each kept q head's own kv head
        v = v.index_select(1, self.kv_index)
        if self.attn_kernel == "sdpa" and not output_attentions:
            # Opt-in: the SAME call as the kernel used by the MASKED model (transformers 4.44.2 *SdpaAttention.forward):
            # attn_mask if a mask is given, otherwise is_causal when q_len > 1. On Qwen2.5 the sdpa-eager rounding gap grows with
            # the large residual-stream activations (fp16/bf16), so a masked-vs-physical comparison is only meaningful with the same kernel.
            m = attention_mask
            if m is not None and m.dim() == 4:
                m = m[:, :, :, :k.shape[-2]]
            if m is not None and q.device.type == "cuda":
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=0.0, is_causal=m is None and q_len > 1)
            out = self.o_proj(out.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.head_dim))
            return (out, None, cache) if self._legacy else (out, None)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        kv_len = k.shape[-2]
        if attention_mask is not None:
            m = attention_mask[:, :, :, :kv_len] if attention_mask.dim() == 4 else attention_mask
            scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min) if m.dtype == torch.bool else scores + m
        elif q_len > 1:
            causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=scores.device).tril(kv_len - q_len)
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        attn = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        if not output_attentions:
            attn = None
        return (out, attn, cache) if self._legacy else (out, attn)


ATTN_KERNELS = ("eager", "sdpa", "auto")


def resolve_attn_kernel(attn_kernel: str, inner: nn.Module) -> str:
    """
    Attention kernel of PrunedHeadAttention. "eager" (default) = matmul + fp32 softmax;
    "sdpa" = torch SDPA (the same kernel as a masked model loaded with sdpa); "auto" = "sdpa" if the wrapped module uses sdpa,
    otherwise "eager" (transformers 4.44.2: class name *SdpaAttention; 5.x: config._attn_implementation).
    """
    if attn_kernel not in ATTN_KERNELS:
        raise ValueError(f"attn_kernel={attn_kernel!r} is invalid; options: {ATTN_KERNELS}")
    if attn_kernel != "auto":
        return attn_kernel
    impl = getattr(getattr(inner, "config", None), "_attn_implementation", None)
    return "sdpa" if "Sdpa" in type(inner).__name__ or impl == "sdpa" else "eager"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _slice_linear(lin: nn.Linear, *, rows: Optional[torch.Tensor] = None, cols: Optional[torch.Tensor] = None) -> nn.Linear:
    w = lin.weight.detach()
    if rows is not None:
        w = w.index_select(0, rows.to(w.device))
    if cols is not None:
        w = w.index_select(1, cols.to(w.device))
    new = nn.Linear(w.shape[1], w.shape[0], bias=lin.bias is not None, device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new.weight.copy_(w)
        if lin.bias is not None:
            b = lin.bias.detach()
            new.bias.copy_(b.index_select(0, rows.to(b.device)) if rows is not None else b)
    new.weight.requires_grad_(lin.weight.requires_grad)
    return new


@torch.no_grad()
def physically_prune_heads(model: nn.Module, heads: Iterable, verbose: bool = True, attn_kernel: Optional[str] = None) -> Dict[str, Any]:
    """
    PHYSICALLY remove heads: the corresponding ROWS of q_proj and COLUMNS of o_proj are cut
    (matrices shrink), k/v_proj are untouched (shared under GQA), the layer's attention module is
    wrapped in PrunedHeadAttention and the per-layer head count is updated;
    `xai_jqp_pruned_heads` / `xai_jqp_num_heads_per_layer` are written to the config (persisted by
    save_pretrained, restored by load_physically_pruned). Numerically equivalent to masking
    (apply_structural_pruning) in terms of perplexity; the size and speed gains come from here.

    Must be called BEFORE QUANTIZATION: a module converted to 4-bit (bitsandbytes / integer weights)
    cannot be sliced -> ValueError. Successive calls on the same layer are cumulative (head indices are
    always given in the ORIGINAL numbering). Note: config.num_attention_heads is unchanged; functions
    that compare the o_proj shape against the config (xai_engine, head_*_scores) do not work on a
    physically pruned model.

    Args:
        model: HF causal LM (Mistral/Llama).
        heads: (layer, head) pairs or "layer_{i}.attn.head_{h}" names.
        attn_kernel: None (default) = new wrappers use "eager" and existing wrappers are left as is;
            with "eager" | "sdpa" | "auto" the kernel of the layers touched by this call is set (resolve_attn_kernel).
            Use "auto"/"sdpa" for numerical comparison with a masked model loaded with sdpa: on Qwen2.5 the
            sdpa-eager rounding gap grows under fp16/bf16.
    Returns:
        {"pruned_heads", "params_before", "params_after", "params_removed", "params_removed_expected",
         "num_heads_per_layer": {layer: remaining}}
    """
    layers = _decoder_layers(model)
    cfg = model.config
    n_heads = cfg.num_attention_heads
    d = _head_dim(model)
    by_layer: Dict[int, set] = {}
    for h in heads:
        if isinstance(h, str):
            bk = parse_block_key(h)
            if bk.kind != "attn":
                raise ValueError(f"not a head: {h}")
            layer, idx = bk.layer, bk.index
        else:
            layer, idx = int(h[0]), int(h[1])
        if not 0 <= layer < len(layers) or not 0 <= idx < n_heads:
            raise ValueError(f"invalid head: layer_{layer}.head_{idx}")
        by_layer.setdefault(layer, set()).add(idx)
    params_before = count_parameters(model)
    pruned_cfg = dict(getattr(cfg, PRUNED_HEADS_CONFIG_KEY, None) or {})
    has_bias = getattr(layers[0].self_attn.q_proj, "bias", None) is not None
    n_pruned = 0
    for layer_idx, idxs in sorted(by_layer.items()):
        layer = layers[layer_idx]
        attn = layer.self_attn
        if _is_quantized_linear(attn.q_proj) or _is_quantized_linear(attn.o_proj):
            raise ValueError(f"layer_{layer_idx}: q_proj/o_proj is quantized; physical pruning must be done BEFORE quantization.")
        if isinstance(attn, PrunedHeadAttention):
            wrapper, inner, current = attn, attn._inner, list(attn.kept_heads)
        else:
            wrapper, inner, current = None, attn, list(range(n_heads))
        already = set(range(n_heads)) - set(current)
        new_prunes = idxs - already
        if not new_prunes:
            continue
        kept = [h for h in current if h not in new_prunes]
        if not kept:
            raise ValueError(f"layer_{layer_idx}: cannot remove all heads (at least 1 head must remain).")
        pos = [current.index(h) for h in kept]  # positions in the current (possibly already sliced) matrix
        rows = torch.tensor([p * d + j for p in pos for j in range(d)], dtype=torch.long)
        inner.q_proj = _slice_linear(inner.q_proj, rows=rows)
        inner.o_proj = _slice_linear(inner.o_proj, cols=rows)
        inner.num_heads = len(kept)
        if wrapper is None:
            wrapper = PrunedHeadAttention(inner, kept, n_heads, attn_kernel=attn_kernel or "eager")
            layer.self_attn = wrapper
        else:
            if attn_kernel is not None:
                wrapper.attn_kernel = resolve_attn_kernel(attn_kernel, inner)
            wrapper.q_proj, wrapper.o_proj = inner.q_proj, inner.o_proj
            wrapper.kept_heads, wrapper.num_heads = kept, len(kept)
            wrapper.kv_index = wrapper._build_kv_index(inner.q_proj.weight.device)
        n_pruned += len(new_prunes)
        pruned_cfg[str(layer_idx)] = sorted(already | new_prunes)
    setattr(cfg, PRUNED_HEADS_CONFIG_KEY, pruned_cfg)
    setattr(cfg, NUM_HEADS_CONFIG_KEY, {k: n_heads - len(v) for k, v in sorted(pruned_cfg.items(), key=lambda kv: int(kv[0]))})
    params_after = count_parameters(model)
    result = {"pruned_heads": n_pruned, "params_before": params_before, "params_after": params_after,
              "params_removed": params_before - params_after,
              "params_removed_expected": n_pruned * (2 * cfg.hidden_size * d + (d if has_bias else 0)),
              "num_heads_per_layer": {int(k): v for k, v in getattr(cfg, NUM_HEADS_CONFIG_KEY).items()}}
    if verbose:
        print(f"[COMPRESSOR] Physical head pruning: {n_pruned} heads, {result['params_removed']:,} parameters removed "
              f"({params_before:,} -> {params_after:,}).")
    return result


def pruned_heads_from_config(cfg) -> List[Tuple[int, int]]:
    """xai_jqp_pruned_heads from config.json -> [(layer, head), ...] (empty if absent)."""
    raw = getattr(cfg, PRUNED_HEADS_CONFIG_KEY, None) or {}
    return sorted((int(l), int(h)) for l, hs in raw.items() for h in hs)


def load_physically_pruned(path: str, *, dtype: torch.dtype = torch.float16, device=None, attn_kernel: Optional[str] = None,
                           **kwargs) -> nn.Module:
    """
    Load a physically pruned model saved with save_pretrained: the skeleton is built from the config
    (on CPU, needs RAM for the full model), the same heads are removed (so shapes match the checkpoint),
    and weights are loaded strictly from the safetensors files. Without pruning info in the config this
    is a plain from_pretrained (`kwargs` are forwarded to it). If `device` is given the model is moved there.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(path)
    heads = pruned_heads_from_config(cfg)
    if not heads:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        return model.to(device) if device is not None else model
    import glob

    from safetensors.torch import load_file

    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = AutoModelForCausalLM.from_config(cfg)
    finally:
        torch.set_default_dtype(prev)
    physically_prune_heads(model, heads, verbose=False, attn_kernel=attn_kernel)
    state: Dict[str, torch.Tensor] = {}
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"{path}: no safetensors file (save with save_pretrained(safe_serialization=True))")
    for f in files:
        state.update(load_file(f))
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    missing = [k for k in missing if not (tied and k.endswith("lm_head.weight"))]
    if missing or unexpected:
        raise RuntimeError(f"{path}: state_dict mismatch; missing={missing[:3]} unexpected={unexpected[:3]}")
    if tied and hasattr(model, "tie_weights"):
        model.tie_weights()
    model.eval()
    return model.to(device) if device is not None else model


# --------------------------------------------------------------------------- #
# CLI: tier assignment + natural-cluster analysis on real scores (CPU only)
# --------------------------------------------------------------------------- #
def load_scores(path: str) -> Dict[str, float]:
    """Read an importance_scores JSON (a 'scores' dict or a flat dict)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = data.get("scores", data)
    if not isinstance(scores, dict):
        raise ValueError(f"{path}: 'scores' dict not found")
    return {k: float(v) for k, v in scores.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default=os.path.join("results", "importance_scores_gun3.json"))
    parser.add_argument("--n-tiers", type=int, default=3, choices=sorted(TIER_LABELS))
    parser.add_argument("--fractions", type=float, nargs="+", default=None, help="tier fractions (summing to 1)")
    parser.add_argument("--mlp-min-tier", default="int4", choices=("prune", "int4", "int8", "fp16"))
    parser.add_argument("--output", default=None, help="write the analysis result as JSON")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console

    scores = load_scores(args.scores)
    print(f"Reading: {args.scores} ({len(scores)} scores)")

    tiers = allocate_compression_tiers(
        scores, args.n_tiers, tier_fractions=args.fractions, mlp_min_tier=args.mlp_min_tier
    )
    plan = build_compression_plan(tiers)
    budget = estimate_compression_budget(plan)
    print(f"\n=== Tier assignment (n_tiers={args.n_tiers}, percentile) ===")
    print(format_tier_report(scores, tiers))
    print(
        f"\n  estimated size ratio (decoder, nominal bits): {budget['size_ratio']:.3f}  "
        f"| avg bits/param: {budget['avg_bits_per_param']:.2f}  | pruned param ratio: {budget['pruned_ratio']:.3f}"
    )

    print("\n=== Natural-cluster analysis (k-means + silhouette) ===")
    analysis_log = analyze_natural_clusters(scores, log_scale=True)
    print(format_cluster_report(analysis_log))
    analysis_raw = analyze_natural_clusters(scores, log_scale=False)
    print("\n  --- comparison: raw scale ---")
    print(format_cluster_report(analysis_raw))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        tier_counts = {
            kind: {
                l: sum(1 for n, t in tiers.items() if t == l and parse_block_key(n).kind == kind)
                for l in TIER_LABELS[args.n_tiers]
            }
            for kind in ("attn", "mlp")
        }
        payload = {
            "source": args.scores,
            "n_tiers": args.n_tiers,
            "tier_fractions": list(args.fractions or DEFAULT_TIER_FRACTIONS[args.n_tiers]),
            "mlp_min_tier": args.mlp_min_tier,
            "tier_counts": tier_counts,
            "tier_score_ranges": tier_thresholds(scores, tiers),
            "budget": budget,
            "clusters_log10": analysis_log,
            "clusters_raw": analysis_raw,
            "tiers": tiers,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nAnalysis saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
