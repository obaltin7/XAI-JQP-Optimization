"""
Drift / stability metrics between importance-score vectors (CPU only, numpy).

Between two importance-score dictionaries (in xai_engine output format:
{"layer_i.attn.head_h": s, "layer_i.mlp": s}) the following are computed per
block kind (heads, MLPs) and jointly:
  * Spearman rank correlation (average rank for ties; no scipy dependency)
  * Pearson correlation
  * top-k Jaccard (set overlap of the k highest-scoring blocks)
Because pruned heads receive exactly 0 on re-scoring (tested behaviour), two
views are reported: "all heads" and "surviving heads" (excluding the exclusion list).

Used by: compute_drift.py (iterative rounds), compare_scores.py (two attribution
runs), tests/test_iterative_gun7_mini.py.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compressor import parse_block_key


def rankdata(x: Sequence[float]) -> np.ndarray:
    """Ascending ranks starting at 1; ties receive the average rank (scipy.stats.rankdata 'average')."""
    a = np.asarray(x, dtype=float)
    n = len(a)
    ranks = np.empty(n, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho = Pearson(rank(a), rank(b)); NaN for n < 2 or a constant vector."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        return float("nan")
    x = np.asarray(a, dtype=float) - float(np.mean(a))
    y = np.asarray(b, dtype=float) - float(np.mean(b))
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / denom) if denom else float("nan")


def topk_names(scores: Dict[str, float], k: int) -> List[str]:
    """Names of the k highest-scoring blocks (ties broken by dictionary order)."""
    items = list(scores.items())
    order = sorted(range(len(items)), key=lambda i: (-items[i][1], i))
    return [items[i][0] for i in order[:k]]


def topk_jaccard(s0: Dict[str, float], s1: Dict[str, float], k: int) -> float:
    """|top_k(s0) ∩ top_k(s1)| / |top_k(s0) ∪ top_k(s1)|; k is capped at the dictionary size."""
    k = min(k, len(s0), len(s1))
    if k <= 0:
        return float("nan")
    a, b = set(topk_names(s0, k)), set(topk_names(s1, k))
    return len(a & b) / len(a | b)


def split_kinds(scores: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(heads, MLPs); dictionary order is preserved."""
    heads: Dict[str, float] = {}
    mlps: Dict[str, float] = {}
    for k, v in scores.items():
        (heads if parse_block_key(k).kind == "attn" else mlps)[k] = float(v)
    return heads, mlps


def _aligned(s0: Dict[str, float], s1: Dict[str, float], keys: Iterable[str]) -> Tuple[List[float], List[float]]:
    ks = list(keys)
    return [s0[k] for k in ks], [s1[k] for k in ks]


def compare_score_dicts(
    s0: Dict[str, float],
    s1: Dict[str, float],
    *,
    exclude: Optional[Iterable[str]] = None,
    ks: Sequence[int] = (100, 200),
) -> Dict[str, object]:
    """
    Drift metrics between s0 (reference, e.g. round 0 = initial importance scores) and s1.

    Both key sets must be identical (missing/extra keys raise an error, since a silent
    intersection would produce a wrong comparison). `exclude`: head names to drop from
    the surviving view (e.g. heads pruned up to this round; their score in s1 is 0).

    Returns:
        {"n_heads", "n_mlp", "n_excluded", "excluded_zero_in_s1",
         "spearman": {"heads_all", "heads_surviving", "mlp", "all_blocks", "all_surviving"},
         "pearson": {"heads_all", "heads_surviving"},
         "jaccard": {"heads_all": {"top100": ..}, "heads_surviving": {...}, "all_surviving": {...}}}
    """
    if set(s0) != set(s1):
        missing = sorted(set(s0) ^ set(s1))
        raise ValueError(f"score key sets differ ({len(missing)} differences; first: {missing[:3]})")
    excluded = set(exclude or ())
    unknown = sorted(excluded - set(s0))
    if unknown:
        raise ValueError(f"exclude contains blocks without a score: {unknown[:3]}")
    heads0, mlp0 = split_kinds(s0)
    surv = [k for k in heads0 if k not in excluded]
    surv_all = [k for k in s0 if k not in excluded]

    def sp(keys: Iterable[str]) -> float:
        a, b = _aligned(s0, s1, keys)
        return spearman(a, b)

    def pe(keys: Iterable[str]) -> float:
        a, b = _aligned(s0, s1, keys)
        return pearson(a, b)

    def jac(keys: List[str]) -> Dict[str, float]:
        sub0 = {k: s0[k] for k in keys}
        sub1 = {k: s1[k] for k in keys}
        return {f"top{k}": topk_jaccard(sub0, sub1, k) for k in ks}

    return {
        "n_heads": len(heads0),
        "n_mlp": len(mlp0),
        "n_excluded": len(excluded),
        "excluded_zero_in_s1": sum(1 for k in excluded if s1[k] == 0.0),
        "spearman": {
            "heads_all": sp(heads0),
            "heads_surviving": sp(surv),
            "mlp": sp(mlp0),
            "all_blocks": sp(s0),
            "all_surviving": sp(surv_all),
        },
        "pearson": {"heads_all": pe(heads0), "heads_surviving": pe(surv)},
        "jaccard": {
            "heads_all": jac(list(heads0)),
            "heads_surviving": jac(surv),
            "all_surviving": jac(surv_all),
        },
    }
