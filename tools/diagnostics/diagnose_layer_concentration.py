# -*- coding: utf-8 -*-
"""
XAI-JQP: layer-concentration diagnosis (CPU only, descriptive; the criterion below was fixed before the analysis).

Question: does xAI pruning concentrate the pruned heads in a few layers compared with random pruning? On Qwen at 10%, random
pruning (single seed) reached a better WikiText perplexity than xAI; one candidate explanation is that xAI spends its budget on
a few layers. This script only measures the DISTRIBUTION and makes no causal claim.

For each head set: per-layer counts c_l and the metrics Gini(c_l) (PRIMARY), pruned fraction of the most affected layer, number
of layers with at least half of their heads pruned, and share of the 3 most affected layers. Reference: 10,000 uniform random
draws of the same size (seed 0) -> mean and 2.5%-97.5% interval. "Concentrated" = Gini of the set above the reference 97.5%.

Usage:
    python tools/diagnostics/diagnose_layer_concentration.py            # -> results/layer_concentration_gun16.json, figures/layer_concentration_gun16.png
    python tools/diagnostics/diagnose_layer_concentration.py --dry-run  # outputs go to dryrun_out/
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

Head = Tuple[int, int]

OUTPUT_FILE = os.path.join("results", "layer_concentration_gun16.json")
FIGURE_FILE = os.path.join("figures", "layer_concentration_gun16.png")
DRYRUN_DIR = "dryrun_out"
N_REFERENCE = 10000
REFERENCE_SEED = 0

# (label, plan JSON, ratio key (schema of results/iterative_gun7.json) or None (schema of results/model2_qwen_gun9.json),
#  fallback model shape (n_layers, n_heads))
SOURCES = [
    ("qwen_f010", os.path.join("results", "model2_qwen_gun9_f010.json"), None, (28, 28)),
    ("qwen_f020", os.path.join("results", "model2_qwen_gun9.json"), None, (28, 28)),
    ("mistral_f020", os.path.join("results", "iterative_gun7.json"), "0.2", (32, 32)),
]
# one-shot xAI set: prune_only in the model2 schema, xai_single in the iterative_gun7 schema (same head set regardless of quantization)
XAI_CONFIGS = ("prune_only", "xai_single", "xai_iter_fixedq")
RANDOM_CONFIG = "prune_random"


def layer_counts(heads: Sequence[Head], n_layers: int) -> List[int]:
    counts = [0] * n_layers
    for layer, _ in heads:
        counts[int(layer)] += 1
    return counts


def gini(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    xs = sorted(counts)
    n = len(xs)
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * total)


def concentration(counts: Sequence[int], n_heads: int) -> Dict[str, float]:
    total = sum(counts)
    top3 = sum(sorted(counts, reverse=True)[:3])
    return {"gini": gini(counts), "max_layer_fraction": max(counts) / n_heads if counts else 0.0,
            "n_layers_ge_half": sum(1 for c in counts if 2 * c >= n_heads), "top3_layer_share": top3 / total if total else 0.0}


def random_reference(n_layers: int, n_heads: int, n_pruned: int, n_draws: int = N_REFERENCE, seed: int = REFERENCE_SEED) -> Dict[str, Dict[str, float]]:
    rng = random.Random(seed)
    population = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    draws = [concentration(layer_counts(rng.sample(population, n_pruned), n_layers), n_heads) for _ in range(n_draws)]
    out: Dict[str, Dict[str, float]] = {}
    for key in draws[0]:
        xs = sorted(d[key] for d in draws)
        out[key] = {"mean": sum(xs) / len(xs), "p2_5": xs[int(0.025 * (len(xs) - 1))], "p97_5": xs[int(round(0.975 * (len(xs) - 1)))]}
    return out


def collect_sets(plan: Dict[str, Any], fraction: Optional[str]) -> Dict[str, List[Head]]:
    """{"<config>" or "prune_random_s<seed>": head list} from a plan JSON; incomplete configs / repeats are skipped."""
    configs = plan["fractions"][fraction]["configs"] if fraction is not None else plan["configs"]
    sets: Dict[str, List[Head]] = {}
    for name in XAI_CONFIGS + (RANDOM_CONFIG,):
        cfg = configs.get(name)
        if not cfg or cfg.get("status") != "completed":
            continue
        for rep in cfg.get("repeats", []):
            if rep.get("status") != "completed" or not rep.get("pruned_heads"):
                continue
            heads = sorted((int(l), int(h)) for l, h in rep["pruned_heads"])
            if name == RANDOM_CONFIG:
                sets[f"{name}_s{rep.get('seed')}"] = heads
            elif name not in sets:  # deterministic configs: first completed repeat
                sets[name] = heads
    return sets


def diagnose_source(plan: Dict[str, Any], fraction: Optional[str], fallback_shape: Tuple[int, int], n_draws: int = N_REFERENCE) -> Dict[str, Any]:
    info = plan.get("model_info") or {}
    n_layers, n_heads = int(info.get("n_layers", fallback_shape[0])), int(info.get("n_heads", fallback_shape[1]))
    sets = collect_sets(plan, fraction)
    sizes = sorted({len(h) for h in sets.values()})
    refs = {n: random_reference(n_layers, n_heads, n, n_draws) for n in sizes}
    entries: Dict[str, Any] = {}
    for name, heads in sets.items():
        counts = layer_counts(heads, n_layers)
        m = concentration(counts, n_heads)
        entries[name] = {"n_pruned": len(heads), "per_layer": counts, **m,
                         "gini_above_random_p97_5": m["gini"] > refs[len(heads)]["gini"]["p97_5"]}
    return {"n_layers": n_layers, "n_heads": n_heads, "sets": entries,
            "random_reference": {str(n): r for n, r in refs.items()}, "reference": {"n_draws": n_draws, "seed": REFERENCE_SEED}}


def make_figure(result: Dict[str, Any], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(result["sources"])
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 3.0 * len(labels)), squeeze=False)
    for ax, label in zip(axes[:, 0], labels):
        src = result["sources"][label]
        names = list(src["sets"])
        width = 0.8 / max(len(names), 1)
        for i, name in enumerate(names):
            e = src["sets"][name]
            xs = [l + (i - (len(names) - 1) / 2) * width for l in range(src["n_layers"])]
            ax.bar(xs, e["per_layer"], width=width, label=f"{name} (Gini {e['gini']:.2f})", alpha=0.5 if name.startswith(RANDOM_CONFIG) else 1.0)
        ax.axhline(src["n_heads"] / 2, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{label}: pruned heads per layer (dotted line = half of the layer)", fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylabel("pruned heads")
        ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Layer-concentration diagnosis: pruned heads per layer for xAI vs random pruning (CPU only)")
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--figure", default=FIGURE_FILE)
    p.add_argument("--n-draws", type=int, default=N_REFERENCE, help="number of random reference draws")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="write outputs to dryrun_out/")
    args = p.parse_args(argv)
    if args.dry_run:
        args.output = os.path.join(DRYRUN_DIR, os.path.basename(args.output))
        args.figure = os.path.join(DRYRUN_DIR, os.path.basename(args.figure))

    result: Dict[str, Any] = {"created_at": datetime.now().isoformat(timespec="seconds"),
                              "definition": "Gini(c_l) is PRIMARY; concentration = Gini > 97.5th percentile of a uniform random reference of the same size (descriptive, no causal claim)",
                              "sources": {}, "missing": []}
    for label, path, fraction, shape in SOURCES:
        if not os.path.exists(path):
            result["missing"].append(path)
            continue
        with open(path, encoding="utf-8") as f:
            plan = json.load(f)
        result["sources"][label] = {"plan": path, **diagnose_source(plan, fraction, shape, args.n_draws)}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"{'source':<14}{'set':<22}{'n':>5}{'Gini':>8}{'ref mean':>9}{'ref p97.5':>11}{'max/H':>8}{'≥half':>7}{'top3':>7}  concentrated?")
    for label, src in result["sources"].items():
        for name, e in src["sets"].items():
            ref = src["random_reference"][str(e["n_pruned"])]["gini"]
            print(f"{label:<14}{name:<22}{e['n_pruned']:>5}{e['gini']:>8.3f}{ref['mean']:>9.3f}{ref['p97_5']:>11.3f}{e['max_layer_fraction']:>8.3f}"
                  f"{e['n_layers_ge_half']:>7}{e['top3_layer_share']:>7.3f}  {'YES' if e['gini_above_random_p97_5'] else '-'}")
    if result["missing"]:
        print(f"missing sources (skipped): {result['missing']}", file=sys.stderr)
    if not args.no_figure and result["sources"]:
        make_figure(result, args.figure)
        print(f"Written: {args.output}, {args.figure}")
    else:
        print(f"Written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
