"""
Calibration-domain diagnosis: where do the head sets pruned by the 20% plans calibrated on WikiText-2 and on C4 differ?

The one-shot plan calibrated on C4 reaches a WikiText-2 perplexity of 11.64 (6.32 with WikiText calibration) while downstream task
accuracy is unchanged (E-6). Hypothesis (a): heads that process WikiText-specific formatting patterns (" @-@ ", "= = heading = =")
receive low scores under C4 calibration and are pruned ("in-domain perplexity bias").

  (default, CPU only)  set difference (pruned only under C4 / only under WikiText / common), layer distribution, the ranks of these
                       heads in both score files, overlap with the prune_reverse (the 205 MOST important heads) and prune_magnitude
                       control sets of results/ablation_gun6.json, and E-7 groups (<= 6 layer blocks)
                       -> results/diagnose_calib_domain_gun15.json + figures/calib_domain_gun15.png
  --ablate (GPU, E-7)  masks each group ON ITS OWN (FP16, no quantization; fresh model load per group) and measures WikiText-2 and
                       C4 perplexity: which layer block causes the +5 ppl? Controls: the full "only C4" set and the full
                       "only WikiText" set. -> results/group_ablation_gun15.json (~2 min/group; 8 entries + fp16 ≈ 18 min;
                       peak VRAM ~15 GB)

Usage:
    python tools/diagnostics/diagnose_calib_domain.py
    python tools/diagnostics/diagnose_calib_domain.py --ablate --dry-run            # mini model
    python tools/diagnostics/diagnose_calib_domain.py --ablate
"""
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WIKI_PLAN = os.path.join("results", "iterative_gun7.json")
C4_PLAN = os.path.join("results", "calib_c4_gun9.json")
WIKI_SCORES = os.path.join("results", "importance_scores_gun3.json")
C4_SCORES = os.path.join("results", "importance_scores_c4.json")
GUN6_FILE = os.path.join("results", "ablation_gun6.json")
OUTPUT_FILE = os.path.join("results", "diagnose_calib_domain_gun15.json")
ABLATION_FILE = os.path.join("results", "group_ablation_gun15.json")
FIGURE_FILE = os.path.join("figures", "calib_domain_gun15.png")
LAYER_BLOCKS = ((0, 5), (6, 10), (11, 15), (16, 20), (21, 26), (27, 31))  # at most 6 groups
Head = Tuple[int, int]


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pruned_set(plan: Dict[str, Any], fraction: str = "0.2", config: str = "xai_single") -> List[Head]:
    rep = next(r for r in plan["fractions"][fraction]["configs"][config]["repeats"] if r.get("status") == "completed")
    return sorted((int(l), int(h)) for l, h in rep["pruned_heads"])


def head_ranks(scores: Dict[str, float]) -> Dict[Head, int]:
    """Rank 0 = LEAST important head (pruning order); only attention-head keys are ranked."""
    items = [(k, float(v)) for k, v in scores.items() if ".attn.head_" in k]
    order = sorted(range(len(items)), key=lambda i: (items[i][1], i))
    out: Dict[Head, int] = {}
    for rank, i in enumerate(order):
        layer, head = items[i][0].split(".attn.head_")
        out[(int(layer.split("_")[1]), int(head))] = rank
    return out


def layer_groups(heads: Sequence[Head], blocks: Sequence[Tuple[int, int]] = LAYER_BLOCKS) -> Dict[str, List[Head]]:
    groups = {f"L{lo}-{hi}": [h for h in heads if lo <= h[0] <= hi] for lo, hi in blocks}
    return {k: v for k, v in groups.items() if v}


def diagnose(wiki: Sequence[Head], c4: Sequence[Head], rank_w: Dict[Head, int], rank_c: Dict[Head, int],
             reference_sets: Optional[Dict[str, Sequence[Head]]] = None, n_layers: int = 32) -> Dict[str, Any]:
    sw, sc = set(wiki), set(c4)
    sets = {"only_c4": sorted(sc - sw), "only_wikitext": sorted(sw - sc), "common": sorted(sw & sc)}

    def med(vals: List[int]) -> Optional[float]:
        v = sorted(vals)
        return None if not v else float(v[len(v) // 2])

    ranks = {name: {"n": len(hs), "median_rank_wikitext": med([rank_w[h] for h in hs]), "median_rank_c4": med([rank_c[h] for h in hs]),
                    "max_rank_wikitext": max((rank_w[h] for h in hs), default=None), "max_rank_c4": max((rank_c[h] for h in hs), default=None)}
             for name, hs in sets.items()}
    per_layer = {name: [sum(1 for h in hs if h[0] == l) for l in range(n_layers)] for name, hs in sets.items()}
    overlaps = {ref: {name: len(set(hs) & set(map(tuple, ref_heads))) for name, hs in sets.items()} for ref, ref_heads in (reference_sets or {}).items()}
    return {"n_pruned": {"wikitext": len(sw), "c4": len(sc)}, "jaccard": len(sw & sc) / len(sw | sc), "sets": {k: [list(h) for h in v] for k, v in sets.items()},
            "rank_definition": "0 = least important head; pruning threshold = number of pruned heads (205): rank < 205 -> pruned under that calibration",
            "ranks": ranks, "per_layer": per_layer, "overlaps_with_gun6": overlaps,
            "groups_only_c4": {k: [list(h) for h in v] for k, v in layer_groups(sets["only_c4"]).items()}}


def make_figure(result: Dict[str, Any], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(result["per_layer"]["common"])
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.bar(x - 0.2, result["per_layer"]["only_wikitext"], 0.4, color="#0072B2", label="pruned only under WikiText-2 calibration")
    ax.bar(x + 0.2, result["per_layer"]["only_c4"], 0.4, color="#E69F00", hatch="//", edgecolor="black", linewidth=0.3, label="pruned only under C4 calibration")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Number of heads")
    ax.set_xticks(x[::2])
    ax.legend(fontsize=7, frameon=False)
    fig.text(0.01, 0.005, f"Source: {os.path.basename(WIKI_PLAN)}, {os.path.basename(C4_PLAN)} · Generated: {datetime.now():%Y-%m-%d}", fontsize=6, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_ablation(groups: Dict[str, List[Head]], args: argparse.Namespace) -> Dict[str, Any]:
    """Per group: fresh load -> mask only those heads (FP16) -> WikiText-2 + C4 perplexity (sliding windows, eval_ppl_windows)."""
    import torch

    import eval_ppl_windows as epw
    from compressor import apply_structural_pruning
    from run_ablation_tests import DryRun, prune_plan_from_heads
    from run_baselines import release_gpu
    from run_e2e_pipeline import Logger, load_model, load_wikitext_text

    log = Logger(args.log, to_file=not args.dry_run)
    dry = DryRun(42) if args.dry_run else None
    texts = {} if dry else {"wikitext2": load_wikitext_text(), "c4": epw.load_c4_eval_text()[0]}
    out: Dict[str, Any] = {"created_at": datetime.now().isoformat(timespec="seconds"), "model": "mini (dry-run)" if dry else "mistralai/Mistral-7B-Instruct-v0.3",
                           "note": "FP16, no quantization; fresh model load per group; delta = group − fp16", "groups": {}}
    for name, heads in [("fp16", [])] + list(groups.items()):
        t0 = time.time()
        tok, model, _ = dry.load_model(log) if dry else load_model(log)
        n_layers = model.config.num_hidden_layers
        if dry:  # the real plan is 32 × 32 heads; keep those that fit the mini model (exercises the code path only)
            heads = [h for h in heads if h[0] < n_layers and h[1] < model.config.num_attention_heads]
        if heads:
            apply_structural_pruning(model, prune_plan_from_heads([tuple(h) for h in heads], n_layers), verbose=False)
        ppl = {}
        for i, ds in enumerate(("wikitext2", "c4")):
            if dry:
                ids = torch.randint(1, model.config.vocab_size, (1, 120), generator=torch.Generator().manual_seed(42 + i))
                ppl[ds] = epw.window_nlls_from_ids(model, ids, "cpu", max_length=32, stride=16)["perplexity"]
            else:
                ppl[ds] = epw.window_nlls(model, tok, texts[ds], next(model.parameters()).device)["perplexity"]
        del model
        release_gpu()
        base = out["groups"].get("fp16", {}).get("ppl", ppl)
        out["groups"][name] = {"n_heads": len(heads), "heads": [list(h) for h in heads], "ppl": ppl,
                               "delta_vs_fp16": {ds: ppl[ds] - base[ds] for ds in ppl}, "seconds": time.time() - t0}
        log(f"{name}: {len(heads)} heads -> ppl WikiText-2 {ppl['wikitext2']:.4f}, C4 {ppl['c4']:.4f}", tag="E7")
        os.makedirs(os.path.dirname(args.ablation_output) or ".", exist_ok=True)
        with open(args.ablation_output, "w", encoding="utf-8") as f:  # partial save after each group
            json.dump(out, f, indent=1, ensure_ascii=False)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Calibration-domain diagnosis (WikiText-2 vs C4 plans) with optional group ablation (E-7)")
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--figure", default=FIGURE_FILE)
    p.add_argument("--ablate", action="store_true", help="(GPU) mask each group separately and measure perplexity on both domains")
    p.add_argument("--ablation-output", default=ABLATION_FILE)
    p.add_argument("--log", default="log_gun15_group_ablation.txt")
    p.add_argument("--dry-run", action="store_true", help="mini model for --ablate; output goes to dryrun_out/")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    g6 = _load(GUN6_FILE)["configs"] if os.path.exists(GUN6_FILE) else {}
    refs = {n: [tuple(h) for h in g6[n]["repeats"][0]["pruned_heads"]] for n in ("prune_reverse", "prune_magnitude") if n in g6}
    result = diagnose(pruned_set(_load(WIKI_PLAN)), pruned_set(_load(C4_PLAN)), head_ranks(_load(WIKI_SCORES)["scores"]),
                      head_ranks(_load(C4_SCORES)["scores"]), refs)
    result.update({"created_at": datetime.now().isoformat(timespec="seconds"), "sources": [WIKI_PLAN, C4_PLAN, WIKI_SCORES, C4_SCORES, GUN6_FILE]})
    if not args.ablate:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        make_figure(result, args.figure)
        s = result["sets"]
        print(f"common {len(s['common'])}, only C4 {len(s['only_c4'])}, only WikiText {len(s['only_wikitext'])}; Jaccard {result['jaccard']:.3f}")
        print("ranks:", json.dumps(result["ranks"], ensure_ascii=False))
        print("overlap with ablation control sets:", result["overlaps_with_gun6"], "| groups:", {k: len(v) for k, v in result["groups_only_c4"].items()})
        return 0
    if args.dry_run:
        args.ablation_output = os.path.join("dryrun_out", "group_ablation_gun15_dry.json")
        os.makedirs("dryrun_out", exist_ok=True)
    groups = dict(result["groups_only_c4"])
    groups["all_only_c4"] = result["sets"]["only_c4"]
    groups["all_only_wikitext"] = result["sets"]["only_wikitext"]
    run_ablation(groups, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
