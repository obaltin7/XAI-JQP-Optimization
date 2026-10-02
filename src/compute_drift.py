"""
Explanation drift computed from the score files written by run_iterative_pruning.py (CPU only).

Applies drift_metrics.compare_score_dicts between round 0 (the initial importance scores,
results/importance_scores_gun3.json) and every score file under results/gun7_scores/:
  * Spearman: heads (all / surviving), MLPs, all blocks
  * Pearson (heads)
  * top-100 / top-200 Jaccard (heads; surviving blocks)
Files: f{fraction}_round{k}.json (xai_iter, after the k-th pruning round) and
f{fraction}_{config}_after[_seed{s}].json (one-shot configurations, after pruning / before
quantization; --rescore-after). Since pruned heads score exactly 0 on re-scoring, the
"surviving" view is the actual drift measure; the "all" view also reflects the pruning itself.

Usage:
    python src/compute_drift.py                                   # default paths
    python src/compute_drift.py --scores-dir dryrun_out/gun7_scores --round0 <dry-run score json> --output dryrun_out/drift.json
Output: results/drift_gun7.json plus a short table on the console.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compressor import load_scores
from drift_metrics import compare_score_dicts

ROUND0_FILE = os.path.join("results", "importance_scores_gun3.json")
SCORES_DIR = os.path.join("results", "gun7_scores")
OUTPUT_FILE = os.path.join("results", "drift_gun7.json")
FILE_RE = re.compile(r"^f(?P<fraction>[0-9.]+)_(?:round(?P<round>\d+)|(?P<config>[a-z_]+)_after(?:_seed(?P<seed>\d+))?)\.json$")


def collect_score_files(scores_dir: str) -> List[Dict[str, Any]]:
    """Recognised score files in the directory with (fraction, config, round, seed), sorted; unrecognised names are skipped."""
    out: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(scores_dir, "*.json"))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        out.append({"file": path, "fraction": m.group("fraction"),
                    "config": "xai_iter" if m.group("round") else m.group("config"),
                    "round": int(m.group("round")) if m.group("round") else None,
                    "seed": int(m.group("seed")) if m.group("seed") else None})
    out.sort(key=lambda e: (float(e["fraction"]), e["config"], e["round"] or 0, e["seed"] or 0))
    return out


def compute_drift(round0_path: str, scores_dir: str, ks=(100, 200)) -> Dict[str, Any]:
    s0 = load_scores(round0_path)
    entries: List[Dict[str, Any]] = []
    for e in collect_score_files(scores_dir):
        with open(e["file"], "r", encoding="utf-8") as f:
            data = json.load(f)
        s1 = {k: float(v) for k, v in data["scores"].items()}
        pruned = list(data.get("pruned_heads", []))
        metrics = compare_score_dicts(s0, s1, exclude=pruned, ks=ks)
        entries.append({**e, "n_pruned": len(pruned), "attribution_seconds": data.get("seconds"), "metrics": metrics})
    return {"round0_file": round0_path, "scores_dir": scores_dir, "ks": list(ks),
            "created_at": datetime.now().isoformat(timespec="seconds"), "n_entries": len(entries),
            "definition": "round 0 = initial importance scores; each entry's metrics compare round 0 with that file; "
                          "'surviving' = excluding heads pruned up to that file (score 0)",
            "entries": entries}


def format_table(result: Dict[str, Any]) -> str:
    ks = result["ks"]
    head = f"{'frac':<6}{'config':<22}{'rnd':>4}{'seed':>5}{'pruned':>8}{'ρ head(surv)':>12}{'ρ head(all)':>12}{'ρ mlp':>8}{'ρ all':>8}"
    head += "".join(f"{'J' + str(k) + ' surv':>10}" for k in ks)
    lines = [head]
    for e in result["entries"]:
        m = e["metrics"]
        sp, jc = m["spearman"], m["jaccard"]["heads_surviving"]
        row = (f"{e['fraction']:<6}{e['config']:<22}{str(e['round'] if e['round'] is not None else '-'):>4}"
               f"{str(e['seed'] if e['seed'] is not None else '-'):>5}{e['n_pruned']:>8}"
               f"{sp['heads_surviving']:>12.4f}{sp['heads_all']:>12.4f}{sp['mlp']:>8.4f}{sp['all_surviving']:>8.4f}")
        row += "".join(f"{jc['top' + str(k)]:>10.3f}" for k in ks)
        lines.append(row)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Explanation drift: round-0 vs per-round / rescored scores (Spearman, top-k Jaccard)")
    p.add_argument("--round0", default=ROUND0_FILE)
    p.add_argument("--scores-dir", default=SCORES_DIR)
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--ks", default="100,200", help="k values for top-k Jaccard")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    if not os.path.isdir(args.scores_dir):
        print(f"score directory not found: {args.scores_dir} (run run_iterative_pruning.py first)")
        return 1
    result = compute_drift(args.round0, args.scores_dir, ks)
    if not result["entries"]:
        print(f"no recognised score files in {args.scores_dir}")
        return 1
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp, args.output)
    print(format_table(result))
    print(f"\nDrift results saved: {args.output} ({result['n_entries']} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
