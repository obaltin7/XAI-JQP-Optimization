"""
PAIRED comparison of task results (CPU only): correct/incorrect pattern of two configurations on the same 500 questions,
with multiple-comparison correction.

run_eval_from_plans.py stores per-question records (HellaSwag / ARC: per_question[].correct_norm; MMLU: per_question[].correct).
With 500 examples an independent 95% CI is about ±4.4 points, so differences of 3-4 points cannot be interpreted with an
unpaired comparison; an exact (two-sided binomial) McNemar test on the same questions is used instead, counting only the
discordant questions (only A correct / only B correct).

Multiple comparisons: --correction {none, bonferroni, holm}, default HOLM; raw and adjusted p are reported TOGETHER.
  PRIMARY family (pre-registered; significance is claimed only for these): for each ratio × task
      xai_single vs prune_random,  xai_single vs prune_taylor,  xai_iter_fixedq vs xai_single
  (in the run_qwen_experiments format -- Qwen -- the counterparts are: prune_only vs prune_random [both without INT4],
  both vs prune_taylor, xai_iter_fixedq vs both). The correction is applied over the WHOLE family
  (Mistral: 3 ratios × 3 tasks × 3 comparisons = 27 tests).
  SECONDARY family: all other comparisons -- descriptive; adjusted p is computed within its own family but no significance
  is claimed in the table.
  Comparisons given with --pair form a separate "extra" family (e.g. E-3: C4-calibrated iterative vs WikiText-calibrated iterative).

Usage:
    python tools/evaluation/compare_tasks_paired.py --merge results/tasks_gun9_iterative_gun7_wanda_ln.json      # Mistral + Wanda_ln -> tasks_gun9_paired.json
    python tools/evaluation/compare_tasks_paired.py --tasks results/tasks_gun9_model2_qwen_gun9.json --mmlu-from results/model2_qwen_gun9.json
    python tools/evaluation/compare_tasks_paired.py --merge results/tasks_gun9_calib_c4_gun14_f06.json:c4 \\
           --pair c4:f0.6/xai_iter_fixedq f0.6/xai_iter_fixedq --pair c4:f0.6/xai_iter_fixedq f0.6/xai_single \\
           --output results/tasks_gun9_paired_e3.json                                            # E-3 hypothesis test
"""
import argparse
import json
import os
import sys
from datetime import datetime
from math import comb
from typing import Any, Dict, List, Optional, Sequence, Tuple

TASKS_FILE = os.path.join("results", "tasks_gun9.json")
OUTPUT_FILE = os.path.join("results", "tasks_gun9_paired.json")
METRICS = (("hellaswag", "correct_norm"), ("arc_challenge", "correct_norm"), ("mmlu", "correct"))
CORRECTIONS = ("none", "bonferroni", "holm")
ALPHA = 0.05
# "{p}" is the ratio prefix: "f<ratio>/" for run_iterative_pruning plans, empty for run_qwen_experiments (Qwen) plans. Missing entries are skipped.
PRIMARY_TEMPLATES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "fractions": (("{p}xai_single", "{p}prune_random"), ("{p}xai_single", "{p}prune_taylor"), ("{p}xai_iter_fixedq", "{p}xai_single")),
    "configs": (("prune_only", "prune_random"), ("both", "prune_taylor"), ("xai_iter_fixedq", "both")),
}
SECONDARY_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    ("fp16", "{p}xai_single"), ("fp16", "{p}xai_iter_fixedq"), ("fp16", "{p}az_buda_cok_kuantize"), ("{p}xai_iter_fixedq", "{p}prune_taylor"),
    ("{p}xai_single", "{p}prune_wanda_ln"), ("{p}prune_taylor", "{p}prune_wanda_ln"), ("{p}prune_wanda_ln", "{p}prune_random"),
    ("{p}xai_single", "{p}prune_wanda"), ("fp16", "{p}both"), ("fp16", "{p}prune_only"), ("fp16", "{p}quant_only"),
    ("{p}prune_only", "{p}prune_reverse"), ("{p}both", "{p}prune_random"))


def paired_path_for(tasks_path: str) -> str:
    """results/tasks_gun9.json -> results/tasks_gun9_paired.json; results/tasks_gun9_<name>.json -> results/tasks_gun9_paired_<name>.json."""
    base = os.path.basename(tasks_path)
    if base.startswith("tasks_gun9"):
        return os.path.join(os.path.dirname(tasks_path), "tasks_gun9_paired" + base[len("tasks_gun9"):])
    return os.path.join(os.path.dirname(tasks_path), "paired_" + base)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact binomial McNemar p-value (discordant n = only_a + only_b, p = 0.5); 1.0 if n = 0."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def adjust_pvalues(pvals: Sequence[float], method: str = "holm") -> List[float]:
    """
    Family-wise adjusted p-values (in input order). holm: for ascending p_(1..m), p_adj(i) = max_{j<=i} min(1, (m-j+1)·p_(j))
    (step-down; always at least as powerful as Bonferroni); bonferroni: min(1, m·p); none: p.
    """
    if method not in CORRECTIONS:
        raise ValueError(f"correction={method!r} is invalid; expected one of {CORRECTIONS}")
    m = len(pvals)
    if method == "none" or m == 0:
        return [float(p) for p in pvals]
    if method == "bonferroni":
        return [min(1.0, m * float(p)) for p in pvals]
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(pvals[i])))
        out[i] = running
    return out


def correctness(entry: Dict[str, Any], task: str, field: str) -> Optional[Dict[Any, bool]]:
    """{question id: is correct}; None if the task/records are missing. Id: (subject, index) -- in MMLU the index is WITHIN a subject (not unique on its own)."""
    block = entry.get("mmlu") if task == "mmlu" else (entry.get("tasks") or {}).get(task)
    rows = (block or {}).get("per_question")
    if not rows:
        return None
    out = {(r.get("subject"), r.get("index", i)): bool(r[field]) for i, r in enumerate(rows)}
    if len(out) != len(rows):
        raise ValueError(f"{task}: question ids are not unique ({len(out)} keys, {len(rows)} records)")
    return out


def compare(entries: Dict[str, Any], a: str, b: str, task: str, field: str) -> Optional[Dict[str, Any]]:
    ca, cb = (correctness(entries[x], task, field) if x in entries else None for x in (a, b))
    if ca is None or cb is None:
        return None
    if set(ca) != set(cb):
        raise ValueError(f"{a} and {b}: {task} question sets differ (paired test not possible)")
    only_a = sum(1 for q in ca if ca[q] and not cb[q])
    only_b = sum(1 for q in ca if cb[q] and not ca[q])
    n = len(ca)
    return {"a": a, "b": b, "task": task, "metric": "acc_norm" if field == "correct_norm" else "acc", "n": n,
            "acc_a": sum(ca.values()) / n, "acc_b": sum(cb.values()) / n, "diff_a_minus_b": (sum(ca.values()) - sum(cb.values())) / n,
            "both_correct": sum(1 for q in ca if ca[q] and cb[q]), "only_a": only_a, "only_b": only_b,
            "both_wrong": sum(1 for q in ca if not ca[q] and not cb[q]), "p_mcnemar_exact": mcnemar_exact(only_a, only_b)}


def fraction_keys(entries: Dict[str, Any]) -> List[str]:
    keys = set()
    for k in entries:
        head = k.split("/", 1)[0]
        if "/" in k and ":" not in head and head.startswith("f"):
            try:
                float(head[1:])
            except ValueError:
                continue
            keys.add(head[1:])
    return sorted(keys, key=float)


def inject_mmlu(entries: Dict[str, Any], model2: Dict[str, Any]) -> int:
    """
    Add the PER-QUESTION MMLU records from run_qwen_experiments output (baseline.mmlu, configs.<name>.repeats[].mmlu) to the task
    entries (the Qwen task evaluation does not re-measure MMLU). For stochastic configs the repeat matching the entry's seed is used.
    Returns the number of entries updated.
    """
    n = 0
    for key, e in entries.items():
        if (e.get("mmlu") or {}).get("per_question"):
            continue
        if key == "fp16":
            block = (model2.get("baseline") or {}).get("mmlu")
        else:
            reps = [r for r in (model2.get("configs", {}).get(key) or {}).get("repeats", []) if r.get("status") == "completed"]
            match = [r for r in reps if e.get("seed") is None or r.get("seed") == e.get("seed")] or reps
            block = match[0].get("mmlu") if match else None
        if block and block.get("per_question"):
            e["mmlu"] = {"per_question": block["per_question"], "mmlu_subset_acc": block.get("mmlu_subset_acc"), "injected_from": "model2"}
            n += 1
    return n


def overlay_mmlu(entries: Dict[str, Any], prefix: str, extra: Dict[str, Any], source: str) -> int:
    """
    Add separately measured per-question MMLU records to the entries with the SAME key (<tasks>_seed<N>_mmlu.json -> 's<N>:' entries;
    the E-4 run was made without --with-mmlu). Writes only to entries without MMLU; touches no other field and leaves the source files
    unchanged. Returns the number of entries updated.
    """
    n = 0
    for key, e in extra.items():
        cur = entries.get(f"{prefix}{key}")
        block = e.get("mmlu") or {}
        if cur is None or (cur.get("mmlu") or {}).get("per_question") or not block.get("per_question"):
            continue
        cur["mmlu"] = {"per_question": block["per_question"], "mmlu_subset_acc": block.get("mmlu_subset_acc"), "injected_from": source}
        n += 1
    return n


def build_pairs(entries: Dict[str, Any], extra_pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    prefixes = [f"f{fk}/" for fk in fraction_keys(entries)]
    fmt = "fractions" if prefixes else "configs"
    pairs: List[Dict[str, Any]] = []
    seen = set()

    def add(a: str, b: str, family: str) -> None:
        for task, field in METRICS:
            if (a, b, task) in seen or (b, a, task) in seen:
                continue
            res = compare(entries, a, b, task, field)
            if res is not None:
                seen.add((a, b, task))
                pairs.append({**res, "family": family})

    for family, templates in (("primary", PRIMARY_TEMPLATES[fmt]), ("secondary", SECONDARY_TEMPLATES)):
        for prefix in (prefixes or [""]):
            for ta, tb in templates:
                add(ta.format(p=prefix), tb.format(p=prefix), family)
                if family == "primary" and tb.endswith("prune_random"):  # one SEPARATE test per random seed (the family grows); s<N>: = extra seed file
                    for seed in EXTRA_RANDOM_SEEDS:
                        add(ta.format(p=prefix), f"s{seed}:" + tb.format(p=prefix), family)
    for a, b in extra_pairs:
        missing = [x for x in (a, b) if x not in entries]
        if missing:
            raise ValueError(f"--pair: entry not found: {missing}; available: {sorted(entries)[:8]}…")
        add(a, b, "extra")
    return pairs


EXTRA_RANDOM_SEEDS = (43, 44)  # E-4: <tasks>_seed<N>.json is merged automatically if present (key prefix "s<N>:")


def random_claims(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pre-registered 'better than random' claim: holds only if significant after Holm correction against ALL available seeds, in favour of A."""
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for pr in pairs:
        if pr["family"] == "primary" and pr["b"].endswith("prune_random"):
            groups.setdefault((pr["a"], pr["b"].split(":")[-1], pr["task"]), []).append(pr)
    return [{"a": a, "b": b, "task": t, "n_seeds": len(g), "all_significant": all(x["significant"] and x["diff_a_minus_b"] > 0 for x in g)}
            for (a, b, t), g in groups.items()]


def apply_correction(pairs: List[Dict[str, Any]], method: str, alpha: float = ALPHA) -> None:
    for family in sorted({p["family"] for p in pairs}):
        fam = [p for p in pairs if p["family"] == family]
        for p, adj in zip(fam, adjust_pvalues([x["p_mcnemar_exact"] for x in fam], method)):
            p.update({"p_adjusted": adj, "correction": method, "family_size": len(fam), "significant": adj < alpha})


def parse_default_correction() -> str:
    """CLI default (testable): 'holm'."""
    return "holm"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Paired (McNemar) comparison of task results + Holm correction")
    p.add_argument("--tasks", default=TASKS_FILE)
    p.add_argument("--output", default=None, help="default: derived from --tasks (tasks_gun9[_x].json -> tasks_gun9_paired[_x].json)")
    p.add_argument("--correction", choices=CORRECTIONS, default=parse_default_correction(), help="within-family multiple-comparison correction (default holm)")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--merge", action="append", default=[], metavar="PATH[:TAG]",
                   help="add the entries of another task JSON; if TAG is given, keys become 'TAG:<key>' (avoids collisions)")
    p.add_argument("--pair", action="append", nargs=2, default=[], metavar=("A", "B"), help="additional comparison (family: extra); repeatable")
    p.add_argument("--mmlu-from", default=None, help="run_qwen_experiments output: per-question MMLU records are added to the entries from here (Qwen)")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(args.tasks, "r", encoding="utf-8") as f:
        entries = {k: e for k, e in json.load(f)["entries"].items() if e.get("status", "completed") == "completed"}
    merged: List[str] = []
    for spec in args.merge:
        path, tag = (spec, "") if os.path.exists(spec) else spec.rsplit(":", 1)
        with open(path, "r", encoding="utf-8") as f:
            extra = json.load(f)["entries"]
        for k, e in extra.items():
            key = f"{tag}:{k}" if tag else k
            if key in entries:
                raise ValueError(f"--merge {spec}: '{key}' already exists; give a TAG (PATH:TAG)")
            if e.get("status", "completed") == "completed":
                entries[key] = e
        merged.append(spec)
    n_overlay = 0
    for seed in EXTRA_RANDOM_SEEDS:  # E-4 seed files are picked up automatically when present
        seed_path = args.tasks.replace(".json", f"_seed{seed}.json")
        if os.path.exists(seed_path):
            with open(seed_path, "r", encoding="utf-8") as f:
                entries.update({f"s{seed}:{k}": e for k, e in json.load(f)["entries"].items() if e.get("status", "completed") == "completed"})
            merged.append(seed_path)
            mmlu_path = seed_path.replace(".json", "_mmlu.json")  # separately measured seed MMLU (used if present; _seed<N>.json is not modified)
            if os.path.exists(mmlu_path):
                with open(mmlu_path, "r", encoding="utf-8") as f:
                    n_overlay += overlay_mmlu(entries, f"s{seed}:", json.load(f)["entries"], mmlu_path)
                merged.append(mmlu_path)
    n_injected = 0
    if args.mmlu_from:
        with open(args.mmlu_from, "r", encoding="utf-8") as f:
            n_injected = inject_mmlu(entries, json.load(f))
    pairs = build_pairs(entries, [tuple(x) for x in args.pair])
    apply_correction(pairs, args.correction, args.alpha)
    output = args.output or paired_path_for(args.tasks)
    data = {"created_at": datetime.now().isoformat(timespec="seconds"), "source": args.tasks, "merged": merged, "mmlu_from": args.mmlu_from,
            "mmlu_injected_entries": n_injected, "mmlu_overlay_entries": n_overlay, "correction": args.correction, "alpha": args.alpha,
            "definition": "exact two-sided binomial McNemar; only_a / only_b = number of questions answered correctly only by A / only by B; "
                          "HellaSwag / ARC acc_norm, MMLU acc; p_adjusted = within-family adjusted p; significant = p_adjusted < alpha",
            "families": {"primary": "pre-registered: for each ratio × task xai_single vs prune_random, xai_single vs prune_taylor, "
                                    "xai_iter_fixedq vs xai_single (model2 format: prune_only vs prune_random, both vs prune_taylor, "
                                    "xai_iter_fixedq vs both); significance is claimed only for this family",
                         "secondary": "all other comparisons -- descriptive", "extra": "comparisons given with --pair (separate family)"},
            "random_claims": random_claims(pairs), "n_pairs": len(pairs), "n_by_family": {f: sum(1 for x in pairs if x["family"] == f) for f in sorted({x["family"] for x in pairs})},
            "pairs": pairs}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, output)
    print(f"{'family':<10}{'A':<26}{'B':<28}{'task':<14}{'A':>7}{'B':>7}{'only A':>10}{'only B':>10}{'p raw':>10}{'p adj.':>10}  significant")
    for r in pairs:
        print(f"{r['family']:<10}{r['a']:<26}{r['b']:<28}{r['task']:<14}{r['acc_a']:>7.3f}{r['acc_b']:>7.3f}{r['only_a']:>10}{r['only_b']:>10}"
              f"{r['p_mcnemar_exact']:>10.2g}{r['p_adjusted']:>10.2g}  {'YES' if r['significant'] else '-'}")
    print(f"Written: {output} ({len(pairs)} comparisons; correction {args.correction}, family sizes {data['n_by_family']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
