"""
E-5: windowed perplexity (WikiText-2 + C4) and paired bootstrap for cross-domain evaluation.

Motivation: in B-3 the C4-calibrated plan has a much worse WikiText-2 perplexity (11.64 vs 6.32) yet matches the
WikiText-calibrated plan on downstream tasks (E-6). Hypothesis (a): domain-specific heads (WikiText formatting patterns)
receive low scores on C4 and are pruned, i.e. an "in-domain perplexity bias". To test this, every model is measured on
TWO domains (2×2: calibration domain × evaluation domain).

  window_nlls        the SAME sliding window as baseline_eval.compute_perplexity (max_length 1024, stride 512); returns per-window
                     (NLL sum, number of target tokens); ppl = exp(Σnll / Σtoken), the same formula as
                     tools/evaluation/baseline_eval.py (Σtoken = sequence length).
  load_c4_eval_text  fixed subset of the allenai/c4 en validation stream that does NOT overlap the calibration passages: the first
                     SKIP_DOCS (1000) documents are skipped (calibration stream indices 0–24), then the next N_DOCS documents with
                     >= MIN_CHARS characters are taken in order; ids (stream index, url, sha1) are written to
                     results/c4_ppl_subset_ids.json and the sha1 hashes are verified on later loads (no silent drift).
  paired_bootstrap   the SAME windows of two configurations are resampled (10,000 draws, seed 42): 95% percentile CI for
                     Δppl = ppl_A − ppl_B.

CLI (CPU only): pre-registered comparisons from stored window NLLs
    python src/eval_ppl_windows.py --tasks results/tasks_gun15_e5_wikitext.json --merge results/tasks_gun15_e5_c4.json:c4 \\
           --pair c4:f0.2/xai_single f0.2/xai_single --pair f0.2/xai_iter_fixedq f0.2/xai_single \\
           --pair c4:f0.2/xai_iter_fixedq c4:f0.2/xai_single --output results/ppl_bootstrap_gun15.json
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

MAX_LENGTH, STRIDE = 1024, 512  # same as baseline_eval
PPL_DATASETS = ("wikitext2", "c4")
C4_IDS_FILE = os.path.join("results", "c4_ppl_subset_ids.json")
C4_EVAL = dict(path="allenai/c4", name="en", split="validation", skip_docs=1000, n_docs=256, min_chars=500)
N_BOOT, BOOT_SEED = 10000, 42


@torch.no_grad()
def window_nlls_from_ids(model, input_ids: torch.Tensor, device, max_length: int = MAX_LENGTH, stride: int = STRIDE) -> Dict[str, Any]:
    """input_ids [1, T] -> {"nll": [per-window NLL sums], "tokens": [per-window target token counts], "perplexity", "n_tokens"}."""
    seq_len = input_ids.size(1)
    nll: List[float] = []
    tokens: List[int] = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        target = ids.clone()
        target[:, :-trg] = -100
        loss = model(ids, labels=target).loss
        nll.append(float(loss) * trg)
        tokens.append(int(trg))
        prev_end = end
        if end == seq_len:
            break
    return {"nll": nll, "tokens": tokens, "n_tokens": int(sum(tokens)), "perplexity": math.exp(sum(nll) / sum(tokens))}


def window_nlls(model, tokenizer, text: str, device, max_length: int = MAX_LENGTH, stride: int = STRIDE) -> Dict[str, Any]:
    return window_nlls_from_ids(model, tokenizer(text, return_tensors="pt").input_ids, device, max_length, stride)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def select_c4_eval_docs(stream, skip_docs: int, n_docs: int, min_chars: int) -> List[Dict[str, Any]]:
    """First n_docs documents with >= min_chars characters in stream order, after skipping skip_docs (deterministic, no seed)."""
    docs: List[Dict[str, Any]] = []
    for i, row in enumerate(stream):
        if i < skip_docs:
            continue
        text = (row.get("text") or "").strip()
        if len(text) >= min_chars:
            docs.append({"stream_index": i, "url": row.get("url"), "sha1": _sha1(text), "n_chars": len(text), "text": text})
            if len(docs) == n_docs:
                break
    return docs


def load_c4_eval_text(ids_path: str = C4_IDS_FILE, *, stream=None, **overrides) -> Tuple[str, Dict[str, Any]]:
    """C4 evaluation text (joined with "\\n\\n") + meta; writes the id file if absent, otherwise verifies the sha1 hashes."""
    cfg = {**C4_EVAL, **overrides}
    if stream is None:
        from datasets import load_dataset  # lazy import: datasets is optional locally

        stream = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], streaming=True)
    docs = select_c4_eval_docs(stream, cfg["skip_docs"], cfg["n_docs"], cfg["min_chars"])
    if len(docs) < cfg["n_docs"]:
        raise ValueError(f"only {len(docs)} documents could be selected from the C4 stream (required: {cfg['n_docs']})")
    ids = [{k: d[k] for k in ("stream_index", "url", "sha1", "n_chars")} for d in docs]
    meta = {"dataset": f"{cfg['path']} {cfg['name']} {cfg['split']} (streaming)", "skip_docs": cfg["skip_docs"], "n_docs": cfg["n_docs"],
            "min_chars": cfg["min_chars"], "note": "calibration passages use stream indices 0–24; no overlap since the first skip_docs documents are skipped"}
    if os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            stored = json.load(f)
        if [d["sha1"] for d in stored["docs"]] != [d["sha1"] for d in ids]:
            raise ValueError(f"C4 evaluation subset does not match {ids_path} (the dataset may have changed)")
    else:
        os.makedirs(os.path.dirname(ids_path) or ".", exist_ok=True)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {**meta, "created_at": datetime.now().isoformat(timespec="seconds")}, "docs": ids}, f, indent=1, ensure_ascii=False)
    return "\n\n".join(d["text"] for d in docs), {**meta, "ids_file": ids_path, "first_stream_index": ids[0]["stream_index"]}


def paired_bootstrap(a: Dict[str, Any], b: Dict[str, Any], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Dict[str, Any]:
    """Δppl = ppl_A − ppl_B over the same windows with a 95% percentile CI; per-window target token counts must match."""
    if list(a["tokens"]) != list(b["tokens"]):
        raise ValueError("window structures differ (same text / tokenizer / window settings required)")
    na, nb, tok = np.asarray(a["nll"], float), np.asarray(b["nll"], float), np.asarray(a["tokens"], float)
    idx = np.random.default_rng(seed).integers(0, len(tok), size=(n_boot, len(tok)))
    t = tok[idx].sum(axis=1)
    delta = np.exp(na[idx].sum(axis=1) / t) - np.exp(nb[idx].sum(axis=1) / t)
    lo, hi = np.percentile(delta, [2.5, 97.5])
    pa, pb = math.exp(na.sum() / tok.sum()), math.exp(nb.sum() / tok.sum())
    return {"ppl_a": pa, "ppl_b": pb, "delta": pa - pb, "ci95": [float(lo), float(hi)], "n_windows": int(len(tok)), "n_boot": n_boot, "seed": seed,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def parse_ppl_datasets(text: str) -> List[str]:
    """'wikitext2,c4' -> ['wikitext2', 'c4']; an unknown domain is an explicit error (shared validation for the D-1 / D-2 flag)."""
    out = [d.strip() for d in (text or "").split(",") if d.strip()]
    bad = [d for d in out if d not in PPL_DATASETS]
    if bad:
        raise SystemExit(f"--window-nll-datasets: unknown domain {bad}; valid: {', '.join(PPL_DATASETS)}")
    return out


def load_ppl_texts(datasets: Sequence[str], wikitext_text: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """{domain: text} + meta. The WikiText text is taken as is from the calling run (not reloaded); C4 = the fixed C4 evaluation subset (sha1-verified)."""
    texts: Dict[str, str] = {}
    meta: Dict[str, Any] = {}
    for ds in datasets:
        if ds == "wikitext2":
            if wikitext_text is None:
                raise ValueError("no text given for wikitext2 (the run must have loaded the WikiText test set)")
            texts[ds] = wikitext_text
        else:
            texts[ds], meta["c4_ppl_subset"] = load_c4_eval_text()
    return texts, meta


def record_window_nll(rep: Dict[str, Any], model, tokenizer, texts: Dict[str, Optional[str]], dry: bool,
                      seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """For the D-1 / D-2 runs: writes each domain's window NLLs to rep["ppl"][<domain>] with the SAME schema as run_eval_from_plans
    (E-5), so the same bootstrap CLI applies. The existing rep["perplexity"] measurement is NOT modified (separate pass; the
    difference serves as a consistency check).
    texts: {domain: text} (values are None in dry-run; fake windows with a fixed per-domain seed -> pairable across runs)."""
    out: Dict[str, Dict[str, Any]] = {}
    for i, (ds, text) in enumerate(texts.items()):
        t0 = time.time()
        if dry:
            g = torch.Generator().manual_seed(seed + i)
            w = window_nlls_from_ids(model, torch.randint(1, model.config.vocab_size, (1, 120), generator=g), "cpu", max_length=32, stride=16)
        else:
            w = window_nlls(model, tokenizer, text, next(model.parameters()).device)
        out[ds] = {"perplexity": w["perplexity"], "n_windows": len(w["nll"]), "n_tokens": w["n_tokens"], "seconds": time.time() - t0,
                   "windows": {"nll": w["nll"], "tokens": w["tokens"]}}
        rep.setdefault("ppl", {})[ds] = out[ds]
    if not dry and "wikitext2" in out and isinstance(rep.get("perplexity"), (int, float)):
        rep["perplexity_minus_windows"] = rep["perplexity"] - out["wikitext2"]["perplexity"]
    return out


def entries_from_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """{key: entry} for the bootstrap. The run_eval_from_plans format ("entries") is used as is; in the D-1 (run_iterative_pruning:
    "fractions") and D-2 / model2 ("configs") formats every completed repeat becomes "<key>@s<seed>", and the first completed
    repeat is additionally exposed as "<key>"."""
    if "entries" in data:
        return dict(data["entries"])
    groups = ([(f"f{fk}/", fr.get("configs", {})) for fk, fr in data["fractions"].items()] if "fractions" in data else [("", data.get("configs", {}))])
    entries: Dict[str, Any] = {}
    for prefix, configs in groups:
        for name, cfg in configs.items():
            for rep in [r for r in cfg.get("repeats", []) if r.get("status") == "completed"]:
                entries.setdefault(f"{prefix}{name}", rep)
                entries[f"{prefix}{name}@s{rep.get('seed')}"] = rep
    return entries


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Paired bootstrap from stored window NLLs (Δppl, 95% CI)")
    p.add_argument("--tasks", required=True, help="run_eval_from_plans output (run with --ppl-datasets --save-window-nll)")
    p.add_argument("--merge", action="append", default=[], metavar="PATH:LABEL")
    p.add_argument("--pair", action="append", nargs=2, default=[], metavar=("A", "B"), help="Δ = ppl_A − ppl_B; repeatable")
    p.add_argument("--datasets", default=",".join(PPL_DATASETS))
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    with open(args.tasks, "r", encoding="utf-8") as f:
        entries = entries_from_json(json.load(f))
    for spec in args.merge:
        path, tag = spec.rsplit(":", 1)
        with open(path, "r", encoding="utf-8") as f:
            entries.update({f"{tag}:{k}": e for k, e in entries_from_json(json.load(f)).items()})
    pairs = []
    for a, b in args.pair:
        for ds in [d for d in args.datasets.split(",") if d]:
            wa, wb = [((entries[x].get("ppl") or {}).get(ds) or {}).get("windows") for x in (a, b)]
            if not wa or not wb:
                raise ValueError(f"{a} / {b}: no {ds} window NLL record (run with --ppl-datasets {ds} --save-window-nll)")
            pairs.append({"a": a, "b": b, "dataset": ds, **paired_bootstrap(wa, wb)})
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"created_at": datetime.now().isoformat(timespec="seconds"), "source": args.tasks, "merged": args.merge,
                   "definition": "paired window bootstrap; delta = ppl_A − ppl_B; ci95 percentile; excludes_zero = CI does not contain zero",
                   "pairs": pairs}, f, indent=1, ensure_ascii=False)
    for r in pairs:
        print(f"{r['a']:<28}{r['b']:<28}{r['dataset']:<10}{r['ppl_a']:>9.4f}{r['ppl_b']:>9.4f}  Δ {r['delta']:+.4f}  CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
