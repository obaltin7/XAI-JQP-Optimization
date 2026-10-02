"""
Figures and tables for the manuscript: results/*.json -> figures/*.png + tables/*.md|.tex

No GPU required; only matplotlib + numpy. Figures carry no titles (captions belong to the manuscript text),
axis labels include units, and the style survives black-and-white printing (colour combined with marker/hatch;
Okabe-Ito colour-blind-safe palette with a FIXED colour per entity), 300 dpi. Every figure has a small footer
with the source JSON name(s) and the generation date.

Ablation and importance-score outputs (results/ablation_gun6.json, results/importance_scores_gun3.json):
  ablasyon_tablo   ablation table (8 rows: ppl, Δ, ±std, heads, INT4 modules, GB)                       tables/
  ppl_log          log-scale perplexity bar chart of the ablation configurations                          figures/
  olcut_katman     per-layer histogram by criterion: xAI / magnitude / reverse / random (3-seed mean)     figures/
  rastgele_seed    spread of the random seeds (3 points + xAI line)                                       figures/
  ayristirma_tablo loss decomposition table (prune / quant / interaction)                                 tables/
  medyan_kurali    per-layer downgrades/upgrades caused by the median rule                                figures/
  onem_heatmap     redraw of the importance heatmap (the existing PNG is left untouched)                  figures/
Fraction-sweep outputs (results/iterative_gun7.json and companions; a missing source JSON is reported as "skipped",
a value absent from a JSON as "not found"):
  gun7_oran         ppl (log) and MMLU vs ratio per method; Wanda dashed "(unverified adaptation)"; + table        figures/, tables/
  gun7_iteratif     gain of xai_iter_fixedq over xai_single (absolute, % of total loss) + pruned-set overlap       figures/, tables/
  gun7_drift        SURVIVING-heads view only: Spearman and top-100 Jaccard, single-shot vs iterative rounds; controls   figures/, tables/
  baseline_pareto   ppl vs GB: fp16, NF4, GPTQ, AWQ, XAI-JQP 20/40/60% (+ physical + INT4 size)                     figures/
  olcut_katman_gun7 per-layer histogram of heads pruned at 20%: xAI single/iterative, Taylor, Wanda, random (seed 42)  figures/
  baseline_tablo    off-the-shelf methods + XAI-JQP rows (single-shot, iterative, prune less, physical + INT4)      tables/
  speed_tablo       physical-pruning speed/size table (results/speed_gun7.json, 4 variants)                        tables/
  n16_tablo         n_steps=8 vs 16 stability summary (results/compare_..._n16.json)                                tables/
  'XAI-JQP (iterative)' = xai_iter_fixedq (final configuration; INT4 plan identical to single-shot); xai_iter is shown separately.
  If results/iterative_gun7_wanda_ln.json exists, the prune_wanda_ln configuration is added to gun7_oran.
Generalisation (second model, additional tasks):
  model2            Mistral vs second model (results/model2_qwen_gun9.json), same rows side by side: ppl/FP16 (log) + MMLU   figures/, tables/
  gorevler          task vs ratio: MMLU / HellaSwag / ARC-Challenge side by side (results/tasks_gun9.json; Qwen via --tasks-source)  figures/, tables/
  mmlu_konu         per-subject MMLU analysis of the drift: 10 subjects × configuration accuracy table + heatmap of the difference
                    to FP16 (from the per-subject summaries in iterative_gun7.json; no per-question records needed)  figures/, tables/
  --speed-source F  source selection for speed_tablo (and the physical row of baseline_tablo/pareto); the table name follows the
                    file name (e.g. tables/speed_gun8_n3); with the default source tables/speed_gun7 is unchanged (archived table)

Usage:
    python tools/visualization/make_figures.py                       # everything (missing source -> skipped)
    python tools/visualization/make_figures.py --which ppl_log,olcut_katman
    python tools/visualization/make_figures.py --dry-run             # outputs go to dryrun_out/figures, dryrun_out/tables
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compressor import parse_block_key  # noqa: E402

RESULTS = "results"
FIGURES_DIR = "figures"
TABLES_DIR = "tables"
DRYRUN_DIR = "dryrun_out"
DPI = 300
ENGLISH_NOTATION = False  # --english-notation: '20%' instead of '%20' in figures (default output unchanged)

ABLATION_FILE = os.path.join(RESULTS, "ablation_gun6.json")
SCORES_FILE = os.path.join(RESULTS, "importance_scores_gun3.json")
GUN7_FILE = os.path.join(RESULTS, "iterative_gun7.json")
DRIFT_FILE = os.path.join(RESULTS, "drift_gun7.json")
BASELINES_FILE = os.path.join(RESULTS, "baselines_gun7.json")
SPEED_FILE = os.path.join(RESULTS, "speed_gun7.json")
N16_FILE = os.path.join(RESULTS, "compare_importance_scores_gun3_vs_importance_scores_gun3_n16.json")
OFFSET16_FILE = os.path.join(RESULTS, "compare_importance_scores_gun3_vs_importance_scores_gun3_offset16.json")  # calibration-sample stability run (A-5)
GUN7_WANDA_LN_FILE = os.path.join(RESULTS, "iterative_gun7_wanda_ln.json")  # optional Wanda_ln run
MODEL2_FILE = os.path.join(RESULTS, "model2_qwen_gun9.json")  # second model (run_qwen_experiments.py)
# Separate outputs of opt-in configurations (merged into gun7_oran / gun7_drift / mmlu_konu when present; never overwrite existing ones)
GUN7_EXTRA_FILES = [os.path.join(RESULTS, "iterative_gun7_attnconf.json"), os.path.join(RESULTS, "iterative_gun7_4tier.json")]

# Okabe-Ito (colour-blind safe); FIXED assignment per entity, no colour cycling
C = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
     "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F"}
STYLE = {  # name -> (label, colour, marker, hatch)
    "fp16": ("FP16 (reference)", C["black"], "s", ""),
    "quant_only": ("quantization only", C["sky"], "D", ""),
    "prune_only": ("pruning only (xAI)", C["blue"], "o", ""),
    "both": ("pruning + quantization (xAI)", C["orange"], "^", ""),
    "both_biascorr": ("+ bias correction", C["yellow"], "v", ""),
    "prune_random": ("random", C["green"], "x", "//"),
    "prune_magnitude": ("magnitude", C["vermillion"], "P", "\\\\"),
    "prune_reverse": ("reverse (highest xAI)", C["purple"], "*", "xx"),
    # fraction-sweep configurations
    "xai_single": ("XAI-JQP (single-shot)", C["blue"], "o", ""),
    "xai_iter": ("xAI iterative (own INT4 plan)", C["sky"], "v", ""),
    "xai_iter_fixedq": ("XAI-JQP (iterative)", C["orange"], "^", ""),  # final configuration
    "prune_wanda": ("Wanda", C["vermillion"], "P", "\\\\"),
    "prune_wanda_ln": ("Wanda, within-layer z-score", C["vermillion"], "X", "\\\\"),
    # opt-in configurations and baselines
    "prune_attnconf": ("attention confidence (Voita)", C["gray"], "h", ".."),
    "xai_single_4tier": ("XAI-JQP single-shot, 4 tiers (+INT8)", C["blue"], "8", ""),
    "xai_iter_fixedq_4tier": ("XAI-JQP iterative, 4 tiers (+INT8)", C["orange"], "p", ""),
    "prune_taylor": ("Taylor", C["purple"], "*", "xx"),
    "az_buda_cok_kuantize": ("prune less / quantize more", C["yellow"], "D", ""),
    "physical_int4": ("XAI-JQP physical + INT4", C["black"], "^", ""),
    "nf4_uniform": ("uniform NF4", C["sky"], "D", ""),
    "gptq_4bit": ("GPTQ 4-bit", C["green"], "x", "//"),
    "awq_4bit": ("AWQ 4-bit", C["purple"], "*", "xx"),
}
ABLATION_ORDER = ["fp16", "quant_only", "prune_only", "both_biascorr", "both", "prune_random", "prune_magnitude", "prune_reverse"]
CRITERIA_ORDER = ["prune_only", "prune_magnitude", "prune_reverse", "prune_random"]
GUN7_ORDER = ["xai_single", "xai_iter", "xai_iter_fixedq", "prune_taylor", "prune_wanda", "prune_wanda_ln", "prune_attnconf", "prune_random",
              "az_buda_cok_kuantize", "xai_single_4tier", "xai_iter_fixedq_4tier"]
SINGLE_POINT_CONFIGS = ("az_buda_cok_kuantize", "xai_single_4tier", "xai_iter_fixedq_4tier")  # evaluated at 20% only: drawn as points, not lines

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": DPI, "figure.dpi": 100, "lines.linewidth": 1.5, "lines.markersize": 6,
})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def label(name: str) -> str:
    return STYLE.get(name, (name, C["gray"], "o", ""))[0]


def style(name: str) -> Tuple[str, str, str]:
    _, color, marker, hatch = STYLE.get(name, (name, C["gray"], "o", ""))
    return color, marker, hatch


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def footer(fig: plt.Figure, sources: Sequence[str]) -> None:
    text = "Source: " + ", ".join(os.path.basename(s) for s in sources) + f" · Generated: {date.today():%Y-%m-%d}"
    fig.text(0.01, 0.005, text, fontsize=6, color="#666666", ha="left", va="bottom")


def _english_text(s: str) -> str:
    """'%20' -> '20%' and no internal table names in the footer (used only with --english-notation)."""
    s = re.sub(r"(, kapanis_\w+)+\)", ", …", s)  # footer() keeps only the basename of the '… full list in tables/…' entry
    return re.sub(r"%(\d+(?:\.\d+)?)", r"\1%", s)


def _apply_english_notation(fig: plt.Figure) -> None:
    for ax in fig.axes:  # fixed tick labels are re-read from the formatter at draw time
        for axis in (ax.xaxis, ax.yaxis):
            fmt_ = axis.get_major_formatter()
            if isinstance(fmt_, matplotlib.ticker.FixedFormatter):
                fmt_.seq = [_english_text(str(t)) for t in fmt_.seq]
            elif isinstance(fmt_, matplotlib.ticker.FuncFormatter):  # set_ticklabels in newer matplotlib
                axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos, g=fmt_.func: _english_text(str(g(x, pos)))))
    for t in fig.findobj(matplotlib.text.Text):
        t.set_text(_english_text(t.get_text()))


def save_fig(fig: plt.Figure, out_dir: str, name: str, sources: Sequence[str]) -> str:
    footer(fig, sources)
    if ENGLISH_NOTATION:
        _apply_english_notation(fig)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def _tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("±", r"$\pm$").replace("Δ", r"$\Delta$")
            .replace("ρ", r"$\rho$").replace("−", "$-$").replace("→", r"$\to$"))


def write_table(out_dir: str, name: str, headers: Sequence[str], rows: Sequence[Sequence[Any]],
                sources: Sequence[str], note: str = "", align: Optional[str] = None) -> List[str]:
    """Write the same table as Markdown (.md) and LaTeX (.tex, booktabs), with a source + date footnote."""
    os.makedirs(out_dir, exist_ok=True)
    src = "Source: " + ", ".join(os.path.basename(s) for s in sources) + f" · Generated: {date.today():%Y-%m-%d}"
    align = align or ("l" + "r" * (len(headers) - 1))
    md = ["| " + " | ".join(headers) + " |",
          "|" + "|".join(":--" if a == "l" else "--:" for a in align) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        md.append(f"\n{note}")
    md.append(f"\n<sub>{src}</sub>")
    md_path = os.path.join(out_dir, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    tex = [f"% {src}", "% requires \\usepackage{booktabs}", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
           " & ".join(_tex_escape(h) for h in headers) + r" \\", "\\midrule"]
    tex += [" & ".join(_tex_escape(c) for c in r) + r" \\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    if note:
        tex.append(f"% {note}")
    tex_path = os.path.join(out_dir, f"{name}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    return [md_path, tex_path]


def fmt(v: Any, spec: str, default: str = "–") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else default


def per_layer_counts(pruned_heads: Sequence[Sequence[int]], n_layers: int) -> np.ndarray:
    counts = np.zeros(n_layers, dtype=float)
    for layer, _ in pruned_heads:
        counts[int(layer)] += 1
    return counts


# --------------------------------------------------------------------------- #
# Ablation run (results/ablation_gun6.json)
# --------------------------------------------------------------------------- #
def fig_ablasyon_tablo(paths: Dict[str, str]) -> List[str]:
    d = load_json(ABLATION_FILE)
    if d is None:
        return []
    fp = d["fp16_rerun"]
    rows: List[List[Any]] = [[label("fp16"), fmt(fp["perplexity"], ".4f"), "–", "–", 0, 0, fmt(fp["model_bytes"]["gb"], ".2f")]]
    for name in ABLATION_ORDER[1:]:
        c = d["configs"].get(name)
        if not c or c.get("status") != "completed":
            continue
        std = c.get("perplexity_std") or 0.0
        rows.append([label(name) + (f" ({c['n_completed_repeats']} seeds)" if c.get("stochastic") else ""),
                     fmt(c["perplexity_mean"], ".4f"), fmt(c["delta_vs_fp16"], "+.4f"),
                     fmt(std, ".4f") if c.get("stochastic") else "–", c["n_pruned_heads"], c["int4_modules"],
                     fmt(c["model_bytes_after_gb"], ".2f")])
    note = "ppl: WikiText-2 test perplexity (baseline evaluation harness); Δ: difference to FP16; ±std: sample std (n−1); " \
           "heads: number of pruned heads (out of 1024); INT4: number of bitsandbytes NF4 modules; GB: parameters+buffers (excluding quant_state)."
    return write_table(paths["tables"], "ablasyon_gun6", ["configuration", "ppl", "Δ ppl", "±std", "heads", "INT4 modules", "GB"],
                       rows, [ABLATION_FILE], note)


def fig_ppl_log(paths: Dict[str, str]) -> List[str]:
    d = load_json(ABLATION_FILE)
    if d is None:
        return []
    names = [n for n in ABLATION_ORDER if n == "fp16" or d["configs"].get(n, {}).get("status") == "completed"]
    vals = [d["fp16_rerun"]["perplexity"] if n == "fp16" else d["configs"][n]["perplexity_mean"] for n in names]
    errs = [0.0 if n == "fp16" else (d["configs"][n].get("perplexity_std") or 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(names))
    for i, n in enumerate(names):
        color, _, hatch = style(n)
        ax.bar(x[i], vals[i], width=0.7, color=color, hatch=hatch, edgecolor="black", linewidth=0.6,
               yerr=errs[i] or None, capsize=3, ecolor="black")
        ax.text(x[i], (vals[i] + errs[i]) * 1.12, f"{vals[i]:.2f}" + (f" ± {errs[i]:.2f}" if errs[i] else ""),
                ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(4, max(vals) * 2.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label(n).replace(" (", "\n(") for n in names], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Perplexity, WikiText-2 (log scale)")
    ax.set_xlabel("Configuration (20% block ratio, 205-head budget)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return [save_fig(fig, paths["figures"], "ppl_log_gun6", [ABLATION_FILE])]


def fig_olcut_katman(paths: Dict[str, str]) -> List[str]:
    d = load_json(ABLATION_FILE)
    if d is None:
        return []
    n_layers = len(d["reference"]["plan_summary_both"])
    fig, axes = plt.subplots(len(CRITERIA_ORDER), 1, figsize=(6.4, 6.0), sharex=True, sharey=True)
    x = np.arange(n_layers)
    for ax, name in zip(axes, CRITERIA_ORDER):
        c = d["configs"].get(name)
        if not c or c.get("status") != "completed":
            ax.set_visible(False)
            continue
        color, _, hatch = style(name)
        reps = [r for r in c["repeats"] if r.get("status") == "completed"]
        mats = np.stack([per_layer_counts(r["pruned_heads"], n_layers) for r in reps])
        mean, std = mats.mean(axis=0), (mats.std(axis=0, ddof=1) if len(reps) > 1 else None)
        ax.bar(x, mean, width=0.8, color=color, hatch=hatch, edgecolor="black", linewidth=0.4,
               yerr=std, capsize=1.5, ecolor="black", error_kw={"linewidth": 0.6})
        title = label(name) + (f" ({len(reps)} seeds, mean ± std)" if len(reps) > 1 else "")
        bbox = {"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.9}
        ax.text(0.01, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=bbox)
        ax.text(0.99, 0.95, f"ppl {c['perplexity_mean']:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=bbox)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 32 * 1.15)  # 32 heads; headroom for the label
    axes[-1].set_xlabel("Layer index")
    axes[-1].set_xticks(x[::2])
    fig.text(0.0, 0.55, "Pruned heads per layer (out of 32)", rotation=90, va="center", ha="left", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    return [save_fig(fig, paths["figures"], "olcut_katman_hist_gun6", [ABLATION_FILE])]


def fig_rastgele_seed(paths: Dict[str, str]) -> List[str]:
    d = load_json(ABLATION_FILE)
    if d is None or d["configs"].get("prune_random", {}).get("status") != "completed":
        return []
    c = d["configs"]["prune_random"]
    reps = [r for r in c["repeats"] if r.get("status") == "completed"]
    seeds = [r["seed"] for r in reps]
    ppls = [r["perplexity"] for r in reps]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    color, marker, _ = style("prune_random")
    ax.axhspan(c["perplexity_mean"] - c["perplexity_std"], c["perplexity_mean"] + c["perplexity_std"],
               color=color, alpha=0.12, lw=0, label="random mean ± std")
    ax.axhline(c["perplexity_mean"], color=color, ls="--", lw=1)
    ax.scatter(seeds, ppls, color=color, marker=marker, s=60, zorder=3, label="random seed (single run)")
    for s, p in zip(seeds, ppls):
        ax.annotate(f"{p:.2f}", (s, p), textcoords="offset points", xytext=(8, 0), fontsize=7, va="center")
    xai = d["configs"].get("prune_only", {}).get("perplexity_mean")
    if xai is not None:
        ax.axhline(xai, color=style("prune_only")[0], ls="-", lw=1.5, label=f"xAI selection ({xai:.2f})")
    fp = d["fp16_rerun"]["perplexity"]
    ax.axhline(fp, color=C["black"], ls=":", lw=1, label=f"FP16 ({fp:.2f})")
    ax.set_xticks(seeds)
    ax.set_xlabel("Seed (random head selection, 205 heads)")
    ax.set_ylabel("Perplexity, WikiText-2")
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.95, edgecolor="none")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return [save_fig(fig, paths["figures"], "rastgele_seed_gun6", [ABLATION_FILE])]


def fig_ayristirma_tablo(paths: Dict[str, str]) -> List[str]:
    d = load_json(ABLATION_FILE)
    if d is None:
        return []
    dec = d["decomposition"]
    both = dec.get("both_delta")

    def share(v: Any) -> str:
        return fmt(100 * v / both, ".1f") + " %" if isinstance(v, (int, float)) and both else "–"

    rows: List[List[Any]] = [["FP16 reference", fmt(dec.get("fp16"), ".4f"), "–", "–"]]
    for key, name in (("prune_delta", "pruning (xAI, 205 heads)"), ("quant_delta", "quantization (129 INT4 modules)"),
                      ("interaction", "interaction (both − pruning − quant.)"), ("both_delta", "total (pruning + quantization)")):
        v = dec.get(key)
        rows.append([name, "–", fmt(v, "+.4f"), share(v)])
    if dec.get("biascorr_gain") is not None:
        rows.append(["bias-correction gain", "–", fmt(-dec["biascorr_gain"], "+.4f"), share(dec["biascorr_gain"])])
    note = "Δ ppl: perplexity difference to FP16; share: percentage of the total loss (both − FP16); " \
           "interaction = both_delta − prune_delta − quant_delta (0 = additive)."
    return write_table(paths["tables"], "ayristirma_gun6", ["component", "ppl", "Δ ppl", "share"], rows, [ABLATION_FILE], note)


def fig_medyan_kurali(paths: Dict[str, str]) -> List[str]:
    d = load_json(ABLATION_FILE)
    if d is None:
        return []
    per = d["median_rule"]["per_layer"]
    n_layers = len(d["reference"]["plan_summary_both"])
    x = np.arange(n_layers)
    down = np.array([per.get(str(i), {}).get("downgraded", 0) for i in range(n_layers)], dtype=float)
    up = np.array([per.get(str(i), {}).get("upgraded", 0) for i in range(n_layers)], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x, up, width=0.8, color=C["blue"], edgecolor="black", linewidth=0.4, label="upgraded (int4 → fp16)")
    ax.bar(x, -down, width=0.8, color=C["vermillion"], hatch="//", edgecolor="black", linewidth=0.4,
           label="downgraded (fp16 → int4)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-float(down.max()) * 1.15, float(up.max()) * 1.45)  # headroom for the summary text
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Heads with a changed tier")
    ax.set_xticks(x[::2])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.grid(axis="x", visible=False)
    m = d["median_rule"]
    ax.text(0.99, 0.97, f"total: {m['downgraded']} downgraded, {m['upgraded']} upgraded, {m['unchanged']} unchanged "
                        f"({m['n_unpruned_heads']} unpruned heads)", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return [save_fig(fig, paths["figures"], "medyan_kurali_gun6", [ABLATION_FILE])]


# --------------------------------------------------------------------------- #
# Importance heatmap redraw (results/importance_scores_gun3.json)
# --------------------------------------------------------------------------- #
def fig_onem_heatmap(paths: Dict[str, str]) -> List[str]:
    d = load_json(SCORES_FILE)
    if d is None:
        return []
    scores = d["scores"]
    layers = sorted({parse_block_key(k).layer for k in scores})
    n_layers = len(layers)
    n_heads = max(parse_block_key(k).index for k in scores if parse_block_key(k).kind == "attn") + 1
    heads = np.full((n_layers, n_heads), np.nan)
    mlp = np.full(n_layers, np.nan)
    for k, v in scores.items():
        bk = parse_block_key(k)
        if bk.kind == "attn":
            heads[bk.layer, bk.index] = v
        else:
            mlp[bk.layer] = v
    attn_mean = np.nanmean(heads, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 0.28 * n_layers + 1.4), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.55})
    for ax, col, ttl in ((axes[0], mlp, "MLP block score"), (axes[1], attn_mean, "Attention (mean over heads)")):
        vmin, vmax = float(np.nanmin(col)), float(np.nanmax(col))
        im = ax.imshow(col.reshape(-1, 1), cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_xlabel(ttl, fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([str(i) for i in range(n_layers)], fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        span = (vmax - vmin) or 1.0
        for i, v in enumerate(col):
            ax.text(0, i, f"{v:.3f}" if v < 1 else f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if (v - vmin) / span > 0.6 else "#1b1b1b")
        cbar = fig.colorbar(im, ax=ax, fraction=0.10, pad=0.06, aspect=40)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_visible(False)
    axes[0].set_ylabel("Layer index")
    fig.text(0.5, 0.965, "LIG importance score (mean |attribution| per token); separate colour scale per panel",
             ha="center", va="top", fontsize=7, color="#555555")
    fig.subplots_adjust(top=0.94, bottom=0.05, left=0.12, right=0.9)
    return [save_fig(fig, paths["figures"], "onem_heatmap_gun3", [SCORES_FILE])]


# --------------------------------------------------------------------------- #
# Fraction sweep (results/iterative_gun7.json and companions): ratio sweep, iterative gain, drift, pareto,
# layer histogram, speed, n16
# --------------------------------------------------------------------------- #
MISSING = "not found"
MISSING_LOG: List[str] = []  # values absent from the JSONs; listed at the end of main (numbers are NEVER invented)
FRACTION_STYLE = {"0.2": (C["blue"], "o"), "0.4": (C["orange"], "^"), "0.6": (C["vermillion"], "s")}
DRIFT_CONFIG_ORDER = ["xai_single", "xai_iter", "prune_taylor", "prune_wanda", "prune_wanda_ln", "prune_attnconf", "prune_random",
                      "az_buda_cok_kuantize", "xai_single_4tier"]
WANDA_NOTE = " (unverified adaptation)"


def missing(what: str) -> str:
    MISSING_LOG.append(what)
    return MISSING


def frac_label(fk: Any) -> str:
    return f"%{round(float(fk) * 100)}"


def method_label(name: str) -> str:
    return label(name) + (WANDA_NOTE if name == "prune_wanda" else "")


def load_gun7_merged() -> Optional[Dict[str, Any]]:
    """iterative_gun7.json plus the EXTRA configurations of iterative_gun7_wanda_ln.json and GUN7_EXTRA_FILES when present
    (existing configurations are never overwritten)."""
    d = load_json(GUN7_FILE)
    if d is None:
        return None
    for path in [GUN7_WANDA_LN_FILE] + GUN7_EXTRA_FILES:  # the attnconf / 4-tier runs are merged the same way
        extra = load_json(path)
        if not extra:
            continue
        for fk, fr in extra.get("fractions", {}).items():
            target = d["fractions"].setdefault(fk, {"configs": {}})["configs"]
            for name, c in fr.get("configs", {}).items():
                if name not in target and c.get("status") == "completed":
                    target[name] = {**c, "source_file": path}
    return d


def _cfg(d: Dict[str, Any], fk: str, name: str) -> Optional[Dict[str, Any]]:
    c = d.get("fractions", {}).get(fk, {}).get("configs", {}).get(name)
    return c if c and c.get("status") == "completed" else None


def _rep0(c: Dict[str, Any], seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
    reps = [r for r in c.get("repeats", []) if r.get("status") == "completed" and (seed is None or r.get("seed") == seed)]
    return reps[0] if reps else None


def _gun7_series(d: Dict[str, Any], key: str) -> Dict[str, Tuple[List[float], List[float], List[float]]]:
    """config -> (ratios, values, std); completed configurations only; std only for ppl."""
    out: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for fk in sorted(d["fractions"], key=float):
        for name, c in d["fractions"][fk]["configs"].items():
            if c.get("status") != "completed" or c.get(key) is None:
                continue
            xs, ys, es = out.setdefault(name, ([], [], []))
            xs.append(float(fk))
            ys.append(c[key])
            es.append((c.get("perplexity_std") or 0.0) if key == "perplexity_mean" else 0.0)
    return out


def fig_gun7_oran(paths: Dict[str, str]) -> List[str]:
    d = load_gun7_merged()
    if d is None:
        return []
    fp = d.get("fp16_rerun", {})
    panels = (("perplexity_mean", "Perplexity, WikiText-2 (log scale)", fp.get("perplexity"), True),
              ("mmlu_subset_acc_mean", "MMLU subset accuracy (500 questions)", fp.get("mmlu_subset_acc"), False))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7))
    handles: Dict[str, Any] = {}
    n_drawn = 0
    for ax, (key, ylabel, ref, logy) in zip(axes, panels):
        series = _gun7_series(d, key)
        for name in GUN7_ORDER:
            if name not in series:
                continue
            xs, ys, es = series[name]
            color, marker, _ = style(name)
            lab = method_label(name)
            if name in SINGLE_POINT_CONFIGS:
                h = ax.scatter(xs, ys, color=color, marker=marker, s=45, zorder=4, edgecolor="black", linewidth=0.6, label=lab)
            else:
                h = ax.errorbar(xs, ys, yerr=(es if any(es) else None), color=color, marker=marker, capsize=3,
                                ls="--" if name == "prune_wanda" else "-", label=lab)  # only raw (unverified) Wanda is dashed; wanda_ln is solid
            handles.setdefault(lab, h)
            n_drawn += 1
        if ref is not None:
            h = ax.axhline(ref, color=C["black"], ls=":", lw=1, label="FP16 reference")
            handles.setdefault("FP16 reference", h)
            ax.annotate(f"FP16 {ref:.3f}", (0.99, ref), xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(0, 2), ha="right", va="bottom", fontsize=6.5)
        if key == "mmlu_subset_acc_mean":
            h = ax.axhline(0.25, color=C["gray"], ls="-.", lw=0.8, label="random baseline (4 options)")
            handles.setdefault("random baseline (4 options)", h)
        if logy:
            ax.set_yscale("log")
        xt = sorted({x for s in series.values() for x in s[0]})
        ax.set_xticks(xt)
        ax.set_xticklabels([frac_label(x) for x in xt])
        ax.set_xlabel("Pruned-head ratio (block ratio)")
        ax.set_ylabel(ylabel)
    if n_drawn == 0:
        plt.close(fig)
        return []
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    oran_sources = [GUN7_FILE] + sorted({c["source_file"] for fr in d["fractions"].values() for c in fr["configs"].values()
                                         if c.get("source_file")})
    outs = [save_fig(fig, paths["figures"], "gun7_oran", oran_sources)]

    rows: List[List[Any]] = []
    if fp.get("perplexity") is not None:
        rows.append(["–", label("fp16"), fmt(fp["perplexity"], ".4f"), "–", "–", fmt(fp.get("mmlu_subset_acc"), ".3f"), 0, 0,
                     fmt(fp.get("model_bytes", {}).get("gb"), ".3f"), 0, fmt(fp.get("seconds"), ".0f")])
    for fk in sorted(d["fractions"], key=float):
        for name in GUN7_ORDER:
            c = _cfg(d, fk, name)
            if not c:
                continue
            lab = method_label(name) + (f" ({c['n_completed_repeats']} seeds, mean)" if c.get("stochastic") else "")
            rows.append([frac_label(fk), lab, fmt(c.get("perplexity_mean"), ".4f"),
                         fmt(c.get("perplexity_std"), ".4f") if c.get("stochastic") else "–",
                         fmt(c.get("delta_vs_fp16"), "+.4f"), fmt(c.get("mmlu_subset_acc_mean"), ".3f"),
                         c.get("n_pruned_heads", "–"), c.get("int4_modules", "–"), fmt(c.get("model_bytes_after_gb"), ".3f"),
                         c.get("attribution_calls", "–"), fmt(c.get("seconds"), ".0f")])
    note = ("ppl: WikiText-2 test perplexity (baseline evaluation harness); Δ: relative to the FP16 rerun; ±std: random control only (3 seeds, n−1); "
            "MMLU: 500 questions (random baseline 0.25); heads: pruned heads (out of 1024); INT4: bitsandbytes NF4 modules; GB: parameters+buffers "
            "(excluding quant_state); attr.: number of attribution (LIG) calls — the extra cost of the iterative variant; time: configuration total "
            "(including loading + rescoring). 'XAI-JQP (iterative)' = xai_iter_fixedq (INT4 plan identical to single-shot); "
            "'xAI iterative (own INT4 plan)' = xai_iter. Wanda: head-level adaptation, unverified (diagnosis: tools/diagnostics/diagnose_wanda.py).")
    outs += write_table(paths["tables"], "gun7_oran", ["ratio", "configuration", "ppl", "±std", "Δ ppl", "MMLU", "heads", "INT4", "GB",
                                                       "attr.", "time (s)"], rows, oran_sources, note, align="llrrrrrrrrr")
    return outs


def fig_gun7_iteratif(paths: Dict[str, str]) -> List[str]:
    d = load_json(GUN7_FILE)
    if d is None:
        return []
    fp = d.get("fp16_rerun", {}).get("perplexity")
    rows: List[Dict[str, Any]] = []
    for fk in sorted(d["fractions"], key=float):
        s, it = _cfg(d, fk, "xai_single"), _cfg(d, fk, "xai_iter_fixedq")
        if not (s and it):
            missing(f"gun7_iteratif: ratio {fk}: xai_single/xai_iter_fixedq not completed")
            continue
        gain = s["perplexity_mean"] - it["perplexity_mean"]
        total = (s["perplexity_mean"] - fp) if fp is not None else None
        r0 = _rep0(it)
        overlap = r0.get("overlap_with_xai_heads") if r0 else None
        rows.append({"fk": fk, "single": s["perplexity_mean"], "iter": it["perplexity_mean"], "gain": gain,
                     "pct": (100.0 * gain / total) if total else None, "overlap": overlap, "n": it["n_pruned_heads"]})
    if not rows:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.1))
    x = np.arange(len(rows))
    ticks = [frac_label(r["fk"]) for r in rows]
    ax = axes[0]
    vals = [r["gain"] for r in rows]
    ax.bar(x, vals, width=0.6, color=C["orange"], edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylim(min(vals) * 0.5, max(vals) * 3)
    for i, v in enumerate(vals):
        ax.text(x[i], v * 1.15, f"−{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("ppl gain: single-shot − iterative (log scale)")
    ax = axes[1]
    pcts = [r["pct"] if r["pct"] is not None else 0.0 for r in rows]
    ax.bar(x, pcts, width=0.6, color=C["blue"], edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 100)
    for i, r in enumerate(rows):
        ax.text(x[i], pcts[i] + 2, f"%{pcts[i]:.0f}" if r["pct"] is not None else MISSING, ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Recovered share of the total loss (%)\n(relative to single-shot − FP16)")
    ax = axes[2]
    common = [r["overlap"] if r["overlap"] is not None else 0 for r in rows]
    changed = [r["n"] - c for r, c in zip(rows, common)]
    ax.bar(x, common, width=0.6, color=C["blue"], edgecolor="black", linewidth=0.6, label="shared heads (iterative ∩ single-shot)")
    ax.bar(x, changed, bottom=common, width=0.6, color=C["vermillion"], hatch="//", edgecolor="black", linewidth=0.6, label="changed heads")
    for i, r in enumerate(rows):
        ax.text(x[i], r["n"] + 8, f"{r['overlap']}/{r['n']}" if r["overlap"] is not None else MISSING, ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(r["n"] for r in rows) * 1.2)
    ax.set_ylabel("Number of pruned heads")
    ax.legend(fontsize=6.5, loc="upper left")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ticks)
        ax.set_xlabel("Pruned-head ratio")
        ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    outs = [save_fig(fig, paths["figures"], "gun7_iteratif", [GUN7_FILE])]
    trows = [[frac_label(r["fk"]), fmt(r["single"], ".4f"), fmt(r["iter"], ".4f"), fmt(-r["gain"], "+.4f"),
              fmt(r["pct"], ".0f") + " %" if r["pct"] is not None else MISSING,
              f"{r['overlap']}/{r['n']}" if r["overlap"] is not None else MISSING] for r in rows]
    note = ("Iterative = xai_iter_fixedq (3 rounds, rescored every round; INT4 plan and size identical to single-shot, only the pruning "
            "selection differs). Share = gain / (single-shot − FP16). Overlap = number of heads pruned by both methods.")
    outs += write_table(paths["tables"], "gun7_iteratif", ["ratio", "ppl single-shot", "ppl iterative", "Δ (iterative − single-shot)",
                                                           "recovered share", "overlap"], rows=trows, sources=[GUN7_FILE], note=note)
    return outs


def _drift_get(e: Dict[str, Any], mkey: str) -> Optional[float]:
    m = e["metrics"]
    return m["spearman"]["heads_surviving"] if mkey == "spearman" else m["jaccard"]["heads_surviving"].get("top100")


def fig_gun7_drift(paths: Dict[str, str]) -> List[str]:
    """SURVIVING-heads view only (pruned heads excluded): the 'all' view measures selection disagreement and is not plotted."""
    d = load_json(DRIFT_FILE)
    if d is None:
        return []
    entries = d["entries"]
    fracs = sorted({e["fraction"] for e in entries}, key=float)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex="col")
    for row, (mkey, ylabel) in enumerate((("spearman", "Spearman ρ (surviving heads)"),
                                          ("j100", "Top-100 Jaccard (surviving heads)"))):
        ax = axes[row][0]
        for fk in fracs:
            col, mk = FRACTION_STYLE.get(fk, (C["gray"], "o"))
            it = sorted((e for e in entries if e["fraction"] == fk and e["config"] == "xai_iter"), key=lambda e: e["round"])
            if it:
                ax.plot([e["round"] for e in it], [_drift_get(e, mkey) for e in it], color=col, marker=mk,
                        label=f"XAI-JQP iterative, {frac_label(fk)}")
            for e in (e for e in entries if e["fraction"] == fk and e["config"] == "xai_single"):
                ax.scatter([0.5], [_drift_get(e, mkey)], facecolors="none", edgecolors=col, marker=mk, s=55, zorder=3,
                           linewidth=1.2, label=f"after single-shot, {frac_label(fk)}")
        ax.set_xticks([0.5, 1, 2, 3])
        ax.set_xticklabels(["single-shot\n(after)", "round 1", "round 2", "round 3"])
        ax.set_ylabel(ylabel)
        ax = axes[row][1]
        xs = [float(f) for f in fracs]
        for name in ("xai_single", "prune_taylor", "prune_wanda", "prune_wanda_ln", "prune_attnconf", "prune_random"):
            ys, es = [], []
            for fk in fracs:
                vals = [_drift_get(e, mkey) for e in entries if e["fraction"] == fk and e["config"] == name]
                vals = [v for v in vals if v is not None]
                ys.append(float(np.mean(vals)) if vals else np.nan)
                es.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            if all(np.isnan(ys)):
                continue
            color, marker, _ = style(name)
            lab = method_label(name) + (" (3 seeds, mean ± std)" if name == "prune_random" else "")
            ax.errorbar(xs, ys, yerr=(es if any(es) else None), color=color, marker=marker, capsize=3,
                        ls="--" if name == "prune_wanda" else "-", label=lab)  # only raw (unverified) Wanda is dashed; wanda_ln is solid
        ax.set_xticks(xs)
        ax.set_xticklabels([frac_label(f) for f in fracs])
    axes[1][0].set_xlabel("Round (xAI iterative; single-shot = rescoring after pruning)")
    axes[1][1].set_xlabel("Pruned-head ratio (single-shot selectors, rescoring after pruning)")
    axes[0][0].legend(fontsize=6, ncol=2, loc="lower left")
    axes[0][1].legend(fontsize=6, loc="lower left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    outs = [save_fig(fig, paths["figures"], "gun7_drift", [DRIFT_FILE])]

    def order(e: Dict[str, Any]) -> Tuple[float, int, int, int]:
        cfg_i = DRIFT_CONFIG_ORDER.index(e["config"]) if e["config"] in DRIFT_CONFIG_ORDER else len(DRIFT_CONFIG_ORDER)
        return float(e["fraction"]), cfg_i, e["round"] or 0, e["seed"] or 0

    drift_names = {"xai_iter": "XAI-JQP (iterative), per-round pruning"}  # the per-round files follow one procedure for xai_iter and xai_iter_fixedq
    rows = [[frac_label(e["fraction"]), drift_names.get(e["config"], method_label(e["config"])), e["round"] if e["round"] is not None else "–",
             e["seed"] if e["seed"] is not None else "–", e["n_pruned"],
             fmt(e["metrics"]["spearman"]["heads_surviving"], ".3f"), fmt(e["metrics"]["spearman"]["mlp"], ".3f"),
             fmt(e["metrics"]["jaccard"]["heads_surviving"].get("top100"), ".3f"),
             fmt(e["metrics"]["jaccard"]["heads_surviving"].get("top200"), ".3f")] for e in sorted(entries, key=order)]
    note = ("Surviving-heads view only: ρ = Spearman rank correlation with round 0 (results/importance_scores_gun3.json), excluding the "
            "heads pruned up to that file (scored 0 when rescored); J = top-k Jaccard (surviving heads). The 'all' view (pruned heads "
            "included with score 0) measures selection disagreement rather than drift and is therefore not tabulated. Drift was not "
            "computed separately for xai_iter_fixedq; it selects the same heads, so the xai_iter values apply. Wanda: unverified adaptation.")
    outs += write_table(paths["tables"], "gun7_drift", ["ratio", "configuration", "round", "seed", "pruned", "ρ head (surviving)",
                                                        "ρ MLP", "J100", "J200"], rows, [DRIFT_FILE], note, align="llrrrrrrr")
    return outs


def _xai_jqp_rows(g: Optional[Dict[str, Any]], s: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """XAI-JQP points for the baseline table/pareto (iterative_gun7.json + speed_gun7.json); missing ones go to MISSING_LOG."""
    out: List[Dict[str, Any]] = []
    if g is None:
        missing("XAI-JQP rows: iterative_gun7.json missing")
        return out
    spec = (("0.2", "xai_single", "XAI-JQP %20 single-shot (205 heads + 129 INT4)", "XAI-JQP %20 single-shot"),
            ("0.2", "xai_iter_fixedq", "XAI-JQP %20 iterative (xai_iter_fixedq)", "XAI-JQP %20 iterative"),
            ("0.2", "az_buda_cok_kuantize", "prune less / quantize more %20 (102 heads + 129 INT4)", "prune less / quantize more %20"),
            ("0.4", "xai_iter_fixedq", "XAI-JQP %40 iterative (xai_iter_fixedq)", "XAI-JQP %40 iterative"),
            ("0.6", "xai_iter_fixedq", "XAI-JQP %60 iterative (xai_iter_fixedq)", "XAI-JQP %60 iterative"))
    for fk, name, long, short in spec:
        c = _cfg(g, fk, name)
        if not c:
            missing(f"XAI-JQP row: ratio {fk} {name} (iterative_gun7.json)")
            continue
        out.append({"name": name, "long": long, "short": short, "ppl": c["perplexity_mean"], "delta": c.get("delta_vs_fp16"),
                    "mmlu": c.get("mmlu_subset_acc_mean"), "gb": c.get("model_bytes_after_gb"), "ratio": c.get("measured_model_bytes_ratio"),
                    "vram": c.get("peak_vram_gb"), "ppl_note": ""})
    xs = _cfg(g, "0.2", "xai_single")
    pi = (s or {}).get("summary", {}).get("physical_int4")
    pv = (s or {}).get("variants", {}).get("physical_int4", {})
    if pi and pi.get("model_gb") is not None and xs:
        out.insert(min(3, len(out)), {"name": "physical_int4", "long": "XAI-JQP %20 physical pruning + INT4 (physical_int4)*",
                                      "short": "XAI-JQP %20 physical + INT4*", "ppl": xs["perplexity_mean"], "delta": xs.get("delta_vs_fp16"),
                                      "mmlu": None, "gb": pi["model_gb"], "ratio": pi.get("size_ratio_vs_fp16"),
                                      "vram": pv.get("peak_vram_gb"), "ppl_note": "*"})
    else:
        missing("XAI-JQP physical_int4 row: speed_gun7.json summary.physical_int4 (or 20% xai_single) missing")
    return out


PARETO_OFFSETS = {"FP16 (reference)": (-6, 4, "right"), "uniform NF4": (-6, -9, "right"), "GPTQ 4-bit": (6, 5, "left"),
                  "AWQ 4-bit": (6, -9, "left"), "XAI-JQP %20 single-shot": (6, 4, "left"), "XAI-JQP %20 iterative": (6, -2, "left"),
                  "prune less / quantize more %20": (6, -9, "left"), "XAI-JQP %20 physical + INT4*": (-6, 6, "right")}


def fig_baseline_pareto(paths: Dict[str, str]) -> List[str]:
    b = load_json(BASELINES_FILE)
    if b is None:
        return []
    g, s = load_json(GUN7_FILE), load_json(SPEED_FILE)
    pts: List[Tuple[str, float, float, str, str]] = []
    for name in ("fp16", "nf4_uniform", "gptq_4bit", "awq_4bit"):
        r = b.get("summary", {}).get("configs", {}).get(name)
        if r and r.get("perplexity") is not None and r.get("model_gb") is not None:
            color, marker, _ = style(name)
            pts.append((label(name), r["model_gb"], r["perplexity"], color, marker))
        else:
            missing(f"baseline_pareto: {name} (baselines_gun7.json summary)")
    for row in _xai_jqp_rows(g, s):
        if row["gb"] is None:
            missing(f"baseline_pareto: {row['short']} size")
            continue
        color, marker, _ = style(row["name"])
        pts.append((row["short"], row["gb"], row["ppl"], color, marker))
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for lab, gb, ppl, color, marker in pts:
        edge = {} if marker in ("x", "+") else {"edgecolor": "black", "linewidth": 0.5}  # unfilled markers take no edge colour
        ax.scatter(gb, ppl, color=color, marker=marker, s=55, zorder=3, **edge)
        dx, dy, ha = PARETO_OFFSETS.get(lab, (6, 3, "left"))
        ax.annotate(lab, (gb, ppl), textcoords="offset points", xytext=(dx, dy), fontsize=6.5, ha=ha, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("Measured model size (GB; parameters + buffers, excluding bnb quant_state)")
    ax.set_ylabel("Perplexity, WikiText-2 (log scale)")
    ax.set_xlim(min(p[1] for p in pts) - 1.0, max(p[1] for p in pts) + 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.text(0.01, 0.035, "* Physical pruning + INT4: size from speed_gun7.json (same 205 heads + 129 INT4); perplexity = masked single-shot "
                          "value of the same head set. Masked = physical equivalence verified on the 7B model (5.8174/5.8173, tables/speed_gun8_fixedq).",
             fontsize=5.8, color="#444444", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return [save_fig(fig, paths["figures"], "baseline_pareto", [BASELINES_FILE, GUN7_FILE, SPEED_FILE])]


def fig_olcut_katman_gun7(paths: Dict[str, str]) -> List[str]:
    d = load_json(GUN7_FILE)
    if d is None:
        return []
    fk = "0.2"
    panels = [("xai_single", None), ("xai_iter_fixedq", None), ("prune_taylor", None), ("prune_wanda", None), ("prune_random", 42)]
    reps: List[Tuple[str, Optional[int], Dict[str, Any], Dict[str, Any]]] = []
    for name, seed in panels:
        c = _cfg(d, fk, name)
        r = _rep0(c, seed) if c else None
        if r is None:
            missing(f"olcut_katman_gun7: ratio {fk} {name}" + (f" seed {seed}" if seed else "") + " not completed")
            continue
        reps.append((name, seed, c, r))
    if not reps:
        return []
    n_layers = len(reps[0][3]["plan_summary"])
    n_heads_layer = d["fractions"][fk]["budget"]["n_heads_total"] // n_layers
    fig, axes = plt.subplots(len(reps), 1, figsize=(6.4, 1.45 * len(reps) + 0.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(n_layers)
    bbox = {"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.9}
    for ax, (name, seed, c, r) in zip(axes, reps):
        color, _, hatch = style(name)
        counts = per_layer_counts(r["pruned_heads"], n_layers)
        layers = sorted(int(l) for l, _ in r["pruned_heads"])
        med = layers[len(layers) // 2] if layers else None
        early = sum(1 for l in layers if l <= 5)
        ax.bar(x, counts, width=0.8, color=color, hatch=hatch, edgecolor="black", linewidth=0.4)
        title = method_label(name) + (f" (seed {seed})" if seed is not None else "")
        ax.text(0.01, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=bbox)
        ax.text(0.99, 0.95, f"ppl {r['perplexity']:.2f} · median layer {med} · layers 0–5: {early}/{len(layers)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, bbox=bbox)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, n_heads_layer * 1.15)
    axes[-1].set_xlabel("Layer index")
    axes[-1].set_xticks(x[::2])
    fig.text(0.0, 0.55, f"Pruned heads per layer (out of {n_heads_layer}; 20% budget)", rotation=90, va="center",
             ha="left", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    return [save_fig(fig, paths["figures"], "olcut_katman_gun7", [GUN7_FILE])]


def fig_baseline_tablo(paths: Dict[str, str]) -> List[str]:
    d = load_json(BASELINES_FILE)
    if d is None:
        return []
    rows: List[List[Any]] = []
    for name in ("fp16", "nf4_uniform", "gptq_4bit", "awq_4bit"):
        r = d.get("summary", {}).get("configs", {}).get(name)
        if not r:
            missing(f"baseline_tablo: {name} (baselines_gun7.json summary)")
            continue
        rows.append([label(name), fmt(r.get("perplexity"), ".4f"), fmt(r.get("perplexity_delta_vs_fp16"), "+.4f"),
                     fmt(r.get("mmlu_subset_acc"), ".3f"), fmt(r.get("model_gb"), ".3f"),
                     fmt(r.get("measured_model_bytes_ratio"), ".3f"), fmt(r.get("peak_vram_gb"), ".1f")])
    g, s = load_json(GUN7_FILE), load_json(SPEED_FILE)
    for x in _xai_jqp_rows(g, s):
        rows.append([x["long"], fmt(x["ppl"], ".4f") + x["ppl_note"], fmt(x["delta"], "+.4f") + x["ppl_note"],
                     fmt(x["mmlu"], ".3f") if x["mmlu"] is not None else missing(f"baseline_tablo: {x['short']} MMLU not measured"),
                     fmt(x["gb"], ".3f"), fmt(x["ratio"], ".3f"), fmt(x["vram"], ".1f")])
    if not rows:
        return []
    sources = [BASELINES_FILE] + ([GUN7_FILE] if g else []) + ([SPEED_FILE] if s else [])
    note = ("GB: parameters + buffers (excluding bnb quant_state; GPTQ/AWQ include qweight+scales+qzeros — slightly biased in favour of bnb); "
            "GPTQ/AWQ are calibrated on WikiText-2 train (same-domain advantage). For XAI-JQP rows, peak VRAM is the run peak including the "
            f"post-pruning rescoring (attribution), not the inference peak. * physical_int4: size and peak VRAM from {os.path.basename(SPEED_FILE)}; "
            "perplexity/Δ is the MASKED single-shot value of the same head set (205 heads + 129 INT4) — masked = physical equivalence "
            "verified on the 7B model (5.8174/5.8173, tables/speed_gun8_fixedq); MMLU not measured.")
    return write_table(paths["tables"], "baseline_gun7", ["method", "ppl", "Δ ppl", "MMLU", "GB", "size ratio", "peak VRAM (GB)"],
                       rows, sources, note)


SPEED_LABELS = {"fp16": "FP16 (reference)", "masked": "masked pruning", "physical": "physical pruning",
                "physical_int4": "physical pruning + INT4", "masked_int4": "masked pruning + INT4"}


def fig_speed_tablo(paths: Dict[str, str]) -> List[str]:
    d = load_json(SPEED_FILE)
    if d is None:
        return []
    src = d.get("source", {})
    n_heads = src.get("n_heads", "?")
    rows: List[List[Any]] = []
    extra_rows: List[List[str]] = []
    for name in ("fp16", "masked", "physical", "physical_int4", "masked_int4"):
        v, r = d.get("variants", {}).get(name), d.get("summary", {}).get(name)
        if not r or not v or v.get("status") != "completed":
            continue
        lab = SPEED_LABELS.get(name, name)
        if name in ("masked", "physical"):
            lab += f" ({n_heads} heads)"
        if name.endswith("_int4"):
            lab += f" ({n_heads} heads + {v.get('quantization', {}).get('int4_modules', '?')} INT4 modules)"
        std = v.get("ms_per_token_std")
        rows.append([lab, fmt(r.get("ms_per_token"), ".2f") + (f" ± {std:.2f}" if isinstance(std, (int, float)) and std else ""),
                     fmt(r.get("tokens_per_second"), ".2f"), fmt(r.get("speedup_vs_fp16"), ".3f") + "×",
                     fmt(r.get("model_gb"), ".3f"), fmt(r.get("size_ratio_vs_fp16"), ".3f"), f"{r.get('n_params', 0):,}".replace(",", " "),
                     fmt(r.get("peak_vram_gb"), ".2f")])
        extra_rows.append([fmt(v.get("peak_vram_generation_gb"), ".2f"), fmt(v.get("perplexity"), ".4f"), str(v.get("attention_class") or "–")])
    if not rows:
        return []
    # Optional fields (generation-time peak VRAM, perplexity, attention class) become columns only when present in the
    # source, so the speed_gun7 table stays byte-identical
    variants_done = [x for x in d.get("variants", {}).values() if x.get("status") == "completed"]
    extra_headers: List[str] = []
    for idx, (key, header) in enumerate((("peak_vram_generation_gb", "generation peak VRAM (GB)"), ("perplexity", "ppl (WikiText-2)"),
                                         ("attention_class", "attention class"))):
        if any(x.get(key) is not None for x in variants_done):
            extra_headers.append(header)
            for row, ex in zip(rows, extra_rows):
                row.append(ex[idx])
    a = d.get("run", {}).get("args", {})
    note = (f"L40S; {a.get('n_prompts', '?')} prompts × {a.get('max_new_tokens', '?')} new tokens, greedy, KV cache on, 1 warm-up, "
            f"seed {a.get('seed', '?')}, repeats {a.get('n_repeats', 1)}; head set {src.get('config')} ({src.get('plan_file')}). "
            "ms/token = total generation time / generated tokens. Parameters: in the INT4 row not comparable with FP16 because of the "
            "bnb Params4bit uint8 packing (use the size column). Peak VRAM = fp16 loading peak before quantization (the memory reserved "
            "while running INT4 is lower). ")
    if (a.get("n_repeats") or 1) > 1:
        note += "± = std across repeats (n−1)."
    else:
        note += "Speed differences lie within ±2–3% noise in a single run (masked > fp16 in this run)."
    if extra_headers:
        note += (f" attn_implementation={a.get('attn_implementation') or 'default (4.44.2: sdpa)'}; physically pruned layers are always "
                 "computed with PrunedHeadAttention (eager matmul + softmax). ppl: WikiText-2 perplexity of the masked and the physical "
                 "model on the same head set (numerical equivalence); generation peak VRAM: peak reset after loading/applying.")
    # --speed-source: the table name follows the source file; default source -> speed_gun7 (the old table is kept as an archive)
    table_name = os.path.splitext(os.path.basename(SPEED_FILE))[0]
    return write_table(paths["tables"], table_name, ["variant", "ms/token", "tokens/s", "speedup (FP16=1)", "GB", "size ratio",
                                                       "parameters", "peak VRAM (GB)"] + extra_headers, rows, [SPEED_FILE], note)


# Repeated speed measurements with matched attention paths in ONE table, eager / sdpa as SEPARATE rows
# (same head set: the 'both' configuration of the ablation run)
SPEED_PATH_FILES = (("eager", os.path.join(RESULTS, "speed_gun8_eager.json")),   # run A-3
                    ("sdpa", os.path.join(RESULTS, "speed_gun8_sdpa.json")),     # run C-5
                    # the default kernel stays "eager"; "auto" (the wrapper also uses sdpa) is an EXTRA row only
                    ("sdpa, wrapper auto", os.path.join(RESULTS, "speed_gun8_sdpa_auto.json")))


def fig_speed_yollar(paths: Dict[str, str]) -> List[str]:
    rows: List[List[Any]] = []
    sources: List[str] = []
    for path_name, file in SPEED_PATH_FILES:
        d = load_json(file)
        if d is None:
            missing(f"speed_yollar: no measurement for the {path_name} path ({os.path.basename(file)})")
            continue
        sources.append(file)
        n_rep = d.get("run", {}).get("args", {}).get("n_repeats", 1)
        for name in ("fp16", "masked", "physical"):
            v, r = d.get("variants", {}).get(name), d.get("summary", {}).get(name)
            if name not in d.get("variants", {}):
                continue  # variant not requested in that run (e.g. the extra auto measurement covers only fp16 + physical)
            if not r or not v or v.get("status") != "completed":
                missing(f"speed_yollar: {path_name} / {name} not completed")
                continue
            std = v.get("ms_per_token_std")
            rows.append([path_name, SPEED_LABELS.get(name, name) + (f" [kernel {v['attn_kernel']}]" if v.get("attn_kernel") else ""),
                         fmt(r.get("ms_per_token"), ".2f") + (f" ± {std:.2f}" if isinstance(std, (int, float)) and std else ""),
                         fmt(r.get("tokens_per_second"), ".2f"), fmt(r.get("speedup_vs_fp16"), ".3f") + "×", fmt(r.get("model_gb"), ".3f"),
                         str(v.get("attention_class") or "–"), str(n_rep)])
    if not rows:
        return []
    note = ("'path' = attn_implementation used when loading the fp16 / masked model; by default, physically pruned layers are computed on "
            "BOTH paths with the eager kernel of PrunedHeadAttention (matmul + softmax) — in the sdpa rows the slowdown of the physical variant "
            "relative to FP16 includes this kernel difference, in the eager rows it does not. 'wrapper auto' rows (extra measurement): "
            "with --attn-kernel auto the wrapper also uses sdpa; the main reported numbers use the default (eager) kernel. Speedup is relative to each path's OWN fp16 row. ± = std across repeats (n−1). L40S, 8 prompts × 64 "
            "new tokens, greedy, KV cache on; head set: 'both' configuration of the ablation run (205 heads).")
    return write_table(paths["tables"], "speed_gun8_yollar", ["path", "variant", "ms/token", "tokens/s", "speedup (path's FP16 = 1)", "GB",
                                                               "attention class", "repeats"], rows, sources, note, align="llrrrrlr")


def fig_n16_tablo(paths: Dict[str, str]) -> List[str]:
    d = load_json(N16_FILE)
    if d is None:
        return []
    m, t = d.get("metrics", {}), d.get("tiers", {})
    sp, pe, jc = m.get("spearman", {}), m.get("pearson", {}), m.get("jaccard", {}).get("heads_all", {})
    ma, mb = d.get("meta_a", {}), d.get("meta_b", {})
    heads, plan = t.get("heads", {}), t.get("plan", {})

    def g(v: Any, spec: str) -> str:
        return fmt(v, spec) if isinstance(v, (int, float)) else missing("n16_kararlilik: missing field")

    rows = [["IG steps (a → b)", f"{ma.get('attribution', {}).get('n_steps', MISSING)} → {mb.get('attribution', {}).get('n_steps', MISSING)}"],
            ["Calibration", f"{ma.get('calibration', {}).get('n_passages', MISSING)} passages, same passages, T={ma.get('calibration', {}).get('max_length', MISSING)}"],
            [f"Spearman ρ, head ({m.get('n_heads', '?')})", g(sp.get("heads_all"), ".4f")],
            [f"Spearman ρ, MLP ({m.get('n_mlp', '?')})", g(sp.get("mlp"), ".4f")],
            ["Spearman ρ, all blocks", g(sp.get("all_blocks"), ".4f")],
            ["Pearson r, head", g(pe.get("heads_all"), ".4f")],
            ["Top-100 Jaccard (head)", g(jc.get("top100"), ".3f")],
            ["Top-200 Jaccard (head)", g(jc.get("top200"), ".3f")],
            [f"Heads changing tier (plan {tuple(t.get('tier_fractions', []))})",
             f"{heads.get('n_changed', MISSING)}/{heads.get('n', MISSING)} (%{100 * heads.get('changed_ratio', 0):.2f})"],
            ["Tier transitions", ", ".join(f"{k}: {v}" for k, v in heads.get("transitions", {}).items()) or "none"],
            ["MLPs changing tier", f"{t.get('mlp', {}).get('n_changed', MISSING)}/{t.get('mlp', {}).get('n', MISSING)}"],
            ["Pruned-set overlap", f"{plan.get('pruned_heads_common', MISSING)}/{plan.get('pruned_heads_a', MISSING)} "
                                       f"(Jaccard {g(plan.get('pruned_set_jaccard'), '.3f')})"],
            ["INT4 modules (a / b)", f"{plan.get('int4_modules_a', MISSING)} / {plan.get('int4_modules_b', MISSING)}; "
                                   f"layers with a changed INT4 decision: {plan.get('layers_attn_quant_changed', MISSING)} attn, "
                                   f"{plan.get('layers_mlp_quant_changed', MISSING)} MLP"],
            ["Attribution time (s, a → b)", f"{g(ma.get('timing', {}).get('attribution_seconds'), '.1f')} → "
                                        f"{g(mb.get('timing', {}).get('attribution_seconds'), '.1f')}"]]
    note = ("a = scores of results/importance_scores_gun3.json (n_steps=8), b = same settings with n_steps=16. Sensitivity to the "
            "calibration sample and to the IG baseline (token 0) is not measured in this table.")
    return write_table(paths["tables"], "n16_kararlilik", ["metric", "n_steps=8 vs 16"], rows, [N16_FILE], note, align="ll")


# --------------------------------------------------------------------------- #
# Second model (Qwen2.5-7B-Instruct) — same rows as Mistral, side by side
# --------------------------------------------------------------------------- #
MODEL2_ORDER = ["fp16", "quant_only", "prune_only", "both", "prune_random", "prune_reverse", "prune_taylor", "xai_iter_fixedq"]
MODEL2_LABELS = {"fp16": "FP16 (reference)", "quant_only": "quantization only (INT4 plan)", "prune_only": "pruning only (xAI)",
                 "both": "XAI-JQP (single-shot): pruning + INT4", "prune_random": "random pruning, no INT4",
                 "prune_reverse": "reverse (highest xAI), no INT4", "prune_taylor": "Taylor + INT4",
                 "xai_iter_fixedq": "XAI-JQP (iterative) + INT4"}
MODEL2_GUN6 = ("quant_only", "prune_only", "both", "prune_random", "prune_reverse")  # Mistral rows from the ablation run (ablation_gun6.json)
MODEL2_GUN7 = ("prune_taylor", "xai_iter_fixedq")  # Mistral rows from the fraction sweep (iterative_gun7.json, 20%)


def _short_model(name: Any) -> str:
    s = str(name or "model 2").split("/")[-1]
    return s.replace("-Instruct", "").replace("-v0.3", "")


def _model2_row(ppl: Any, std: Any, mmlu: Any, heads: Any, int4: Any, gb: Any, stochastic: bool = False) -> Dict[str, Any]:
    return {"ppl": ppl, "std": std if stochastic else None, "mmlu": mmlu, "heads": heads, "int4": int4, "gb": gb}


def _mistral_model2_rows() -> Dict[str, Dict[str, Any]]:
    """Mistral side: ablation run (MMLU not measured; the MMLU of 'both' comes from the identical xai_single configuration of the
    fraction sweep) + fraction sweep at 20%."""
    g6, g7 = load_json(ABLATION_FILE), load_json(GUN7_FILE)
    rows: Dict[str, Dict[str, Any]] = {}
    fp7 = (g7 or {}).get("fp16_rerun", {})
    fp6 = (g6 or {}).get("fp16_rerun", {})
    fp = fp7 if fp7.get("perplexity") is not None else fp6
    if fp.get("perplexity") is not None:
        rows["fp16"] = _model2_row(fp["perplexity"], None, fp7.get("mmlu_subset_acc"), 0, 0, fp.get("model_bytes", {}).get("gb"))
    for name in MODEL2_GUN6:
        c = (g6 or {}).get("configs", {}).get(name)
        if not c or c.get("status") != "completed":
            continue
        mmlu = c.get("mmlu_subset_acc_mean")
        if mmlu is None and name == "both" and g7:  # fraction-sweep xai_single at 20% = ablation 'both' (reproduced with 0.0 difference)
            mmlu = (_cfg(g7, "0.2", "xai_single") or {}).get("mmlu_subset_acc_mean")
        rows[name] = _model2_row(c.get("perplexity_mean"), c.get("perplexity_std"), mmlu, c.get("n_pruned_heads"),
                                 c.get("int4_modules"), c.get("model_bytes_after_gb"), bool(c.get("stochastic")))
    for name in MODEL2_GUN7:
        c = _cfg(g7, "0.2", name) if g7 else None
        if c:
            rows[name] = _model2_row(c.get("perplexity_mean"), None, c.get("mmlu_subset_acc_mean"), c.get("n_pruned_heads"),
                                     c.get("int4_modules"), c.get("model_bytes_after_gb"))
    return rows


def _second_model_rows(d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    b = d.get("baseline", {})
    if b.get("status") == "completed":
        rows["fp16"] = _model2_row(b.get("perplexity"), None, b.get("mmlu_subset_acc"), 0, 0, b.get("model_bytes", {}).get("gb"))
    for name, c in d.get("configs", {}).items():
        if c.get("status") == "completed":
            rows[name] = _model2_row(c.get("perplexity_mean"), c.get("perplexity_std"), c.get("mmlu_subset_acc_mean"), c.get("n_pruned_heads"),
                                     c.get("int4_modules"), c.get("model_bytes_after_gb"), bool(c.get("stochastic")))
    return rows


def fig_model2(paths: Dict[str, str]) -> List[str]:
    d = load_json(MODEL2_FILE)
    if d is None:
        return []
    second, first = _second_model_rows(d), _mistral_model2_rows()
    if not second or "fp16" not in second:
        return []
    m1, m2 = "Mistral-7B", _short_model(d.get("run", {}).get("model"))
    names = [n for n in MODEL2_ORDER if n in second or n in first]
    models = ((m1, first, C["blue"], ""), (m2, second, C["orange"], "//"))

    def ratio(rows: Dict[str, Dict[str, Any]], n: str) -> Optional[float]:
        r, ref = rows.get(n), rows.get("fp16")
        if not r or not ref or r["ppl"] is None or not ref["ppl"]:
            return None
        return r["ppl"] / ref["ppl"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(names))
    width = 0.38
    for k, (mname, rows, color, hatch) in enumerate(models):
        off = (k - 0.5) * width
        rs = [ratio(rows, n) for n in names]
        errs = [(rows[n]["std"] / rows["fp16"]["ppl"]) if rs[i] is not None and rows[n].get("std") else 0.0 for i, n in enumerate(names)]
        axes[0].bar(x + off, [r if r is not None else np.nan for r in rs], width, yerr=errs, capsize=2, color=color, hatch=hatch,
                    edgecolor="black", linewidth=0.5, label=mname)
        for xi, r in zip(x + off, rs):
            if r is not None:
                axes[0].annotate(f"{r:.2f}", (xi, r), textcoords="offset points", xytext=(0, 2), ha="center", va="bottom", fontsize=5.5)
        accs = [rows.get(n, {}).get("mmlu") for n in names]
        axes[1].bar(x + off, [a if a is not None else np.nan for a in accs], width, color=color, hatch=hatch, edgecolor="black",
                    linewidth=0.5, label=mname)
    axes[0].axhline(1.0, color=C["black"], ls=":", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Perplexity / FP16 perplexity (log scale)")
    axes[1].axhline(0.25, color=C["gray"], ls="-.", lw=0.8)
    axes[1].annotate("random baseline (4 options)", (0.99, 0.25), xycoords=("axes fraction", "data"), textcoords="offset points",
                     xytext=(0, 2), ha="right", va="bottom", fontsize=6.5, color=C["gray"])
    axes[1].set_ylabel("MMLU subset accuracy (500 questions)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL2_LABELS.get(n, n).replace(": ", ":\n").replace(", ", ",\n").replace(" (", "\n(") for n in names],
                           rotation=60, ha="right", fontsize=6.5)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].legend(loc="upper left", fontsize=7)  # the upper left of the left panel is empty (low bars); avoids covering bars in the MMLU panel
    fig.tight_layout()
    sources = [MODEL2_FILE] + [f for f in (ABLATION_FILE, GUN7_FILE) if os.path.exists(f)]
    outs = [save_fig(fig, paths["figures"], "model2_karsilastirma", sources)]

    def cells(rows: Dict[str, Dict[str, Any]], n: str) -> List[str]:
        r = rows.get(n)
        if not r:
            return ["–"] * 5
        ppl = fmt(r["ppl"], ".4f") + (f" ± {r['std']:.4f}" if isinstance(r.get("std"), (int, float)) and r["std"] else "")
        return [ppl, fmt(ratio(rows, n), ".3f") + "×", fmt(r["mmlu"], ".3f"), f"{fmt(r['heads'], 'd')} / {fmt(r['int4'], 'd')}", fmt(r["gb"], ".3f")]

    table = [[MODEL2_LABELS.get(n, n)] + cells(first, n) + cells(second, n) for n in names]
    ref = d.get("reference", {})
    bh, info = ref.get("budget_heads", {}), d.get("model_info", {})
    note = (f"Rows are defined identically for both models: quantization only / pruning only / single-shot / random / reverse follow the "
            f"ablation definition (random and reverse without quantization; random ± std, n−1); Taylor and iterative follow the fraction-sweep "
            f"definition (INT4 plan identical to single-shot). ppl/FP16: ratio to the model's own FP16 perplexity (comparable across models). "
            f"heads / INT4: number of pruned heads / bitsandbytes NF4 modules. {m1}: %20 = 205/1024 heads, 129 INT4 modules; source: ablation run "
            f"(MMLU not measured: –; the single-shot MMLU comes from the identical xai_single configuration of the fraction sweep) and the fraction "
            f"sweep at 20%. {m2}: %{round(100 * (ref.get('fraction') or 0))} = "
            f"{bh.get('n_prune_heads', '?')}/{bh.get('n_heads_total', '?')} heads, {ref.get('int4_modules_in_plan', '?')} INT4 modules; "
            f"{info.get('n_layers', '?')} layers × {info.get('n_heads', '?')} heads, {info.get('n_kv_heads', '?')} kv-heads, q/k/v with bias; "
            f"precision {d.get('run', {}).get('dtype', '?')}. GB: parameters + buffers (excluding bnb quant_state).")
    if d.get("run", {}).get("dtype") == "bf16":  # Mistral is not re-run in bf16
        note += (f" Precision differs: {m2} bf16, {m1} fp16; the 'FP16 (reference)' row and the ppl/FP16 column refer to each model's OWN "
                 "uncompressed (fp16 / bf16) baseline.")
    headers = ["configuration"] + [f"{m} {h}" for m in (m1, m2) for h in ("ppl", "ppl/FP16", "MMLU", "heads / INT4", "GB")]
    outs += write_table(paths["tables"], "model2_gun9", headers, table, sources, note, align="l" + "r" * 10)
    return outs


# --------------------------------------------------------------------------- #
# Additional tasks (run_eval_from_plans.py) — MMLU / HellaSwag / ARC-Challenge side by side
# --------------------------------------------------------------------------- #
TASKS_FILE = os.path.join(RESULTS, "tasks_gun9.json")
TASK_PANELS = (("mmlu", "MMLU (500 questions)"), ("hellaswag", "HellaSwag (500 examples, acc_norm)"), ("arc_challenge", "ARC-Challenge (500 examples, acc_norm)"))


def _task_value(e: Dict[str, Any], task: str) -> Tuple[Optional[float], bool]:
    """(value, from_source): if MMLU was not measured in this run, the value of the plan's source run; acc_norm for the other tasks."""
    if task == "mmlu":
        if isinstance(e.get("mmlu_subset_acc"), (int, float)):
            return e["mmlu_subset_acc"], False
        v = (e.get("source") or {}).get("mmlu_subset_acc")
        return (v, True) if isinstance(v, (int, float)) else (None, False)
    v = ((e.get("tasks") or {}).get(task) or {}).get("acc_norm")
    return (v if isinstance(v, (int, float)) else None), False


TASKS_WANDA_LN_FILE = os.path.join(RESULTS, "tasks_gun9_iterative_gun7_wanda_ln.json")  # run B-4b (merged into the default source only)
TASKS_EXTRA_SEEDS = (43, 44)  # E-4: if <source>_seed<N>.json files exist, the random row becomes a 3-seed mean ± std
SIGNIF_REFERENCES = ("xai_single", "both", "prune_only")  # reference side in the primary family (compare_tasks_paired.PRIMARY_TEMPLATES)
SIGNIF_TASKS = (("hellaswag", "HS"), ("arc_challenge", "ARC"), ("mmlu", "MMLU"))


GOREVLER_OUT_NAME: Optional[str] = None  # output-name override for gorevler (used only by the gun17_qwen_gorevler wrapper)


def _seed_tasks_file(tasks_file: str, seed: int) -> str:
    """<source>_seed<N>.json; if the source is '…_mmlu.json' (per-question Qwen MMLU file), the seed files are looked up under the
    name without '_mmlu'."""
    direct = tasks_file.replace(".json", f"_seed{seed}.json")
    if os.path.exists(direct) or not tasks_file.endswith("_mmlu.json"):
        return direct
    return tasks_file[: -len("_mmlu.json")] + f"_seed{seed}.json"


def _random_seed_values(tasks_file: str, key: str, base_entry: Dict[str, Any]) -> Dict[str, List[float]]:
    """{task: [values of seeds 42, 43, 44]} — single-element lists (seed 42) when no extra seed files exist."""
    sources = [base_entry]
    for seed in TASKS_EXTRA_SEEDS:
        extra = load_json(_seed_tasks_file(tasks_file, seed))
        e = ((extra or {}).get("entries") or {}).get(key)
        if e and e.get("status") == "completed":
            ov = ((load_json(_seed_tasks_file(tasks_file, seed).replace(".json", "_mmlu.json")) or {}).get("entries") or {}).get(key)
            if tasks_file.endswith("_mmlu.json") and ov and (ov.get("mmlu") or {}).get("accuracy") is not None and e.get("mmlu_subset_acc") is None:
                e = {**e, "mmlu_subset_acc": ov["mmlu"]["accuracy"]}  # seed MMLU overlay (…_seed<N>_mmlu.json); files are not modified
            sources.append(e)
    out: Dict[str, List[float]] = {}
    for task, _ in TASK_PANELS:
        vals = [_task_value(e, task)[0] for e in sources]
        out[task] = [v for v in vals if v is not None]
    for task in ("hellaswag", "arc_challenge"):
        out[task + "_acc"] = [v for v in (((e.get("tasks") or {}).get(task) or {}).get("acc") for e in sources) if isinstance(v, (int, float))]
    return out


def _mean_std(vals: Sequence[float], spec: str = ".3f") -> str:
    if not vals:
        return "–"
    if len(vals) == 1:
        return format(vals[0], spec)
    return f"{np.mean(vals):{spec}} ± {np.std(vals, ddof=1):{spec}}"


def _significance_cells(tasks_file: str) -> Dict[str, str]:
    """
    {entry key: "HS yes · ARC no · MMLU yes (→ reference)"} — Holm-adjusted result of the PRIMARY family in the compare_tasks_paired
    output (α taken from the file). Row = the NON-reference side of the comparison. Without the file: empty dict (the column stays "–"
    and a 'not found' entry is logged).
    """
    from tools.evaluation import compare_tasks_paired as ctp

    path = ctp.paired_path_for(tasks_file)
    if not os.path.exists(path) and tasks_file.endswith("_mmlu.json"):  # the paired comparison of the MMLU file is stored under the name without '_mmlu'
        path = ctp.paired_path_for(tasks_file[: -len("_mmlu.json")] + ".json")
    d = load_json(path)
    if d is None:
        missing(f"gorevler: no paired-comparison file ({os.path.basename(path)}); significance column left empty")
        return {}
    by_row: Dict[str, Dict[str, Any]] = {}
    for pr in d.get("pairs", []):
        if pr.get("family") != "primary":
            continue
        a_ref = pr["a"].split("/")[-1] in SIGNIF_REFERENCES
        row, ref = (pr["b"], pr["a"]) if a_ref else (pr["a"], pr["b"])
        cell = by_row.setdefault(row, {"ref": ref.split("/")[-1], "tasks": {}})
        cell["tasks"][pr["task"]] = bool(pr.get("significant"))
    out: Dict[str, str] = {}
    for row, c in by_row.items():
        parts = [f"{short} {'yes' if c['tasks'][t] else 'no'}" for t, short in SIGNIF_TASKS if t in c["tasks"]]
        out[row] = " · ".join(parts) + f" (→ {MODEL2_LABELS.get(c['ref'], method_label(c['ref']))})"
    return out


def fig_gorevler(paths: Dict[str, str]) -> List[str]:
    d = load_json(TASKS_FILE)
    if d is None:
        return []
    entries = {k: e for k, e in d.get("entries", {}).items() if e.get("status") == "completed"}
    if not entries:
        return []
    default_source = os.path.normpath(TASKS_FILE) == os.path.normpath(os.path.join(RESULTS, "tasks_gun9.json"))
    table_sources = [TASKS_FILE]
    if default_source:  # the Wanda_ln task measurement (B-4b) lives in a separate file; existing entries are never overwritten
        extra = load_json(TASKS_WANDA_LN_FILE)
        for k, e in ((extra or {}).get("entries") or {}).items():
            if e.get("status") == "completed" and k not in entries:
                entries[k] = e
        if extra:
            table_sources.append(TASKS_WANDA_LN_FILE)
    random_vals = {k: _random_seed_values(TASKS_FILE, k, e) for k, e in entries.items() if e.get("config", k) == "prune_random"}
    n_random_seeds = max([len(v["hellaswag"]) for v in random_vals.values()] or [1])
    random_label = f" ({n_random_seeds} seeds, mean ± std)" if n_random_seeds > 1 else " (seed 42)"
    table_sources += [_seed_tasks_file(TASKS_FILE, s) for s in TASKS_EXTRA_SEEDS if os.path.exists(_seed_tasks_file(TASKS_FILE, s))]
    out_name = GOREVLER_OUT_NAME or ("gun9_gorevler" if default_source else "gun9_gorevler_" + os.path.splitext(os.path.basename(TASKS_FILE))[0])
    fp = entries.get("fp16")
    by_fraction = d.get("source", {}).get("plan_format") == "fractions"
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    handles: Dict[str, Any] = {}
    if by_fraction:
        for ax, (task, title) in zip(axes, TASK_PANELS):
            series: Dict[str, Tuple[List[float], List[float]]] = {}
            random_err: Dict[float, float] = {}
            for k, e in entries.items():
                v, _ = _task_value(e, task)
                if k in random_vals and len(random_vals[k][task]) > 1:  # 3 seeds -> mean + std
                    v = float(np.mean(random_vals[k][task]))
                    random_err[float(e["fraction"])] = float(np.std(random_vals[k][task], ddof=1))
                if e.get("fraction") is None or v is None:
                    continue
                xs, ys = series.setdefault(e["config"], ([], []))
                xs.append(float(e["fraction"]))
                ys.append(v)
            for name in GUN7_ORDER:
                if name not in series:
                    continue
                xs, ys = (list(t) for t in zip(*sorted(zip(*series[name]))))
                color, marker, _ = style(name)
                lab = method_label(name) + (random_label if name == "prune_random" else "")
                if name == "prune_random" and random_err:
                    ax.errorbar(xs, ys, yerr=[random_err.get(x, 0.0) for x in xs], color=color, ls="none", capsize=2.5, zorder=3)
                if name in SINGLE_POINT_CONFIGS:
                    h = ax.scatter(xs, ys, color=color, marker=marker, s=45, zorder=4, edgecolor="black", linewidth=0.6, label=lab)
                else:
                    h, = ax.plot(xs, ys, color=color, marker=marker, ls="--" if name == "prune_wanda" else "-", label=lab)  # only raw (unverified) Wanda is dashed; wanda_ln is solid
                handles.setdefault(lab, h)
            xt = sorted({x for s in series.values() for x in s[0]})
            ax.set_xticks(xt)
            ax.set_xticklabels([frac_label(x) for x in xt])
            ax.set_xlabel("Pruned-head ratio (block ratio)")
    else:  # 'configs' format (second model): one bar per configuration
        names = [k for k in entries if k != "fp16"]
        x = np.arange(len(names))
        for ax, (task, title) in zip(axes, TASK_PANELS):
            vals = [_task_value(entries[n], task)[0] for n in names]
            ax.bar(x, [v if v is not None else np.nan for v in vals], 0.6, color=[style(n)[0] for n in names], edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL2_LABELS.get(n, label(n)).replace(": ", ":\n").replace(", ", ",\n").replace(" (", "\n(") for n in names],
                               rotation=60, ha="right", fontsize=6.5)
    for ax, (task, title) in zip(axes, TASK_PANELS):
        ref = _task_value(fp, task)[0] if fp else None
        if ref is not None:
            h = ax.axhline(ref, color=C["black"], ls=":", lw=1, label="FP16 reference")
            handles.setdefault("FP16 reference", h)
            ax.annotate(f"FP16 {ref:.3f}", (0.99, ref), xycoords=("axes fraction", "data"), textcoords="offset points", xytext=(0, 2),
                        ha="right", va="bottom", fontsize=6.5)
        h = ax.axhline(0.25, color=C["gray"], ls="-.", lw=0.8, label="random baseline (≈0.25)")
        handles.setdefault("random baseline (≈0.25)", h)
        ax.set_ylabel(f"Accuracy — {title}", fontsize=8)  # figures have no titles, so the task name goes into the axis label
        ax.tick_params(labelleft=True)
    if by_fraction:
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.13, 1, 1))
    else:
        fig.tight_layout()
    outs = [save_fig(fig, paths["figures"], out_name, [TASKS_FILE])]

    rows: List[List[Any]] = []
    any_source_mmlu = False
    signif = _significance_cells(TASKS_FILE)
    if by_fraction:  # merged (Wanda_ln) entries go under their own ratio; order within a ratio is preserved (stable sort)
        ordered = sorted(entries.items(), key=lambda kv: -1.0 if kv[0] == "fp16" or kv[1].get("fraction") is None else float(kv[1]["fraction"]))
    else:
        ordered = list(entries.items())
    for key, e in ordered:
        t = e.get("tasks") or {}
        mm, from_src = _task_value(e, "mmlu")
        any_source_mmlu |= from_src
        name = e.get("config", key)
        lab = (MODEL2_LABELS.get(name, label(name)) if not by_fraction else method_label(name)) + (random_label if name == "prune_random" else "")
        cells = [fmt((t.get("hellaswag") or {}).get("acc"), ".3f"), fmt((t.get("hellaswag") or {}).get("acc_norm"), ".3f"),
                 fmt((t.get("arc_challenge") or {}).get("acc"), ".3f"), fmt((t.get("arc_challenge") or {}).get("acc_norm"), ".3f"), fmt(mm, ".3f")]
        if key in random_vals and n_random_seeds > 1:
            rv = random_vals[key]
            cells = [_mean_std(rv["hellaswag_acc"]), _mean_std(rv["hellaswag"]), _mean_std(rv["arc_challenge_acc"]), _mean_std(rv["arc_challenge"]),
                     _mean_std(rv["mmlu"])]
        rows.append([frac_label(e["fraction"]) if e.get("fraction") is not None and name != "fp16" else "–", lab, *cells[:4],
                     cells[4] + ("†" if from_src else ""), fmt((e.get("source") or {}).get("perplexity"), ".4f"),
                     e.get("n_pruned_heads", "–"), e.get("int4_modules", "–"), signif.get(key, "–")])
    note = (f"Model: {d.get('run', {}).get('model', '?')}; plans from {os.path.basename(str(d.get('source', {}).get('plan_file', '?')))} are applied "
            "unchanged from the source run (no attribution). HellaSwag / ARC-Challenge: 500 fixed examples each, 0-shot, option log-likelihood; "
            "acc = raw sum, acc_norm = normalised by the option's byte length (lm-eval-harness definition); ±4.4 points (95% CI) at 500 examples. "
            "ppl: WikiText-2 perplexity of the source run. "
            + ("Random pruning: seed 42 only." if n_random_seeds == 1 else
               f"Random pruning: {n_random_seeds} seeds (42 + extra seed files), mean ± std (n−1); the ppl column is the seed-42 value.")
            + (" † MMLU not measured in this run; value from the source run." if any_source_mmlu else "")
            + " Last column: exact McNemar test on the same 500 questions, Holm correction over the PRE-SPECIFIED primary family (xAI vs "
              "random, xAI vs Taylor, iterative vs single-shot; every ratio × task), α = 0.05; 'yes' = Holm-adjusted p < α, the compared "
              "reference in parentheses. '–' for rows outside the primary family (secondary comparisons are descriptive: "
              "results/tasks_gun9_paired*.json). The random row is tested with seed 42.")
    outs += write_table(paths["tables"], out_name, ["ratio", "configuration", "HellaSwag acc", "HellaSwag acc_norm", "ARC-C acc", "ARC-C acc_norm",
                                                    "MMLU", "ppl (source)", "heads", "INT4", "Holm-adjusted significant (α=0.05)"], rows,
                        table_sources, note, align="llrrrrrrrrl")
    return outs


# --------------------------------------------------------------------------- #
# Per-subject MMLU analysis (no GPU) — from the per-subject summaries in iterative_gun7.json (repeats[].mmlu.per_subject)
# --------------------------------------------------------------------------- #
SUBJECT_LABELS = {"abstract_algebra": "abstract algebra", "anatomy": "anatomy", "astronomy": "astronomy",
                  "college_computer_science": "college CS", "high_school_mathematics": "mathematics (HS)",
                  "philosophy": "philosophy", "world_religions": "world religions", "us_foreign_policy": "US foreign pol.",
                  "econometrics": "econometrics", "moral_scenarios": "moral scenarios"}
MMLU_KONU_ORDER = ["xai_single", "xai_iter_fixedq", "az_buda_cok_kuantize", "xai_single_4tier", "xai_iter_fixedq_4tier", "prune_taylor",
                   "prune_attnconf", "prune_random", "prune_wanda", "prune_wanda_ln"]


def mmlu_subject_matrix(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    {"subjects": [...], "n_per_subject": {...}, "fp16": {subject: acc}, "rows": [{"fraction", "config", "n_repeats", "acc": {subject: acc}, "total"}]}.
    Subject accuracy = pooled correct/n over the completed repeats (random: 3 seeds); no per-question records needed.
    """
    fp = (d.get("fp16_rerun", {}).get("mmlu") or {}).get("per_subject")
    if not fp:
        return None
    subjects = list(fp)

    def pooled(reps: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        per = [(r.get("mmlu") or {}).get("per_subject") for r in reps]
        per = [p for p in per if p]
        if not per:
            return None
        return {s: sum(p[s]["correct"] for p in per) / sum(p[s]["n"] for p in per) for s in subjects}

    rows: List[Dict[str, Any]] = []
    for fk in sorted(d.get("fractions", {}), key=float):
        for name in MMLU_KONU_ORDER:
            c = _cfg(d, fk, name)
            if not c:
                continue
            reps = [r for r in c.get("repeats", []) if r.get("status") == "completed"]
            acc = pooled(reps)
            if acc:
                rows.append({"fraction": fk, "config": name, "n_repeats": len(reps), "acc": acc, "total": c.get("mmlu_subset_acc_mean"),
                             "top_letter": top_letter(reps)})
    if not rows:
        return None
    return {"subjects": subjects, "n_per_subject": {s: fp[s]["n"] for s in subjects}, "fp16": {s: fp[s]["accuracy"] for s in subjects},
            "fp16_total": d["fp16_rerun"].get("mmlu_subset_acc"), "fp16_top_letter": top_letter([d["fp16_rerun"]]),
            "gold_letter_counts": d["fp16_rerun"]["mmlu"].get("gold_letter_counts"), "rows": rows}


def top_letter(reps: Sequence[Dict[str, Any]]) -> Optional[Tuple[str, float]]:
    """Most frequently predicted letter and its share (repeats pooled): a collapsed model locks onto one letter, so its accuracy equals
    that letter's share of the correct answers."""
    counts: Dict[str, int] = {}
    for r in reps:
        for k, v in ((r.get("mmlu") or {}).get("predicted_letter_counts") or {}).items():
            counts[k] = counts.get(k, 0) + int(v)
    total = sum(counts.values())
    if not total:
        return None
    k = max(counts, key=lambda x: counts[x])
    return k, counts[k] / total


def fig_mmlu_konu(paths: Dict[str, str]) -> List[str]:
    d = load_gun7_merged()
    m = mmlu_subject_matrix(d) if d else None
    if not m:
        return []
    subjects, rows = m["subjects"], m["rows"]
    fractions = sorted({r["fraction"] for r in rows}, key=float)
    heights = [max(sum(1 for r in rows if r["fraction"] == fk), 1) for fk in fractions]
    fig, axes = plt.subplots(len(fractions), 1, figsize=(8.2, 0.42 * sum(heights) + 1.9), gridspec_kw={"height_ratios": heights}, squeeze=False)
    im = None
    for ax, fk in zip(axes[:, 0], fractions):
        sub = [r for r in rows if r["fraction"] == fk]
        diff = np.array([[100.0 * (r["acc"][s] - m["fp16"][s]) for s in subjects] for r in sub])
        im = ax.imshow(diff, cmap="RdBu", vmin=-40, vmax=40, aspect="auto")
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                ax.text(j, i, f"{diff[i, j]:+.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(diff[i, j]) > 26 else "black")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([method_label(r["config"]) + (f" ({r['n_repeats']} seeds)" if r["n_repeats"] > 1 else "") for r in sub], fontsize=7)
        ax.set_ylabel(frac_label(fk))
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels([])
        ax.grid(False)
    axes[-1, 0].set_xticklabels([f"{SUBJECT_LABELS.get(s, s)}\n(FP16 {m['fp16'][s]:.2f})" for s in subjects], rotation=45, ha="right", fontsize=7)
    cbar = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy difference to FP16 (points)")
    outs = [save_fig(fig, paths["figures"], "gun9_mmlu_konu", [GUN7_FILE])]

    def tl(v: Optional[Tuple[str, float]]) -> str:
        return f"{v[0]} (%{100 * v[1]:.0f})" if v else "–"

    table = [["–", label("fp16")] + [fmt(m["fp16"][s], ".2f") for s in subjects] + [fmt(m["fp16_total"], ".3f"), tl(m["fp16_top_letter"])]]
    for r in rows:
        lab = method_label(r["config"]) + (f" ({r['n_repeats']} seeds)" if r["n_repeats"] > 1 else "")
        table.append([frac_label(r["fraction"]), lab] + [fmt(r["acc"][s], ".2f") for s in subjects] + [fmt(r["total"], ".3f"), tl(r["top_letter"])])
    n_s = sorted(set(m["n_per_subject"].values()))
    gold = m.get("gold_letter_counts") or {}
    n_gold = sum(gold.values()) or 1
    gold_note = (" Shares of the correct options: " + ", ".join(f"{k}: {v / n_gold:.3f}" for k, v in gold.items()) + " — the accuracy of a "
                 "model locked onto one letter (collapsed) equals that letter's share and measures no knowledge.") if gold else ""
    note = (f"Questions per subject: {'/'.join(str(n) for n in n_s)} (total {sum(m['n_per_subject'].values())}); 95% confidence interval for a single subject ≈ ±14 points "
            "(n=50, p≈0.5) — differences within ±14 points in the heatmap cannot be told apart from noise; for random pruning the 3 seeds are pooled "
            "(n=150, ≈ ±8 points). Source: per-subject summary stored per configuration (mmlu.per_subject); per-QUESTION records are not in iterative_gun7.json "
            "(run_eval_from_plans.py --with-mmlu stores them). Random baseline 0.25. 'most frequent prediction': most chosen letter and its share." + gold_note)
    outs += write_table(paths["tables"], "gun9_mmlu_konu", ["ratio", "configuration"] + [SUBJECT_LABELS.get(s, s) for s in subjects]
                        + ["total", "most frequent prediction"], table, [GUN7_FILE], note, align="ll" + "r" * (len(subjects) + 2))
    return outs


# --------------------------------------------------------------------------- #
# Combined stability table (IG step count + calibration sample)
# --------------------------------------------------------------------------- #
def fig_kararlilik(paths: Dict[str, str]) -> List[str]:
    """n_steps 8→16 (same passages) and passage_offset 0→16 (non-overlapping passages, n_steps 8) in one table; a column whose file is missing is skipped."""
    cols = [(h, load_json(f), f) for h, f in (("n_steps 8 → 16 (same 16 passages)", N16_FILE),
                                              ("passages 0–15 → 16–31 (n_steps 8)", OFFSET16_FILE))]
    cols = [(h, d, f) for h, d, f in cols if d]
    if not cols:
        return []

    def cell(d: Dict[str, Any], fn: Callable[[Dict[str, Any]], Any], spec: str = "") -> str:
        try:
            v = fn(d)
        except (KeyError, TypeError):
            v = None
        if v is None:
            return missing("kararlilik: missing field")
        return format(v, spec) if spec else str(v)

    spec_rows = [
        ("Calibration (run b)", lambda d: f"{d['meta_b']['calibration']['n_passages']} passages, offset "
                                          f"{d['meta_b']['calibration'].get('passage_offset', 0)}, {d['meta_b']['calibration']['n_valid_tokens']} tokens", ""),
        ("IG steps (a → b)", lambda d: f"{d['meta_a']['attribution']['n_steps']} → {d['meta_b']['attribution']['n_steps']}", ""),
        ("Spearman ρ, head (1024)", lambda d: d["metrics"]["spearman"]["heads_all"], ".4f"),
        ("Spearman ρ, MLP (32)", lambda d: d["metrics"]["spearman"]["mlp"], ".4f"),
        ("Pearson r, head", lambda d: d["metrics"]["pearson"]["heads_all"], ".4f"),
        ("Top-100 Jaccard (head)", lambda d: d["metrics"]["jaccard"]["heads_all"]["top100"], ".3f"),
        ("Top-200 Jaccard (head)", lambda d: d["metrics"]["jaccard"]["heads_all"]["top200"], ".3f"),
        ("Heads changing tier (plan 0.2/0.4/0.4)", lambda d: f"{d['tiers']['heads']['n_changed']}/{d['tiers']['heads']['n']} "
                                                             f"(%{100 * d['tiers']['heads']['changed_ratio']:.2f})", ""),
        ("Tier transitions", lambda d: ", ".join(f"{k}: {v}" for k, v in d["tiers"]["heads"]["transitions"].items()) or "none", ""),
        ("MLPs changing tier", lambda d: f"{d['tiers']['mlp']['n_changed']}/{d['tiers']['mlp']['n']}", ""),
        ("Pruned-set overlap (%20)", lambda d: f"{d['tiers']['plan']['pruned_heads_common']}/{d['tiers']['plan']['pruned_heads_a']} "
                                               f"(Jaccard {d['tiers']['plan']['pruned_set_jaccard']:.3f})", ""),
        ("INT4 modules (a / b)", lambda d: f"{d['tiers']['plan']['int4_modules_a']} / {d['tiers']['plan']['int4_modules_b']}", ""),
        ("Layers with a changed INT4 decision (attn / MLP)", lambda d: f"{d['tiers']['plan']['layers_attn_quant_changed']} / "
                                                                       f"{d['tiers']['plan']['layers_mlp_quant_changed']}", ""),
        ("Attribution time (s, a → b)", lambda d: f"{d['meta_a']['timing']['attribution_seconds']:.1f} → {d['meta_b']['timing']['attribution_seconds']:.1f}", ""),
    ]
    rows = [[name] + [cell(d, fn, spec) for _, d, _ in cols] for name, fn, spec in spec_rows]
    note = ("a = scores of results/importance_scores_gun3.json (WikiText-2 test, first 16 eligible paragraphs, n_steps 8, IG baseline token 0). "
            "The scores are insensitive to the number of IG steps and MODERATELY sensitive to the calibration sample: the ranking is largely "
            "preserved, but the pruned set and the most important heads partly change (right column; reported as a limitation). "
            "Sensitivity to the IG baseline was not measured.")
    return write_table(paths["tables"], "kararlilik", ["metric"] + [h for h, _, _ in cols], rows, [f for _, _, f in cols], note,
                       align="l" + "l" * len(cols))


# --------------------------------------------------------------------------- #
# LoRA recovery (run_lora_recovery.py) — before / after recovery
# --------------------------------------------------------------------------- #
LORA_FILE = os.path.join(RESULTS, "lora_recovery_gun10.json")


def fig_lora_telafi(paths: Dict[str, str]) -> List[str]:
    d = load_json(LORA_FILE)
    if d is None:
        return []
    done = {fk: r for fk, r in d.get("fractions", {}).items() if r.get("status") == "completed"}
    if not done:
        return []
    ref = d.get("reference", {})
    keys = sorted(done, key=lambda k: float(k) if k.replace(".", "", 1).isdigit() else 0.0)
    labels = [frac_label(k) if k.replace(".", "", 1).isdigit() else k for k in keys]
    x = np.arange(len(keys))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    panels = ((axes[0], [done[k].get("ppl_before") for k in keys], [done[k].get("ppl_after") for k in keys], ref.get("fp16_perplexity"),
               "Perplexity, WikiText-2 (log scale)", ".2f"),
              (axes[1], [(done[k].get("source") or {}).get("mmlu_subset_acc") for k in keys], [done[k].get("mmlu_after_acc") for k in keys],
               ref.get("fp16_mmlu_subset_acc"), "MMLU subset accuracy (500 questions)", ".3f"))
    for ax, before, after, fp16, ylabel, spec in panels:
        for off, vals, lab, color, hatch in ((-width / 2, before, "before recovery (XAI-JQP iterative)", C["orange"], ""),
                                             (width / 2, after, "after LoRA recovery", C["green"], "//")):
            ax.bar(x + off, [v if isinstance(v, (int, float)) else np.nan for v in vals], width, color=color, hatch=hatch,
                   edgecolor="black", linewidth=0.5, label=lab)
            for xi, v in zip(x + off, vals):
                if isinstance(v, (int, float)):
                    ax.annotate(format(v, spec), (xi, v), textcoords="offset points", xytext=(0, 2), ha="center", va="bottom", fontsize=6.5)
        if isinstance(fp16, (int, float)):
            ax.axhline(fp16, color=C["black"], ls=":", lw=1, label="FP16 reference")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Pruned-head ratio (block ratio)")
        ax.set_ylabel(ylabel)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].axhline(0.25, color=C["gray"], ls="-.", lw=0.8, label="random baseline (4 options)")
    handles, labs = axes[1].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    outs = [save_fig(fig, paths["figures"], "gun10_lora_telafi", [LORA_FILE])]

    rows: List[List[Any]] = []
    for k, lab in zip(keys, labels):
        r = done[k]
        tr, ad = r.get("train") or {}, r.get("adapter") or {}
        share = r.get("recovered_share_of_loss")
        rows.append([lab, fmt(r.get("ppl_before"), ".4f"), fmt(r.get("ppl_after"), ".4f"), fmt(r.get("ppl_recovered"), "+.4f"),
                     f"%{100 * share:.0f}" if isinstance(share, (int, float)) else "–",
                     fmt((r.get("source") or {}).get("mmlu_subset_acc"), ".3f"), fmt(r.get("mmlu_after_acc"), ".3f"),
                     f"{fmt(tr.get('loss_mean_first10'), '.3f')} → {fmt(tr.get('loss_mean_last10'), '.3f')}",
                     fmt(tr.get("seconds"), ".0f"), fmt(r.get("peak_vram_gb"), ".1f"),
                     fmt(ad.get("bytes_fp16", 0) / 1e6 if ad.get("bytes_fp16") else None, ".0f")])
    lo, tn = ref.get("lora", {}), ref.get("training", {})
    note = (f"SUPPLEMENTARY EXPERIMENT — the main claim of XAI-JQP is training-free. Plan: {ref.get('plan_config', '?')} (masked pruning + NF4), followed by QLoRA "
            f"r={lo.get('r', '?')}, alpha={lo.get('alpha', '?')}, dropout={lo.get('dropout', '?')}, targets q/k/v/o/gate/up/down; {tn.get('steps', '?')} steps × "
            f"{tn.get('batch_size', '?')} × {tn.get('seq_len', '?')} tokens, lr {tn.get('lr', '?')}, {tn.get('schedule', '?')}, seed {tn.get('seed', '?')}. "
            "Training on WikiText-2 TRAIN, perplexity on WikiText-2 TEST (same-domain advantage; MMLU as out-of-domain check). MMLU before = value "
            "from the fraction-sweep run. 'loss share' = (before − after) / (before − FP16). Pruned head slices stay zero throughout training (mask verified). "
            "The adapter is not merged; its size is reported separately (fp16 MB). Single seed, single run.")
    outs += write_table(paths["tables"], "gun10_lora_telafi", ["ratio", "ppl before", "ppl after", "Δ ppl", "loss share", "MMLU before", "MMLU after",
                                                                "training loss (first → last 10 steps)", "training (s)", "peak VRAM (GB)", "adapter (MB)"],
                        rows, [LORA_FILE], note)
    return outs


# --------------------------------------------------------------------------- #
# Mixed precision (D-1, FP16/NF4) + sub-4-bit bit allocation (D-2, HQQ) — perplexity vs GB
# --------------------------------------------------------------------------- #
MIXED_FILE = os.path.join(RESULTS, "mixed_precision_gun11.json")   # run_iterative_pruning.py mixed_* configurations (D-1)
HQQ_FILE = os.path.join(RESULTS, "mixed_bits_gun11.json")          # run_mixed_precision.py (D-2)
SPEED_FIXEDQ_FILE = os.path.join(RESULTS, "speed_gun8_fixedq.json")  # final model: physical_int4 (ppl + GB from the same run)
FOUR_TIER_FILE = os.path.join(RESULTS, "iterative_gun7_4tier.json")
MIXED_XAI_CURVE = ("mixed_xai_k5", "mixed_xai_k10", "mixed_xai_k20")
MIXED_STYLE = {  # name -> (label, colour, marker)
    "mixed_xai": ("mixed precision, xAI (k = 5/10/20)", C["orange"], "o"),
    "mixed_random_k10": ("mixed precision, random k=10 (mean ± std)", C["green"], "x"),
    "mixed_magnitude_k10": ("mixed precision, magnitude k=10", C["vermillion"], "P"),
    "mixed_wanda_ln_k10": ("mixed precision, Wanda_ln k=10", C["purple"], "X"),
}
HQQ_STYLE = {"xai": ("HQQ bit allocation, xAI", C["orange"], "v"), "uniform": ("HQQ uniform", C["sky"], "v"),
             "random": ("HQQ bit allocation, random (mean ± std)", C["green"], "v"), "magnitude": ("HQQ bit allocation, magnitude", C["vermillion"], "v")}


def pareto_gun11_rows() -> List[Dict[str, Any]]:
    """Shared rows of the pareto_gun11 figure and table; only values present in the JSONs (missing ones go to MISSING_LOG)."""
    rows: List[Dict[str, Any]] = []
    b = load_json(BASELINES_FILE) or {}
    fp16_ppl = (b.get("summary", {}).get("configs", {}).get("fp16") or {}).get("perplexity")
    for name in ("fp16", "nf4_uniform", "gptq_4bit", "awq_4bit"):
        r = b.get("summary", {}).get("configs", {}).get(name)
        if r and r.get("perplexity") is not None and r.get("model_gb") is not None:
            rows.append({"group": "baseline", "name": name, "label": label(name), "ppl": r["perplexity"], "std": None,
                         "mmlu": r.get("mmlu_subset_acc"), "gb": r["model_gb"], "detail": "", "source": BASELINES_FILE})
        else:
            missing(f"pareto_gun11: {name} (baselines_gun7.json summary)")
    m = load_json(MIXED_FILE)
    mcfgs = (((m or {}).get("fractions") or {}).get("0.2") or {}).get("configs", {})
    for name, c in mcfgs.items():
        if c.get("status") != "completed" or c.get("perplexity_mean") is None:
            missing(f"pareto_gun11: {name} not completed ({os.path.basename(MIXED_FILE)})")
            continue
        mp = c.get("mixed_precision") or {}
        nb = mp.get("n_fp16_blocks") or {}
        rows.append({"group": "mixed", "name": name, "label": f"mixed precision {mp.get('selector', '?')} k={mp.get('k_percent', '?')}",
                     "ppl": c["perplexity_mean"], "std": c.get("perplexity_std") if c.get("stochastic") else None,
                     "mmlu": c.get("mmlu_subset_acc_mean"), "gb": c.get("model_bytes_after_gb"),
                     "detail": f"FP16 blocks: {nb.get('attn', '?')} attn + {nb.get('mlp', '?')} MLP", "source": MIXED_FILE})
    h = load_json(HQQ_FILE)
    for name, c in ((h or {}).get("configs") or {}).items():
        if c.get("status") != "completed" or c.get("perplexity_mean") is None:
            missing(f"pareto_gun11: {name} not completed ({os.path.basename(HQQ_FILE)})")
            continue
        rows.append({"group": "hqq", "name": name, "label": c.get("label") or name, "selector": c.get("selector"),
                     "ppl": c["perplexity_mean"], "std": c.get("perplexity_std") if c.get("stochastic") else None,
                     "mmlu": c.get("mmlu_subset_acc_mean"), "gb": c.get("model_gb"),
                     "detail": f"mean {fmt(c.get('avg_bits_nominal'), '.2f')} bits nominal / {fmt(c.get('avg_bits_effective'), '.2f')} effective",
                     "source": HQQ_FILE})
    s = load_json(SPEED_FIXEDQ_FILE)
    pi = ((s or {}).get("summary") or {}).get("physical_int4") or {}
    if pi.get("perplexity") is not None and pi.get("model_gb") is not None:
        rows.append({"group": "xai_jqp", "name": "physical_int4", "label": "XAI-JQP final (%20 iterative, physical + INT4)", "ppl": pi["perplexity"],
                     "std": None, "mmlu": None, "gb": pi["model_gb"], "detail": "205 heads physically pruned + 129 INT4", "source": SPEED_FIXEDQ_FILE})
    else:
        missing("pareto_gun11: final physical_int4 (speed_gun8_fixedq.json summary)")
    g = load_json(GUN7_FILE)
    az = _cfg(g, "0.2", "az_buda_cok_kuantize") if g else None
    if az:
        rows.append({"group": "xai_jqp", "name": "az_buda_cok_kuantize", "label": "prune less / quantize more %20", "ppl": az["perplexity_mean"], "std": None,
                     "mmlu": az.get("mmlu_subset_acc_mean"), "gb": az.get("model_bytes_after_gb"), "detail": "102 heads masked + INT4",
                     "source": GUN7_FILE})
    t4 = load_json(FOUR_TIER_FILE)
    for name in ("xai_single_4tier", "xai_iter_fixedq_4tier"):
        c = _cfg(t4, "0.2", name) if t4 else None
        if c:
            rows.append({"group": "xai_jqp", "name": name, "label": label(name), "ppl": c["perplexity_mean"], "std": None,
                         "mmlu": c.get("mmlu_subset_acc_mean"), "gb": c.get("model_bytes_after_gb"),
                         "detail": f"{c.get('int4_modules')} INT4 + {c.get('int8_modules')} INT8 (masked)", "source": FOUR_TIER_FILE})
    for r in rows:
        r["delta"] = (r["ppl"] - fp16_ppl) if fp16_ppl is not None else None
    return rows


def fig_pareto_gun11(paths: Dict[str, str]) -> List[str]:
    if load_json(MIXED_FILE) is None and load_json(HQQ_FILE) is None:
        return []  # no mixed-precision / HQQ outputs: baseline_pareto already covers this, so it is not duplicated
    rows = [r for r in pareto_gun11_rows() if r["gb"] is not None]
    if not rows:
        return []
    by = {r["name"]: r for r in rows}
    fp16 = by.get("fp16")
    near = [r for r in rows if r["group"] == "mixed" or r["name"] in ("nf4_uniform", "gptq_4bit", "awq_4bit")
            or (r["group"] == "hqq" and fp16 is not None and r["ppl"] <= fp16["ppl"] * 1.05)]  # 4-bit region: zoomed in the right panel
    near_names = {r["name"] for r in near}
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(10.0, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    edge = {"edgecolor": "black", "linewidth": 0.5}

    def draw(a, zoom: bool) -> None:
        for r in rows:
            if r["group"] in ("baseline", "xai_jqp") and (not zoom or r["name"] in near_names):
                color, marker, _ = style(r["name"])
                kw = {} if marker in ("x", "+") else edge
                a.scatter(r["gb"], r["ppl"], color=color, marker=marker, s=55, zorder=3, **kw)
                if zoom or r["name"] not in near_names:  # the 4-bit cluster is unlabelled in the left panel (labels would overlap)
                    a.annotate(r["label"], (r["gb"], r["ppl"]), textcoords="offset points", xytext=(6, -8 if zoom else 4), fontsize=6.5,
                               ha="left", va="center")
        curve = [by[n] for n in ("nf4_uniform",) + MIXED_XAI_CURVE if n in by]
        if any(r["group"] == "mixed" for r in curve):
            lab, color, marker = MIXED_STYLE["mixed_xai"]
            a.plot([r["gb"] for r in curve], [r["ppl"] for r in curve], color=color, marker=marker, markersize=5, zorder=4, label=lab)
            if fp16 is not None and not zoom:  # k=100 = FP16: end of the curve (dotted; no measurements in between)
                a.plot([curve[-1]["gb"], fp16["gb"]], [curve[-1]["ppl"], fp16["ppl"]], color=color, linestyle=":", linewidth=1.0, zorder=2)
            if zoom:
                for r in curve[1:]:
                    a.annotate(f"k={r['name'].rsplit('_k', 1)[-1]}", (r["gb"], r["ppl"]), textcoords="offset points", xytext=(0, 7),
                               fontsize=6.5, ha="center")
        for name in ("mixed_random_k10", "mixed_magnitude_k10", "mixed_wanda_ln_k10"):
            if name in by:
                lab, color, marker = MIXED_STYLE[name]
                a.errorbar(by[name]["gb"], by[name]["ppl"], yerr=by[name]["std"] or None, color=color, marker=marker, markersize=6,
                           linestyle="none", capsize=2.5, zorder=5, label=lab)
        seen = set()
        for r in rows:
            if r["group"] != "hqq" or (zoom and r["name"] not in near_names):
                continue
            lab, color, marker = HQQ_STYLE.get(r.get("selector"), (r["label"], C["gray"], "v"))
            a.errorbar(r["gb"], r["ppl"], yerr=r["std"] or None, color=color, marker=marker, markersize=6, markeredgecolor="black",
                       markeredgewidth=0.5, linestyle="none", capsize=2.5, zorder=4, label=None if lab in seen else lab)
            seen.add(lab)
            if zoom or r["name"] not in near_names:
                a.annotate(r["label"], (r["gb"], r["ppl"]), textcoords="offset points", xytext=(5, -7), fontsize=5.8, ha="left", va="center")

    draw(ax, zoom=False)
    ax.set_yscale("log")
    ax.set_xlabel("Measured model size (GB)")
    ax.set_ylabel("Perplexity, WikiText-2 (log scale)")
    ax.set_xlim(min(r["gb"] for r in rows) - 0.8, max(r["gb"] for r in rows) + 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="y", which="minor", labelsize=6)
    ax.set_title("(a) overview", fontsize=8, loc="left")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=6)
    if near:
        draw(axz, zoom=True)
        lo = min([r["ppl"] for r in near] + ([fp16["ppl"]] if fp16 else []))
        hi = max(r["ppl"] + (r["std"] or 0.0) for r in near)
        pad = max(0.15 * (hi - lo), 1e-3)
        axz.set_ylim(lo - pad, hi + pad)
        axz.set_xlim(min(r["gb"] for r in near) - 0.3, max(r["gb"] for r in near) + 0.9)
        if fp16 is not None:
            axz.axhline(fp16["ppl"], color=C["black"], linestyle="--", linewidth=0.8, zorder=1)
            axz.annotate(f"FP16 {fp16['ppl']:.4f} ({fp16['gb']:.1f} GB)", (axz.get_xlim()[1], fp16["ppl"]), textcoords="offset points",
                         xytext=(-3, 4), fontsize=6.5, ha="right")
        axz.set_xlabel("Measured model size (GB)")
        axz.set_ylabel("Perplexity (linear scale)")
        axz.set_title("(b) 4-bit region (zoom)", fontsize=8, loc="left")
    else:
        axz.set_axis_off()
    fig.text(0.01, 0.035, "GB: parameters + buffers; HQQ includes W_q + scale + zero point, bnb points exclude quant_state.", fontsize=5.8,
             color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    sources = sorted({r["source"] for r in rows})
    outs = [save_fig(fig, paths["figures"], "pareto_gun11", sources)]
    table_rows = [[r["label"], fmt(r["ppl"], ".4f") + (f" ± {r['std']:.4f}" if r["std"] is not None else ""), fmt(r["delta"], "+.4f"),
                   fmt(r["mmlu"], ".3f"), fmt(r["gb"], ".3f"), r["detail"] or "–"] for r in rows]
    note = ("Mixed precision (D-1): no pruning; the top k% of modules within each type (attention block / MLP block) stay FP16, all other decoder "
            "Linear layers are NF4; k=0 is uniform NF4 (baselines_gun7.json, not re-run). The random row is the mean ± std over 3 seeds "
            "(same number of blocks per type). HQQ (D-2): group_size 64, axis 1; GB includes the W_q + scale + zero-point tensors "
            "(bnb rows exclude quant_state, a slight bias in favour of bnb). The final XAI-JQP row is pruning + INT4 (speed_gun8_fixedq.json); "
            "MMLU was not measured in that run.")
    return outs + write_table(paths["tables"], "pareto_gun11", ["method", "ppl", "Δ ppl (FP16)", "MMLU", "GB", "details"], table_rows, sources,
                              note, align="lrrrrl")


# --------------------------------------------------------------------------- #
# Two-domain perplexity (E-5) + C4-calibrated task rows + Qwen 10% (E-2) — writes NEW files only
# --------------------------------------------------------------------------- #
E5_FILES = [("WikiText-2", os.path.join(RESULTS, "tasks_gun15_e5_wikitext.json")), ("C4", os.path.join(RESULTS, "tasks_gun15_e5_c4.json"))]
E5_BOOTSTRAP_FILE = os.path.join(RESULTS, "ppl_bootstrap_gun15.json")
QWEN_F010_FILE = os.path.join(RESULTS, "model2_qwen_gun9_f010.json")
QWEN_F010_TASKS_FILE = os.path.join(RESULTS, "tasks_gun9_model2_qwen_gun9_f010.json")  # from run E-2b; task columns are "–" when absent
E5_METHODS = [("f0.2/xai_single", "XAI-JQP (single-shot)"), ("f0.2/xai_iter_fixedq", "XAI-JQP iterative (fixed INT4)"), ("f0.2/prune_random", "random (seed 42)")]


def fig_e5_iki_alan(paths: Dict[str, str]) -> List[str]:
    loaded = [(calib, file, load_json(file)) for calib, file in E5_FILES]
    if any(d is None for _, _, d in loaded):
        missing("e5_iki_alan: E-5 task files missing (tasks_gun15_e5_wikitext.json / _c4.json)")
        return []

    def cells(e: Dict[str, Any]) -> List[str]:
        s = e.get("summary") or {}
        return [fmt(((e.get("ppl") or {}).get(ds) or {}).get("perplexity"), ".4f") for ds in ("wikitext2", "c4")] + \
               [fmt((s.get(t) or {}).get("acc_norm"), ".3f") for t in ("hellaswag", "arc_challenge")] + [fmt(e.get("mmlu_subset_acc"), ".3f")]

    rows: List[List[Any]] = []
    bars: List[Tuple[str, float, float]] = []
    fp = loaded[0][2]["entries"].get("fp16")
    if fp:
        rows.append(["–", "FP16", "0"] + cells(fp))
        bars.append(("FP16", fp["ppl"]["wikitext2"]["perplexity"], fp["ppl"]["c4"]["perplexity"]))
    for calib, _, d in loaded:
        for key, name in E5_METHODS:
            e = d["entries"].get(key)
            if not e or e.get("status") != "completed":
                missing(f"e5_iki_alan: {calib} / {key} not completed")
                continue
            rows.append([calib, name, str(e.get("int4_modules", "–"))] + cells(e))
            bars.append((f"{name}\n[calib.: {calib}]", e["ppl"]["wikitext2"]["perplexity"], e["ppl"]["c4"]["perplexity"]))
    sources = [f for _, f, _ in loaded]
    boot = load_json(E5_BOOTSTRAP_FILE)
    ci = ""
    if boot:
        sources.append(E5_BOOTSTRAP_FILE)
        p = {(r["a"], r["b"], r["dataset"]): r for r in boot["pairs"]}
        r_w, r_c = p.get(("c4:f0.2/xai_single", "f0.2/xai_single", "wikitext2")), p.get(("c4:f0.2/xai_single", "f0.2/xai_single", "c4"))
        if r_w and r_c:
            ci = (f" Paired bootstrap (95% CI), C4-calibrated − WikiText-calibrated single-shot: WikiText-2 Δ {r_w['delta']:+.4f} "
                  f"[{r_w['ci95'][0]:+.4f}, {r_w['ci95'][1]:+.4f}], C4 Δ {r_c['delta']:+.4f} [{r_c['ci95'][0]:+.4f}, {r_c['ci95'][1]:+.4f}].")
    note = ("20% block ratio (205 heads), Mistral-7B-Instruct-v0.3. 'calibration' = domain on which the importance scores were computed; the "
            "perplexity columns are the EVALUATION domain (WikiText-2 test; C4 = fixed 256-document subset disjoint from the calibration data; "
            "window 1024 / stride 512 for both). HellaSwag / ARC: acc_norm, 500 examples, 0-shot; MMLU: 500-question subset. Random: single "
            "seed (42); the three-seed task row is in the gun9_gorevler table." + ci)
    out = write_table(paths["tables"], "gun15_e5_iki_alan", ["calibration", "method", "INT4 modules", "ppl WikiText-2", "ppl C4", "HellaSwag", "ARC-C", "MMLU"],
                      rows, sources, note, align="llrrrrrr")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(bars))
    for off, idx, lab, color in ((-0.19, 1, "WikiText-2 perplexity", "#4477AA"), (0.19, 2, "C4 perplexity", "#EE6677")):
        vals = [b[idx] for b in bars]
        ax.bar(x + off, vals, width=0.36, color=color, edgecolor="black", linewidth=0.6, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0].replace(" (", "\n(") for b in bars], fontsize=6.5)
    ax.set_ylabel("Perplexity (evaluation domain)")
    ax.set_ylim(0, max(max(b[1], b[2]) for b in bars) * 1.15)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out.append(save_fig(fig, paths["figures"], "gun15_e5_iki_alan", sources))

    q = load_json(QWEN_F010_FILE)
    if q is None:
        missing("e5_iki_alan: Qwen 10% file missing (model2_qwen_gun9_f010.json)")
        return out
    qt = (load_json(QWEN_F010_TASKS_FILE) or {}).get("entries", {})
    qrows: List[List[Any]] = []

    def task_cells(key: str) -> List[str]:
        s = (qt.get(key) or {}).get("summary") or {}
        return [fmt((s.get(t) or {}).get("acc_norm"), ".3f") for t in ("hellaswag", "arc_challenge")]

    base = q.get("baseline") or {}
    qrows.append(["uncompressed (bf16)", "0", "0", fmt(base.get("perplexity"), ".4f"), fmt(base.get("mmlu_subset_acc"), ".3f")] + task_cells("fp16"))
    for name, cfg in q.get("configs", {}).items():
        if cfg.get("status") != "completed":
            continue
        qrows.append([label(name), str(cfg.get("n_pruned_heads", "–")), str(cfg.get("int4_modules", "–")), fmt(cfg.get("perplexity_mean"), ".4f"),
                      fmt(cfg.get("mmlu_subset_acc_mean"), ".3f")] + task_cells(name))
    qnote = ("Qwen2.5-7B-Instruct, 10% block ratio (78 / 784 heads), bf16; perplexity on WikiText-2 (= calibration domain). Random: SINGLE seed "
             "(42; the best of the three seeds at 20%) — full three-seed table (E-2c / E-2b, two-domain ppl): tables/kapanis_qwen_tam. "
             "No post-hoc tuning was added to the method.")
    return out + write_table(paths["tables"], "gun16_qwen_f010", ["configuration", "heads", "INT4 modules", "ppl WikiText-2", "MMLU", "HellaSwag", "ARC-C"],
                             qrows, [QWEN_F010_FILE] + ([QWEN_F010_TASKS_FILE] if qt else []), qnote, align="lrrrrrr")


# --------------------------------------------------------------------------- #
# Final summary tables (NEW file names: tables/kapanis_*.md|.tex; existing tables are unchanged)
# A number absent from the JSONs is NEVER invented: the cell becomes "not measured" or "–" and is logged in MISSING_LOG.
# --------------------------------------------------------------------------- #
OLCULMEDI = "not measured"
KAP_E5_WT = os.path.join(RESULTS, "tasks_gun15_e5_wikitext.json")
KAP_E5_C4 = os.path.join(RESULTS, "tasks_gun15_e5_c4.json")
KAP_CALIB_C4_GUN9 = os.path.join(RESULTS, "calib_c4_gun9.json")
KAP_TASKS_CALIB_C4 = os.path.join(RESULTS, "tasks_gun9_calib_c4_gun9.json")
KAP_E3_FILE = os.path.join(RESULTS, "calib_c4_gun16_f06.json")
KAP_E3_TASKS = os.path.join(RESULTS, "tasks_gun9_calib_c4_gun16_f06.json")
KAP_PAIRED_MISTRAL = os.path.join(RESULTS, "tasks_gun9_paired.json")
KAP_PAIRED_QWEN = os.path.join(RESULTS, "tasks_gun9_paired_model2_qwen_gun9.json")
KAP_PAIRED_QWEN_F010 = os.path.join(RESULTS, "tasks_gun9_paired_model2_qwen_gun9_f010.json")
KAP_PAIRED_E3 = os.path.join(RESULTS, "tasks_gun9_paired_e3.json")
KAP_D2_TASKS = os.path.join(RESULTS, "tasks_gun9_mixed_bits_gun11.json")  # Chain C: D-2 tasks + per-question MMLU (+ _seed43/44)
KAP_4TIER_TASKS = os.path.join(RESULTS, "tasks_gun9_iterative_gun7_4tier.json")  # Chain C: C-3 tasks
KAP_PAIRED_D2 = os.path.join(RESULTS, "tasks_gun9_paired_mixed_bits_gun11.json")
KAP_PAIRED_4TIER = os.path.join(RESULTS, "tasks_gun9_paired_iterative_gun7_4tier.json")
KAP_BOOT_FILES = [("E-5 cross-domain (Mistral %20)", os.path.join(RESULTS, "ppl_bootstrap_gun15.json")),
                  ("E-2b Qwen %10", os.path.join(RESULTS, "ppl_bootstrap_gun16_qwen_f010.json")),
                  ("D-1 mixed precision", os.path.join(RESULTS, "ppl_bootstrap_gun16_d1.json")),
                  ("D-2 HQQ bit allocation", os.path.join(RESULTS, "ppl_bootstrap_gun16_d2.json"))]
KAP_GROUP_FILE = os.path.join(RESULTS, "group_ablation_gun15.json")
KAP_CALIB_DIAG_FILE = os.path.join(RESULTS, "diagnose_calib_domain_gun15.json")
KAP_D1_BOOT = os.path.join(RESULTS, "ppl_bootstrap_gun16_d1.json")
KAP_D2_BOOT = os.path.join(RESULTS, "ppl_bootstrap_gun16_d2.json")
KAP_QWEN_TASKS_MMLU = os.path.join(RESULTS, "tasks_gun9_model2_qwen_gun9_mmlu.json")
KAP_QWEN_F010_SEEDS = os.path.join(RESULTS, "model2_qwen_gun9_f010_seeds.json")
KAP_C4_COMPARE = os.path.join(RESULTS, "compare_wikitext_vs_c4_gun9.json")
KAP_SPEED_FILES = [("speed_gun7", "sdpa", os.path.join(RESULTS, "speed_gun7.json")), ("speed_gun8_eager", "eager", os.path.join(RESULTS, "speed_gun8_eager.json")),
                   ("speed_gun8_fixedq", "sdpa (fixedq)", os.path.join(RESULTS, "speed_gun8_fixedq.json")), ("speed_gun8_sdpa", "sdpa", os.path.join(RESULTS, "speed_gun8_sdpa.json")),
                   ("speed_gun8_sdpa_auto", "sdpa, wrapper auto", os.path.join(RESULTS, "speed_gun8_sdpa_auto.json")),
                   ("speed_gun12_qwen_physical (Qwen)", "sdpa", os.path.join(RESULTS, "speed_gun12_qwen_physical.json"))]
KAP_LAYER_GROUPS = [("L0-5", 0, 5), ("L6-10", 6, 10), ("L11-15", 11, 15), ("L16-20", 16, 20), ("L21-26", 21, 26), ("L27-31", 27, 31)]
KAP_SCRIPT_BY_PREFIX = [  # result-file prefix -> producing script (experiment inventory)
    ("ablation_gun6", "run_ablation_tests.py"), ("baselines_gun7", "run_baselines.py"), ("calib_c4", "run_iterative_pruning.py --calib-dataset c4"),
    ("e2e_prototype_gun5", "run_e2e_pipeline.py"), ("gun7_smoke", "run_iterative_pruning.py (smoke)"), ("iterative_gun7_4tier", "run_iterative_pruning.py (4 tiers)"),
    ("iterative_gun7_attnconf", "run_iterative_pruning.py (prune_attnconf)"), ("iterative_gun7_wanda_ln", "run_iterative_pruning.py (prune_wanda_ln)"),
    ("iterative_gun7", "run_iterative_pruning.py"), ("lora_recovery_gun10", "run_lora_recovery.py"), ("mixed_bits_gun11", "run_mixed_precision.py"),
    ("mixed_precision_gun11", "run_iterative_pruning.py (mixed_*)"), ("model2_qwen_gun9", "run_qwen_experiments.py"), ("speed_", "measure_speed.py"),
    ("tasks_gun15_e5", "run_eval_from_plans.py (E-5)"), ("tasks_gun9", "run_eval_from_plans.py"), ("importance_scores", "run_xai_on_mistral.py"),
    ("diagnose_qwen_kernel", "diagnose_qwen_kernel.py"),
]


def _ms(vals: Sequence[Any], spec: str = ".4f") -> str:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return _mean_std(vals, spec) if vals else "–"


def _entry_tasks(e: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    """(HellaSwag, ARC, MMLU) cells; 'not measured' when the entry is missing."""
    if not e:
        return (OLCULMEDI, OLCULMEDI, OLCULMEDI)
    hs = ((e.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm")
    arc = ((e.get("tasks") or {}).get("arc_challenge") or {}).get("acc_norm")
    mm = e.get("mmlu_subset_acc") if isinstance(e.get("mmlu_subset_acc"), (int, float)) else (e.get("mmlu") or {}).get("accuracy")
    return (fmt(hs, ".3f", OLCULMEDI), fmt(arc, ".3f", OLCULMEDI), fmt(mm, ".3f", OLCULMEDI))


def _entry_ppl(e: Optional[Dict[str, Any]], ds: str) -> Optional[float]:
    v = (((e or {}).get("ppl") or {}).get(ds) or {}).get("perplexity")
    return v if isinstance(v, (int, float)) else None


def _seed_entries(tasks_file: str, key: str, base_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Entries of the random control: seed 42 (main file or base_file) + seeds 43 / 44 (<source>_seed<N>.json); an MMLU overlay file is applied when present."""
    out: List[Dict[str, Any]] = []
    base = ((load_json(base_file or tasks_file) or {}).get("entries") or {}).get(key)
    if base and base.get("status") == "completed":
        out.append(base)
    for seed in TASKS_EXTRA_SEEDS:
        sf = tasks_file.replace(".json", f"_seed{seed}.json")
        e = ((load_json(sf) or {}).get("entries") or {}).get(key)
        if e and e.get("status") == "completed":
            e = dict(e)
            ov = ((load_json(sf.replace(".json", "_mmlu.json")) or {}).get("entries") or {}).get(key)  # E-4' overlay
            if ov and (ov.get("mmlu") or {}).get("accuracy") is not None and (e.get("mmlu") or {}).get("accuracy") is None:
                e["mmlu"] = ov["mmlu"]; e["mmlu_subset_acc"] = ov["mmlu"]["accuracy"]
            out.append(e)
    return out


def _tasks_mean_std(entries: Sequence[Dict[str, Any]]) -> Tuple[str, str, str]:
    def col(get):
        return _ms([get(e) for e in entries], ".3f")
    return (col(lambda e: ((e.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm")),
            col(lambda e: ((e.get("tasks") or {}).get("arc_challenge") or {}).get("acc_norm")),
            col(lambda e: e.get("mmlu_subset_acc") if isinstance(e.get("mmlu_subset_acc"), (int, float)) else (e.get("mmlu") or {}).get("accuracy")))


def fig_kapanis_ana_sonuc(paths: Dict[str, str]) -> List[str]:
    """(1) Main result: FP16 -> XAI-JQP iterative 20%, two models; size + two-domain ppl + three tasks."""
    e5 = load_json(KAP_E5_WT); g = load_json(GUN7_FILE); sp = load_json(SPEED_FIXEDQ_FILE)
    q = load_json(MODEL2_FILE); qt = (load_json(KAP_QWEN_TASKS_MMLU) or {}).get("entries") or {}
    qf = (load_json(QWEN_F010_TASKS_FILE) or {}).get("entries") or {}
    if e5 is None or g is None or q is None:
        return []
    rows: List[List[Any]] = []
    ent = e5.get("entries") or {}
    fp = ent.get("fp16"); it = ent.get("f0.2/xai_iter_fixedq")
    it_cfg = _cfg(g, "0.2", "xai_iter_fixedq") or {}
    fp_gb = fmt(((g.get("fp16_rerun") or {}).get("model_bytes_before") or {}).get("gb") or 14.496055296, ".3f")
    phys = ((sp or {}).get("summary") or {}).get("physical_int4") or {}
    if phys.get("model_gb") is None:
        missing("kapanis_ana_sonuc: physical + INT4 GB (speed_gun8_fixedq)")
    rows.append(["Mistral-7B-Instruct-v0.3", "FP16 (reference)", "0", "0", fp_gb, fp_gb, fmt(_entry_ppl(fp, "wikitext2"), ".4f", OLCULMEDI),
                 fmt(_entry_ppl(fp, "c4"), ".4f", OLCULMEDI), *_entry_tasks(fp)])
    rows.append(["Mistral-7B-Instruct-v0.3", "XAI-JQP iterative %20 (205 heads + 129 INT4)", str(it_cfg.get("n_pruned_heads", "–")), str(it_cfg.get("int4_modules", "–")),
                 fmt(it_cfg.get("model_bytes_after_gb"), ".3f"), fmt(phys.get("model_gb"), ".3f", OLCULMEDI),
                 fmt(_entry_ppl(it, "wikitext2"), ".4f", OLCULMEDI), fmt(_entry_ppl(it, "c4"), ".4f", OLCULMEDI), *_entry_tasks(it)])
    qb = q.get("baseline") or {}; qit = (q.get("configs") or {}).get("xai_iter_fixedq") or {}
    q_fp_gb = fmt((qb.get("model_bytes_before") or {}).get("gb") or ((q.get("configs") or {}).get("prune_only") or {}).get("model_bytes_after_gb"), ".3f")
    rows.append([_short_model(q.get("run", {}).get("model")), "FP16 (bf16, reference)", "0", "0", q_fp_gb, q_fp_gb, fmt(qb.get("perplexity"), ".4f"),
                 fmt(_entry_ppl(qf.get("fp16"), "c4"), ".4f", OLCULMEDI), *_entry_tasks(qt.get("fp16"))])
    rows.append([_short_model(q.get("run", {}).get("model")), "XAI-JQP iterative %20 (157 heads + 119 INT4)", str(qit.get("n_pruned_heads", "–")), str(qit.get("int4_modules", "–")),
                 fmt(qit.get("model_bytes_after_gb"), ".3f"), OLCULMEDI, fmt(qit.get("perplexity_mean"), ".4f"), OLCULMEDI, *_entry_tasks(qt.get("xai_iter_fixedq"))])
    note = ("Final configuration = xai_iter_fixedq (3 rescoring rounds; INT4 plan identical to single-shot). GB (masked) = parameters + buffers, excluding quant_state; "
            "GB (physical) = 205 heads physically pruned + INT4 (speed_gun8_fixedq.json). ppl: WikiText-2 test (calibration domain) and a fixed C4 subset (outside "
            "calibration), window 1024 / stride 512. HellaSwag / ARC acc_norm, MMLU acc; 500 examples, 0-shot. For Qwen, C4 perplexity and physical pruning + INT4 "
            "were not measured at 20% (C4 only in the 10% run); the Qwen FP16 C4 value comes from the 10% task file.")
    src = [KAP_E5_WT, GUN7_FILE, SPEED_FIXEDQ_FILE, MODEL2_FILE, KAP_QWEN_TASKS_MMLU, QWEN_F010_TASKS_FILE]
    return write_table(paths["tables"], "kapanis_ana_sonuc", ["model", "configuration", "heads", "INT4", "GB (masked)", "GB (physical)", "ppl WikiText-2", "ppl C4",
                                                            "HellaSwag", "ARC-C", "MMLU"], rows, src, note, align="llrrrrrrrrr")


def fig_kapanis_kontroller(paths: Dict[str, str]) -> List[str]:
    """(2) Controls: xAI single-shot / iterative, random (3 seeds), magnitude, Taylor, Wanda_ln, reverse, attention confidence × 3 ratios + C4-calibrated rows."""
    g = load_gun7_merged(); t = load_json(TASKS_FILE); ab = load_json(ABLATION_FILE)
    if g is None or t is None:
        return []
    tw = (load_json(TASKS_WANDA_LN_FILE) or {}).get("entries") or {}
    ent = dict(t.get("entries") or {}); ent.update(tw)
    rows: List[List[Any]] = []
    order = [("xai_single", "XAI-JQP (single-shot)"), ("xai_iter_fixedq", "XAI-JQP (iterative)"), ("prune_random", "random (3 seeds, mean ± std)"),
             ("prune_taylor", "Taylor"), ("prune_wanda_ln", "Wanda, within-layer z-score"), ("prune_attnconf", "attention confidence (Voita)")]
    for fk in ("0.2", "0.4", "0.6"):
        for name, lab in order:
            c = _cfg(g, fk, name)
            if not c:
                continue
            key = f"f{fk}/{name}"
            if name == "prune_random":
                ppl = _ms(c.get("perplexity_values") or [], ".4f")
                hs, arc, mm = _tasks_mean_std(_seed_entries(TASKS_FILE, key))
            else:
                ppl = fmt(c.get("perplexity_mean"), ".4f")
                hs, arc, mm = _entry_tasks(ent.get(key)) if name != "prune_attnconf" else (OLCULMEDI, OLCULMEDI, fmt(c.get("mmlu_subset_acc_mean"), ".3f"))
            rows.append([frac_label(fk), lab, str(c.get("int4_modules", "–")), ppl, hs, arc, mm])
        if fk == "0.2" and ab:
            for name, lab in (("prune_magnitude", "magnitude (pruning only, no INT4)"), ("prune_reverse", "reverse: highest xAI (pruning only, no INT4)")):
                c = (ab.get("configs") or {}).get(name)
                if c:
                    rows.append([frac_label(fk), lab, "0", fmt(c.get("perplexity_mean"), ".4f"), OLCULMEDI, OLCULMEDI, OLCULMEDI])
        else:
            for lab in ("magnitude", "reverse: highest xAI"):
                rows.append([frac_label(fk), lab, "–", OLCULMEDI, OLCULMEDI, OLCULMEDI, OLCULMEDI])
    t4 = load_json(FOUR_TIER_FILE); t4t = (load_json(KAP_4TIER_TASKS) or {}).get("entries") or {}  # Chain C: 4-tier tasks
    for name, lab in (("xai_single_4tier", "XAI-JQP single-shot, 4 tiers (+INT8)"), ("xai_iter_fixedq_4tier", "XAI-JQP iterative, 4 tiers (+INT8)")):
        c = _cfg(t4, "0.2", name) if t4 else None
        if c:
            rows.append(["%20", lab, f"{c.get('int4_modules', '–')} + {c.get('int8_modules', '–')} INT8", fmt(c.get("perplexity_mean"), ".4f"), *_entry_tasks(t4t.get(f"f0.2/{name}"))])
    c4 = load_json(KAP_CALIB_C4_GUN9); c4t = (load_json(KAP_TASKS_CALIB_C4) or {}).get("entries") or {}
    if c4:
        for name, lab in (("xai_single", "C4-calibrated XAI-JQP (single-shot)"), ("xai_iter_fixedq", "C4-calibrated XAI-JQP (iterative)"), ("prune_random", "C4-calibrated random (seed 42)")):
            c = _cfg(c4, "0.2", name)
            if c:
                rows.append(["%20", lab, str(c.get("int4_modules", "–")), fmt(c.get("perplexity_mean"), ".4f"), *_entry_tasks(c4t.get(f"f0.2/{name}"))])
    e3 = load_json(KAP_E3_FILE); e3t = (load_json(KAP_E3_TASKS) or {}).get("entries") or {}
    c = _cfg(e3, "0.6", "xai_iter_fixedq") if e3 else None
    if c:
        rows.append(["%60", "C4-calibrated XAI-JQP (iterative; E-3, bf16 fallback in round 3)", str(c.get("int4_modules", "–")), fmt(c.get("perplexity_mean"), ".4f"),
                     *_entry_tasks(e3t.get("f0.6/xai_iter_fixedq"))])
    note = ("Mistral-7B-Instruct-v0.3; ppl = WikiText-2 test; HellaSwag / ARC acc_norm, MMLU acc (500 examples). All rows within a ratio share the same "
            "head budget; the INT4 plan equals that of xAI single-shot (the magnitude / reverse rows come from the pruning-only ablation run, without INT4, "
            "tasks not measured; these two controls were not run at 40% / 60%). Random: perplexity and tasks as mean ± std over 3 seeds (42 / 43 / 44) (n−1). "
            "Attention confidence: tasks not measured. C4-calibrated rows: scores from C4 calibration, evaluation still on WikiText-2 (in-domain "
            "perplexity bias). Significance: tables/kapanis_holm_anlamlilik.")
    src = [GUN7_FILE, GUN7_WANDA_LN_FILE, GUN7_EXTRA_FILES[0], ABLATION_FILE, TASKS_FILE, TASKS_WANDA_LN_FILE, FOUR_TIER_FILE, KAP_4TIER_TASKS, KAP_CALIB_C4_GUN9,
           KAP_TASKS_CALIB_C4, KAP_E3_FILE, KAP_E3_TASKS]
    return write_table(paths["tables"], "kapanis_kontroller", ["ratio", "configuration", "INT4", "ppl WikiText-2", "HellaSwag", "ARC-C", "MMLU"], rows,
                       [s for s in src if os.path.exists(s)], note, align="llrrrrr")


def _holm_cell(pr: Optional[Dict[str, Any]]) -> str:
    if not pr:
        return "–"
    mark = "✓" if pr.get("significant") else "✗"
    return f"{pr['acc_a']:.3f} / {pr['acc_b']:.3f} · {pr['only_a']}/{pr['only_b']} · p {pr['p_adjusted']:.1e} {mark}"


def _pairs_index(d: Dict[str, Any], family: str = "primary") -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
    idx: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for pr in d.get("pairs", []):
        if pr.get("family") == family:
            idx.setdefault((pr["a"], pr["b"]), {})[pr["task"]] = pr
    return idx


def fig_kapanis_holm_anlamlilik(paths: Dict[str, str]) -> List[str]:
    """(3) Holm-adjusted significance, primary families: Mistral (45), Qwen 20% (15), Qwen 10% (12) + extra families (E-3, C-3, D-2)."""
    rows: List[List[Any]] = []
    srcs: List[str] = []
    for label_, path, fam in (("Mistral", KAP_PAIRED_MISTRAL, "primary"), ("Qwen %20", KAP_PAIRED_QWEN, "primary"), ("Qwen %10", KAP_PAIRED_QWEN_F010, "primary"),
                              ("Mistral %60 E-3 (extra)", KAP_PAIRED_E3, "extra"), ("Mistral %20 4 tiers C-3 (extra)", KAP_PAIRED_4TIER, "extra"),
                              ("Mistral D-2 HQQ (extra)", KAP_PAIRED_D2, "extra")):
        d = load_json(path)
        if d is None:
            missing(f"kapanis_holm_anlamlilik: {os.path.basename(path)} missing")
            continue
        srcs.append(path)
        idx = _pairs_index(d, fam)
        n_fam = (d.get("n_by_family") or {}).get(fam, len(idx) * 3)
        for (a, b), tasks in idx.items():
            rows.append([f"{label_} (family {n_fam})", a, b] + [_holm_cell(tasks.get(t)) for t in ("hellaswag", "arc_challenge", "mmlu")])
    if not rows:
        return []
    note = ("Cell: acc_A / acc_B · only A correct / only B correct · Holm-adjusted p (within family) and significance at α = 0.05 (✓ / ✗). Exact binomial "
            "McNemar on the same 500 questions. 's43:' / 's44:' = seed-43 / 44 entries of the random control (E-4, E-4', E-2c). For Qwen 20%, the MMLU of "
            "seeds 43 / 44 was overlaid from a separate file (mmlu_overlay_entries). The E-3, C-3 (4 tiers vs 3 tiers, 'g7:' = tasks_gun9.json) and D-2 "
            "(xAI vs random 3 seeds / magnitude / uniform, B = 3.0 and 3.5) rows form 'extra' families (--pair; Holm within each family) — they are not part "
            "of the pre-specified primary family and their evidential weight is descriptive.")
    return write_table(paths["tables"], "kapanis_holm_anlamlilik", ["model / family", "A", "B", "HellaSwag", "ARC-C", "MMLU"], rows, srcs, note, align="llllll")


def fig_kapanis_bootstrap_ga(paths: Dict[str, str]) -> List[str]:
    """(4) All paired window-bootstrap CIs in one table (E-5, Qwen 10%, D-1, D-2)."""
    rows: List[List[Any]] = []; srcs: List[str] = []
    for exp, path in KAP_BOOT_FILES:
        d = load_json(path)
        if d is None:
            missing(f"kapanis_bootstrap_ga: {os.path.basename(path)} missing")
            continue
        srcs.append(path)
        for pr in d.get("pairs", []):
            lo, hi = pr.get("ci95") or (None, None)
            rows.append([exp, pr.get("a"), pr.get("b"), pr.get("dataset"), fmt(pr.get("ppl_a"), ".4f"), fmt(pr.get("ppl_b"), ".4f"), fmt(pr.get("delta"), "+.4f"),
                         f"[{lo:+.4f}, {hi:+.4f}]" if isinstance(lo, (int, float)) else "–", "yes" if pr.get("excludes_zero") else "no", str(pr.get("n_windows", "–"))])
    if not rows:
        return []
    note = ("Δ = ppl_A − ppl_B; paired window bootstrap (10 000 resamples, seed 42), 95% percentile CI; 'excludes zero' = the CI does not contain 0. "
            "Window level (does not include the uncertainty of the calibration sample). E-5: WikiText-calibrated vs C4-calibrated plans; D-1 / D-2: "
            "xAI − control (negative = xAI better); Qwen 10%: pruning only − random (positive = random better).")
    return write_table(paths["tables"], "kapanis_bootstrap_ga", ["experiment", "A", "B", "domain", "ppl A", "ppl B", "Δ", "95% CI", "excludes zero", "windows"],
                       rows, srcs, note, align="llllrrrrlr")


def fig_kapanis_e7_grup_ablasyonu(paths: Dict[str, str]) -> List[str]:
    """(5) E-7: group-wise and joint ablation of the 47 C4-only heads (FP16, no quantization)."""
    d = load_json(KAP_GROUP_FILE)
    if d is None:
        return []
    groups = d.get("groups") or {}
    fp = (groups.get("fp16") or {}).get("ppl") or {}
    rows: List[List[Any]] = []
    order = ["fp16", "L0-5", "L11-15", "L16-20", "L21-26", "L27-31", "all_only_c4", "all_only_wikitext"]
    labels = {"fp16": "FP16 (reference)", "all_only_c4": "C4-only 47 heads (joint)", "all_only_wikitext": "WikiText-only 47 heads (joint)"}
    sum_wt = sum_c4 = 0.0
    for gname in order:
        gd = groups.get(gname)
        if not gd:
            continue
        p = gd.get("ppl") or {}; dl = gd.get("delta_vs_fp16") or {}
        rows.append([labels.get(gname, f"C4-only group {gname}"), str(gd.get("n_heads", "–")), fmt(p.get("wikitext2"), ".4f"), fmt(dl.get("wikitext2"), "+.4f"),
                     fmt(100 * dl["wikitext2"] / fp["wikitext2"], "+.1f") + " %" if fp.get("wikitext2") and dl.get("wikitext2") is not None else "–",
                     fmt(p.get("c4"), ".4f"), fmt(dl.get("c4"), "+.4f"),
                     fmt(100 * dl["c4"] / fp["c4"], "+.1f") + " %" if fp.get("c4") and dl.get("c4") is not None else "–"])
        if gname.startswith("L"):
            sum_wt += dl.get("wikitext2") or 0.0; sum_c4 += dl.get("c4") or 0.0
    if sum_wt:
        rows.append(["sum of Δ over the C4-only groups (additive expectation)", "47", "–", fmt(sum_wt, "+.4f"), "–", "–", fmt(sum_c4, "+.4f"), "–"])
    note = ("Mistral FP16, no quantization, masked pruning; every group measured separately after a clean reload. Groups = layer ranges of the 47 heads "
            "pruned only by the C4 calibration. The 'joint' row is ~2.7× the group sum (super-additive → redundancy). Δ% relative to FP16.")
    return write_table(paths["tables"], "kapanis_e7_grup_ablasyonu", ["group", "heads", "ppl WikiText-2", "Δ", "Δ %", "ppl C4", "Δ", "Δ %"], rows, [KAP_GROUP_FILE], note,
                       align="lrrrrrrr")


def fig_kapanis_alan_teshisi(paths: Dict[str, str]) -> List[str]:
    """(6) Domain diagnosis: pruned sets under WikiText vs C4 calibration (47 / 47 / 158), rank statistics and layer distribution."""
    d = load_json(KAP_CALIB_DIAG_FILE)
    if d is None:
        return []
    per = d.get("per_layer") or {}; ranks = d.get("ranks") or {}; ov = d.get("overlaps_with_gun6") or {}
    rows: List[List[Any]] = []
    for key, lab in (("only_c4", "pruned by C4 only"), ("only_wikitext", "pruned by WikiText only"), ("common", "pruned by both (shared)")):
        pl = per.get(key) or []; r = ranks.get(key) or {}
        groups = [str(sum(pl[a:b + 1])) if pl else "–" for _, a, b in KAP_LAYER_GROUPS]
        rows.append([lab, str(r.get("n", len(pl) and sum(pl))), fmt(r.get("median_rank_wikitext"), ".0f"), fmt(r.get("max_rank_wikitext"), ".0f"),
                     fmt(r.get("median_rank_c4"), ".0f"), fmt(r.get("max_rank_c4"), ".0f")] + groups +
                    [str((ov.get("prune_reverse") or {}).get(key, "–")), str((ov.get("prune_magnitude") or {}).get(key, "–"))])
    note = (f"20% (205 heads), Mistral; Jaccard {fmt(d.get('jaccard'), '.3f')}. Rank: 0 = least important head (by that calibration's scores); rank < 205 → "
            "pruned under that calibration; median / highest rank = position of the set under the other calibration (the C4-only set has medium-low "
            "importance under WikiText, not critical). Layer columns: number of heads of the set. Overlap: intersection with the reverse (top-205) and "
            "magnitude sets of the ablation run.")
    return write_table(paths["tables"], "kapanis_alan_teshisi", ["set", "n", "median rank (WikiText)", "highest rank (WikiText)", "median rank (C4)", "highest rank (C4)"]
                       + [g for g, _, _ in KAP_LAYER_GROUPS] + ["∩ reverse", "∩ magnitude"], rows, [KAP_CALIB_DIAG_FILE], note, align="l" + "r" * 13)


def _boot_lookup(path: str, a_key: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    d = load_json(path) or {}
    return {(pr["b"], pr["dataset"]): pr for pr in d.get("pairs", []) if pr.get("a") == a_key}


def _ga(pr: Optional[Dict[str, Any]]) -> str:
    if not pr:
        return "–"
    lo, hi = pr["ci95"]
    return f"{pr['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]{' *' if pr.get('excludes_zero') else ''}"


def fig_kapanis_d1(paths: Dict[str, str]) -> List[str]:
    """(7) D-1 mixed precision: configuration × (FP16 blocks, GB, two-domain ppl, MMLU, bootstrap Δ vs xAI k10)."""
    m = load_json(MIXED_FILE)
    cfgs = (((m or {}).get("fractions") or {}).get("0.2") or {}).get("configs") or {}
    if not cfgs:
        return []
    boot = _boot_lookup(KAP_D1_BOOT, "f0.2/mixed_xai_k10")
    rows: List[List[Any]] = []
    for name, c in cfgs.items():
        if c.get("status") != "completed":
            continue
        mp = c.get("mixed_precision") or {}; nb = mp.get("n_fp16_blocks") or {}
        reps = [r for r in c.get("repeats", []) if r.get("status") == "completed"]
        wt = _ms([_entry_ppl(r, "wikitext2") for r in reps]); c4 = _ms([_entry_ppl(r, "c4") for r in reps])
        mm = _ms([(r.get("mmlu") or {}).get("accuracy") for r in reps], ".3f")
        keys = [f"f0.2/{name}"] + [f"f0.2/{name}@s{r.get('seed')}" for r in reps]
        ga_wt = " · ".join(_ga(boot[(k, "wikitext2")]) for k in keys if (k, "wikitext2") in boot) or ("–" if name != "mixed_xai_k10" else "reference")
        ga_c4 = " · ".join(_ga(boot[(k, "c4")]) for k in keys if (k, "c4") in boot) or ("–" if name != "mixed_xai_k10" else "reference")
        rows.append([name, str(mp.get("selector", "–")), str(mp.get("k_percent", "–")), f"{nb.get('attn', '–')} + {nb.get('mlp', '–')}", fmt(c.get("model_bytes_after_gb"), ".3f"),
                     wt, c4, mm, ga_wt, ga_c4, OLCULMEDI, OLCULMEDI])
    note = ("NO PRUNING; the top-k% attention / MLP blocks of each type stay in FP16, the remaining decoder Linear layers use NF4 (double quantization). "
            "k=0 = uniform NF4 (control). Random: 3 seeds, mean ± std. Bootstrap: Δ = xAI k=10 − the row's configuration (negative = xAI better), 95% CI, "
            "* = excludes zero; for random, one CI per seed. The pre-registered criterion (xAI better than every seed and than magnitude) was not met. "
            "HellaSwag / ARC: NOT MEASURED (D-1 was skipped in Chain C; perplexity effect ≤ 0.01 ppl, configurations indistinguishable); the MMLU "
            "column is the per-subject summary of the source run.")
    return write_table(paths["tables"], "kapanis_d1_karisik_hassasiyet", ["config", "selector", "k %", "FP16 blocks (attn + MLP)", "GB", "ppl WikiText-2", "ppl C4", "MMLU",
                                                                          "Δ xAI k10 − config (WikiText)", "Δ (C4)", "HellaSwag", "ARC-C"], rows, [MIXED_FILE, KAP_D1_BOOT], note,
                       align="lllrrrrrllrr")


def fig_kapanis_d2(paths: Dict[str, str]) -> List[str]:
    """(8) D-2 HQQ: configuration × (budget, effective bits, tier counts, GB, two-domain ppl, MMLU, bootstrap Δ vs xAI at the same budget)."""
    h = load_json(HQQ_FILE)
    cfgs = (h or {}).get("configs") or {}
    if not cfgs:
        return []
    boot = {"3.0": _boot_lookup(KAP_D2_BOOT, "hqq_xai_b3.0"), "3.5": _boot_lookup(KAP_D2_BOOT, "hqq_xai_b3.5")}
    d2t = (load_json(KAP_D2_TASKS) or {}).get("entries") or {}  # Chain C: tasks + per-question MMLU (random: 3 seed files)
    rows: List[List[Any]] = []
    for name, c in cfgs.items():
        if c.get("status") != "completed":
            continue
        reps = [r for r in c.get("repeats", []) if r.get("status") == "completed"]
        if not d2t:
            task_cells = (OLCULMEDI, OLCULMEDI, OLCULMEDI)
        elif c.get("stochastic"):
            task_cells = _tasks_mean_std(_seed_entries(KAP_D2_TASKS, name))
        else:
            task_cells = _entry_tasks(d2t.get(name))
        bc = (c.get("bit_counts") or {}).get("attn") or {}
        tiers = " / ".join(str(bc.get(b, 0)) for b in ("8", "4", "3", "2"))
        b = str(c.get("budget_bits")); bl = boot.get(b) or boot.get("3.5" if "4bit" in name else "3.0") or {}
        keys = [name] + [f"{name}@s{r.get('seed')}" for r in reps]
        ga_wt = " · ".join(_ga(bl[(k, "wikitext2")]) for k in keys if (k, "wikitext2") in bl) or ("reference" if c.get("selector") == "xai" else "–")
        ga_c4 = " · ".join(_ga(bl[(k, "c4")]) for k in keys if (k, "c4") in bl) or ("reference" if c.get("selector") == "xai" else "–")
        rows.append([c.get("label") or name, str(c.get("selector", "–")), fmt(c.get("avg_bits_nominal"), ".2f"), fmt(c.get("avg_bits_effective"), ".2f"), tiers,
                     fmt(c.get("model_gb"), ".3f"), _ms(c.get("perplexity_values") or []), _ms([_entry_ppl(r, "c4") for r in reps]),
                     _ms(c.get("mmlu_subset_acc_values") or [], ".3f"), ga_wt, ga_c4, *task_cells])
    note = ("HQQ 0.2.8.post1, group_size 64, axis 1; decoder Linear modules (32 attn + 32 MLP); tier counts per type at 8 / 4 / 3 / 2 bits (attn = MLP). "
            "Budget B = parameter-weighted nominal mean bits; effective bits include packing + scale / zero point. GB includes the HQQ meta tensors "
            "(comparisons with bnb rows favour bnb). Random: 3 seeds, mean ± std. Bootstrap: Δ = xAI (same B) − the row's configuration, WikiText / C4, "
            "* = CI excludes zero; uniform 4-bit is NOT the same-budget control of B=3.5 (B=4.0). Task columns (Chain C): HellaSwag / ARC acc_norm, "
            "per-question MMLU; random 3 seeds, mean ± std; Holm (extra family, 30 tests): tables/kapanis_holm_anlamlilik. Result: in ppl xAI ≫ random "
            "≈ magnitude, uniform > xAI; on the tasks xAI ≈ uniform.")
    src = [HQQ_FILE, KAP_D2_BOOT] + ([KAP_D2_TASKS] + [_seed_tasks_file(KAP_D2_TASKS, s) for s in TASKS_EXTRA_SEEDS if os.path.exists(_seed_tasks_file(KAP_D2_TASKS, s))]
                                     if d2t else [])
    return write_table(paths["tables"], "kapanis_d2_bit_dagilimi", ["config", "selector", "B nominal", "effective bits", "8 / 4 / 3 / 2 bit (attn)", "GB", "ppl WikiText-2", "ppl C4",
                                                                    "MMLU", "Δ xAI − config (WikiText)", "Δ (C4)", "HellaSwag", "ARC-C", "MMLU (per-question)"], rows, src, note,
                       align="llrrrrrrrllrrr")


def fig_kapanis_kayip_ayrisimi(paths: Dict[str, str]) -> List[str]:
    """(9) Loss decomposition: pruning / quantization / interaction share — Mistral 20%, Qwen 20%, Qwen 10% (partial), Mistral 40% / 60% (not measured)."""
    ab = load_json(ABLATION_FILE); q = load_json(MODEL2_FILE); qf = load_json(QWEN_F010_FILE)
    if ab is None:
        return []
    rows: List[List[Any]] = []

    def row(model: str, frac: str, fp, prune, quant, both, src: str) -> List[Any]:
        if fp is None or both is None:
            return [model, frac, OLCULMEDI, OLCULMEDI, OLCULMEDI, OLCULMEDI, OLCULMEDI, OLCULMEDI, OLCULMEDI, src]
        inter = (both - prune - quant) if (prune is not None and quant is not None) else None
        share = (prune / (prune + quant)) if (prune is not None and quant is not None and prune + quant) else None
        return [model, frac, fmt(fp, ".4f"), fmt(prune, "+.4f", OLCULMEDI), fmt(quant, "+.4f", OLCULMEDI), fmt(both, "+.4f"), fmt(inter, "+.4f", OLCULMEDI),
                fmt(100 * share, ".1f") + " %" if share is not None else OLCULMEDI, fmt(100 * prune / both, ".1f") + " %" if prune is not None and both else OLCULMEDI, src]
    dc = ab.get("decomposition") or {}
    rows.append(row("Mistral-7B", "%20", dc.get("fp16"), dc.get("prune_delta"), dc.get("quant_delta"), dc.get("both_delta"), "ablation_gun6.json"))
    for fk in ("0.4", "0.6"):
        g7 = load_json(GUN7_FILE) or {}
        xs = _cfg(g7, fk, "xai_single") if g7.get("fractions") else None
        fp7 = (g7.get("fp16_rerun") or {}).get("perplexity")
        both7 = (xs["perplexity_mean"] - fp7) if (xs and isinstance(fp7, (int, float))) else None
        rows.append(["Mistral-7B", frac_label(fk), fmt(fp7, ".4f"), OLCULMEDI, OLCULMEDI, fmt(both7, "+.4f", OLCULMEDI), OLCULMEDI, OLCULMEDI, OLCULMEDI,
                     "iterative_gun7.json (no pruning-only run)"])
    if q:
        cp = q.get("comparison") or {}
        rows.append(row("Qwen2.5-7B", "%20", cp.get("baseline"), cp.get("prune_delta"), cp.get("quant_delta"), cp.get("both_delta"), "model2_qwen_gun9.json"))
    if qf:
        base = (qf.get("baseline") or {}).get("perplexity"); cf = qf.get("configs") or {}
        po = (cf.get("prune_only") or {}).get("perplexity_mean"); bo = (cf.get("both") or {}).get("perplexity_mean")
        if base and po and bo:
            rows.append(["Qwen2.5-7B", "%10", fmt(base, ".4f"), fmt(po - base, "+.4f"), OLCULMEDI + f" (both − pruning = {bo - po:+.4f})", fmt(bo - base, "+.4f"), OLCULMEDI,
                         OLCULMEDI, fmt(100 * (po - base) / (bo - base), ".1f") + " %", "model2_qwen_gun9_f010.json (no quantization-only run)"])
    note = ("Δ = perplexity − FP16 (WikiText-2). Interaction = both − pruning − quantization (0 = additive). Share 1 = pruning / (pruning + quantization), "
            "share 2 = pruning / both. For Mistral 40% / 60% and Qwen 10% the pruning-only / quantization-only runs required for the decomposition do not "
            "exist → not measured (not invented).")
    return write_table(paths["tables"], "kapanis_kayip_ayrisimi", ["model", "ratio", "FP16 ppl", "Δ pruning", "Δ quantization", "Δ both", "interaction", "share 1", "share 2", "source"],
                       rows, [ABLATION_FILE, GUN7_FILE, MODEL2_FILE, QWEN_F010_FILE], note, align="llrrrrrrrl")


def fig_kapanis_qwen_tam(paths: Dict[str, str]) -> List[str]:
    """(10) Full Qwen table: 20% + 10%, random with 3 seeds; two-domain ppl (C4 only at 10%), three tasks."""
    q = load_json(MODEL2_FILE); qf = load_json(QWEN_F010_FILE)
    if q is None:
        return []
    qt = (load_json(KAP_QWEN_TASKS_MMLU) or {}).get("entries") or {}
    rows: List[List[Any]] = []
    base = q.get("baseline") or {}
    rows.append(["–", "uncompressed (bf16)", "0", "0", fmt(((q.get("configs") or {}).get("prune_only") or {}).get("model_bytes_after_gb"), ".3f"), fmt(base.get("perplexity"), ".4f"),
                 OLCULMEDI, *_entry_tasks(qt.get("fp16"))])
    for name, c in (q.get("configs") or {}).items():
        if c.get("status") != "completed":
            continue
        if name == "prune_random":
            hs, arc, mm = _tasks_mean_std(_seed_entries(KAP_QWEN_TASKS_MMLU.replace("_mmlu.json", ".json"), name, base_file=KAP_QWEN_TASKS_MMLU))
            ppl = _ms(c.get("perplexity_values") or [])
        else:
            hs, arc, mm = _entry_tasks(qt.get(name)); ppl = fmt(c.get("perplexity_mean"), ".4f")
        rows.append(["%20", MODEL2_LABELS.get(name, label(name)) + (" (3 seeds, mean ± std)" if name == "prune_random" else ""), str(c.get("n_pruned_heads", "–")),
                     str(c.get("int4_modules", "–")), fmt(c.get("model_bytes_after_gb"), ".3f"), ppl, OLCULMEDI, hs, arc, mm])
    if qf:
        ft = (load_json(QWEN_F010_TASKS_FILE) or {}).get("entries") or {}
        seeds = load_json(KAP_QWEN_F010_SEEDS)
        for name, c in (qf.get("configs") or {}).items():
            if c.get("status") != "completed":
                continue
            if name == "prune_random":
                ents = _seed_entries(QWEN_F010_TASKS_FILE, name)
                vals = list(c.get("perplexity_values") or [])
                if seeds:
                    vals += list((((seeds.get("configs") or {}).get("prune_random") or {}).get("perplexity_values") or []))
                ppl = _ms(vals); c4 = _ms([_entry_ppl(e, "c4") for e in ents]); hs, arc, mm = _tasks_mean_std(ents)
                lab = MODEL2_LABELS.get(name, label(name)) + f" ({len(vals)} seeds, mean ± std)"
            else:
                e = ft.get(name); ppl = fmt(c.get("perplexity_mean"), ".4f"); c4 = fmt(_entry_ppl(e, "c4"), ".4f", OLCULMEDI); hs, arc, mm = _entry_tasks(e)
                lab = MODEL2_LABELS.get(name, label(name))
            rows.append(["%10", lab, str(c.get("n_pruned_heads", "–")), str(c.get("int4_modules", "–")), fmt(c.get("model_bytes_after_gb"), ".3f"), ppl, c4, hs, arc, mm])
    note = ("Qwen2.5-7B-Instruct (bf16); 20% = 157 / 784 heads + 119 INT4, 10% = 78 heads + 104 INT4 (plans of model2_qwen_gun9.json / model2_qwen_gun9_f010.json). "
            "ppl WikiText-2 = calibration domain; C4 measured only in the 10% run (E-2b). Random: 3 seeds (42 / 43 / 44), mean ± std; the 20% MMLU of seeds 43 / 44 "
            "comes from the E-4' overlay files. Significance: tables/kapanis_holm_anlamlilik (Qwen 20%: HellaSwag + MMLU significant against all three seeds; "
            "10%: no comparison significant).")
    src = [MODEL2_FILE, KAP_QWEN_TASKS_MMLU, QWEN_F010_FILE, QWEN_F010_TASKS_FILE, KAP_QWEN_F010_SEEDS]
    return write_table(paths["tables"], "kapanis_qwen_tam", ["ratio", "configuration", "heads", "INT4", "GB", "ppl WikiText-2", "ppl C4", "HellaSwag", "ARC-C", "MMLU"], rows,
                       [s for s in src if os.path.exists(s)], note, align="llrrrrrrrr")


def fig_kapanis_kararlilik(paths: Dict[str, str]) -> List[str]:
    """(11) Stability: score comparisons (n_steps, passage sample, calibration domain) + post-pruning drift Spearman correlations (surviving heads)."""
    rows: List[List[Any]] = []; srcs: List[str] = []
    for lab, path in (("IG steps 8 → 16 (same 16 passages)", N16_FILE), ("calibration passages 0–15 → 16–31 (n_steps 8)", OFFSET16_FILE),
                      ("calibration domain WikiText-2 → C4 (16 passages)", KAP_C4_COMPARE)):
        d = load_json(path)
        if d is None:
            missing(f"kapanis_kararlilik: {os.path.basename(path)} missing")
            continue
        srcs.append(path); m = d.get("metrics") or {}
        sp = m.get("spearman") or {}; pe = m.get("pearson") or {}; ja = (m.get("jaccard") or {}).get("heads_all") or {}
        rows.append(["score comparison", lab, fmt(sp.get("heads_all"), ".4f"), fmt(sp.get("mlp"), ".4f"), fmt(sp.get("all_blocks"), ".4f"), fmt(pe.get("heads_all"), ".4f"),
                     fmt(ja.get("top100"), ".3f"), fmt(ja.get("top200"), ".3f")])
    dr = load_json(DRIFT_FILE)
    if dr:
        srcs.append(DRIFT_FILE)
        ents = dr.get("entries") or []
        for fk in ("0.2", "0.4", "0.6"):
            def pick(cfg: str, rnd=None, seed=None):
                for e in ents:
                    if e.get("fraction") == fk and e.get("config") == cfg and (rnd is None or e.get("round") == rnd) and (seed is None or e.get("seed") == seed):
                        return e
                return None
            for cfg, lab, rnd in (("xai_single", "after XAI-JQP single-shot", None), ("xai_iter", "after XAI-JQP iterative, round 3", 3), ("prune_taylor", "after Taylor", None),
                                  ("prune_wanda_ln", "after Wanda_ln", None), ("prune_attnconf", "after attention confidence", None)):
                e = pick(cfg, rnd)
                if not e:
                    continue
                m = e.get("metrics") or {}; sp = m.get("spearman") or {}; ja = (m.get("jaccard") or {}).get("heads_surviving") or {}
                rows.append([f"drift {frac_label(fk)}", lab, fmt(sp.get("heads_surviving"), ".4f"), fmt(sp.get("mlp"), ".4f"), fmt(sp.get("all_surviving"), ".4f"), "–",
                             fmt(ja.get("top100"), ".3f"), fmt(ja.get("top200"), ".3f")])
            rs = [pick("prune_random", None, s) for s in (42, 43, 44)]
            rs = [e for e in rs if e]
            if rs:
                rows.append([f"drift {frac_label(fk)}", f"after random ({len(rs)} seeds, mean ± std)",
                             _ms([(e["metrics"].get("spearman") or {}).get("heads_surviving") for e in rs]), _ms([(e["metrics"].get("spearman") or {}).get("mlp") for e in rs]),
                             _ms([(e["metrics"].get("spearman") or {}).get("all_surviving") for e in rs]), "–",
                             _ms([((e["metrics"].get("jaccard") or {}).get("heads_surviving") or {}).get("top100") for e in rs], ".3f"),
                             _ms([((e["metrics"].get("jaccard") or {}).get("heads_surviving") or {}).get("top200") for e in rs], ".3f")])
    if not rows:
        return []
    note = ("Score-comparison rows: Spearman ρ (1024 heads / 32 MLPs / 1056 blocks), Pearson r (heads) and top-k Jaccard between the reference scores of "
            "results/importance_scores_gun3.json (WikiText-2, 16 passages, n_steps 8) and the stated change. Drift rows: between the rescoring AFTER pruning "
            "and round 0, surviving heads only (pruned heads excluded); Jaccard over the surviving heads. Random: 3 seeds, mean ± std.")
    return write_table(paths["tables"], "kapanis_kararlilik", ["type", "comparison", "ρ head", "ρ MLP", "ρ all blocks", "Pearson r (head)", "J100", "J200"], rows, srcs, note,
                       align="llrrrrrr")


def fig_kapanis_hiz(paths: Dict[str, str]) -> List[str]:
    """(12) Speed: all measurement files (speed_gun7 sdpa, speed_gun8 eager / fixedq / sdpa / auto, Qwen) in one table."""
    rows: List[List[Any]] = []; srcs: List[str] = []
    var_label = {"fp16": "FP16 (reference)", "masked": "masked pruning", "physical": "physical pruning", "physical_int4": "physical pruning + INT4"}
    for day, path_label, path in KAP_SPEED_FILES:
        d = load_json(path)
        if d is None:
            missing(f"kapanis_hiz: {os.path.basename(path)} missing")
            continue
        srcs.append(path)
        args = (d.get("run") or {}).get("args") or {}
        n_rep = args.get("n_repeats", 1); kernel = args.get("attn_kernel")
        for vname, v in (d.get("summary") or {}).items():
            ms = v.get("ms_per_token"); sd = v.get("ms_per_token_std")
            cell = f"{ms:.2f} ± {sd:.2f}" if isinstance(sd, (int, float)) and n_rep > 1 else fmt(ms, ".2f")
            kern = f" [kernel {(d.get('variants') or {}).get(vname, {}).get('attn_kernel')}]" if vname == "physical" and kernel else ""
            rows.append([day, path_label, var_label.get(vname, vname) + kern, cell, fmt(v.get("tokens_per_second"), ".2f"), (fmt(v.get("speedup_vs_fp16"), ".3f") + "×") if isinstance(v.get("speedup_vs_fp16"), (int, float)) else "–",
                         fmt(v.get("model_gb"), ".3f"), str(n_rep), _short_model((d.get("run") or {}).get("model") or args.get("model") or "Mistral-7B")])
    if not rows:
        return []
    note = ("L40S; 8 prompts × 64 new tokens (Qwen: 2 × 16), greedy, KV cache on; ms/token ± std over repeats (no std for a single repeat). Speedup relative to "
            "the FP16 row of the same file. 'path' = attn_implementation used when loading the fp16 / masked model; physically pruned layers run through the "
            "PrunedHeadAttention wrapper (eager kernel; sdpa with auto). Result: physical pruning stays within ±3% (no speed gain), INT4 (bnb) 0.80×.")
    return write_table(paths["tables"], "kapanis_hiz", ["run", "path", "variant", "ms/token", "tokens/s", "speedup", "GB", "repeats", "model"], rows, srcs, note, align="lllrrrrrl")


def fig_kapanis_deney_envanteri(paths: Dict[str, str]) -> List[str]:
    """(13) Experiment inventory: run records in results/*.json (script, model, date, duration, status) + attribution runs; total GPU hours."""
    import glob

    rows: List[List[Any]] = []; total_h = 0.0; n_runs = 0
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))) + sorted(glob.glob(os.path.join(RESULTS, "gun9_scores", "importance_scores_*.json"))):
        d = load_json(path)
        if not isinstance(d, dict):
            continue
        name = os.path.basename(path)
        run = d.get("run") if isinstance(d.get("run"), dict) else None
        script = next((s for pfx, s in KAP_SCRIPT_BY_PREFIX if name.startswith(pfx)), None)
        if run:
            sec = run.get("total_seconds")
            if isinstance(sec, (int, float)) and sec > 0:
                total_h += sec / 3600; n_runs += 1
            args = run.get("args") or {}
            rows.append([name, script or "–", _short_model(run.get("model") or args.get("model") or "–"), str(run.get("started_at") or "–")[:16], fmt(sec / 3600, ".2f") if isinstance(sec, (int, float)) else "–",
                         str(run.get("status") or "–")])
        elif name.startswith("importance_scores") and isinstance(d.get("timing"), dict):
            sec = (d["timing"].get("attribution_seconds") or 0) + (d["timing"].get("model_load_seconds") or 0)
            total_h += sec / 3600; n_runs += 1
            rows.append([name, script or "run_xai_on_mistral.py", _short_model(d.get("model") or "–"), "–", fmt(sec / 3600, ".2f"), "completed (attribution)"])
    rows.append(["**total (recorded run durations)**", f"{n_runs} runs", "–", "–", fmt(total_h, ".2f"), "–"])
    note = ("Duration = JSON run.total_seconds (wall clock = GPU time, single L40S) or attribution + loading time; smoke / preflight / dry-run outputs and idle "
            "pod time are not included → the actual pod rental exceeds this total (no hourly-rate record exists → cost not computed, not invented). "
            "Derived files (paired, bootstrap, diagnose, drift, subset ids) need no GPU and are not listed.")
    return write_table(paths["tables"], "kapanis_deney_envanteri", ["result file", "script", "model", "start", "duration (h)", "status"], rows,
                       [os.path.join(RESULTS, "*.json")], note, align="llllrl")


KAPANIS_TABLES = {
    "kapanis_ana_sonuc": (fig_kapanis_ana_sonuc, KAP_E5_WT),
    "kapanis_kontroller": (fig_kapanis_kontroller, GUN7_FILE),
    "kapanis_holm_anlamlilik": (fig_kapanis_holm_anlamlilik, KAP_PAIRED_MISTRAL),
    "kapanis_bootstrap_ga": (fig_kapanis_bootstrap_ga, KAP_BOOT_FILES[0][1]),
    "kapanis_e7_grup_ablasyonu": (fig_kapanis_e7_grup_ablasyonu, KAP_GROUP_FILE),
    "kapanis_alan_teshisi": (fig_kapanis_alan_teshisi, KAP_CALIB_DIAG_FILE),
    "kapanis_d1_karisik_hassasiyet": (fig_kapanis_d1, MIXED_FILE),
    "kapanis_d2_bit_dagilimi": (fig_kapanis_d2, HQQ_FILE),
    "kapanis_kayip_ayrisimi": (fig_kapanis_kayip_ayrisimi, ABLATION_FILE),
    "kapanis_qwen_tam": (fig_kapanis_qwen_tam, MODEL2_FILE),
    "kapanis_kararlilik": (fig_kapanis_kararlilik, N16_FILE),
    "kapanis_hiz": (fig_kapanis_hiz, KAP_SPEED_FILES[0][2]),
    "kapanis_deney_envanteri": (fig_kapanis_deney_envanteri, GUN7_FILE),
}


# --------------------------------------------------------------------------- #
# Final summary figures (NEW file names: figures/kapanis_*.png; existing figures are unchanged)
# --------------------------------------------------------------------------- #
KAP_FAMILY_STYLE = {  # point family -> (label, colour)
    "xai": ("XAI-JQP (single-shot / iterative)", C["blue"]), "xai_c4": ("XAI-JQP, C4 calibration", C["orange"]), "random": ("random (per seed)", C["green"]),
    "taylor": ("Taylor", C["purple"]), "wanda_ln": ("Wanda, within-layer z-score", C["vermillion"]), "wanda": ("Wanda (raw)", C["gray"]),
    "az_buda": ("prune less / quantize more", C["yellow"]), "lora": ("XAI-JQP + LoRA recovery", C["sky"]), "attnconf": ("attention confidence (Voita)", "#8C564B"),
    "d2": ("HQQ bit allocation (D-2, no pruning)", "#17BECF"), "d1": ("mixed precision (D-1, no pruning)", "#BCBD22"), "uniform": ("uniform 4-bit (NF4 / GPTQ / AWQ)", C["black"]),
}
KAP_MODEL_MARKER = {"Mistral": "o", "Qwen": "s"}


def _kap_point(model: str, family: str, label_: str, fp_ppl: float, ppl: Any, hs: Any, mmlu: Any, **extra) -> Optional[Dict[str, Any]]:
    if not isinstance(ppl, (int, float)) or not isinstance(fp_ppl, (int, float)):
        return None
    return {"model": model, "family": family, "label": label_, "dppl": ppl - fp_ppl, "ratio": ppl / fp_ppl,
            "dhs": (hs - extra.pop("fp_hs")) if isinstance(hs, (int, float)) and isinstance(extra.get("fp_hs"), (int, float)) else None,
            "dmmlu": (mmlu - extra.pop("fp_mmlu")) if isinstance(mmlu, (int, float)) and isinstance(extra.get("fp_mmlu"), (int, float)) else None, **extra}


def kapanis_points() -> Tuple[List[Dict[str, Any]], List[str]]:
    """All (Δppl WikiText, ΔHellaSwag, ΔMMLU) points for the perplexity–task divergence; two models, all configurations; only values present in the JSONs."""
    pts: List[Dict[str, Any]] = []; srcs: List[str] = []

    def fam(name: str) -> str:
        n = name.split("/")[-1]
        if n.startswith("prune_random"):
            return "random"
        if n in ("prune_taylor",):
            return "taylor"
        if n == "prune_wanda_ln":
            return "wanda_ln"
        if n == "prune_wanda":
            return "wanda"
        if n == "az_buda_cok_kuantize":
            return "az_buda"
        if n == "prune_attnconf":
            return "attnconf"
        return "xai"

    def tasks_file_points(model: str, path: str, fp_from: Optional[str] = None, family_override: Optional[str] = None, tag: str = "", skip: Sequence[str] = ()) -> None:
        d = load_json(path)
        if d is None:
            return
        srcs.append(path)
        ents = {k: e for k, e in (d.get("entries") or {}).items() if e.get("status") == "completed"}
        fp = ents.get("fp16") or ((load_json(fp_from) or {}).get("entries") or {}).get("fp16") if fp_from else ents.get("fp16")
        if not fp:
            return
        fp_ppl = (fp.get("source") or {}).get("perplexity"); fp_hs = ((fp.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm"); fp_mm = fp.get("mmlu_subset_acc")
        for key, e in ents.items():
            if key == "fp16" or key.split("/")[-1] in skip:
                continue
            frac = key.split("/")[0] if "/" in key else None
            lab = (frac_label(frac[1:]) + " " if frac else "") + MODEL2_LABELS.get(key.split("/")[-1], label(key.split("/")[-1])) + tag
            p = _kap_point(model, family_override or fam(key), lab, fp_ppl, (e.get("source") or {}).get("perplexity"), ((e.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm"),
                           e.get("mmlu_subset_acc"), fp_hs=fp_hs, fp_mmlu=fp_mm, seed=e.get("seed"), key=key)
            if p:
                pts.append(p)

    # Mistral: task results (+ Wanda_ln, random seeds 43 / 44, C4-calibrated, E-3, LoRA)
    tasks_file_points("Mistral", TASKS_FILE, skip=("prune_reverse",))
    tasks_file_points("Mistral", TASKS_WANDA_LN_FILE, fp_from=TASKS_FILE)
    for seed in TASKS_EXTRA_SEEDS:
        tasks_file_points("Mistral", TASKS_FILE.replace(".json", f"_seed{seed}.json"), fp_from=TASKS_FILE, tag=f" (s{seed})")
    tasks_file_points("Mistral", KAP_TASKS_CALIB_C4, fp_from=TASKS_FILE, family_override="xai_c4", tag=" [C4-calibrated]")
    tasks_file_points("Mistral", KAP_E3_TASKS, fp_from=TASKS_FILE, family_override="xai_c4", tag=" [C4-calibrated, E-3]")
    lora = load_json(LORA_FILE); t9 = load_json(TASKS_FILE)
    if lora and t9:
        srcs.append(LORA_FILE)
        fp = (t9.get("entries") or {}).get("fp16") or {}
        fp_ppl = (fp.get("source") or {}).get("perplexity"); fp_hs = ((fp.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm"); fp_mm = fp.get("mmlu_subset_acc")
        for fk, f in (lora.get("fractions") or {}).items():
            if f.get("status") != "completed":
                continue
            p = _kap_point("Mistral", "lora", f"{frac_label(fk)} iterative + LoRA", fp_ppl, f.get("ppl_after"), ((f.get("tasks_after_summary") or {}).get("hellaswag") or {}).get("acc_norm"),
                           f.get("mmlu_after_acc"), fp_hs=fp_hs, fp_mmlu=fp_mm, key=f"lora/{fk}")
            if p:
                pts.append(p)
    # Qwen: 20% (per-question MMLU file + seeds 43 / 44 + MMLU overlay) and 10% (+ seeds 43 / 44)
    q20 = KAP_QWEN_TASKS_MMLU; q20_base = q20.replace("_mmlu.json", ".json")
    tasks_file_points("Qwen", q20, tag=" [Qwen %20]", skip=("prune_reverse",))
    for seed in TASKS_EXTRA_SEEDS:
        sf = q20_base.replace(".json", f"_seed{seed}.json"); d = load_json(sf)
        if not d:
            continue
        srcs.append(sf)
        e = (d.get("entries") or {}).get("prune_random")
        ov = ((load_json(sf.replace(".json", "_mmlu.json")) or {}).get("entries") or {}).get("prune_random") or {}
        fp = ((load_json(q20) or {}).get("entries") or {}).get("fp16") or {}
        if e:
            p = _kap_point("Qwen", "random", f"random (s{seed}) [Qwen %20]", (fp.get("source") or {}).get("perplexity"), (e.get("source") or {}).get("perplexity"),
                           ((e.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm"), (ov.get("mmlu") or {}).get("accuracy"),
                           fp_hs=((fp.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm"), fp_mmlu=fp.get("mmlu_subset_acc"), seed=seed, key=f"q20/prune_random@s{seed}")
            if p:
                pts.append(p)
    tasks_file_points("Qwen", QWEN_F010_TASKS_FILE, tag=" [Qwen %10]")
    for seed in TASKS_EXTRA_SEEDS:
        tasks_file_points("Qwen", QWEN_F010_TASKS_FILE.replace(".json", f"_seed{seed}.json"), fp_from=QWEN_F010_TASKS_FILE, tag=f" (s{seed}) [Qwen %10]")
    # No-pruning points (MMLU only): D-2 HQQ, D-1, uniform 4-bit baselines, attention confidence / 4 tiers (MMLU of the fraction sweep)
    g7 = load_json(GUN7_FILE); fp7 = (g7 or {}).get("fp16_rerun") or {}
    fp_ppl7 = fp7.get("perplexity"); fp_mm7 = fp7.get("mmlu_subset_acc")
    h = load_json(HQQ_FILE); d2t = load_json(KAP_D2_TASKS)
    fp9 = ((load_json(TASKS_FILE) or {}).get("entries") or {}).get("fp16") or {}
    fp_hs7 = ((fp9.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm")
    if h and fp_ppl7:
        srcs.append(HQQ_FILE)
        if d2t:
            srcs.append(KAP_D2_TASKS)
        for name, c in (h.get("configs") or {}).items():
            if c.get("status") != "completed":
                continue
            fam_ = "uniform" if c.get("selector") == "uniform" else "d2"
            tes = _seed_entries(KAP_D2_TASKS, name) if d2t else []
            if tes:  # Chain C: tasks were measured -> HellaSwag + per-question MMLU points (per seed)
                for e in tes:
                    p = _kap_point("Mistral", fam_, (c.get("label") or name) + (f" (s{e.get('seed')})" if c.get("stochastic") else ""), fp_ppl7,
                                   (e.get("source") or {}).get("perplexity"), ((e.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm"), e.get("mmlu_subset_acc"),
                                   fp_hs=fp_hs7, fp_mmlu=fp_mm7, key=f"d2/{name}@s{e.get('seed')}")
                    if p:
                        pts.append(p)
                continue
            for i, (pv, mv) in enumerate(zip(c.get("perplexity_values") or [], c.get("mmlu_subset_acc_values") or [])):
                p = _kap_point("Mistral", fam_, (c.get("label") or name) + (f" (s{42 + i})" if len(c.get("perplexity_values") or []) > 1 else ""), fp_ppl7, pv, None, mv,
                               fp_hs=None, fp_mmlu=fp_mm7, key=f"d2/{name}@{i}")
                if p:
                    pts.append(p)
    m = load_json(MIXED_FILE)
    if m and fp_ppl7:
        srcs.append(MIXED_FILE)
        for name, c in ((((m.get("fractions") or {}).get("0.2") or {}).get("configs")) or {}).items():
            if c.get("status") != "completed":
                continue
            for i, (pv, mv) in enumerate(zip(c.get("perplexity_values") or [], c.get("mmlu_subset_acc_values") or [])):
                p = _kap_point("Mistral", "d1", f"mixed precision {name.split('mixed_')[-1]}", fp_ppl7, pv, None, mv, fp_hs=None, fp_mmlu=fp_mm7, key=f"d1/{name}@{i}")
                if p:
                    pts.append(p)
    b = load_json(BASELINES_FILE)
    if b and fp_ppl7:
        srcs.append(BASELINES_FILE)
        for name in ("nf4_uniform", "gptq_4bit", "awq_4bit"):
            r = ((b.get("summary") or {}).get("configs") or {}).get(name) or {}
            p = _kap_point("Mistral", "uniform", label(name), fp_ppl7, r.get("perplexity"), None, r.get("mmlu_subset_acc"), fp_hs=None, fp_mmlu=fp_mm7, key=f"base/{name}")
            if p:
                pts.append(p)
    if g7 and fp_ppl7:
        merged = load_gun7_merged() or g7
        t4t = (load_json(KAP_4TIER_TASKS) or {}).get("entries") or {}
        if t4t:
            srcs.append(KAP_4TIER_TASKS)
        for fk in ("0.2", "0.4", "0.6"):
            for name, fam_ in (("prune_attnconf", "attnconf"), ("xai_single_4tier", "xai"), ("xai_iter_fixedq_4tier", "xai")):
                c = _cfg(merged, fk, name)
                if c:
                    te = t4t.get(f"f{fk}/{name}")  # Chain C: HellaSwag + per-question MMLU when the 4-tier tasks exist
                    hs = ((te or {}).get("tasks") or {}).get("hellaswag", {}).get("acc_norm") if te else None
                    mm = te.get("mmlu_subset_acc") if te else c.get("mmlu_subset_acc_mean")
                    p = _kap_point("Mistral", fam_, f"{frac_label(fk)} {label(name)}", fp_ppl7, c.get("perplexity_mean"), hs, mm, fp_hs=fp_hs7 if te else None,
                                   fp_mmlu=fp_mm7, key=f"g7/{fk}/{name}")
                    if p:
                        pts.append(p)
    return pts, sorted(set(srcs))


KAP_ANNOTATE_KEYS = ("f0.2/prune_wanda_ln", "f0.4/prune_wanda_ln", "f0.6/xai_iter_fixedq", "f0.6/xai_single", "lora/0.2", "lora/0.6", "f0.4/prune_random",
                     "q20/prune_random@s43", "f0.2/prune_random", "d2/hqq_xai_b3.0@0", "d2/hqq_random_b3.0@0", "d2/hqq_uniform_3bit@0")


def _kap_scatter(paths: Dict[str, str], ykey: str, ylabel: str, name: str, annotate_extra: Sequence[str]) -> List[str]:
    pts, srcs = kapanis_points()
    pts = [p for p in pts if p.get(ykey) is not None]
    if not pts:
        return []
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axhline(0, color=C["black"], lw=0.8, ls=":"); ax.axvline(0, color=C["black"], lw=0.8, ls=":")
    ax.axhspan(-0.044, 0.044, color="#DDDDDD", alpha=0.35, lw=0, label="±4.4 points (95% CI, 500 examples)")
    seen: Dict[str, Any] = {}
    for p in pts:
        lab, color = KAP_FAMILY_STYLE.get(p["family"], (p["family"], C["gray"]))
        marker = KAP_MODEL_MARKER.get(p["model"], "o")
        key = (p["family"], p["model"])
        h = ax.scatter(p["dppl"], p[ykey], color=color, marker=marker, s=34 if p["model"] == "Mistral" else 30, alpha=0.9, zorder=3,
                       edgecolors="black" if marker == "s" else "none", linewidths=0.4)
        seen.setdefault(key, (h, f"{lab} — {p['model']}"))
        k = p.get("key", "")
        if any(k == a or k.endswith(a) for a in KAP_ANNOTATE_KEYS + tuple(annotate_extra)) or ("E-3" in p["label"]) or (p["family"] == "xai_c4" and "single-shot" in p["label"]):
            ax.annotate(p["label"], (p["dppl"], p[ykey]), textcoords="offset points", xytext=(4, 3), fontsize=6.3, ha="left", va="bottom", color="#333333")
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.8)
    ax.set_xlabel("Δ perplexity (WikiText-2, relative to FP16; symlog)")
    ax.set_ylabel(ylabel)
    handles = [v[0] for v in seen.values()] + [ax.collections[0]] if False else [v[0] for v in seen.values()]
    labels_ = [v[1] for v in seen.values()]
    ax.legend(handles, labels_, fontsize=6.2, loc="lower left", ncol=2, frameon=True, framealpha=0.95, edgecolor="none")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    short = srcs[:4] + [f"… (+{len(srcs) - 4} files; full list in tables/kapanis_kontroller, kapanis_qwen_tam)"] if len(srcs) > 5 else srcs
    return [save_fig(fig, paths["figures"], name, short)]


def fig_kapanis_ayrisma_hellaswag(paths: Dict[str, str]) -> List[str]:
    """(a) x = Δppl WikiText, y = ΔHellaSwag acc_norm; all configurations, two models."""
    return _kap_scatter(paths, "dhs", "Δ HellaSwag acc_norm (relative to FP16)", "kapanis_ayrisma_hellaswag", ())


def fig_kapanis_ayrisma_mmlu(paths: Dict[str, str]) -> List[str]:
    """(b) MMLU version of the same plot; also contains the no-pruning points (D-1, D-2, uniform 4-bit) and attention confidence."""
    return _kap_scatter(paths, "dmmlu", "Δ MMLU accuracy (relative to FP16)", "kapanis_ayrisma_mmlu", ("g7/0.2/prune_attnconf", "d2/hqq_xai_b3.5@0", "base/nf4_uniform"))


def fig_kapanis_e7_gruplar(paths: Dict[str, str]) -> List[str]:
    """(c) E-7 group-ablation bars: WikiText vs C4 Δppl; sum of groups vs joint removal."""
    d = load_json(KAP_GROUP_FILE)
    if d is None:
        return []
    groups = d.get("groups") or {}
    order = [g for g in ("L0-5", "L11-15", "L16-20", "L21-26", "L27-31") if g in groups]
    if not order:
        return []
    labels_ = [f"{g}\n({groups[g].get('n_heads')} heads)" for g in order] + ["group\nsum", "47 heads\njoint", "WikiText-only\n47 heads"]
    wt = [groups[g]["delta_vs_fp16"]["wikitext2"] for g in order]; c4 = [groups[g]["delta_vs_fp16"]["c4"] for g in order]
    wt += [sum(wt), groups.get("all_only_c4", {}).get("delta_vs_fp16", {}).get("wikitext2", 0), groups.get("all_only_wikitext", {}).get("delta_vs_fp16", {}).get("wikitext2", 0)]
    c4 += [sum(c4), groups.get("all_only_c4", {}).get("delta_vs_fp16", {}).get("c4", 0), groups.get("all_only_wikitext", {}).get("delta_vs_fp16", {}).get("c4", 0)]
    x = np.arange(len(labels_)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    b1 = ax.bar(x - w / 2, wt, w, color=C["blue"], label="Δ ppl WikiText-2 (calibration domain)", edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w / 2, c4, w, color=C["orange"], hatch="//", label="Δ ppl C4 (outside calibration)", edgecolor="black", linewidth=0.4)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():+.3f}", (r.get_x() + r.get_width() / 2, r.get_height()), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6)
    ax.axvline(len(order) - 0.5, color=C["gray"], lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels_, fontsize=7)
    ax.set_ylabel("Δ perplexity (relative to FP16, no quantization)")
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.95, edgecolor="none")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return [save_fig(fig, paths["figures"], "kapanis_e7_gruplar", [KAP_GROUP_FILE])]


def fig_kapanis_d2_pareto(paths: Dict[str, str]) -> List[str]:
    """(d) D-2 bit–perplexity Pareto: xAI, uniform, random ± std, magnitude; two domains (WikiText, C4), x = GB (labelled with nominal bits)."""
    h = load_json(HQQ_FILE)
    cfgs = {k: c for k, c in ((h or {}).get("configs") or {}).items() if c.get("status") == "completed"}
    if not cfgs:
        return []
    b = load_json(BASELINES_FILE); fp = (((b or {}).get("summary") or {}).get("configs") or {}).get("fp16") or {}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    sel_style = {"xai": ("xAI-guided allocation", C["orange"], "v"), "uniform": ("uniform", C["sky"], "D"), "random": ("random allocation (3 seeds, mean ± std)", C["green"], "x"),
                 "magnitude": ("magnitude allocation", C["vermillion"], "P")}
    for ax, ds, ttl in zip(axes, ("wikitext2", "c4"), ("WikiText-2 (calibration domain)", "C4 (outside calibration)")):
        for sel, (lab, color, marker) in sel_style.items():
            xs, ys, es, names = [], [], [], []
            for name, c in cfgs.items():
                if c.get("selector") != sel:
                    continue
                reps = [r for r in c.get("repeats", []) if r.get("status") == "completed"]
                vals = [_entry_ppl(r, ds) for r in reps]; vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                xs.append(c.get("model_gb")); ys.append(float(np.mean(vals))); es.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                names.append(f"B={c.get('budget_bits')}")
            if not xs:
                continue
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]; es = [es[i] for i in order]; names = [names[i] for i in order]
            ax.errorbar(xs, ys, yerr=es if any(es) else None, color=color, marker=marker, markersize=6, linestyle="-" if sel in ("xai", "uniform") else "none",
                        linewidth=1.2, capsize=3, label=lab, zorder=4 if sel == "xai" else 3)
            for x_, y_, n_ in zip(xs, ys, names):
                ax.annotate(n_, (x_, y_), textcoords="offset points", xytext=(5, -9 if sel == "random" else 4), fontsize=6.2)
        ppl_fp = fp.get("perplexity") if ds == "wikitext2" else None
        if ppl_fp:
            ax.axhline(ppl_fp, color=C["black"], ls=":", lw=1, label=f"FP16 ({ppl_fp:.2f}, 14.5 GB)")
        ax.set_yscale("log"); ax.set_xlabel("model size (GB; including HQQ meta tensors)"); ax.set_title(ttl, fontsize=9)
        ax.set_ylabel("perplexity (log)")
        ax.legend(fontsize=6.5, loc="upper right", frameon=True, framealpha=0.95, edgecolor="none")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return [save_fig(fig, paths["figures"], "kapanis_d2_pareto", [HQQ_FILE, BASELINES_FILE])]


def fig_kapanis_rastgele_seed(paths: Dict[str, str]) -> List[str]:
    """(e) Spread of the 3 random seeds vs xAI per ratio (Mistral 20% / 40% / 60%, Qwen 10% / 20%): ppl (log), HellaSwag, MMLU."""
    panels: List[Dict[str, Any]] = []  # {"title", "ppl": {"random": [...], "xai": v, "fp16": v}, "hs": {...}, "mmlu": {...}}
    g = load_json(GUN7_FILE); t = load_json(TASKS_FILE)
    srcs: List[str] = []
    if g and t:
        srcs += [GUN7_FILE, TASKS_FILE]
        fp = (t["entries"].get("fp16") or {})
        for fk in ("0.2", "0.4", "0.6"):
            rc = _cfg(g, fk, "prune_random"); xc = _cfg(g, fk, "xai_single")
            if not rc or not xc:
                continue
            ents = _seed_entries(TASKS_FILE, f"f{fk}/prune_random")
            xe = t["entries"].get(f"f{fk}/xai_single") or {}
            panels.append({"title": f"Mistral {frac_label(fk)}",
                           "ppl": {"random": rc.get("perplexity_values") or [], "xai": xc.get("perplexity_mean"), "fp16": (g.get("fp16_rerun") or {}).get("perplexity")},
                           "hs": {"random": [((e.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm") for e in ents], "xai": ((xe.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm"),
                                  "fp16": ((fp.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm")},
                           "mmlu": {"random": [e.get("mmlu_subset_acc") for e in ents], "xai": xe.get("mmlu_subset_acc"), "fp16": fp.get("mmlu_subset_acc")}})
    qf = load_json(QWEN_F010_FILE); qs = load_json(KAP_QWEN_F010_SEEDS); qft = load_json(QWEN_F010_TASKS_FILE)
    if qf and qft:
        srcs += [QWEN_F010_FILE, QWEN_F010_TASKS_FILE] + ([KAP_QWEN_F010_SEEDS] if qs else [])
        rc = (qf.get("configs") or {}).get("prune_random") or {}; xc = (qf.get("configs") or {}).get("prune_only") or {}
        vals = list(rc.get("perplexity_values") or []) + list((((qs or {}).get("configs") or {}).get("prune_random") or {}).get("perplexity_values") or [])
        ents = _seed_entries(QWEN_F010_TASKS_FILE, "prune_random"); fp = qft["entries"].get("fp16") or {}; xe = qft["entries"].get("prune_only") or {}
        panels.append({"title": "Qwen %10", "ppl": {"random": vals, "xai": xc.get("perplexity_mean"), "fp16": (qf.get("baseline") or {}).get("perplexity")},
                       "hs": {"random": [((e.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm") for e in ents], "xai": ((xe.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm"),
                              "fp16": ((fp.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm")},
                       "mmlu": {"random": [e.get("mmlu_subset_acc") for e in ents], "xai": xe.get("mmlu_subset_acc"), "fp16": fp.get("mmlu_subset_acc")}})
    q = load_json(MODEL2_FILE); qt = load_json(KAP_QWEN_TASKS_MMLU)
    if q and qt:
        srcs += [MODEL2_FILE, KAP_QWEN_TASKS_MMLU]
        rc = (q.get("configs") or {}).get("prune_random") or {}; xc = (q.get("configs") or {}).get("prune_only") or {}
        ents = _seed_entries(KAP_QWEN_TASKS_MMLU.replace("_mmlu.json", ".json"), "prune_random", base_file=KAP_QWEN_TASKS_MMLU)
        fp = qt["entries"].get("fp16") or {}; xe = qt["entries"].get("prune_only") or {}
        panels.append({"title": "Qwen %20", "ppl": {"random": rc.get("perplexity_values") or [], "xai": xc.get("perplexity_mean"), "fp16": (q.get("baseline") or {}).get("perplexity")},
                       "hs": {"random": [((e.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm") for e in ents], "xai": ((xe.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm"),
                              "fp16": ((fp.get("tasks") or {}).get("hellaswag") or {}).get("acc_norm")},
                       "mmlu": {"random": [e.get("mmlu_subset_acc") if isinstance(e.get("mmlu_subset_acc"), (int, float)) else (e.get("mmlu") or {}).get("accuracy") for e in ents],
                                "xai": xe.get("mmlu_subset_acc"), "fp16": fp.get("mmlu_subset_acc")}})
    if not panels:
        return []
    metrics = [("ppl", "perplexity WikiText-2 (log)"), ("hs", "HellaSwag acc_norm"), ("mmlu", "MMLU accuracy")]
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(2.1 * len(panels) + 1.2, 7.2), sharex="col")
    rc_color, rc_marker, _ = style("prune_random")
    for i, (mk, ylab) in enumerate(metrics):
        for j, pn in enumerate(panels):
            ax = axes[i][j]
            m = pn[mk]; vals = [v for v in m["random"] if isinstance(v, (int, float))]
            xs = np.arange(len(vals))
            if vals:
                ax.scatter(xs, vals, color=rc_color, marker=rc_marker, s=42, zorder=3, label="random (seeds 42 / 43 / 44)")
                if len(vals) > 1:
                    mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
                    ax.axhspan(mu - sd, mu + sd, color=rc_color, alpha=0.12, lw=0); ax.axhline(mu, color=rc_color, ls="--", lw=0.9)
            if isinstance(m.get("xai"), (int, float)):
                ax.axhline(m["xai"], color=C["blue"], lw=1.6, label="xAI selection")
            if isinstance(m.get("fp16"), (int, float)):
                ax.axhline(m["fp16"], color=C["black"], ls=":", lw=1, label="FP16")
            if mk == "ppl":
                ax.set_yscale("log"); ax.set_title(pn["title"], fontsize=8.5)
            if mk == "mmlu":
                ax.axhline(0.25, color=C["gray"], ls="-.", lw=0.8)
            ax.set_xticks(xs if len(vals) else []); ax.set_xticklabels([f"s{42 + k}" for k in range(len(vals))], fontsize=7)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper left", frameon=True, framealpha=0.95, edgecolor="none")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return [save_fig(fig, paths["figures"], "kapanis_rastgele_seed", sorted(set(srcs)))]


def fig_kapanis_yontem_akis(paths: Dict[str, str]) -> List[str]:
    """(f) XAI-JQP method flow chart (matplotlib; settings of the scoring run and the fraction sweep)."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.5, 6.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec=C["blue"], fs=6.4, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, text="", color=C["black"], ls="-"):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1, "linestyle": ls, "shrinkA": 1, "shrinkB": 1})
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.5, text, ha="center", va="bottom", fontsize=6.0, color=color)

    box(2, 80, 26, 14, "1. Calibration\n16 WikiText-2 passages (~2.7k tokens)\n[alternative: 16 C4 documents]", ec=C["gray"])
    box(32, 80, 31, 14, "2. Structural xAI scoring\nCaptum LayerIntegratedGradients, n_steps 8\n1024 heads + 32 MLPs = one importance spectrum", ec=C["blue"], bold=True)
    box(67, 80, 31, 14, "3. Within-type percentile tiering\nhead: prune f / INT4 / FP16 (+INT8)\nMLP: INT4 / FP16", ec=C["blue"], bold=True)
    box(67, 58, 31, 12, "4. Median rule\nattention block → lower median of\nits head tiers (~27% of heads change)", ec=C["orange"])
    box(32, 58, 31, 12, "5. Compression\nmasked / physical head pruning\n+ bitsandbytes NF4 [HQQ 2–8 bit: D-2]", ec=C["orange"], bold=True)
    box(2, 58, 26, 12, "Iterative (3 rounds)\nsurviving heads are\nrescored every round", fc="#FFF4E0", ec=C["orange"])
    box(2, 34, 30, 16, "Controls (same budget)\nrandom × 3 seeds · magnitude · reverse\nTaylor · Wanda / Wanda_ln · attention confidence\nuniform NF4 / GPTQ / AWQ / HQQ", fc="#F2F2F2", ec=C["gray"])
    box(36, 34, 31, 16, "6. Evaluation\nppl: WikiText-2 (in-domain) + C4 (out-of-domain)\nHellaSwag / ARC-C / MMLU (500 each, 0-shot)\nsize (GB), latency (ms/token)", ec=C["green"], bold=True)
    box(71, 34, 27, 16, "7. Statistics (pre-registered)\nper-seed McNemar + Holm\npaired window bootstrap\n'significant only against all seeds'", ec=C["green"])
    box(36, 10, 31, 14, "8. Explainability drift\nrescoring after pruning vs round 0\nSpearman ρ (surviving), top-k Jaccard", ec=C["purple"])
    box(71, 10, 27, 14, "Output: Mistral-7B, Qwen2.5-7B\n20% iterative: 14.50 → 8.20 GB, no task loss\nlimitation: uniform 4-bit leads in ppl", fc="#E8F1FA", ec=C["blue"])
    arrow(28, 87, 32, 87); arrow(63, 87, 67, 87); arrow(82, 80, 82, 70); arrow(67, 64, 63, 64)
    arrow(32, 64, 28, 64, "rescore", color=C["orange"]); arrow(15, 70, 15, 80, "round k+1", color=C["orange"], ls="--")
    arrow(47, 58, 51, 50); arrow(17, 50, 17, 58, "same head / bit budget", color=C["gray"], ls="--"); arrow(32, 42, 36, 42, color=C["gray"])
    arrow(67, 42, 71, 42); arrow(51, 34, 51, 24, color=C["purple"]); arrow(84, 34, 84, 24)
    ax.text(50, 3, "XAI-JQP: pruning + tier decisions from a single IG spectrum; the separation of calibration and evaluation domains and the "
                   "perplexity–task divergence are part of the design.", ha="center", va="bottom", fontsize=6.2, color="#333333")
    return [save_fig(fig, paths["figures"], "kapanis_yontem_akis", [SCORES_FILE, GUN7_FILE])]


KAPANIS_FIGS = {
    "kapanis_ayrisma_hellaswag": (fig_kapanis_ayrisma_hellaswag, TASKS_FILE),
    "kapanis_ayrisma_mmlu": (fig_kapanis_ayrisma_mmlu, TASKS_FILE),
    "kapanis_e7_gruplar": (fig_kapanis_e7_gruplar, KAP_GROUP_FILE),
    "kapanis_d2_pareto": (fig_kapanis_d2_pareto, HQQ_FILE),
    "kapanis_rastgele_seed": (fig_kapanis_rastgele_seed, GUN7_FILE),
    "kapanis_yontem_akis": (fig_kapanis_yontem_akis, SCORES_FILE),
}


FIGURES: Dict[str, Tuple[Callable[[Dict[str, str]], List[str]], str]] = {
    "ablasyon_tablo": (fig_ablasyon_tablo, ABLATION_FILE),
    "ppl_log": (fig_ppl_log, ABLATION_FILE),
    "olcut_katman": (fig_olcut_katman, ABLATION_FILE),
    "rastgele_seed": (fig_rastgele_seed, ABLATION_FILE),
    "ayristirma_tablo": (fig_ayristirma_tablo, ABLATION_FILE),
    "medyan_kurali": (fig_medyan_kurali, ABLATION_FILE),
    "onem_heatmap": (fig_onem_heatmap, SCORES_FILE),
    "gun7_oran": (fig_gun7_oran, GUN7_FILE),
    "gun7_iteratif": (fig_gun7_iteratif, GUN7_FILE),
    "gun7_drift": (fig_gun7_drift, DRIFT_FILE),
    "baseline_pareto": (fig_baseline_pareto, BASELINES_FILE),
    "olcut_katman_gun7": (fig_olcut_katman_gun7, GUN7_FILE),
    "baseline_tablo": (fig_baseline_tablo, BASELINES_FILE),
    "speed_tablo": (fig_speed_tablo, SPEED_FILE),
    "n16_tablo": (fig_n16_tablo, N16_FILE),
    "model2": (fig_model2, MODEL2_FILE),
    "gorevler": (fig_gorevler, TASKS_FILE),
    "mmlu_konu": (fig_mmlu_konu, GUN7_FILE),
    "kararlilik": (fig_kararlilik, N16_FILE),
    "lora_telafi": (fig_lora_telafi, LORA_FILE),
    "speed_yollar": (fig_speed_yollar, SPEED_PATH_FILES[0][1]),  # eager / sdpa as separate rows (sdpa: run C-5)
    "pareto_gun11": (fig_pareto_gun11, BASELINES_FILE),  # skipped when mixed_precision_gun11.json / mixed_bits_gun11.json are absent
    "e5_iki_alan": (fig_e5_iki_alan, E5_FILES[0][1]),  # two-domain perplexity + tasks (E-5) and the Qwen 10% table; NEW files
}
FIGURES.update(KAPANIS_TABLES)  # 13 final summary tables, tables/kapanis_*; existing targets unchanged
FIGURES.update(KAPANIS_FIGS)  # 6 final summary figures, figures/kapanis_*


def fig_gun17_qwen_gorevler(paths: Dict[str, str]) -> List[str]:
    """Qwen 20% task table / figure from the per-question MMLU file (B-5') + seed 43 / 44 MMLU overlay; NEW name gun17_qwen_gorevler,
    the existing gun9_gorevler_tasks_gun9_model2_qwen_gun9 files are unchanged; MMLU was measured, so no † footnote is emitted."""
    global TASKS_FILE, GOREVLER_OUT_NAME
    saved = (TASKS_FILE, GOREVLER_OUT_NAME)
    TASKS_FILE, GOREVLER_OUT_NAME = KAP_QWEN_TASKS_MMLU, "gun17_qwen_gorevler"
    try:
        return fig_gorevler(paths)
    finally:
        TASKS_FILE, GOREVLER_OUT_NAME = saved


FIGURES["gun17_qwen_gorevler"] = (fig_gun17_qwen_gorevler, KAP_QWEN_TASKS_MMLU)
SOURCE_OVERRIDES = {"speed_source": ("SPEED_FILE", ["speed_tablo"]), "model2_source": ("MODEL2_FILE", ["model2"]),
                    "tasks_source": ("TASKS_FILE", ["gorevler"]), "lora_source": ("LORA_FILE", ["lora_telafi"])}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Figures/tables for the manuscript (results/*.json -> figures/, tables/)")
    p.add_argument("--which", default="all", help="comma-separated: " + ",".join(FIGURES) + " (default: all)")
    p.add_argument("--figures-dir", default=FIGURES_DIR)
    p.add_argument("--tables-dir", default=TABLES_DIR)
    p.add_argument("--dry-run", action="store_true", help="write outputs under dryrun_out/")
    p.add_argument("--speed-source", default=None,
                   help="speed JSON for speed_tablo / baseline_tablo / baseline_pareto (default results/speed_gun7.json); "
                        "the table name follows the file name (e.g. speed_gun8_n3); the old speed_gun7 table is kept as an archive")
    p.add_argument("--model2-source", default=None, help="JSON for the model2 table/figure (default results/model2_qwen_gun9.json)")
    p.add_argument("--tasks-source", default=None,
                   help="JSON for the gorevler figure/table (default results/tasks_gun9.json; for another source the output name is "
                        "gun9_gorevler_<file name>, e.g. Qwen: results/tasks_gun9_model2_qwen_gun9.json)")
    p.add_argument("--lora-source", default=None, help="JSON for the lora_telafi figure/table (default results/lora_recovery_gun10.json)")
    p.add_argument("--english-notation", action="store_true",
                   help="figures use English percent notation (20%% instead of %%20) and a footer without internal table names")
    args = p.parse_args(argv)
    global ENGLISH_NOTATION
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    saved_consts = {const: globals()[const] for const, _ in SOURCE_OVERRIDES.values()}
    saved_figs = {n: FIGURES[n] for _, names_ in SOURCE_OVERRIDES.values() for n in names_}
    for arg_name, (const, fig_names) in SOURCE_OVERRIDES.items():  # source selection: module constant + registry entry
        value = getattr(args, arg_name, None)
        if value:
            globals()[const] = value
            for n in fig_names:
                FIGURES[n] = (FIGURES[n][0], value)
    ENGLISH_NOTATION = args.english_notation
    try:
        return _generate(args)
    finally:  # later calls in the same process (tests) fall back to the default sources
        globals().update(saved_consts)
        FIGURES.update(saved_figs)
        ENGLISH_NOTATION = False


def _generate(args: argparse.Namespace) -> int:
    if args.dry_run:
        if args.figures_dir == FIGURES_DIR:
            args.figures_dir = os.path.join(DRYRUN_DIR, "figures")
        if args.tables_dir == TABLES_DIR:
            args.tables_dir = os.path.join(DRYRUN_DIR, "tables")
    names = list(FIGURES) if args.which == "all" else [w.strip() for w in args.which.split(",") if w.strip()]
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s): {unknown}; valid: {list(FIGURES)}")
    paths = {"figures": args.figures_dir, "tables": args.tables_dir}
    MISSING_LOG.clear()
    n_ok = 0
    for name in names:
        fn, source = FIGURES[name]
        if not os.path.exists(source):
            print(f"[{name:<16}] skipped — source missing: {source}")
            continue
        outs = fn(paths)
        if not outs:
            print(f"[{name:<16}] skipped — required field/configuration missing in source: {source}")
            continue
        n_ok += 1
        print(f"[{name:<16}] " + ", ".join(outs))
    print(f"{n_ok}/{len(names)} generated")
    if MISSING_LOG:
        print(f"{MISSING} ({len(MISSING_LOG)} value(s) absent from the JSONs; no numbers were invented):")
        for m in MISSING_LOG:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
