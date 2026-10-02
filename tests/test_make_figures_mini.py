"""
GPU-free tests for make_figures: generation from the real ablation / importance-score JSONs into a tmp directory,
"skipped" reporting for missing fraction-sweep sources, and the Markdown/LaTeX output of the table writer.

Run:
    pytest tests/test_make_figures_mini.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.visualization import make_figures
from tools.visualization.make_figures import ABLATION_FILE, BASELINES_FILE, DRIFT_FILE, FIGURES, GUN7_FILE, N16_FILE, SCORES_FILE, SPEED_FILE  # noqa: E402
from tools.visualization.make_figures import MISSING, MISSING_LOG, write_table  # noqa: E402
from tools.visualization.make_figures import main as figures_main  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_write_table_markdown_and_latex(tmp_path):
    paths = write_table(str(tmp_path), "t", ["a_b", "Δ %"], [["x", "+1.5"], ["y_z", "–"]], ["results/k.json"], note="not")
    md = open(paths[0], encoding="utf-8").read()
    tex = open(paths[1], encoding="utf-8").read()
    assert "| a_b | Δ % |" in md and "| y_z | – |" in md and "k.json" in md and "not" in md
    assert r"\begin{tabular}{lr}" in tex and r"a\_b & $\Delta$ \%" in tex and r"y\_z & –" in tex and "\\toprule" in tex


def test_generates_gun6_outputs_and_skips_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)  # results/ paths are relative
    if not (os.path.exists(ABLATION_FILE) and os.path.exists(SCORES_FILE)):
        return  # meaningless without the real result files (they ship with the repository)
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    out = capsys.readouterr().out
    for name in ("ppl_log_gun6", "olcut_katman_hist_gun6", "rastgele_seed_gun6", "medyan_kurali_gun6", "onem_heatmap_gun3"):
        p = figs / f"{name}.png"
        assert p.exists() and p.stat().st_size > 10_000, name
    for name in ("ablasyon_gun6", "ayristirma_gun6"):
        assert (tabs / f"{name}.md").exists() and (tabs / f"{name}.tex").exists(), name
    md = (tabs / "ablasyon_gun6.md").read_text(encoding="utf-8")
    assert md.count("\n|") >= 9 and "6.3171" in md  # header + separator + 8 rows; ablation 'both'
    # skipped when the input JSON is missing, otherwise the generator's REAL output exists (baseline_tablo writes a table only)
    produced = {"gun7_oran": figs / "gun7_oran.png", "gun7_drift": figs / "gun7_drift.png",
                "baseline_tablo": tabs / "baseline_gun7.md"}
    for name, path in produced.items():
        assert f"[{name:<16}] skipped" in out or path.exists(), name
    assert set(FIGURES) >= {"ablasyon_tablo", "ppl_log", "olcut_katman", "rastgele_seed", "ayristirma_tablo",
                            "medyan_kurali", "onem_heatmap", "gun7_oran", "gun7_iteratif", "gun7_drift", "baseline_pareto",
                            "olcut_katman_gun7", "baseline_tablo", "speed_tablo", "n16_tablo"}
    # the existing importance-heatmap PNG was not touched
    assert not (figs / "importance_heatmap.png").exists()


GUN7_WHICH = "gun7_oran,gun7_iteratif,gun7_drift,baseline_pareto,olcut_katman_gun7,baseline_tablo,speed_tablo,n16_tablo"


def _run(tmp_path, which):
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--which", which, "--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    return figs, tabs


def test_gun7_figures_and_tables_from_real_json(tmp_path, monkeypatch, capsys):
    """Figures/tables from the fraction-sweep JSONs; numbers match the recorded results; the 'all' drift view is not tabulated."""
    monkeypatch.chdir(ROOT)
    if not all(os.path.exists(f) for f in (GUN7_FILE, DRIFT_FILE, BASELINES_FILE, SPEED_FILE, N16_FILE)):
        return  # without the GPU results (they ship with the repository)
    figs, tabs = _run(tmp_path, GUN7_WHICH)
    out = capsys.readouterr().out
    for name in ("gun7_oran", "gun7_iteratif", "gun7_drift", "baseline_pareto", "olcut_katman_gun7"):
        p = figs / f"{name}.png"
        assert p.exists() and p.stat().st_size > 20_000, name
    for name in ("gun7_oran", "gun7_iteratif", "gun7_drift", "baseline_gun7", "speed_gun7", "n16_kararlilik"):
        assert (tabs / f"{name}.md").exists() and (tabs / f"{name}.tex").exists(), name
    oran = (tabs / "gun7_oran.md").read_text(encoding="utf-8")
    assert "5.8654" in oran and "XAI-JQP (iterative)" in oran and "60.0156" in oran  # fixedq 20%, wanda 20%
    has_ln = os.path.exists(make_figures.GUN7_WANDA_LN_FILE)  # Wanda_ln run present: +1 row per ratio
    has_attn = os.path.exists(make_figures.GUN7_EXTRA_FILES[0])  # prune_attnconf: +1 row at each of 3 ratios (4 tiers at 20% only)
    assert "Wanda (unverified adaptation)" in oran and oran.count("| %60 |") == 6 + has_ln + has_attn
    assert "| – | FP16 (reference) | 5.2200 |" in oran
    it = (tabs / "gun7_iteratif.md").read_text(encoding="utf-8")
    assert "196/205" in it and "390/410" in it and "576/614" in it and "-38.6003" in it and "72 %" in it
    drift = (tabs / "gun7_drift.md").read_text(encoding="utf-8")
    header = drift.splitlines()[0]
    assert "all" not in header and "surviving" in header  # surviving-heads view only
    assert "| %60 | Wanda (unverified adaptation) | – | – | 614 | 0.736 |" in drift  # not the 'all' view value −0.023
    import json

    n_drift = json.load(open(DRIFT_FILE, encoding="utf-8"))["n_entries"]  # 28 (base), 31 (+ wanda_ln), 34 (+ attnconf)
    assert n_drift in (28, 31, 34) and sum(1 for ln in drift.splitlines() if ln.startswith("|")) == 2 + n_drift
    base = (tabs / "baseline_gun7.md").read_text(encoding="utf-8")
    assert sum(1 for ln in base.splitlines() if ln.startswith("| XAI-JQP")) == 5 and "8.198" in base and "6.3171*" in base and MISSING in base  # physical MMLU not measured
    assert "prune less / quantize more %20" in base and "5.5018" in base
    speed = (tabs / "speed_gun7.md").read_text(encoding="utf-8")
    assert sum(1 for ln in speed.splitlines() if ln.startswith("|")) == 2 + 4 and "32.87" in speed and "8.198" in speed and "0.796×" in speed
    n16 = (tabs / "n16_kararlilik.md").read_text(encoding="utf-8")
    assert "0.9989" in n16 and "6/1024" in n16 and "204/205" in n16 and MISSING not in n16
    assert "not found (1 value" in out and "physical + INT4* MMLU not measured" in out  # only the unmeasured MMLU


def test_missing_values_are_reported_not_invented(tmp_path, monkeypatch, capsys):
    """Without speed_gun7.json the pareto/baseline table is still produced; the physical_int4 point is reported as 'not found'."""
    monkeypatch.chdir(ROOT)
    if not (os.path.exists(GUN7_FILE) and os.path.exists(BASELINES_FILE)):
        return
    monkeypatch.setattr(make_figures, "SPEED_FILE", str(tmp_path / "missing_speed.json"))
    figs, tabs = _run(tmp_path, "baseline_pareto,baseline_tablo,speed_tablo")
    out = capsys.readouterr().out
    assert (figs / "baseline_pareto.png").exists() and (tabs / "baseline_gun7.md").exists()
    assert "[speed_tablo     ] skipped" in out
    base = (tabs / "baseline_gun7.md").read_text(encoding="utf-8")
    assert sum(1 for ln in base.splitlines() if ln.startswith("| XAI-JQP")) == 4 and "8.198" not in base  # physical row not invented
    assert any("physical_int4" in m for m in MISSING_LOG) and "not found (" in out


def test_speed_source_selects_file_and_keeps_default_table_name(tmp_path, monkeypatch, capsys):
    """--speed-source selects the source; the table name follows the file name, the default source keeps the speed_gun7 (archive) name."""
    import shutil

    monkeypatch.chdir(ROOT)
    if not os.path.exists(SPEED_FILE):
        return
    src = tmp_path / "speed_gun8_n3.json"
    shutil.copy(SPEED_FILE, src)
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--which", "speed_tablo", "--speed-source", str(src), "--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    assert (tabs / "speed_gun8_n3.md").exists() and not (tabs / "speed_gun7.md").exists()
    assert "speed_gun8_n3.json" in (tabs / "speed_gun8_n3.md").read_text(encoding="utf-8")
    assert make_figures.SPEED_FILE == SPEED_FILE and make_figures.FIGURES["speed_tablo"][1] == SPEED_FILE  # restored after the call
    _run(tmp_path, "speed_tablo")
    assert (tabs / "speed_gun7.md").exists()


def test_model2_table_and_figure_from_dry_run_json(tmp_path, monkeypatch):
    """Mistral vs second-model table/figure from run_qwen_experiments --dry-run JSON; the Mistral columns hold the real ablation / fraction-sweep values."""
    import json

    from run_qwen_experiments import main as model2_main

    monkeypatch.chdir(ROOT)
    if not (os.path.exists(ABLATION_FILE) and os.path.exists(GUN7_FILE)):
        return
    out = tmp_path / "m2.json"
    assert model2_main(["--dry-run", "--output", str(out), "--log", str(tmp_path / "log.txt"), "--scores-dir", str(tmp_path / "scores"),
                        "--n-repeats", "2"]) == 0
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--which", "model2", "--model2-source", str(out), "--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    assert (figs / "model2_karsilastirma.png").stat().st_size > 20_000
    md = (tabs / "model2_gun9.md").read_text(encoding="utf-8")
    rows = [ln for ln in md.splitlines() if ln.startswith("| ")]
    assert len(rows) == 1 + 8 and "Mistral-7B ppl/FP16" in rows[0]  # header + fp16 + 7 configurations
    d = json.loads(out.read_text(encoding="utf-8"))
    assert "6.3171" in md and "5.8654" in md and "5.2200" in md  # Mistral: ablation 'both', fraction-sweep fixedq, FP16
    assert f"{d['configs']['both']['perplexity_mean']:.4f}" in md and f"{d['baseline']['perplexity']:.4f}" in md
    both_row = next(ln for ln in rows if ln.startswith("| XAI-JQP (single-shot)"))
    assert "205 / 129" in both_row and "0.548" in both_row and "6 / 7" in both_row  # Mistral heads/INT4 + fraction-sweep MMLU; mini Qwen2 6/7
    assert "±" in next(ln for ln in rows if ln.startswith("| random"))
    assert make_figures.MODEL2_FILE.endswith("model2_qwen_gun9.json")


def test_tasks_figure_and_table_from_dry_run_json(tmp_path, monkeypatch):
    """Task-vs-ratio figure/table from run_eval_from_plans --dry-run JSON (source value marked † when MMLU was not measured)."""
    import json

    import run_eval_from_plans as rp

    monkeypatch.chdir(ROOT)
    out = tmp_path / "tasks_dry.json"
    assert rp.main(["--dry-run", "--output", str(out), "--log", str(tmp_path / "log.txt")]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    d["entries"]["f0.2/xai_single"]["source"]["mmlu_subset_acc"] = 0.4321  # source MMLU (the dry plan is produced with --no-mmlu)
    out.write_text(json.dumps(d), encoding="utf-8")
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--which", "gorevler", "--tasks-source", str(out), "--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    assert (figs / "gun9_gorevler_tasks_dry.png").stat().st_size > 20_000
    md = (tabs / "gun9_gorevler_tasks_dry.md").read_text(encoding="utf-8")
    rows = [ln for ln in md.splitlines() if ln.startswith("| ")]
    assert len(rows) == 1 + len(d["entries"]) and "HellaSwag acc_norm" in rows[0] and "ARC-C acc_norm" in rows[0]
    single = next(ln for ln in rows if ln.startswith("| %20 | XAI-JQP (single-shot)"))
    hs = d["entries"]["f0.2/xai_single"]["tasks"]["hellaswag"]
    assert f"{hs['acc']:.3f}" in single and f"{hs['acc_norm']:.3f}" in single and "0.432†" in single
    assert "| – | FP16 (reference) |" in md and "random (seed 42)" in md and "† MMLU not measured in this run" in md
    assert make_figures.TASKS_FILE.endswith("tasks_gun9.json")


def test_mmlu_subject_table_and_heatmap_from_per_subject_summaries(tmp_path, monkeypatch):
    """10 subjects × configuration table + difference heatmap from the per-subject summaries of iterative_gun7.json, without per-question records."""
    import json

    monkeypatch.chdir(ROOT)
    if not os.path.exists(GUN7_FILE):
        return
    d = json.load(open(GUN7_FILE, encoding="utf-8"))
    assert "per_question" not in d["fp16_rerun"]["mmlu"]  # no records: the analysis uses per_subject
    m = make_figures.mmlu_subject_matrix(d)
    assert len(m["subjects"]) == 10 and set(m["n_per_subject"].values()) == {50} and len(m["rows"]) == 6 + 5 + 5
    for r in m["rows"]:  # mean over subjects = recorded total accuracy (equal n)
        assert abs(sum(r["acc"].values()) / 10 - r["total"]) < 1e-9, (r["fraction"], r["config"])
    rnd = next(r for r in m["rows"] if r["fraction"] == "0.2" and r["config"] == "prune_random")
    assert rnd["n_repeats"] == 3 and abs(rnd["total"] - 0.4533) < 1e-3
    wanda = next(r for r in m["rows"] if r["fraction"] == "0.2" and r["config"] == "prune_wanda")
    assert wanda["top_letter"][0] == "A" and wanda["top_letter"][1] > 0.99  # collapsed model locked onto one letter
    assert abs(wanda["total"] - m["gold_letter_counts"]["A"] / 500) < 1e-9  # accuracy = share of 'A' as the correct option
    figs, tabs = _run(tmp_path, "mmlu_konu")
    assert (figs / "gun9_mmlu_konu.png").stat().st_size > 50_000
    md = (tabs / "gun9_mmlu_konu.md").read_text(encoding="utf-8")
    rows = [ln for ln in md.splitlines() if ln.startswith("| ")]
    n_ln = 3 * os.path.exists(make_figures.GUN7_WANDA_LN_FILE)  # one wanda_ln row per ratio in the merged source
    n_extra = 3 * os.path.exists(make_figures.GUN7_EXTRA_FILES[0]) + 2 * os.path.exists(
        make_figures.GUN7_EXTRA_FILES[1])  # attnconf ×3, 4 tiers ×2
    assert len(rows) == 1 + 1 + 16 + n_ln + n_extra and "moral scenarios" in rows[0] and "most frequent prediction" in rows[0]
    assert "| – | FP16 (reference) | 0.34 | 0.48 | 0.70 |" in md and "| 0.546 |" in md and "A (%100)" in md
    assert "per-QUESTION records are not in iterative_gun7.json" in md and "±14 points" in md


def test_speed_table_gets_extra_columns_only_when_source_has_them(tmp_path, monkeypatch):
    """speed_gun8_fixedq adds columns (ppl, generation peak VRAM, attention class); the speed_gun7 table header is unchanged."""
    monkeypatch.chdir(ROOT)
    fixedq = os.path.join("results", "speed_gun8_fixedq.json")
    if not (os.path.exists(fixedq) and os.path.exists(SPEED_FILE)):
        return
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    for src in (fixedq, SPEED_FILE):
        assert figures_main(["--which", "speed_tablo", "--speed-source", src, "--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    new = (tabs / "speed_gun8_fixedq.md").read_text(encoding="utf-8")
    assert "ppl (WikiText-2)" in new.splitlines()[0] and "5.8174" in new and "5.8173" in new and "5.8656" in new and "PrunedHeadAttention" in new
    old = (tabs / "speed_gun7.md").read_text(encoding="utf-8")
    assert old.splitlines()[0] == "| variant | ms/token | tokens/s | speedup (FP16=1) | GB | size ratio | parameters | peak VRAM (GB) |"


def test_combined_stability_table(tmp_path, monkeypatch):
    """n_steps (n16) and calibration-sample (offset16) comparisons in one table; numbers come from the JSONs."""
    monkeypatch.chdir(ROOT)
    if not (os.path.exists(N16_FILE) and os.path.exists(make_figures.OFFSET16_FILE)):
        return
    _, tabs = _run(tmp_path, "kararlilik")
    md = (tabs / "kararlilik.md").read_text(encoding="utf-8")
    assert (tabs / "kararlilik.tex").exists() and MISSING not in md
    assert "| Spearman ρ, head (1024) | 0.9989 | 0.9811 |" in md and "| Top-100 Jaccard (head) | 1.000 | 0.724 |" in md
    assert "204/205" in md and "188/205" in md and "6/1024" in md and "82/1024" in md and "129 / 125" in md


def test_lora_recovery_figure_and_table_from_synthetic_json(tmp_path, monkeypatch):
    """Before/after-recovery figure + table; only the generation path is exercised, with clearly SYNTHETIC values."""
    import json

    monkeypatch.chdir(ROOT)
    fr = {fk: {"status": "completed", "ppl_before": b, "ppl_after": a, "ppl_recovered": b - a, "recovered_share_of_loss": (b - a) / (b - 5.0),
               "source": {"mmlu_subset_acc": m0}, "mmlu_after_acc": m1, "peak_vram_gb": 25.0,
               "train": {"loss_mean_first10": 2.5, "loss_mean_last10": 2.0, "seconds": 480.0}, "adapter": {"bytes_fp16": 84_000_000}}
          for fk, b, a, m0, m1 in (("0.2", 6.0, 5.5, 0.55, 0.54), ("0.4", 10.0, 8.0, 0.52, 0.50), ("0.6", 20.0, 12.0, 0.24, 0.30))}
    fr["0.6"]["status"] = "failed"  # an incomplete ratio is not drawn
    src = tmp_path / "lora.json"
    src.write_text(json.dumps({"reference": {"fp16_perplexity": 5.0, "fp16_mmlu_subset_acc": 0.546, "plan_config": "xai_iter_fixedq",
                                             "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
                                             "training": {"steps": 200, "batch_size": 4, "seq_len": 512, "lr": 2e-4, "schedule": "cosine", "seed": 42}},
                               "fractions": fr}), encoding="utf-8")
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--which", "lora_telafi", "--lora-source", str(src), "--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    assert (figs / "gun10_lora_telafi.png").stat().st_size > 15_000
    md = (tabs / "gun10_lora_telafi.md").read_text(encoding="utf-8")
    rows = [ln for ln in md.splitlines() if ln.startswith("| ")]
    assert len(rows) == 1 + 2 and "| %20 | 6.0000 | 5.5000 | +0.5000 | %50 | 0.550 | 0.540 |" in md and "| 84 |" in md
    assert "SUPPLEMENTARY EXPERIMENT" in md and "same-domain advantage" in md and make_figures.LORA_FILE.endswith("lora_recovery_gun10.json")


def test_gun10_extra_files_are_merged_like_wanda_ln(tmp_path, monkeypatch):
    """results/iterative_gun7_attnconf.json / _4tier.json configurations are merged into the ratio table (wanda_ln pattern); synthetic ppl."""
    import json

    monkeypatch.chdir(ROOT)
    if not os.path.exists(GUN7_FILE):
        return
    src = json.load(open(GUN7_FILE, encoding="utf-8"))["fractions"]["0.2"]["configs"]["prune_taylor"]
    attn, tier = tmp_path / "attnconf.json", tmp_path / "4tier.json"
    attn.write_text(json.dumps({"fractions": {"0.2": {"configs": {"prune_attnconf": {**src, "perplexity_mean": 7.1234}}}}}), encoding="utf-8")
    tier.write_text(json.dumps({"fractions": {"0.2": {"configs": {"xai_single_4tier": {**src, "perplexity_mean": 6.4321},
                                                                  "xai_single": {**src, "perplexity_mean": 999.0}}}}}), encoding="utf-8")
    monkeypatch.setattr(make_figures, "GUN7_EXTRA_FILES", [str(attn), str(tier), str(tmp_path / "missing.json")])
    c = make_figures.load_gun7_merged()["fractions"]["0.2"]["configs"]
    assert c["prune_attnconf"]["source_file"] == str(attn) and c["xai_single_4tier"]["perplexity_mean"] == 6.4321
    assert c["xai_single"]["perplexity_mean"] != 999.0  # existing configurations are never overwritten
    figs, tabs = _run(tmp_path, "gun7_oran")
    oran = (tabs / "gun7_oran.md").read_text(encoding="utf-8")
    assert "attention confidence (Voita)" in oran and "7.1234" in oran and "4 tiers (+INT8)" in oran and "6.4321" in oran
    assert "attnconf.json" in oran and "4tier.json" in oran  # source footnote


def test_wanda_ln_extra_file_is_merged_without_overwriting(tmp_path, monkeypatch):
    """If results/iterative_gun7_wanda_ln.json exists, prune_wanda_ln is added; existing configurations stay unchanged."""
    import json

    monkeypatch.chdir(ROOT)
    if not os.path.exists(GUN7_FILE):
        return
    d = json.load(open(GUN7_FILE, encoding="utf-8"))
    src = d["fractions"]["0.2"]["configs"]["prune_wanda"]
    extra = {"fractions": {"0.2": {"configs": {"prune_wanda_ln": {**src, "perplexity_mean": 7.77},
                                               "xai_single": {**src, "perplexity_mean": 999.0}}}}}
    path = tmp_path / "wanda_ln.json"
    path.write_text(json.dumps(extra), encoding="utf-8")
    monkeypatch.setattr(make_figures, "GUN7_WANDA_LN_FILE", str(path))
    merged = make_figures.load_gun7_merged()
    c = merged["fractions"]["0.2"]["configs"]
    assert c["prune_wanda_ln"]["perplexity_mean"] == 7.77 and c["prune_wanda_ln"]["source_file"] == str(path)
    assert c["xai_single"]["perplexity_mean"] == d["fractions"]["0.2"]["configs"]["xai_single"]["perplexity_mean"]
    figs, tabs = _run(tmp_path, "gun7_oran")
    oran = (tabs / "gun7_oran.md").read_text(encoding="utf-8")
    assert "Wanda, within-layer z-score" in oran and "7.7700" in oran


def test_speed_paths_table_has_separate_eager_and_sdpa_rows(tmp_path, monkeypatch):
    """Repeated eager / sdpa speed measurements as SEPARATE rows in one table; without the sdpa file only eager rows + a 'not found' entry."""
    import json

    def speed(attn, base):
        variants = {n: {"status": "completed", "ms_per_token_std": 0.05, "attention_class": "PrunedHeadAttention" if n == "physical" else "MistralSdpaAttention"}
                    for n in ("fp16", "masked", "physical")}
        summary = {n: {"ms_per_token": base + i, "tokens_per_second": 1000 / (base + i), "speedup_vs_fp16": base / (base + i), "model_gb": 14.5 - i}
                   for i, n in enumerate(("fp16", "masked", "physical"))}
        return {"run": {"args": {"n_repeats": 3, "attn_implementation": attn}}, "variants": variants, "summary": summary}

    eager, sdpa = tmp_path / "e.json", tmp_path / "s.json"
    eager.write_text(json.dumps(speed("eager", 28.0)), encoding="utf-8")
    monkeypatch.setattr(make_figures, "SPEED_PATH_FILES", (("eager", str(eager)), ("sdpa", str(sdpa))))
    paths = {"figures": str(tmp_path / "f"), "tables": str(tmp_path / "t")}
    MISSING_LOG.clear()
    outs = make_figures.fig_speed_yollar(paths)
    text = open(outs[0], encoding="utf-8").read()
    assert text.count("| eager |") == 3 and "| sdpa |" not in text and any("sdpa" in m for m in MISSING_LOG)
    sdpa.write_text(json.dumps(speed("sdpa", 25.0)), encoding="utf-8")
    text = open(make_figures.fig_speed_yollar(paths)[0], encoding="utf-8").read()
    assert text.count("| eager |") == 3 and text.count("| sdpa |") == 3 and "25.00 ± 0.05" in text and "28.00 ± 0.05" in text
    # the "auto" kernel is an EXTRA row (only fp16 + physical are measured; the missing masked variant is not reported as 'not found')
    auto = speed("sdpa", 24.0)
    del auto["variants"]["masked"], auto["summary"]["masked"]
    auto["variants"]["physical"]["attn_kernel"] = "sdpa"
    auto_file = tmp_path / "a.json"
    auto_file.write_text(json.dumps(auto), encoding="utf-8")
    monkeypatch.setattr(make_figures, "SPEED_PATH_FILES", (("eager", str(eager)), ("sdpa", str(sdpa)), ("sdpa, wrapper auto", str(auto_file))))
    MISSING_LOG.clear()
    text = open(make_figures.fig_speed_yollar(paths)[0], encoding="utf-8").read()
    assert text.count("| sdpa, wrapper auto |") == 2 and "physical pruning [kernel sdpa]" in text and not MISSING_LOG


def test_tasks_table_has_holm_column_and_multi_seed_random(tmp_path, monkeypatch):
    """Holm column (primary family) in the task table + random row as mean ± std when extra seed files exist."""
    import json

    from tools.evaluation import compare_tasks_paired as ctp

    def entry(frac, cfg, hs, n=40, seed=None):
        k = int(round(hs * n))
        pq = [{"index": i, "correct": i < k, "correct_norm": i < k} for i in range(n)]
        return {"status": "completed", "fraction": frac, "config": cfg, "seed": seed, "n_pruned_heads": 5, "int4_modules": 7,
                "source": {"perplexity": 6.0}, "mmlu_subset_acc": hs,
                "mmlu": {"per_question": [{"subject": "s", "index": i, "correct": i < k} for i in range(n)]},
                "tasks": {t: {"acc": hs, "acc_norm": hs, "per_question": pq} for t in ("hellaswag", "arc_challenge")}}

    base = {"fp16": entry(None, "fp16", 0.9), "f0.2/xai_single": entry("0.2", "xai_single", 0.9), "f0.2/xai_iter_fixedq": entry("0.2", "xai_iter_fixedq", 0.9),
            "f0.2/prune_random": entry("0.2", "prune_random", 0.2, seed=42)}
    src = tmp_path / "tasks_gun9_x.json"
    doc = {"run": {"model": "mini"}, "source": {"plan_format": "fractions", "plan_file": "p.json"}, "entries": base}
    src.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(make_figures, "TASKS_FILE", str(src))
    paths = {"figures": str(tmp_path / "f"), "tables": str(tmp_path / "t")}
    MISSING_LOG.clear()
    outs = make_figures.fig_gorevler(paths)
    text = open(outs[1], encoding="utf-8").read()
    assert "Holm-adjusted significant" in text and "random (seed 42)" in text and any("no paired-comparison file" in m for m in MISSING_LOG)
    assert ctp.main(["--tasks", str(src)]) == 0 and os.path.exists(ctp.paired_path_for(str(src)))
    for seed, hs in ((43, 0.3), (44, 0.4)):
        extra = dict(doc, entries={"f0.2/prune_random": entry("0.2", "prune_random", hs, seed=seed)})
        (tmp_path / f"tasks_gun9_x_seed{seed}.json").write_text(json.dumps(extra), encoding="utf-8")
    MISSING_LOG.clear()
    text = open(make_figures.fig_gorevler(paths)[1], encoding="utf-8").read()
    row = next(ln for ln in text.splitlines() if "random (3 seeds, mean ± std)" in ln)
    assert "0.300 ± 0.100" in row and "HS yes · ARC yes · MMLU yes (→ XAI-JQP (single-shot))" in row  # 36/40 vs 8/40: significant after Holm
    it = next(ln for ln in text.splitlines() if "XAI-JQP (iterative)" in ln)
    assert "HS no · ARC no · MMLU no" in it and not MISSING_LOG
    single = next(ln for ln in text.splitlines() if "| XAI-JQP (single-shot) |" in ln)
    assert single.rstrip().endswith("| – |")  # "–" for the reference row and for rows outside the primary family


def test_e5_two_domain_table_new_files_only(tmp_path, monkeypatch):
    """The two-domain perplexity table/figure is written under NEW names only; silently skipped without the source."""
    import json

    def entry(w, c, int4):
        return {"status": "completed", "int4_modules": int4, "ppl": {"wikitext2": {"perplexity": w}, "c4": {"perplexity": c}},
                "summary": {"hellaswag": {"acc_norm": 0.8}, "arc_challenge": {"acc_norm": 0.5}}, "mmlu_subset_acc": 0.5}

    files = []
    for tag, w in (("wiki", 6.25), ("c4", 11.5)):
        p = tmp_path / f"e5_{tag}.json"
        p.write_text(json.dumps({"entries": {"fp16": entry(5.0, 7.0, 0), "f0.2/xai_single": entry(w, 9.0, 129)}}), encoding="utf-8")
        files.append(str(p))
    monkeypatch.setattr(make_figures, "E5_FILES", [("WikiText-2", files[0]), ("C4", files[1])])
    for name in ("E5_BOOTSTRAP_FILE", "QWEN_F010_FILE", "QWEN_F010_TASKS_FILE"):
        monkeypatch.setattr(make_figures, name, str(tmp_path / "missing.json"))
    paths = {"figures": str(tmp_path / "f"), "tables": str(tmp_path / "t")}
    out = make_figures.fig_e5_iki_alan(paths)
    assert sorted(os.path.basename(o) for o in out) == ["gun15_e5_iki_alan.md", "gun15_e5_iki_alan.png", "gun15_e5_iki_alan.tex"]
    text = open(os.path.join(paths["tables"], "gun15_e5_iki_alan.md"), encoding="utf-8").read()
    assert "| C4 | XAI-JQP (single-shot) | 129 | 11.5000 | 9.0000 |" in text and "| WikiText-2 | XAI-JQP (single-shot) | 129 | 6.2500 |" in text
    assert text.count("| FP16 |") == 1 and "e5_iki_alan" in FIGURES
    monkeypatch.setattr(make_figures, "E5_FILES", [("WikiText-2", str(tmp_path / "missing.json")), ("C4", files[1])])
    assert make_figures.fig_e5_iki_alan(paths) == []


def test_english_notation_is_opt_in(tmp_path):
    import matplotlib.pyplot as plt
    assert make_figures._english_text("%20 XAI-JQP [Qwen %10], ~27% of heads") == "20% XAI-JQP [Qwen 10%], ~27% of heads"
    assert make_figures._english_text("a.json, kapanis_kontroller, kapanis_qwen_tam) · Generated") == "a.json, … · Generated"
    fig, ax = plt.subplots()
    ax.set_xticks([0.2, 0.4])
    ax.set_xticklabels(["%20", "%40"])
    ax.annotate("%60 XAI-JQP", (0.4, 0.5))
    make_figures._apply_english_notation(fig)
    fig.canvas.draw()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["20%", "40%"]
    assert ax.texts[0].get_text() == "60% XAI-JQP"
    plt.close(fig)
    assert make_figures.ENGLISH_NOTATION is False  # default output unchanged
    assert figures_main(["--which", "kapanis_d2_pareto", "--english-notation", "--figures-dir", str(tmp_path),
                         "--tables-dir", str(tmp_path)]) == 0
    assert make_figures.ENGLISH_NOTATION is False  # flag is reset after the run
