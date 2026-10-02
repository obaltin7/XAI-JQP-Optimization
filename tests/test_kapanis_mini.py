"""
make_figures.py kapanis_* targets: the 13 final summary tables and the 6 final summary figures, generated from the real results/*.json.
No GPU required. Numbers must match the recorded GPU results; existing table / figure files are not regenerated (only kapanis_* outputs).

Run:
    pytest tests/test_kapanis_mini.py -q
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.visualization import make_figures
from tools.visualization.make_figures import FIGURES, MISSING_LOG  # noqa: E402
from tools.visualization.make_figures import main as figures_main  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_TARGETS = ["kapanis_ana_sonuc", "kapanis_kontroller", "kapanis_holm_anlamlilik", "kapanis_bootstrap_ga", "kapanis_e7_grup_ablasyonu",
                 "kapanis_alan_teshisi", "kapanis_d1_karisik_hassasiyet", "kapanis_d2_bit_dagilimi", "kapanis_kayip_ayrisimi", "kapanis_qwen_tam",
                 "kapanis_kararlilik", "kapanis_hiz", "kapanis_deney_envanteri"]
FIGURE_TARGETS = ["kapanis_ayrisma_hellaswag", "kapanis_ayrisma_mmlu", "kapanis_e7_gruplar", "kapanis_d2_pareto", "kapanis_rastgele_seed", "kapanis_yontem_akis"]


def _have_pod_results() -> bool:
    return all(os.path.exists(os.path.join(ROOT, "results", f)) for f in
               ("calib_c4_gun16_f06.json", "mixed_bits_gun11.json", "tasks_gun9_paired.json", "tasks_gun15_e5_wikitext.json", "lora_recovery_gun10.json"))


def _rows(md: str):
    return [ln for ln in md.splitlines() if ln.startswith("| ") and not ln.startswith("|:")]


def check_tex_table(tex: str) -> None:
    """Structural compilability check (no pdflatex): one tabular open/close, column count = header count, no unescaped special characters."""
    assert tex.count(r"\begin{tabular}") == 1 and tex.count(r"\end{tabular}") == 1
    m = re.search(r"\\begin\{tabular\}\{([lrc]+)\}", tex)
    assert m, "missing column alignment spec"
    n_cols = len(m.group(1))
    body = [ln for ln in tex.splitlines() if ln.endswith(r"\\")]
    assert body, "no rows"
    for ln in body:
        cells = ln[:-2].split(" & ")
        assert len(cells) == n_cols, (n_cols, ln[:80])
        for ch in ("%", "_", "#"):
            assert not re.search(r"(?<!\\)" + re.escape(ch), ln), (ch, ln[:80])  # unescaped special character


def test_kapanis_targets_are_registered_and_do_not_touch_existing_names():
    for name in TABLE_TARGETS + FIGURE_TARGETS:
        assert name in FIGURES, name
    assert "gorevler" in FIGURES and "e5_iki_alan" in FIGURES and "pareto_gun11" in FIGURES  # existing targets still registered


def test_kapanis_tables_from_real_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    if not _have_pod_results():
        return  # the GPU result files are not part of the repository
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--which", ",".join(TABLE_TARGETS), "--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    out = capsys.readouterr().out
    assert f"{len(TABLE_TARGETS)}/{len(TABLE_TARGETS)} generated" in out
    produced = sorted(p.name for p in tabs.iterdir())
    assert all(n.startswith("kapanis_") for n in produced) and len(produced) == 2 * len(TABLE_TARGETS)  # new names only, .md + .tex
    assert not figs.exists() or not any(figs.iterdir())  # table targets write no figures
    for name in TABLE_TARGETS:
        md = (tabs / f"{name}.md").read_text(encoding="utf-8")
        assert "Source:" in md and "Generated:" in md, name
        check_tex_table((tabs / f"{name}.tex").read_text(encoding="utf-8"))

    ana = (tabs / "kapanis_ana_sonuc.md").read_text(encoding="utf-8")
    assert len(_rows(ana)) == 1 + 4 and "| 8.343 | 8.199 | 5.8654 | 8.6431 | 0.828 | 0.578 | 0.552 |" in ana  # Mistral iterative 20% (E-5 + speed_gun8_fixedq)
    assert "| 12.4991 | not measured | 0.788 | 0.526 | 0.678 |" in ana and "11.3900" in ana  # Qwen: C4 not measured, not invented

    kon = (tabs / "kapanis_kontroller.md").read_text(encoding="utf-8")
    assert "| %20 | random (3 seeds, mean ± std) | 129 | 7.5476 ± 1.3096 | 0.766 ± 0.011 | 0.499 ± 0.008 | 0.453 ± 0.059 |" in kon
    assert "| %60 | C4-calibrated XAI-JQP (iterative; E-3, bf16 fallback in round 3) | 138 | 33.8914 | 0.726 | 0.468 | 0.254 |" in kon
    assert "| %20 | magnitude (pruning only, no INT4) | 0 | 50.8672 | not measured |" in kon and "| %40 | magnitude | – | not measured |" in kon
    assert "| %20 | attention confidence (Voita) | 129 | 13.7359 | not measured | not measured | 0.280 |" in kon

    holm = (tabs / "kapanis_holm_anlamlilik.md").read_text(encoding="utf-8")
    assert "Mistral (family 45)" in holm and "Qwen %20 (family 15)" in holm and "Qwen %10 (family 12)" in holm and "(family 6)" in holm
    assert "0.726 / 0.678 · 37/13 · p 5.6e-03 ✓" in holm and "0.678 / 0.492 · 136/43 · p 2.5e-11 ✓" in holm
    assert holm.count("✗") > 0 and sum(1 for ln in _rows(holm) if "Qwen %10" in ln and "✓" in ln) == 0  # no comparison significant at Qwen 10%

    boot = (tabs / "kapanis_bootstrap_ga.md").read_text(encoding="utf-8")
    assert len(_rows(boot)) == 1 + 10 + 8 + 10 + 20 and "| +5.3230 | [+5.1500, +5.4954] | yes |" in boot and "| +0.9986 | [+0.9584, +1.0396] | yes |" in boot

    e7 = (tabs / "kapanis_e7_grup_ablasyonu.md").read_text(encoding="utf-8")
    assert "| C4-only 47 heads (joint) | 47 | 6.6406 | +1.4206 | +27.2 % |" in e7 and "additive expectation" in e7 and "+0.5180" in e7

    alan = (tabs / "kapanis_alan_teshisi.md").read_text(encoding="utf-8")
    assert "| pruned by C4 only | 47 | 279 | 641 | 136 | 202 | 7 | 0 | 1 | 4 | 22 | 13 | 0 | 2 |" in alan  # domain diagnosis + E-7 group sizes

    d1 = (tabs / "kapanis_d1_karisik_hassasiyet.md").read_text(encoding="utf-8")
    assert "| mixed_random_k10 | random | 10 | 3 + 3 | 5.009 | 5.2964 ± 0.0026 |" in d1 and "+0.0082 [+0.0059, +0.0104] *" in d1 and "| reference | reference |" in d1

    d2 = (tabs / "kapanis_d2_bit_dagilimi.md").read_text(encoding="utf-8")
    assert "| HQQ xAI B=3.0 | xai | 3.00 | 3.56 | 0 / 11 / 10 / 11 | 3.645 | 6.9787 | 10.0184 | 0.484 | reference |" in d2
    assert "26.5091 ± 15.6140" in d2 and "+0.9986 [+0.9584, +1.0396] *" in d2  # uniform 3-bit better than xAI

    kay = (tabs / "kapanis_kayip_ayrisimi.md").read_text(encoding="utf-8")
    assert "| Mistral-7B | %20 | 5.2200 | +1.0050 | +0.0475 | +1.0971 | +0.0446 | 95.5 % | 91.6 % |" in kay and "| Qwen2.5-7B | %20 | 7.0868 | +5.8174 | +0.1490 |" in kay
    assert "| Mistral-7B | %40 | 5.2200 | not measured | not measured | +6.8649 |" in kay  # no pruning-only run → not invented

    qw = (tabs / "kapanis_qwen_tam.md").read_text(encoding="utf-8")
    assert "| 13.7083 ± 0.6814 | not measured | 0.683 ± 0.069 | 0.443 ± 0.068 | 0.531 ± 0.034 |" in qw  # 20% random; MMLU s43/s44 overlay
    assert "| 8.0602 ± 0.2269 | 12.7171 ± 0.4938 | 0.776 ± 0.003 | 0.529 ± 0.011 | 0.639 ± 0.005 |" in qw  # 10% random, three seeds

    kar = (tabs / "kapanis_kararlilik.md").read_text(encoding="utf-8")
    assert "| 0.9989 | 1.0000 | 0.9990 | 0.9989 | 1.000 | 0.990 |" in kar and "| 0.9269 | 0.8966 |" in kar and "| drift %60 | after attention confidence | 0.2048 |" in kar

    hiz = (tabs / "kapanis_hiz.md").read_text(encoding="utf-8")
    assert "| 28.02 ± 0.79 | 35.69 | 0.957× |" in hiz and "| 25.45 ± 0.12 | 39.29 | 1.024× |" in hiz and "| 26.18 | 38.20 | 1.000× |" in hiz  # no std for a single repeat
    assert "| 22.67 | 44.11 | – |" in hiz  # no FP16 in the Qwen file → speedup not invented

    env = (tabs / "kapanis_deney_envanteri.md").read_text(encoding="utf-8")
    assert "| iterative_gun7.json | run_iterative_pruning.py | Mistral-7B | 2026-09-19T10:36 | 4.53 | completed |" in env and "total (recorded run durations)" in env
    assert "completed_with_failures" in env and "no hourly-rate record" in env
    assert not any("kapanis_" in m for m in MISSING_LOG), MISSING_LOG  # no target invented or reported a missing value


def test_kapanis_tables_skip_gracefully_without_sources(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    for const in ("KAP_GROUP_FILE", "KAP_CALIB_DIAG_FILE"):
        monkeypatch.setattr(make_figures, const, str(tmp_path / "missing.json"))
    monkeypatch.setitem(make_figures.FIGURES, "kapanis_e7_grup_ablasyonu", (make_figures.fig_kapanis_e7_grup_ablasyonu, str(tmp_path / "missing.json")))
    assert figures_main(["--which", "kapanis_e7_grup_ablasyonu", "--figures-dir", str(tmp_path / "f"), "--tables-dir", str(tmp_path / "t")]) == 0
    assert "skipped" in capsys.readouterr().out and not (tmp_path / "t").exists()


def test_kapanis_figures_from_real_json(tmp_path, monkeypatch, capsys):
    """6 final summary figures; only kapanis_* PNGs, no tables; divergence points cover both models and the LoRA / C4-calibrated / D-2 families."""
    monkeypatch.chdir(ROOT)
    if not _have_pod_results():
        return
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--which", ",".join(FIGURE_TARGETS), "--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    assert f"{len(FIGURE_TARGETS)}/{len(FIGURE_TARGETS)} generated" in capsys.readouterr().out
    produced = sorted(p.name for p in figs.iterdir())
    assert produced == sorted(f"{n}.png" for n in FIGURE_TARGETS) and not tabs.exists()
    for name in FIGURE_TARGETS:
        assert (figs / f"{name}.png").stat().st_size > 50_000, name
    pts, srcs = make_figures.kapanis_points()
    fams = {p["family"] for p in pts}; models = {p["model"] for p in pts}
    assert models == {"Mistral", "Qwen"} and {"xai", "random", "xai_c4", "lora", "d2", "d1", "uniform", "wanda_ln", "attnconf"} <= fams
    e3 = next(p for p in pts if "E-3" in p["label"])
    assert abs(e3["dppl"] - (33.8914 - 5.2200)) < 1e-3 and abs(e3["dhs"] - (0.726 - 0.832)) < 1e-9
    q10 = [p for p in pts if p["family"] == "random" and "Qwen %10" in p["label"]]
    assert len(q10) == 3 and all(p["dppl"] < 8.9530 - 7.0868 for p in q10)  # all three seeds have a lower Δppl than xAI
    assert all(p["dhs"] is None for p in pts if p["family"] == "d1" or p.get("key", "").startswith("base/"))  # D-1 and NF4 / GPTQ / AWQ only in the MMLU plot (D-2 has HellaSwag via Chain C)


def test_all_tex_tables_are_structurally_valid():
    """Compilability pre-check of tables/*.tex (no local pdflatex): tabular open / close, column count, no unescaped special characters."""
    import glob

    if not _have_pod_results():
        return
    files = sorted(glob.glob(os.path.join(ROOT, "tables", "*.tex")))
    if not files:  # tables/ is not shipped; generate it locally with make_figures.py
        return
    assert len(files) >= 34
    for f in files:
        check_tex_table(open(f, encoding="utf-8").read())


def test_eval_from_plans_recognises_c3_d1_d2_plans_dry_run(tmp_path, monkeypatch):
    """C-3 (4 tiers, int8), D-1 (mixed_*) and D-2 (HQQ bit plan) plans run end to end through run_eval_from_plans (mini model, CPU)."""
    import json

    import run_eval_from_plans as rp
    import run_iterative_pruning as g7
    import run_mixed_precision as mb

    monkeypatch.chdir(ROOT)
    logd = str(tmp_path / "log.txt")
    c3 = str(tmp_path / "c3_dry.json")
    assert g7.main(["--dry-run", "--fractions", "0.2", "--configs", "xai_single_4tier,xai_iter_fixedq_4tier", "--n-repeats", "1", "--no-mmlu",
                    "--output", c3, "--log", logd, "--scores-dir", str(tmp_path / "s1")]) == 0
    d1 = str(tmp_path / "d1_dry.json")
    assert g7.main(["--dry-run", "--configs", "mixed_xai_k10,mixed_random_k10,mixed_magnitude_k10", "--n-repeats", "2", "--no-mmlu",
                    "--output", d1, "--log", logd, "--scores-dir", str(tmp_path / "s2")]) == 0
    d2 = str(tmp_path / "d2_dry.json")
    assert mb.main(["--dry-run", "--no-mmlu", "--n-repeats", "2", "--configs", "hqq_xai_b3.0,hqq_random_b3.0,hqq_uniform_3bit",
                    "--output", d2, "--log", logd]) == 0
    outs = {}
    for name, plan, cfgs in (("c3", c3, "xai_single_4tier,xai_iter_fixedq_4tier"), ("d1", d1, "mixed_xai_k10,mixed_random_k10,mixed_magnitude_k10"),
                             ("d2", d2, None)):
        out = str(tmp_path / f"tasks_{name}.json")
        argv = ["--dry-run", "--plan", plan, "--skip-fp16", "--with-mmlu", "--output", out, "--log", logd]
        if cfgs:
            argv += ["--configs", cfgs]
        assert rp.main(argv) == 0, name
        outs[name] = json.load(open(out, encoding="utf-8"))
        assert outs[name]["run"]["status"] == "completed", name
        for k, e in outs[name]["entries"].items():
            assert e["status"] == "completed" and set(e["tasks"]) == {"hellaswag", "arc_challenge"} and "per_question" in e["mmlu"], (name, k)
    assert set(outs["c3"]["entries"]) == {"f0.2/xai_single_4tier", "f0.2/xai_iter_fixedq_4tier"}
    assert all(e["quantization"]["int8_modules"] > 0 for e in outs["c3"]["entries"].values())  # the 4th tier was actually applied
    assert set(outs["d1"]["entries"]) == {"f0.2/mixed_xai_k10", "f0.2/mixed_random_k10", "f0.2/mixed_magnitude_k10"}
    assert all(e["n_pruned_heads"] == 0 and e["int4_modules"] > 0 for e in outs["d1"]["entries"].values())
    assert set(outs["d2"]["entries"]) == {"hqq_xai_b3.0", "hqq_random_b3.0", "hqq_uniform_3bit"}
    src = json.load(open(d2, encoding="utf-8"))["configs"]
    for k, e in outs["d2"]["entries"].items():
        assert e["quantization_path"].startswith("hqq") and e["n_pruned_heads"] == 0
        assert e["hqq_modules_by_bits"] == src[k]["repeats"][0]["quantization"]["modules_by_bits"]  # tier counts identical to the source
    # seed 43 -> separate file name, stochastic configuration only
    a = rp.parse_args(["--plan", d2, "--random-seed", "43", "--configs", "hqq_random_b3.0"])
    assert a.output.endswith("_seed43.json")
    # the real D-2 file can be read without a GPU (plan entries carry bit_plan)
    if not _have_pod_results():
        return
    real = json.load(open(os.path.join(ROOT, "results", "mixed_bits_gun11.json"), encoding="utf-8"))
    ents = rp.collect_entries(real, include_fp16=False)
    assert {e["key"] for e in ents} == set(real["configs"]) and all(e["bit_plan"] and not e["plan"] for e in ents)

def test_gun17_qwen_gorevler_table_uses_mmlu_overlay(tmp_path, monkeypatch):
    """Qwen task table from the per-question MMLU file + seed 43 / 44 MMLU overlay; new name, no †, the existing table is not written."""
    monkeypatch.chdir(ROOT)
    if not _have_pod_results():
        return
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--which", "gun17_qwen_gorevler", "--figures-dir", str(figs), "--tables-dir", str(tabs)]) == 0
    assert sorted(p.name for p in tabs.iterdir()) == ["gun17_qwen_gorevler.md", "gun17_qwen_gorevler.tex"]
    assert sorted(p.name for p in figs.iterdir()) == ["gun17_qwen_gorevler.png"]
    md = (tabs / "gun17_qwen_gorevler.md").read_text(encoding="utf-8")
    assert "†" not in md and "0.531 ± 0.034" in md and "0.683 ± 0.069" in md  # MMLU 3 seeds (0.554 / 0.492 / 0.546), HS 3 seeds
    assert "HS yes · ARC no · MMLU yes" in md and "tasks_gun9_model2_qwen_gun9_seed43_mmlu.json" not in md
    assert make_figures.TASKS_FILE.endswith("tasks_gun9.json") and make_figures.GOREVLER_OUT_NAME is None  # restored
    check_tex_table((tabs / "gun17_qwen_gorevler.tex").read_text(encoding="utf-8"))


def test_zincir_c_task_columns_and_extra_families(tmp_path, monkeypatch):
    """Chain C (C-3 / D-2 tasks) in the tables — D-2 task columns (random, 3 seeds), D-1 'not measured', 4-tier rows in the controls,
    C-3 and D-2 extra families in the Holm table, D-2 HellaSwag in the divergence points."""
    monkeypatch.chdir(ROOT)
    if not os.path.exists(os.path.join(ROOT, "results", "tasks_gun9_mixed_bits_gun11.json")):
        return
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    assert figures_main(["--which", "kapanis_d2_bit_dagilimi,kapanis_d1_karisik_hassasiyet,kapanis_kontroller,kapanis_holm_anlamlilik", "--figures-dir", str(figs),
                         "--tables-dir", str(tabs)]) == 0
    d2 = (tabs / "kapanis_d2_bit_dagilimi.md").read_text(encoding="utf-8")
    assert "| HellaSwag | ARC-C | MMLU (per-question) |" in d2 and "| 0.802 | 0.524 | 0.484 |" in d2  # xAI B=3.0 tasks
    assert "| 0.719 ± 0.028 | 0.448 ± 0.036 | 0.378 ± 0.004 |" in d2 and "| 0.812 | 0.520 | 0.472 |" in d2  # random 3 seeds, uniform 3-bit
    d1 = (tabs / "kapanis_d1_karisik_hassasiyet.md").read_text(encoding="utf-8")
    assert d1.count("| not measured | not measured |") == 6 and "D-1 was skipped" in d1
    kon = (tabs / "kapanis_kontroller.md").read_text(encoding="utf-8")
    assert "| %20 | XAI-JQP iterative, 4 tiers (+INT8) | 91 + 102 INT8 | 5.8711 | 0.828 | 0.578 | 0.566 |" in kon
    holm = (tabs / "kapanis_holm_anlamlilik.md").read_text(encoding="utf-8")
    assert "Mistral %20 4 tiers C-3 (extra) (family 6)" in holm and "Mistral D-2 HQQ (extra) (family 30)" in holm
    assert "0.802 / 0.812 · 17/22 · p 1.0e+00 ✗" in holm and "0.484 / 0.352 · 113/47 · p 5.3e-06 ✓" in holm  # uniform equal, magnitude significant
    pts, _ = make_figures.kapanis_points()
    d2pts = [p for p in pts if p["family"] == "d2" and p["dhs"] is not None]
    assert len(d2pts) == 10 and any("(s43)" in p["label"] for p in d2pts)  # xai ×2 + magnitude ×2 + random 3×2 = 10 points
    t4 = [p for p in pts if p.get("key", "").endswith("_4tier") and p["dhs"] is not None]
    assert len(t4) == 2
