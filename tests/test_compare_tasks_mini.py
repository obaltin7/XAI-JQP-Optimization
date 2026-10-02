"""
Mini tests for compare_tasks_paired.py (paired McNemar comparison of task results).

  * exact binomial McNemar p-value (small hand-checkable cases; n = 0 -> 1.0; symmetric)
  * question id (subject, index): in MMLU the index is within a subject -> keying by index alone collapses 500 records to ~225
    (regression guard)
  * end to end on synthetic JSON; if the real results/tasks_gun9.json exists, MMLU accuracies match the recorded mmlu_subset_acc exactly

Run:
    pytest tests/test_compare_tasks_mini.py -q
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.evaluation import compare_tasks_paired as ct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_mcnemar_exact_values():
    assert ct.mcnemar_exact(0, 0) == 1.0 and ct.mcnemar_exact(5, 5) == 1.0
    assert math.isclose(ct.mcnemar_exact(0, 5), 2 * 0.5 ** 5)  # 0.0625
    assert math.isclose(ct.mcnemar_exact(1, 9), 2 * (1 + 10) / 2 ** 10)
    assert ct.mcnemar_exact(31, 13) == ct.mcnemar_exact(13, 31) and 0.005 < ct.mcnemar_exact(31, 13) < 0.015


def _entry(task_correct, mmlu_correct):
    return {"tasks": {"hellaswag": {"per_question": [{"index": 10 + i, "correct": c, "correct_norm": c} for i, c in enumerate(task_correct)]}},
            "mmlu": {"per_question": [{"subject": "a" if i < 3 else "b", "index": i % 3, "correct": c} for i, c in enumerate(mmlu_correct)]}}


def test_question_key_uses_subject_and_index_and_detects_collisions():
    e = _entry([True, False], [True, True, False, False, True, False])  # index 0..2 repeats across the two subjects
    c = ct.correctness(e, "mmlu", "correct")
    assert len(c) == 6 and c[("a", 0)] is True and c[("b", 0)] is False
    dup = {"mmlu": {"per_question": [{"subject": "a", "index": 0, "correct": True}, {"subject": "a", "index": 0, "correct": False}]}}
    try:
        ct.correctness(dup, "mmlu", "correct")
    except ValueError:
        pass
    else:
        raise AssertionError("a duplicate question id must raise ValueError")
    assert ct.correctness({"tasks": {}}, "hellaswag", "correct_norm") is None


def test_end_to_end_on_synthetic_json(tmp_path):
    entries = {"fp16": _entry([True, True, True, True], [True] * 6),
               "f0.2/xai_single": _entry([True, True, True, False], [True, True, True, False, False, True]),
               "f0.2/xai_iter_fixedq": _entry([True, False, True, True], [True, True, False, False, True, True])}
    src, out = tmp_path / "tasks.json", tmp_path / "paired.json"
    src.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    assert ct.main(["--tasks", str(src), "--output", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    pairs = {(p["a"], p["b"], p["task"]): p for p in d["pairs"]}
    hs = pairs[("f0.2/xai_iter_fixedq", "f0.2/xai_single", "hellaswag")]  # primary family: A = iterative, B = single-shot
    assert (hs["only_a"], hs["only_b"], hs["both_correct"], hs["both_wrong"], hs["n"]) == (1, 1, 2, 0, 4) and hs["p_mcnemar_exact"] == 1.0
    assert hs["family"] == "primary" and hs["correction"] == "holm" and hs["p_adjusted"] == 1.0 and hs["significant"] is False
    assert d["correction"] == "holm" and d["alpha"] == 0.05 and d["n_by_family"]["primary"] == 2  # no random/taylor entries: 1 comparison × 2 tasks
    mm = pairs[("fp16", "f0.2/xai_single", "mmlu")]
    assert (mm["only_a"], mm["only_b"]) == (2, 0) and math.isclose(mm["acc_b"], 4 / 6) and mm["metric"] == "acc"
    assert ("fp16", "f0.2/xai_single", "arc_challenge") not in pairs  # missing task is skipped
    bare = {"fp16": entries["fp16"], "both": entries["f0.2/xai_single"], "xai_iter_fixedq": entries["f0.2/xai_iter_fixedq"]}  # model2 (Qwen) format
    src2, out2 = tmp_path / "tasks_qwen.json", tmp_path / "paired_qwen.json"
    src2.write_text(json.dumps({"entries": bare}), encoding="utf-8")
    assert ct.main(["--tasks", str(src2), "--output", str(out2)]) == 0
    keys = {(p["a"], p["b"]) for p in json.loads(out2.read_text(encoding="utf-8"))["pairs"]}
    assert ("xai_iter_fixedq", "both") in keys and ("fp16", "both") in keys
    entries["f0.2/xai_iter_fixedq"]["tasks"]["hellaswag"]["per_question"][0]["index"] = 999
    src.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    try:
        ct.main(["--tasks", str(src), "--output", str(out)])
    except ValueError:
        pass
    else:
        raise AssertionError("differing question sets must raise ValueError")


def test_real_b4_results_match_recorded_accuracies():
    path = os.path.join(ROOT, "results", "tasks_gun9.json")
    if not os.path.exists(path):
        return
    entries = json.load(open(path, encoding="utf-8"))["entries"]
    for name, e in entries.items():
        if "mmlu" in e:
            c = ct.correctness(e, "mmlu", "correct")
            assert len(c) == 500 and math.isclose(sum(c.values()) / 500, e["mmlu_subset_acc"])
        for task in ("hellaswag", "arc_challenge"):
            c = ct.correctness(e, task, "correct_norm")
            assert math.isclose(sum(c.values()) / len(c), e["summary"][task]["acc_norm"])
    r = ct.compare(entries, "f0.6/xai_single", "f0.6/xai_iter_fixedq", "hellaswag", "correct_norm")
    assert (r["only_a"], r["only_b"]) == (31, 13) and r["p_mcnemar_exact"] < 0.05  # 60%: single-shot is better on HellaSwag by RAW p
    pairs = ct.build_pairs(entries, [])
    ct.apply_correction(pairs, "holm")
    prim = [p for p in pairs if p["family"] == "primary"]
    assert len(prim) == 27 and all(p["family_size"] == 27 for p in prim)  # 3 ratios × 3 tasks × 3 pre-registered comparisons
    it60 = next(p for p in prim if (p["a"], p["b"], p["task"]) == ("f0.6/xai_iter_fixedq", "f0.6/xai_single", "hellaswag"))
    assert it60["p_mcnemar_exact"] < 0.05 <= it60["p_adjusted"] and it60["significant"] is False  # NOT significant after Holm
    rnd = [p for p in prim if p["b"].endswith("prune_random") and p["task"] == "hellaswag"]
    assert len(rnd) == 3 and all(p["significant"] for p in rnd)  # xAI vs random is significant on HellaSwag at every ratio


def test_holm_and_bonferroni_adjustment_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert ct.adjust_pvalues(p, "none") == p
    assert all(math.isclose(a, b) for a, b in zip(ct.adjust_pvalues(p, "bonferroni"), [0.04, 0.16, 0.12, 0.02]))
    # Holm (by hand): sorted 0.005, 0.01, 0.03, 0.04 -> 4·0.005 = 0.02, 3·0.01 = 0.03, 2·0.03 = 0.06, 1·0.04 -> max(0.06, 0.04) = 0.06
    assert all(math.isclose(a, b) for a, b in zip(ct.adjust_pvalues(p, "holm"), [0.03, 0.06, 0.06, 0.02]))
    assert ct.adjust_pvalues([0.5, 0.9], "holm") == [1.0, 1.0] and ct.adjust_pvalues([], "holm") == []  # clipped at 1
    holm, bonf = ct.adjust_pvalues([0.001, 0.02, 0.03], "holm"), ct.adjust_pvalues([0.001, 0.02, 0.03], "bonferroni")
    assert all(h <= b for h, b in zip(holm, bonf))  # Holm is never more conservative than Bonferroni
    try:
        ct.adjust_pvalues([0.1], "fdr")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown correction must raise ValueError")
    assert ct.parse_default_correction() == "holm"


def test_merge_pair_and_mmlu_injection(tmp_path):
    base = {"fp16": _entry([True, True, True, True], [True] * 6), "f0.6/xai_iter_fixedq": _entry([True, False, False, True], [False] * 6)}
    c4 = {"f0.6/xai_iter_fixedq": _entry([True, True, True, True], [True] * 6)}  # same key: a TAG is required
    src, other, out = tmp_path / "tasks_gun9.json", tmp_path / "tasks_gun9_c4.json", tmp_path / "e3.json"
    src.write_text(json.dumps({"entries": base}), encoding="utf-8")
    other.write_text(json.dumps({"entries": c4}), encoding="utf-8")
    try:
        ct.main(["--tasks", str(src), "--merge", str(other), "--output", str(out)])
    except ValueError:
        pass
    else:
        raise AssertionError("colliding keys cannot be merged without a TAG")
    assert ct.main(["--tasks", str(src), "--merge", f"{other}:c4", "--pair", "c4:f0.6/xai_iter_fixedq", "f0.6/xai_iter_fixedq", "--output", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    extra = [p for p in d["pairs"] if p["family"] == "extra"]
    assert len(extra) == 2 and extra[0]["a"] == "c4:f0.6/xai_iter_fixedq" and (extra[0]["only_a"], extra[0]["only_b"]) == (2, 0)
    assert all(p["family_size"] == 2 for p in extra)  # the extra family is corrected within itself
    assert ct.paired_path_for(os.path.join("results", "tasks_gun9.json")) == os.path.join("results", "tasks_gun9_paired.json")
    assert ct.paired_path_for(os.path.join("results", "tasks_gun9_model2_qwen_gun9.json")) == os.path.join("results", "tasks_gun9_paired_model2_qwen_gun9.json")
    # Qwen: the task evaluation does not re-measure MMLU -> per-question records are injected from the model2 output (seed matching for stochastic configs)
    pq = lambda vals: {"per_question": [{"subject": "a", "index": i, "correct": v} for i, v in enumerate(vals)], "mmlu_subset_acc": sum(vals) / len(vals)}
    model2 = {"baseline": {"mmlu": pq([True, True])},
              "configs": {"prune_random": {"repeats": [{"status": "completed", "seed": 42, "mmlu": pq([True, False])},
                                                       {"status": "completed", "seed": 43, "mmlu": pq([False, False])}]}}}
    entries = {"fp16": {"tasks": {}}, "prune_random": {"tasks": {}, "seed": 43}, "both": {"tasks": {}}}
    assert ct.inject_mmlu(entries, model2) == 2 and "mmlu" not in entries["both"]
    assert [r["correct"] for r in entries["prune_random"]["mmlu"]["per_question"]] == [False, False]  # the seed-43 record


def test_random_claim_requires_all_seeds_significant(tmp_path):
    """Pre-registered: 'better than random' only if significant after Holm against ALL seeds; seed files are merged automatically."""
    good, bad = [True] * 40, [False] * 40
    close = [True] * 38 + [False] * 2
    def _entry(task_correct, _unused):  # task records only (the module-level helper's MMLU ids are limited to 6 questions)
        return {"tasks": {"hellaswag": {"per_question": [{"index": i, "correct": c, "correct_norm": c} for i, c in enumerate(task_correct)]}}}

    base = {"f0.2/xai_single": _entry(good, good), "f0.2/prune_random": _entry(bad, bad)}
    src = tmp_path / "tasks_gun9_r.json"
    src.write_text(json.dumps({"entries": base}), encoding="utf-8")
    (tmp_path / "tasks_gun9_r_seed43.json").write_text(json.dumps({"entries": {"f0.2/prune_random": _entry(bad, bad)}}), encoding="utf-8")
    (tmp_path / "tasks_gun9_r_seed44.json").write_text(json.dumps({"entries": {"f0.2/prune_random": _entry(close, close)}}), encoding="utf-8")
    out = tmp_path / "o.json"
    assert ct.main(["--tasks", str(src), "--output", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    hs = next(c for c in d["random_claims"] if c["task"] == "hellaswag")
    assert hs["n_seeds"] == 3 and hs["all_significant"] is False  # not significant against seed 44 -> NO claim
    assert d["n_by_family"]["primary"] == 3 and any(p["b"] == "s44:f0.2/prune_random" for p in d["pairs"])


def test_seed_mmlu_overlay_is_automatic_and_leaves_seed_file_untouched(tmp_path):
    """Qwen seed 43/44 MMLU measured separately -> <tasks>_seed<N>_mmlu.json is read automatically, _seed<N>.json is NOT modified."""
    def pq(vals):
        return {"per_question": [{"subject": "a", "index": i, "correct": v} for i, v in enumerate(vals)], "mmlu_subset_acc": sum(vals) / len(vals)}

    def hs(vals):
        return {"tasks": {"hellaswag": {"per_question": [{"index": i, "correct": v, "correct_norm": v} for i, v in enumerate(vals)]}}}

    good, bad = [True] * 20, [False] * 20
    src = tmp_path / "tasks_gun9_q.json"
    src.write_text(json.dumps({"entries": {"prune_only": {**hs(good), "mmlu": pq(good)}, "prune_random": {**hs(bad), "mmlu": pq(bad)}}}), encoding="utf-8")
    seed_file, seed_payload = tmp_path / "tasks_gun9_q_seed43.json", {"entries": {"prune_random": hs(bad)}}  # the E-4 run was made without --with-mmlu
    seed_file.write_text(json.dumps(seed_payload), encoding="utf-8")
    out = tmp_path / "o.json"
    assert ct.main(["--tasks", str(src), "--output", str(out)]) == 0
    d0 = json.loads(out.read_text(encoding="utf-8"))
    assert d0["mmlu_overlay_entries"] == 0 and not any(p["task"] == "mmlu" and p["b"] == "s43:prune_random" for p in d0["pairs"])
    mmlu_file = tmp_path / "tasks_gun9_q_seed43_mmlu.json"
    mmlu_file.write_text(json.dumps({"entries": {"prune_random": {"mmlu": pq(bad)}}}), encoding="utf-8")
    assert ct.main(["--tasks", str(src), "--output", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["mmlu_overlay_entries"] == 1 and str(mmlu_file) in d["merged"]
    mm = next(p for p in d["pairs"] if p["task"] == "mmlu" and p["b"] == "s43:prune_random")
    assert mm["family"] == "primary" and mm["acc_a"] == 1.0 and mm["acc_b"] == 0.0  # the per-seed MMLU test joined the family
    assert json.loads(seed_file.read_text(encoding="utf-8")) == seed_payload  # source file unchanged
    assert ct.overlay_mmlu({"s43:x": {"mmlu": pq(good)}}, "s43:", {"x": {"mmlu": pq(bad)}}, "k") == 0  # does NOT overwrite existing MMLU
    assert ct.overlay_mmlu({}, "s43:", {"x": {"mmlu": pq(bad)}}, "k") == 0  # entry without a counterpart is skipped
