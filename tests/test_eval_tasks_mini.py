"""
CPU-only, offline mini tests for eval_tasks (HellaSwag + ARC-Challenge) and run_eval_from_plans.

  * record conversion: HellaSwag preprocessing ([title], square brackets), ARC with 3-5 choices and numeric labels
  * fixed subset: deterministic selection, written to file once, id + sha1 verification on reload, error on corruption
  * scoring: batched / padded choice log-likelihoods equal the one-by-one (unbatched) computation; acc = argmax(ll),
    acc_norm = argmax(ll / bytes)
  * per-question records are mandatory (per_question); the real subset files in the repository hold 500 sorted, unique indices
  * run_eval_from_plans: 17 entries from the real results/iterative_gun7.json (prune_random seed 42), dry run end to end
    (+MMLU per question, --verify-ppl reproduces the source perplexity), --resume, model2 (configs) plan format

Usage:
    pytest tests/test_eval_tasks_mini.py -q
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_tasks as et  # noqa: E402
from eval_mmlu import FakeTokenizer  # noqa: E402
from test_xai_engine_mini import build_mini_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hs_row(i):
    return {"ind": str(100 + i), "activity_label": "Roof repair", "ctx_a": f"A man [header] is on roof {i}.", "ctx_b": "he",
            "endings": [f"ending {i} a [title] more", f"ending {i} b", f"ending  {i} c", f"ending {i} d"], "label": str(i % 4)}


def _arc_row(i):
    n = 3 + i % 3
    labels = [str(k + 1) for k in range(n)] if i % 2 else list("ABCDE"[:n])
    return {"id": f"ARC_{i}", "question": f"Which option is number {i}?", "choices": {"text": [f"option {i} {k}" for k in range(n)], "label": labels},
            "answerKey": labels[i % n]}


ROWS = {"hellaswag": [_hs_row(i) for i in range(40)], "arc_challenge": [_arc_row(i) for i in range(30)]}


def test_record_conversion_matches_lm_eval_preprocessing():
    r = et._to_record("hellaswag", 3, _hs_row(3))
    assert r["context"] == "Roof repair: A man is on roof 3. He" and r["gold"] == 3 and r["id"] == "103"
    assert r["choices"][0] == "ending 3 a. more" and r["choices"][2] == "ending 3 c"  # [title] -> ". ", double space collapsed
    a = et._to_record("arc_challenge", 5, _arc_row(5))  # numeric labels, 5 choices
    assert a["context"] == "Question: Which option is number 5?\nAnswer:" and len(a["choices"]) == 5 and a["gold"] == 0 and a["id"] == "ARC_5"
    assert et._to_record("arc_challenge", 4, _arc_row(4))["gold"] == 4 % 4
    try:
        et._to_record("piqa", 0, {})
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown task must raise ValueError")


def test_subset_is_fixed_written_once_and_verified(tmp_path):
    logs = []
    for task, rows in ROWS.items():
        path = str(tmp_path / f"{task}_ids.json")
        s1 = et.load_or_build_task_subset(task, path, n=10, seed=42, row_loader=lambda t: ROWS[t], log=logs.append)
        meta = json.load(open(path, encoding="utf-8"))
        assert meta["indices"] == sorted(meta["indices"]) and len(set(meta["indices"])) == 10 and meta["n_available"] == len(rows)
        assert [r["index"] for r in s1["records"]] == meta["indices"] and len(meta["context_sha1"]) == len(meta["ids"]) == 10
        s2 = et.load_or_build_task_subset(task, path, n=5, seed=7, row_loader=lambda t: ROWS[t], log=logs.append)  # the file takes precedence
        assert [r["id"] for r in s2["records"]] == [r["id"] for r in s1["records"]] and any("using the stored subset" in m for m in logs)
        meta["context_sha1"][0] = "0" * 16  # simulate a changed dataset
        json.dump(meta, open(path, "w", encoding="utf-8"))
        try:
            et.load_or_build_task_subset(task, path, row_loader=lambda t: ROWS[t], log=logs.append)
        except ValueError as e:
            assert "do not match" in str(e)
        else:
            raise AssertionError("a sha1 mismatch must raise ValueError")
    try:  # file of another task
        et.load_or_build_task_subset("hellaswag", str(tmp_path / "arc_challenge_ids.json"), row_loader=lambda t: ROWS[t])
    except ValueError:
        pass
    else:
        raise AssertionError("a subset file of the wrong task must raise ValueError")


def test_repo_subset_files_are_500_sorted_unique():
    for task, spec in et.TASK_SPECS.items():
        path = os.path.join(ROOT, spec["ids_file"])
        if not os.path.exists(path):
            continue
        m = json.load(open(path, encoding="utf-8"))
        assert m["task"] == task and m["split"] == spec["split"] and m["dataset"] == spec["dataset"] and m["seed"] == 42
        assert m["n"] == 500 == len(m["indices"]) == len(set(m["indices"])) == len(m["ids"]) == len(m["context_sha1"])
        assert m["indices"] == sorted(m["indices"]) and max(m["indices"]) < m["n_available"]


@torch.no_grad()
def _manual_ll(model, tok, context, choice):
    ctx, cont, _ = et.encode_pair(tok, context, " " + choice)
    ids = torch.tensor([ctx + cont])
    logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
    return sum(float(logp[len(ctx) + k - 1, t]) for k, t in enumerate(cont))


def test_batched_scores_match_unbatched_and_norm_definition():
    model = build_mini_model()
    tok = FakeTokenizer(model.config.vocab_size)
    for task in et.ALL_TASKS:
        subset = et.build_fake_task_subset(task, seed=3, n=6)
        res = et.evaluate_task(model, tok, task, subset=subset, batch_size=5, max_length=64, log=lambda m: None)
        assert res["n"] == 6 and len(res["per_question"]) == 6 and res["n_boundary_mismatch"] == 0 and res["n_truncated"] == 0
        assert res["n_sequences"] == sum(len(r["choices"]) for r in subset["records"])
        n_acc = n_norm = 0
        for r, q in zip(subset["records"], res["per_question"]):
            manual = [_manual_ll(model, tok, r["context"], c) for c in r["choices"]]
            assert all(abs(a - b) < 1e-4 for a, b in zip(manual, q["ll"])), (task, r["index"])  # pad/batch/position indexing is correct
            nbytes = [len(c.encode("utf-8")) for c in r["choices"]]
            assert q["cont_bytes"] == nbytes and q["gold"] == r["gold"] and q["id"] == r["id"]
            assert q["pred"] == max(range(len(manual)), key=lambda k: q["ll"][k])
            assert q["pred_norm"] == max(range(len(manual)), key=lambda k: q["ll"][k] / nbytes[k])
            n_acc += q["correct"]
            n_norm += q["correct_norm"]
        assert res["acc"] == n_acc / 6 and res["acc_norm"] == n_norm / 6 and 0.0 <= res["mean_gold_prob_norm"] <= 1.0
        json.dumps(res)  # JSON-serialisable
    assert set(et.summarize_tasks(et.evaluate_tasks_dry(model, 1, log=lambda m: None))) == set(et.ALL_TASKS)


def test_encode_pair_splits_whole_sequence():
    tok = FakeTokenizer(128)
    ctx, cont, stable = et.encode_pair(tok, "Question: what is this ?\nAnswer:", " the answer")
    assert stable and ctx[0] == tok.bos_token_id and len(cont) == 2
    assert ctx + cont == tok("Question: what is this ?\nAnswer: the answer", add_special_tokens=True)["input_ids"]


# --------------------------------------------------------------------------- #
# run_eval_from_plans
# --------------------------------------------------------------------------- #
def test_collect_entries_from_real_gun7_plan():
    import run_eval_from_plans as rp

    path = os.path.join(ROOT, "results", "iterative_gun7.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path, encoding="utf-8"))
    entries = rp.collect_entries(d)
    keys = [e["key"] for e in entries]
    assert len(keys) == 17 and keys[0] == "fp16" and "f0.2/az_buda_cok_kuantize" in keys and "f0.4/az_buda_cok_kuantize" not in keys
    assert not any("xai_iter" == k.split("/")[-1] for k in keys)  # xai_iter (own INT4 plan) is not in the default list
    by = {e["key"]: e for e in entries}
    for fk, n_heads, n_int4 in (("0.2", 205, 129), ("0.4", 410, 153), ("0.6", 614, 150)):
        for name in ("xai_single", "xai_iter_fixedq", "prune_taylor", "prune_wanda", "prune_random"):
            e = by[f"f{fk}/{name}"]
            assert len(e["heads"]) == n_heads and rp.planned_int4_modules(e["plan"]) == n_int4 == e["source"]["int4_modules"], e["key"]
    assert by["f0.2/prune_random"]["seed"] == 42 and abs(by["f0.2/prune_random"]["source"]["perplexity"] - 6.9856) < 1e-3
    assert len(by["f0.2/az_buda_cok_kuantize"]["heads"]) == 102 and by["fp16"]["source"]["mmlu_subset_acc"] == 0.546
    assert set(map(tuple, by["f0.2/xai_single"]["heads"])) != set(map(tuple, by["f0.2/xai_iter_fixedq"]["heads"]))
    sub = rp.collect_entries(d, fractions=["0.2"], configs=["xai_single"], include_fp16=False)
    assert [e["key"] for e in sub] == ["f0.2/xai_single"]
    try:
        rp.collect_entries(d, fractions=["0.3"])
    except ValueError:
        pass
    else:
        raise AssertionError("a fraction missing from the plan must raise ValueError")


def test_run_eval_from_plans_dry_run_resume_and_model2_format(tmp_path):
    import run_eval_from_plans as rp
    from run_qwen_experiments import main as model2_main

    out = str(tmp_path / "tasks.json")
    common = ["--dry-run", "--output", out, "--log", str(tmp_path / "log.txt")]
    assert rp.main(common + ["--with-mmlu", "--verify-ppl"]) == 0
    d = json.load(open(out, encoding="utf-8"))
    assert d["run"]["status"] == "completed" and d["source"]["plan_format"] == "fractions" and d["source"]["random_seed"] == 42
    keys = list(d["entries"])
    assert keys[0] == "fp16" and "f0.2/az_buda_cok_kuantize" in keys and "f0.4/prune_random" in keys and len(keys) == 12
    for key, e in d["entries"].items():
        assert e["status"] == "completed", key
        assert set(e["tasks"]) == set(et.ALL_TASKS) and all(len(t["per_question"]) == t["n"] for t in e["tasks"].values())
        assert len(e["mmlu"]["per_question"]) == e["mmlu"]["n_questions"]  # per-question records are mandatory
        assert abs(e["perplexity_minus_source"]) < 1e-3, key  # plan applied EXACTLY as in the source run
        if key != "fp16":
            assert e["int4_modules"] == e["source"]["int4_modules"] and e["n_pruned_heads"] == e["source"]["n_pruned_heads"]
    assert d["entries"]["f0.2/prune_random"]["seed"] == 42
    assert rp.main(common + ["--with-mmlu", "--verify-ppl", "--resume"]) == 0
    d2 = json.load(open(out, encoding="utf-8"))
    assert all(e.get("resumed") for e in d2["entries"].values())
    assert d2["entries"]["f0.2/xai_single"]["summary"] == d["entries"]["f0.2/xai_single"]["summary"]
    # model2 format (configs; mini Qwen2): from the run_qwen_experiments --dry-run output
    m2 = str(tmp_path / "m2.json")
    assert model2_main(["--dry-run", "--output", m2, "--log", str(tmp_path / "m2log.txt"), "--scores-dir", str(tmp_path / "m2scores"),
                        "--n-repeats", "1", "--no-mmlu"]) == 0
    out2 = str(tmp_path / "tasks_m2.json")
    assert rp.main(["--dry-run", "--plan", m2, "--output", out2, "--log", str(tmp_path / "log2.txt"), "--verify-ppl"]) == 0
    t2 = json.load(open(out2, encoding="utf-8"))
    assert t2["source"]["plan_format"] == "configs" and list(t2["entries"])[:4] == ["fp16", "quant_only", "prune_only", "both"]
    assert t2["entries"]["prune_only"]["int4_modules"] == 0 and t2["entries"]["both"]["n_pruned_heads"] == 6
    assert all(abs(e["perplexity_minus_source"]) < 1e-3 for e in t2["entries"].values())
    assert rp.parse_args(["--plan", "results/model2_qwen_gun9.json"]).output == os.path.join("results", "tasks_gun9_model2_qwen_gun9.json")
    assert rp.parse_args([]).output == os.path.join("results", "tasks_gun9.json")


def test_random_seed_selects_other_repeats_and_writes_separate_output():
    """E-4: --random-seed 43/44 -> the prune_random repeat with that seed; default (42) unchanged; SEPARATE output name with _seed<N>."""
    import run_eval_from_plans as rp

    a = rp.parse_args([])
    assert a.random_seed == rp.RANDOM_SEED == 42 and a.output == rp.OUTPUT_FILE
    assert rp.parse_args(["--random-seed", "43"]).output == rp.OUTPUT_FILE.replace(".json", "_seed43.json")
    q = rp.parse_args(["--plan", os.path.join("results", "model2_qwen_gun9.json"), "--random-seed", "44"])
    assert q.output == os.path.join("results", "tasks_gun9_model2_qwen_gun9_seed44.json")
    assert rp.parse_args(["--random-seed", "43", "--output", "x.json"]).output == "x.json"  # an explicit --output is left untouched
    for plan, key, n_heads in ((os.path.join(ROOT, "results", "iterative_gun7.json"), "f0.2/prune_random", 205),
                               (os.path.join(ROOT, "results", "model2_qwen_gun9.json"), "prune_random", 157)):
        if not os.path.exists(plan):
            continue
        d = json.load(open(plan, encoding="utf-8"))
        kw = dict(configs=["prune_random"], include_fp16=False)
        by_seed = {s: {e["key"]: e for e in rp.collect_entries(d, random_seed=s, **kw)} for s in (42, 43, 44)}
        assert {e["key"]: e for e in rp.collect_entries(d, **kw)}[key]["heads"] == by_seed[42][key]["heads"]  # default = seed 42
        heads = [tuple(map(tuple, by_seed[s][key]["heads"])) for s in (42, 43, 44)]
        assert [by_seed[s][key]["seed"] for s in (42, 43, 44)] == [42, 43, 44] and len(set(heads)) == 3  # three distinct sets
        assert all(len(h) == n_heads for h in heads)
        assert rp.collect_entries(d, random_seed=99, **kw) == []  # missing seed: no entry (no silent fallback to seed 42)
