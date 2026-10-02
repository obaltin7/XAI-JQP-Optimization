"""
GPU-free / network-free mini tests for the second calibration set (C4).

  * select_c4_passages: first N documents in stream order with >= min_tokens tokens; deterministic, offset,
    identifiers (stream index, url, sha1)
  * run_xai_on_mistral --calib-dataset: default (wikitext2) path and output name unchanged; c4 -> importance_scores_c4.json,
    load_dataset("allenai/c4", "en", split="validation", streaming=True)
  * run_iterative_pruning --calib-dataset c4: default paths SEPARATE from the default run's files, reproduction check
    against the ablation run skipped, passage sha1 verification
  * end-to-end dry run (xai_single = both, xai_iter_fixedq, prune_random with 1 seed)

Run:
    pytest tests/test_calib_c4_mini.py -q
"""
import importlib.machinery
import importlib.util
import json
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _name, _attrs in (("captum", {}), ("captum.attr", {"LayerIntegratedGradients": None}), ("datasets", {"load_dataset": None})):
    if importlib.util.find_spec(_name.split(".")[0]) is None and _name not in sys.modules:  # local .venv has no datasets package
        _m = types.ModuleType(_name)
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, None)
        for k, v in _attrs.items():
            setattr(_m, k, v)
        sys.modules[_name] = _m
        if "." in _name:
            setattr(sys.modules[_name.split(".")[0]], _name.split(".")[1], _m)

import run_xai_on_mistral as rx, run_iterative_pruning as g7


class WordTokenizer:
    """Word = token (+1 BOS); single text or list (right-padded tensor) — the build_calibration_batches interface."""

    def _ids(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [2 + (hash(w) % 50) for w in text.split()]

    def __call__(self, text, add_special_tokens=True, return_tensors=None, padding=False, truncation=False, max_length=None):
        if isinstance(text, str):
            return {"input_ids": self._ids(text, add_special_tokens)}
        rows = [self._ids(t)[:max_length] if truncation else self._ids(t) for t in text]
        width = max(len(r) for r in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _doc(i: int, n_words: int):
    return {"text": " ".join(f"w{i}_{j}" for j in range(n_words)), "url": f"https://example.org/{i}", "timestamp": "2019-04-01T00:00:00Z"}


STREAM = [_doc(0, 5), {"text": "   ", "url": "empty"}, _doc(2, 12), _doc(3, 9), _doc(4, 30), _doc(5, 10), _doc(6, 11), _doc(7, 3), _doc(8, 40)]


def test_select_c4_passages_first_n_by_token_count_and_ids():
    tok = WordTokenizer()
    texts, ids = rx.select_c4_passages(iter(STREAM), tok, n_passages=3, min_tokens=11)  # tokens = words + 1 (BOS)
    assert [p["stream_index"] for p in ids] == [2, 4, 5]  # 9 words (10 tokens) rejected; 10 words = 11 tokens accepted
    assert texts == [STREAM[i]["text"] for i in (2, 4, 5)]
    assert [p["n_tokens"] for p in ids] == [13, 31, 11] and ids[0]["url"] == "https://example.org/2"
    assert all(len(p["sha1"]) == 16 and p["n_chars"] == len(t) for p, t in zip(ids, texts))
    again = rx.select_c4_passages(iter(STREAM), tok, 3, 11)
    assert again == (texts, ids)  # deterministic, no seed
    t2, i2 = rx.select_c4_passages(iter(STREAM), tok, 2, 11, offset=3)  # the first 3 ELIGIBLE documents are skipped -> no overlap
    assert [p["stream_index"] for p in i2] == [6, 8] and set(t2).isdisjoint(texts)
    t3, _ = rx.select_c4_passages(iter(STREAM), tok, 10, 11)  # not enough documents: warning, no error
    assert len(t3) == 5
    try:
        rx.select_c4_passages(iter(STREAM), tok, 1, 11, offset=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative offset should raise ValueError")


def test_xai_cli_default_unchanged_and_c4_output():
    a = rx.parse_args([])
    assert a.calib_dataset == "wikitext2" and a.output == rx.OUTPUT_FILE and a.output.endswith("importance_scores_gun3.json")
    c = rx.parse_args(["--calib-dataset", "c4"])
    assert c.output == os.path.join("results", "importance_scores_c4.json") and c.n_passages == 16 and c.max_length == 256 and c.seed is None
    assert rx.parse_args(["--calib-dataset", "c4", "--output", "x.json"]).output == "x.json"


def test_load_calibration_passages_dispatch(monkeypatch):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "wikitext":
            return {"text": [" = Title = ", "a" * 250, "short", "b" * 300]}
        return iter(STREAM)

    monkeypatch.setattr(rx, "load_dataset", fake_load_dataset)
    passages, ids = rx.load_calibration_passages(WordTokenizer(), "wikitext2", 2, 200, 256)
    assert passages == ["a" * 250, "b" * 300] and ids is None  # WikiText-2 path: identical to select_passages
    assert calls[-1] == (("wikitext", "wikitext-2-raw-v1"), {"split": "test"})
    passages, ids = rx.load_calibration_passages(WordTokenizer(), "c4", 2, 200, 11)
    assert calls[-1] == (("allenai/c4", "en"), {"split": "validation", "streaming": True})
    assert [p["stream_index"] for p in ids] == [2, 4] and len(passages) == 2
    try:
        rx.load_calibration_passages(WordTokenizer(), "pile", 2, 200, 11)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown calibration set should raise ValueError")


def test_iterative_cli_c4_paths_are_separate_from_gun7():
    d = g7.parse_args([])
    assert (d.calib_dataset, d.scores, d.output, d.log, d.scores_dir) == ("wikitext2", g7.SCORES_FILE, g7.OUTPUT_FILE, g7.LOG_FILE, g7.SCORES_DIR)
    assert g7.reproduction_check_applies(d) and g7.calib_extra(d) == {}
    c = g7.parse_args(["--calib-dataset", "c4"])
    assert c.scores == os.path.join("results", "importance_scores_c4.json") and c.output == os.path.join("results", "calib_c4_gun9.json")
    assert c.log == "log_gun9_c4.txt" and c.scores_dir == os.path.join("results", "gun9_scores", "c4")
    assert not g7.reproduction_check_applies(c) and g7.calib_extra(c) == {"calib_dataset": "c4"}
    e = g7.parse_args(["--calib-dataset", "c4", "--output", "o.json", "--scores", "s.json"])
    assert e.output == "o.json" and e.scores == "s.json"
    assert not g7.reproduction_check_applies(g7.parse_args(["--scores", "results/importance_scores_gun3_n16.json"]))
    dry = g7.parse_args(["--calib-dataset", "c4", "--dry-run"])
    assert dry.output.startswith(g7.DRYRUN_DIR) and dry.scores_dir.startswith(g7.DRYRUN_DIR)  # never writes to the real results/ paths


def test_c4_calibration_batches_verify_passage_ids(tmp_path, monkeypatch):
    docs = [_doc(i, 300) for i in range(20)]
    monkeypatch.setattr(rx, "load_dataset", lambda *a, **k: iter(docs))
    tok = WordTokenizer()
    batches, ids = g7.load_c4_calibration_batches(tok, None)
    assert len(ids) == g7.CALIB["n_passages"] == 16 and len(batches) == 4
    assert all(b["input_ids"].shape == (4, g7.CALIB["max_length"]) and int(b["attention_mask"].sum()) == 4 * 256 for b in batches)  # unpadded
    good = tmp_path / "scores_ok.json"
    good.write_text(json.dumps({"calibration": {"passages": ids}, "scores": {}}), encoding="utf-8")
    assert g7.load_c4_calibration_batches(tok, str(good))[1] == ids
    bad = tmp_path / "scores_bad.json"
    bad.write_text(json.dumps({"calibration": {"passages": [{**p, "sha1": "0" * 16} for p in ids]}, "scores": {}}), encoding="utf-8")
    try:
        g7.load_c4_calibration_batches(tok, str(bad))
    except ValueError as e:
        assert "do not match" in str(e)
    else:
        raise AssertionError("different passage identifiers should raise ValueError")


def test_dry_run_c4_end_to_end(tmp_path):
    out = tmp_path / "calib_c4_dry.json"
    rc = g7.main(["--dry-run", "--calib-dataset", "c4", "--fractions", "0.2", "--configs", "xai_single,xai_iter_fixedq,prune_random",
                  "--n-repeats", "1", "--no-rescore-after", "--output", str(out), "--log", str(tmp_path / "log.txt"),
                  "--scores-dir", str(tmp_path / "scores")])
    assert rc == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["run"]["status"] == "completed" and d["reference"]["calib_dataset"] == "c4"
    cfgs = d["fractions"]["0.2"]["configs"]
    assert list(cfgs) == ["xai_single", "xai_iter_fixedq", "prune_random"] and all(c["status"] == "completed" for c in cfgs.values())
    assert "reproduction_check" not in cfgs["xai_single"] and cfgs["prune_random"]["n_repeats"] == 1
    assert cfgs["xai_iter_fixedq"]["int4_module_set_equals_single"] is True
    round_files = sorted(os.listdir(tmp_path / "scores"))
    assert round_files and all("xai_iter_fixedq_round" in f for f in round_files)
    meta = json.loads((tmp_path / "scores" / round_files[0]).read_text(encoding="utf-8"))
    assert meta["calib_dataset"] == "c4"


class NoPadTokenizer(WordTokenizer):
    """Like the Mistral tokenizer: NO pad_token; like HF, raises ValueError when padding is requested (even if no padding is needed)."""

    pad_token, eos_token, padding_side = None, "</s>", "left"

    def __call__(self, text, add_special_tokens=True, return_tensors=None, padding=False, truncation=False, max_length=None):
        if padding and self.pad_token is None:
            raise ValueError("Asking to pad but the tokenizer does not have a padding token")
        return super().__call__(text, add_special_tokens, return_tensors, padding, truncation, max_length)


def test_c4_batches_work_with_tokenizer_without_pad_token(monkeypatch):
    """Regression: a tokenizer from AutoTokenizer.from_pretrained (not passed through load_model) has no pad_token and broke the preflight."""
    docs = [_doc(i, 300) for i in range(20)]
    monkeypatch.setattr(rx, "load_dataset", lambda *a, **k: iter(docs))
    tok = NoPadTokenizer()
    try:  # root cause of the original failure: build_calibration_batches is called with padding=True
        rx.build_calibration_batches(NoPadTokenizer(), [docs[0]["text"]], 4, 16)
    except ValueError as e:
        assert "padding token" in str(e)
    else:
        raise AssertionError("a tokenizer without a pad token should raise with padding=True (HF behavior)")
    batches, ids = g7.load_c4_calibration_batches(tok, None)
    assert tok.pad_token == tok.eos_token and tok.padding_side == "right"  # same rule as run_xai_on_mistral.main / load_model
    assert len(batches) == 4 and len(ids) == g7.CALIB["n_passages"]
    ref_batches, ref_ids = g7.load_c4_calibration_batches(WordTokenizer(), None)  # path with a pad token: identical result
    assert [p["sha1"] for p in ids] == [p["sha1"] for p in ref_ids]
    assert all(torch.equal(a["input_ids"], b["input_ids"]) and torch.equal(a["attention_mask"], b["attention_mask"]) for a, b in zip(batches, ref_batches))
    ready = NoPadTokenizer()
    ready.pad_token = "<pad>"
    assert g7.ensure_pad_token(ready).pad_token == "<pad>"  # an already configured pad token is left untouched
