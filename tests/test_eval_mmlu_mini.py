"""
CPU-only mini test for eval_mmlu (no datasets dependency).

With a fake subset + FakeTokenizer: result-dict schema, last-position indexing (in a
right-padded batch), per-subject counts; with a fake row loader: the subset file is
written once, read back unchanged on the next call, and verified via sha1.

Run:
    pytest tests/test_eval_mmlu_mini.py -q
"""
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_mmlu import (  # noqa: E402
    LETTERS,
    FakeTokenizer,
    build_fake_subset,
    encode_prompt,
    evaluate_mmlu,
    evaluate_mmlu_dry,
    letter_token_ids,
    load_or_build_subset,
)
from test_xai_engine_mini import VOCAB, build_mini_model  # noqa: E402


class _LastTokenStub(nn.Module):
    """Predicts `last_letter` at each example's LAST real position and `other_letter` everywhere else (including padding)."""

    def __init__(self, last_letter: int, other_letter: int):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.last_letter, self.other_letter = last_letter, other_letter

    def forward(self, input_ids, attention_mask, use_cache=False):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, VOCAB)
        logits[:, :, 1 + self.other_letter] = 5.0  # FakeTokenizer: " A".." D" = 1..4
        last = attention_mask.sum(dim=1) - 1
        logits[torch.arange(B), last, 1 + self.last_letter] = 9.0
        return SimpleNamespace(logits=logits)


def test_fake_tokenizer_letters_and_prompt():
    tok = FakeTokenizer(VOCAB)
    ids, info = letter_token_ids(tok)
    assert ids == [1, 2, 3, 4] and info["context_stable"] and info["tokens"] == [" A", " B", " C", " D"]
    rec = build_fake_subset()["records"][0]
    enc = encode_prompt(tok, rec, "plain")
    assert enc[0] == tok.bos_token_id and all(0 < i < VOCAB for i in enc)
    assert len(enc) < 64  # max_position_embeddings of the mini model


def test_result_schema_and_counts_on_mini_model():
    model = build_mini_model()
    model.train()
    res = evaluate_mmlu_dry(model, seed=42, log=lambda m: None)
    assert model.training  # training mode restored
    assert res["n_questions"] == 18 and res["mmlu_subset_acc"] == res["accuracy"]
    assert 0.0 <= res["accuracy"] <= 1.0 and res["n_correct"] == round(res["accuracy"] * 18)
    assert set(res["per_subject"]) == {"fake_subject_0", "fake_subject_1", "fake_subject_2"}
    assert sum(s["n"] for s in res["per_subject"].values()) == 18
    assert sum(s["correct"] for s in res["per_subject"].values()) == res["n_correct"]
    assert sum(res["predicted_letter_counts"].values()) == 18
    pf = res["prompt_format"]
    assert pf["style"] == "plain" and pf["chat_template_used"] is False and pf["n_shot"] == 0
    assert res["subset"]["ids_file"] is None and res["seconds"] >= 0
    json.dumps(res)  # JSON-serialisable
    # deterministic
    assert evaluate_mmlu_dry(model, seed=42, log=lambda m: None)["n_correct"] == res["n_correct"]


def test_last_position_indexing_with_right_padding():
    """Prompts of different lengths -> right-padded batch; the prediction must always be read at the LAST real position."""
    subset = build_fake_subset(seed=3)
    subset["records"][1]["question"] += " and then some more words to change the length"
    tok = FakeTokenizer(VOCAB)
    assert len({len(encode_prompt(tok, r)) for r in subset["records"][:4]}) > 1
    res = evaluate_mmlu(_LastTokenStub(last_letter=1, other_letter=0), tok, subset=subset, batch_size=4, log=lambda m: None)
    assert res["predicted_letter_counts"] == {"A": 0, "B": 18, "C": 0, "D": 0}
    assert res["n_correct"] == res["gold_letter_counts"]["B"]
    assert res["mean_correct_prob"] > 0


def test_subset_file_written_once_then_reused_and_verified(tmp_path):
    rows = {s: [{"question": f"{s} q{i}", "choices": ["w", "x", "y", "z"], "answer": i % 4} for i in range(n)]
            for s, n in (("alpha", 30), ("beta", 5))}
    calls = []

    def loader(subject):
        calls.append(subject)
        return rows[subject]

    path = str(tmp_path / "ids.json")
    kw = dict(subjects=("alpha", "beta"), per_subject=8, seed=42, row_loader=loader, log=lambda m: None)
    first = load_or_build_subset(path, **kw)
    meta = first["meta"]
    assert [len(meta["items"][s]["indices"]) for s in ("alpha", "beta")] == [8, 5]  # min(per_subject, n)
    assert meta["items"]["alpha"]["indices"] == sorted(meta["items"]["alpha"]["indices"])
    assert [r["subject"] for r in first["records"]] == ["alpha"] * 8 + ["beta"] * 5
    written = open(path, encoding="utf-8").read()

    # second call: the stored subset is used even if a different seed/per_subject is requested; the file is not rewritten
    second = load_or_build_subset(path, **dict(kw, seed=7, per_subject=3))
    assert second["records"] == first["records"]
    assert open(path, encoding="utf-8").read() == written
    # same seed -> same selection (reproducibility)
    third = load_or_build_subset(str(tmp_path / "ids2.json"), **kw)
    assert third["meta"]["items"] == meta["items"]

    # a changed dataset must not pass silently
    idx = meta["items"]["alpha"]["indices"][0]
    rows["alpha"][idx]["question"] = "changed"
    try:
        load_or_build_subset(path, **kw)
    except ValueError as e:
        assert "sha1" in str(e)
    else:
        raise AssertionError("expected ValueError for a sha1 mismatch")
    assert len(LETTERS) == 4 and calls
