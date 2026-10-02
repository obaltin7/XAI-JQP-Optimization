"""
CPU-only mini tests for compressor.head_taylor_scores / head_wanda_scores.

Using the mini MistralForCausalLM (tests/test_xai_engine_mini.build_mini_model) and
dummy calibration batches: output shape/key order match head_magnitude_scores, no
NaN/Inf, a head with zeroed o_proj columns scores exactly 0, and the model state
(weights, requires_grad, .grad, train/eval) is unchanged after the call.

Usage:
    pytest tests/test_head_criteria_mini.py -q
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compressor import (  # noqa: E402
    allocate_compression_tiers,
    head_magnitude_scores,
    head_taylor_scores,
    head_wanda_scores,
)
from test_xai_engine_mini import HEAD_DIM, build_dummy_dataloader, build_mini_model  # noqa: E402

CRITERIA = [head_taylor_scores, head_wanda_scores]


def test_shape_matches_magnitude_and_values_are_finite():
    model = build_mini_model()
    batches = build_dummy_dataloader()
    expected = list(head_magnitude_scores(model))
    for fn in CRITERIA:
        scores = fn(model, batches)
        assert list(scores) == expected, fn.__name__
        assert all(isinstance(v, float) and math.isfinite(v) and v > 0 for v in scores.values()), fn.__name__
        # a head-only dict can be passed directly to tier assignment; the lowest-scoring head is pruned
        tiers = allocate_compression_tiers(scores, 3, tier_fractions=(0.25, 0.25, 0.5))
        assert tiers[min(scores, key=scores.get)] == "prune"


def test_zeroed_head_scores_exactly_zero():
    model = build_mini_model()
    head = 2
    with torch.no_grad():
        model.model.layers[1].self_attn.o_proj.weight[:, head * HEAD_DIM:(head + 1) * HEAD_DIM] = 0
    for fn in CRITERIA:
        scores = fn(model, build_dummy_dataloader())
        assert scores[f"layer_1.attn.head_{head}"] == 0.0, fn.__name__
        assert all(v > 0 for k, v in scores.items() if k != f"layer_1.attn.head_{head}"), fn.__name__


def test_model_state_unchanged_after_call():
    for fn in CRITERIA:
        model = build_mini_model()
        # mixed initial state: one parameter frozen, one o_proj with a pre-existing .grad, model in train mode
        model.model.embed_tokens.weight.requires_grad_(False)
        o0 = model.model.layers[0].self_attn.o_proj.weight
        o0.grad = torch.ones_like(o0)
        model.train()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        flags = {n: p.requires_grad for n, p in model.named_parameters()}

        fn(model, build_dummy_dataloader())

        after = model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before), fn.__name__
        assert {n: p.requires_grad for n, p in model.named_parameters()} == flags, fn.__name__
        assert model.training, fn.__name__
        assert o0.grad is not None and torch.equal(o0.grad, torch.ones_like(o0)), fn.__name__
        others = [p for n, p in model.named_parameters() if p is not o0]
        assert all(p.grad is None for p in others), fn.__name__
        assert not any(layer.self_attn.o_proj._forward_pre_hooks for layer in model.model.layers), fn.__name__


def test_deterministic_and_padding_is_masked():
    """Same input -> same score; token ids at pad positions must not affect the score (attention_mask path)."""
    model = build_mini_model()
    a = build_dummy_dataloader()
    b = build_dummy_dataloader()
    for batch in b:
        batch["input_ids"][-1, -3:] = 7  # different ids at pad positions (mask=0)
    for fn in CRITERIA:
        s1, s2, s3 = fn(model, a), fn(model, a), fn(model, b)
        assert s1 == s2, fn.__name__
        assert all(abs(s1[k] - s3[k]) <= 1e-6 * max(1.0, abs(s1[k])) for k in s1), fn.__name__


def test_empty_calibration_raises():
    model = build_mini_model()
    for fn in CRITERIA:
        try:
            fn(model, [])
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__}: expected ValueError for empty calibration")
