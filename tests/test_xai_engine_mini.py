"""
CPU-only mini-model test for xai_engine.calculate_importance_scores.

Instead of the real Mistral-7B, builds a tiny (~50K parameter) randomly initialised
model with the same architecture (MistralForCausalLM). This verifies within seconds
on CPU that the Captum hooks work on the real Mistral code path
(self_attn.o_proj, mlp.down_proj).

Run:
    python tests/test_xai_engine_mini.py      # prints a report
    pytest tests/test_xai_engine_mini.py -q   # runs the assertions
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import MistralConfig, MistralForCausalLM  # noqa: E402

from xai_engine import calculate_importance_scores, format_importance_report  # noqa: E402

N_LAYERS, N_HEADS, HEAD_DIM, INTER, VOCAB = 2, 4, 8, 64, 128
MLP_N_GROUPS = 2


def build_mini_model(seed: int = 0) -> MistralForCausalLM:
    torch.manual_seed(seed)
    cfg = MistralConfig(
        vocab_size=VOCAB,
        hidden_size=N_HEADS * HEAD_DIM,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=64,
        pad_token_id=0,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model = MistralForCausalLM(cfg).float().eval()
    return model


def build_dummy_dataloader(seed: int = 1, n_batches: int = 2, batch_size: int = 2, seq_len: int = 12):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        input_ids = torch.randint(1, VOCAB, (batch_size, seq_len), generator=g)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[-1, -3:] = 0  # last 3 tokens of the last example are padding: exercises the mask path
        input_ids[-1, -3:] = 0
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return batches


def expected_keys():
    keys = []
    for i in range(N_LAYERS):
        keys += [f"layer_{i}.attn.head_{h}" for h in range(N_HEADS)]
        keys += [f"layer_{i}.mlp.group_{g}" for g in range(MLP_N_GROUPS)]
    return keys


def test_shapes_and_values():
    model = build_mini_model()
    scores = calculate_importance_scores(
        model, build_dummy_dataloader(), n_steps=8, mlp_n_groups=MLP_N_GROUPS, device=torch.device("cpu"), verbose=False
    )
    assert list(scores.keys()) == expected_keys()
    assert all(math.isfinite(v) and v >= 0 for v in scores.values())
    assert sum(scores.values()) > 0
    # the model must be restored to its original state
    assert all(p.requires_grad for p in model.parameters())


def test_block_level_default_mlp():
    model = build_mini_model()
    scores = calculate_importance_scores(model, build_dummy_dataloader(n_batches=1), n_steps=4, verbose=False)
    assert "layer_0.mlp" in scores and "layer_0.mlp.group_0" not in scores
    assert len(scores) == N_LAYERS * (N_HEADS + 1)


def test_ablated_head_gets_zero_importance():
    """Sanity: zeroing a head's columns in o_proj must give that head a score of exactly 0."""
    model = build_mini_model()
    head = 1
    with torch.no_grad():
        model.model.layers[0].self_attn.o_proj.weight[:, head * HEAD_DIM : (head + 1) * HEAD_DIM] = 0.0
    scores = calculate_importance_scores(
        model, build_dummy_dataloader(n_batches=1), n_steps=4, mlp_n_groups=MLP_N_GROUPS, verbose=False
    )
    assert scores[f"layer_0.attn.head_{head}"] == 0.0
    others = [scores[f"layer_0.attn.head_{h}"] for h in range(N_HEADS) if h != head]
    assert all(v > 0 for v in others)


def test_ablated_mlp_group_gets_zero_importance():
    """Sanity: zeroing a neuron group's columns in down_proj must give that group a score of exactly 0."""
    model = build_mini_model()
    group_size = INTER // MLP_N_GROUPS
    with torch.no_grad():
        model.model.layers[1].mlp.down_proj.weight[:, 0:group_size] = 0.0
    scores = calculate_importance_scores(
        model, build_dummy_dataloader(n_batches=1), n_steps=4, mlp_n_groups=MLP_N_GROUPS, verbose=False
    )
    assert scores["layer_1.mlp.group_0"] == 0.0
    assert scores["layer_1.mlp.group_1"] > 0


if __name__ == "__main__":
    model = build_mini_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[TEST] Mini MistralForCausalLM: {N_LAYERS} layers, {N_HEADS} heads x {HEAD_DIM} dim, "
          f"MLP intermediate size {INTER}, {n_params:,} parameters (random weights), device=cpu")

    scores = calculate_importance_scores(
        model, build_dummy_dataloader(), n_steps=8, mlp_n_groups=MLP_N_GROUPS, device=torch.device("cpu")
    )
    print("\n[TEST] Raw score dictionary:")
    for k, v in scores.items():
        print(f"  {k!r}: {v:.6f}")
    print("\n[TEST] Sorted importance report:")
    print(format_importance_report(scores))

    print("\n[TEST] Running ablation sanity checks...")
    test_shapes_and_values()
    test_block_level_default_mlp()
    test_ablated_head_gets_zero_importance()
    test_ablated_mlp_group_gets_zero_importance()
    print("[TEST] 4/4 tests passed.")
