"""
GPU-free mini tests for run_lora_recovery (C-1; requires peft).

  * pruning mask: on a mask-pruned mini Mistral + LoRA training, the effective ΔW = B·A stays EXACTLY 0 on the q_proj rows / o_proj
    columns of the pruned heads (gradient hook + post-step projection); the pruned heads have zero effect: the trained model's
    logits do not change even when the o_proj input of those heads is randomly corrupted
  * unmasked counter-example: the same training without the mask gives ΔW != 0 on the o_proj columns (the test is meaningful)
  * training loss decreases; cosine + warmup schedule; WikiText block selection is deterministic
  * dry run end to end (+ --resume), adapter files are written, JSON schema

peft is installed only in .venv-quant (pinned: peft 0.12.0, transformers 4.44.2); in the .venv (pytest) environment tests that
need peft are skipped when it is missing (early return). Run:
    pytest tests/test_lora_recovery_mini.py -q
    .venv-quant\\Scripts\\python.exe tests/test_lora_recovery_mini.py     # full run with peft (without pytest)
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if importlib.util.find_spec("captum") is None:  # .venv-quant has no captum; the mini model builder imports xai_engine
    _captum = types.ModuleType("captum")
    _captum.attr = types.ModuleType("captum.attr")
    _captum.attr.LayerIntegratedGradients = None
    sys.modules["captum"], sys.modules["captum.attr"] = _captum, _captum.attr

HAS_PEFT = importlib.util.find_spec("peft") is not None

import run_lora_recovery as lr
from compressor import LayerPlan, apply_structural_pruning  # noqa: E402
from test_xai_engine_mini import HEAD_DIM, N_LAYERS, VOCAB, build_mini_model  # noqa: E402

HEADS = [(0, 1), (1, 0), (1, 3)]


def _plan(heads):
    plan = {i: LayerPlan(i) for i in range(N_LAYERS)}
    for l, h in heads:
        plan[l].pruned_heads.append(h)
    return plan


def _batches(n, seed=0, batch=4, seq=16):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (batch, seq), generator=g) for _ in range(n)]


def _pruned_lora_model(seed=0):
    torch.manual_seed(seed)
    model = build_mini_model()
    apply_structural_pruning(model, _plan(HEADS), verbose=False)
    return lr.attach_lora(model, rank=4, alpha=8, dropout=0.0)


def test_cosine_schedule_and_block_selection_are_deterministic():
    assert lr.cosine_lr(0, 200, 2e-4, 10) == 2e-4 / 10 and lr.cosine_lr(9, 200, 2e-4, 10) == 2e-4
    assert math.isclose(lr.cosine_lr(10, 200, 2e-4, 10), 2e-4) and lr.cosine_lr(199, 200, 2e-4, 10) < 2e-6
    assert all(lr.cosine_lr(s, 200, 2e-4, 10) >= lr.cosine_lr(s + 1, 200, 2e-4, 10) for s in range(10, 199))

    class Tok:
        def __call__(self, text, return_tensors=None):
            return types.SimpleNamespace(input_ids=torch.arange(len(text.split())).unsqueeze(0))

    text = " ".join(["w"] * 1000)
    a = lr.build_train_batches(Tok(), text, steps=5, batch_size=4, seq_len=8, seed=42)
    b = lr.build_train_batches(Tok(), text, steps=5, batch_size=4, seq_len=8, seed=42)
    assert len(a) == 5 and all(x.shape == (4, 8) for x in a) and all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[0], lr.build_train_batches(Tok(), text, steps=5, batch_size=4, seq_len=8, seed=43)[0])
    starts = torch.cat([x[:, 0] for x in a])
    assert len(set(starts.tolist())) == 20 and all(int(s) % 8 == 0 for s in starts)  # non-overlapping, block-aligned
    try:
        lr.build_train_batches(Tok(), text, steps=50, batch_size=4, seq_len=8, seed=42)
    except ValueError:
        pass
    else:
        raise AssertionError("insufficient text must raise ValueError")
    assert lr.head_slices([(0, 1), (0, 3), (2, 0)], 4)[0].tolist() == [4, 5, 6, 7, 12, 13, 14, 15]


def test_mask_keeps_pruned_slices_exactly_zero_during_training():
    if not HAS_PEFT:
        return
    model = _pruned_lora_model()
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < n_train < sum(p.numel() for p in model.parameters()) / 2  # only LoRA is trained
    batches = _batches(30)
    with torch.no_grad():
        loss0 = float(model(input_ids=batches[0], labels=batches[0]).loss)
    res = lr.train_lora(model, batches, HEADS, steps=30, lr=5e-3, warmup=2, max_grad_norm=1.0, use_amp=False)
    with torch.no_grad():
        loss1 = float(model(input_ids=batches[0], labels=batches[0]).loss)
    assert res["steps"] == 30 and res["n_tokens"] == 30 * 4 * 16 and res["n_masked_tensors"] == 2 * 2 and loss1 < loss0
    chk = lr.pruned_slice_check(model, HEADS)
    assert chk["ok"] and chk["max_abs_delta_q_rows"] == 0.0 and chk["max_abs_delta_o_cols"] == 0.0
    assert chk["max_abs_base_q_rows"] == 0.0 and chk["max_abs_base_o_cols"] == 0.0 and chk["max_abs_delta_elsewhere"] > 0  # LoRA did learn
    # functional evidence: corrupting the o_proj input of the pruned heads does not change the output (the head is truly disabled)
    hf = model.get_base_model()
    ref = model(input_ids=batches[1]).logits.detach()

    def corrupt(layer, heads):
        def hook(_mod, inputs):
            x = inputs[0].clone()
            for h in heads:
                x[..., h * HEAD_DIM:(h + 1) * HEAD_DIM] += 100.0
            return (x,)
        return hf.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook)

    hooks = [corrupt(0, [1]), corrupt(1, [0, 3])]
    try:
        assert torch.allclose(model(input_ids=batches[1]).logits, ref, atol=1e-5, rtol=0)
    finally:
        for h in hooks:
            h.remove()
    assert not hf.model.layers[0].self_attn.q_proj.lora_B["default"].weight._backward_hooks  # hooks removed


def test_without_mask_pruned_columns_get_reconnected():
    """Counter-example: without the mask the pruned columns of o_proj.lora_A are non-zero -> once B learns, ΔW[:, columns] != 0."""
    if not HAS_PEFT:
        return
    model = _pruned_lora_model()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=0.0)
    model.train()
    for ids in _batches(10):
        opt.zero_grad()
        model(input_ids=ids, labels=ids).loss.backward()
        opt.step()
    chk = lr.pruned_slice_check(model, HEADS)
    assert not chk["ok"] and chk["max_abs_delta_o_cols"] > 0


def test_dry_run_end_to_end_and_resume(tmp_path=None):
    if not HAS_PEFT:
        return
    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    out = os.path.join(tmp_dir, "lora.json")
    common = ["--dry-run", "--output", out, "--log", os.path.join(tmp_dir, "log.txt"), "--adapter-dir", os.path.join(tmp_dir, "adapters")]
    assert lr.main(common + ["--preflight"]) == 0 and not os.path.exists(out)
    assert lr.main(common) == 0
    d = json.load(open(out, encoding="utf-8"))
    assert d["run"]["status"] == "completed" and d["reference"]["lora"] == {"r": 16, "alpha": 32, "dropout": 0.05,
                                                                          "target_modules": list(lr.LORA_TARGETS)}
    assert d["reference"]["training"]["lr"] == 2e-4 and d["reference"]["training"]["seed"] == 42
    r = d["fractions"]["dry"]
    assert r["status"] == "completed" and r["mask_check"]["ok"] and r["train"]["steps"] == 12 and len(r["train"]["losses"]) == 12
    assert r["n_pruned_heads"] > 0 and r["int4_modules"] == r["source"]["int4_modules"]
    assert r["ppl_before"] > 0 and r["ppl_after"] > 0 and abs(r["ppl_recovered"] - (r["ppl_before"] - r["ppl_after"])) < 1e-9
    assert len(r["mmlu_after"]["per_question"]) == r["mmlu_after"]["n_questions"] and r["adapter"]["n_params"] == r["train"]["n_trainable_params"]
    assert any(f.startswith("adapter_model") for f in r["adapter"]["files_bytes"]) and "adapter_config.json" in r["adapter"]["files_bytes"]
    assert "tasks_after" not in r and "tasks_after_summary" not in r  # --with-tasks is OFF by default: schema unchanged
    assert lr.main(common + ["--resume"]) == 0
    assert json.load(open(out, encoding="utf-8"))["fractions"]["dry"].get("resumed") is True


def test_with_tasks_records_per_question_hellaswag_and_arc(tmp_path=None):
    """--with-tasks -> post-training HellaSwag + ARC (eval_tasks), per-question records; independent of MMLU."""
    if not HAS_PEFT:
        return
    tmp_dir = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    out = os.path.join(tmp_dir, "lora_tasks.json")
    assert lr.main(["--dry-run", "--with-tasks", "--no-mmlu", "--no-save-adapter", "--output", out,
                    "--log", os.path.join(tmp_dir, "log.txt"), "--adapter-dir", os.path.join(tmp_dir, "adapters")]) == 0
    r = json.load(open(out, encoding="utf-8"))["fractions"]["dry"]
    assert r["status"] == "completed" and "mmlu_after" not in r and set(r["tasks_after"]) == {"hellaswag", "arc_challenge"}
    for t, res in r["tasks_after"].items():
        assert res["n"] > 0 and 0.0 <= res["acc"] <= 1.0 and 0.0 <= res["acc_norm"] <= 1.0
        assert r["tasks_after_summary"][t] == {k: res[k] for k in ("acc", "acc_norm", "n", "seconds")}
        per_q = next(v for v in res.values() if isinstance(v, list) and v and isinstance(v[0], dict))  # per-question records
        assert len(per_q) == res["n"]


def test_cli_defaults_match_user_spec():
    a = lr.parse_args([])
    assert (a.rank, a.alpha, a.dropout, a.steps, a.batch_size, a.seq_len, a.lr, a.seed) == (16, 32, 0.05, 200, 4, 512, 2e-4, 42)
    assert a.output == lr.OUTPUT_FILE and a.config == "xai_iter_fixedq" and a.mmlu and not a.grad_checkpointing
    assert a.with_tasks is False and lr.parse_args(["--with-tasks"]).with_tasks is True  # opt-in
    dry = lr.parse_args(["--dry-run"])
    assert dry.output.startswith(lr.DRYRUN_DIR) and dry.adapter_dir.startswith(lr.DRYRUN_DIR) and dry.seq_len == 32


TESTS = [test_cosine_schedule_and_block_selection_are_deterministic, test_mask_keeps_pruned_slices_exactly_zero_during_training,
         test_without_mask_pruned_columns_get_reconnected, test_dry_run_end_to_end_and_resume, test_with_tasks_records_per_question_hellaswag_and_arc,
         test_cli_defaults_match_user_spec]

if __name__ == "__main__":  # environment without pytest (.venv-quant: peft 0.12.0 + transformers 4.44.2)
    import transformers

    print(f"[TEST] transformers {transformers.__version__}, torch {torch.__version__}, peft={'yes' if HAS_PEFT else 'MISSING (peft tests skipped)'}")
    for fn in TESTS:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"[TEST] {len(TESTS)}/{len(TESTS)} tests passed.")
