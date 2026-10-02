"""
C-1: LoRA recovery — how much of the compressed model's loss can a short training run recover? (GPU; --dry-run runs on CPU)

The MAIN CLAIM of XAI-JQP is TRAINING-FREE (calibration + plan + application); this script is a separate SUPPLEMENTARY experiment.
The stored plan of the final config `xai_iter_fixedq` (results/iterative_gun7.json: list of pruned heads + INT4 module plan; NO
attribution is run) is applied to a clean model (masked pruning -> bitsandbytes NF4; the same path as the pruning experiments),
followed by a short QLoRA recovery run; perplexity is measured before/after and MMLU after training.

QLoRA settings: r=16, alpha=32, dropout 0.05, targets q/k/v/o/gate/up/down_proj (modules in both the NF4 and fp16 tiers),
WikiText-2 TRAIN, 200 steps × batch 4 × 512 tokens (= 409 600 tokens), AdamW lr 2e-4, cosine (10 warmup steps), seed 42,
fp16 autocast + GradScaler (LoRA weights in fp32). Base weights are frozen; the adapter is not merged (it cannot be merged into NF4).

Pruning mask (mandatory): the q_proj ROWS and o_proj COLUMNS of pruned heads must stay zero throughout training. Since the effective
LoRA weight is W + (alpha/r)·B·A, the corresponding rows of lora_B (q_proj) and columns of lora_A (o_proj) are zeroed at the start,
their gradients are masked with a hook (so the Adam moments stay zero as well), and they are re-zeroed after every optimizer step
(projection). After training, `pruned_slice_check` verifies that the effective ΔW is exactly 0 on these slices (JSON: mask_check).
The o_proj mask is essential: the context vector of a pruned head is not zero (q=0 -> uniform attention), so an unmasked LoRA
would reconnect it.

Caveats (also recorded in the JSON "notes"): (1) training on WikiText-2 TRAIN and measuring on WikiText-2 TEST gives an in-domain
advantage (as with GPTQ/AWQ calibration); MMLU is the out-of-domain check. (2) The adapter adds parameters (~42M, ~84 MB in fp16)
and is carried unmerged at inference; its size is reported separately. (3) Single seed, single run.

Usage:
    python experiments/run_lora_recovery.py --dry-run        # mini model + random tokens, CPU (requires peft)
    python experiments/run_lora_recovery.py --preflight      # CUDA, peft/bnb versions, plan, WikiText-2 train, MMLU subset
    python experiments/run_lora_recovery.py --with-tasks     # full run; --with-tasks (opt-in) adds post-training HellaSwag + ARC
                                                 # (eval_tasks, per question); "before" values = results/tasks_gun9.json
    python experiments/run_lora_recovery.py --resume         # skip completed fractions

Outputs: results/lora_recovery_gun10.json (per fraction: ppl before/after, training curve, mask check, MMLU/tasks after), LoRA
adapters under results/lora_gun10/f<fraction>/ and the log file log_gun10_lora.txt.
"""

# HF_HOME must be set BEFORE transformers/datasets are imported (see run_e2e_pipeline.py)
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compressor
from compressor import LayerPlan, apply_structural_pruning, build_compression_plan
from run_ablation_tests import DryRun, _StoreView, heads_from_tiers, set_seed
from run_baselines import ensure_clean_gpu, release_gpu
from run_e2e_pipeline import (
    MODEL_NAME,
    Logger,
    ResultStore,
    cuda_gb,
    env_info,
    load_model,
    load_wikitext_text,
    module_bytes,
    perplexity_tools,
    quantize_with_fallback,
    should_log_to_file,
)
from run_eval_from_plans import collect_entries, planned_int4_modules, prune_only

DEFAULT_PLAN = os.path.join("results", "iterative_gun7.json")
OUTPUT_FILE = os.path.join("results", "lora_recovery_gun10.json")
ADAPTER_DIR = os.path.join("results", "lora_gun10")
LOG_FILE = "log_gun10_lora.txt"
DRYRUN_DIR = "dryrun_out"
PLAN_CONFIG = "xai_iter_fixedq"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
HeadList = List[Tuple[int, int]]


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C-1: xai_iter_fixedq plan + QLoRA recovery training (ppl before/after, MMLU after)")
    p.add_argument("--plan", default=DEFAULT_PLAN, help="plan source (run_iterative_pruning format)")
    p.add_argument("--config", default=PLAN_CONFIG, help="config whose plan is used (default: the final config xai_iter_fixedq)")
    p.add_argument("--fractions", default=None, help="comma-separated fraction keys (default: all fractions in the plan)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--grad-checkpointing", action="store_true", help="reduce activation memory (slower); off by default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mmlu", action="store_true", help="do NOT evaluate MMLU after training")
    p.add_argument("--mmlu-batch-size", type=int, default=8)
    p.add_argument("--with-tasks", action="store_true",
                   help="also evaluate HellaSwag + ARC-Challenge AFTER training (eval_tasks, per-question records; off by default). "
                        "'Before' values are not measured in this run: see the xai_iter_fixedq rows of results/tasks_gun9.json for the same plan")
    p.add_argument("--tasks-batch-size", type=int, default=16)
    p.add_argument("--no-save-adapter", action="store_true", help="do not write the LoRA weights to disk")
    p.add_argument("--adapter-dir", default=ADAPTER_DIR)
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--log", default=LOG_FILE)
    p.add_argument("--resume", action="store_true", help="skip fractions with status=completed in --output")
    p.add_argument("--preflight", action="store_true", help="check packages/plan/data without loading the model, then exit")
    p.add_argument("--dry-run", action="store_true", help="mini model + fake quantization + random tokens, CPU")
    args = p.parse_args(argv)
    args.mmlu = not args.no_mmlu
    if args.dry_run:  # never touch the real results/ files
        if args.output == OUTPUT_FILE:
            args.output = os.path.join(DRYRUN_DIR, "lora_recovery_gun10_dry.json")
        if args.log == LOG_FILE:
            args.log = os.path.join(DRYRUN_DIR, "log_gun10_lora_dry.txt")
        if args.adapter_dir == ADAPTER_DIR:
            args.adapter_dir = os.path.join(DRYRUN_DIR, "lora_gun10")
        args.seq_len = min(args.seq_len, 32)  # mini model: max_position_embeddings 64
        args.steps = min(args.steps, 12)
        args.warmup_steps = min(args.warmup_steps, 2)
    return args


# --------------------------------------------------------------------------- #
# LoRA + pruning mask (no GPU needed; called directly by the tests)
# --------------------------------------------------------------------------- #
def attach_lora(model: nn.Module, rank: int, alpha: int, dropout: float):
    """Attach a peft LoRA adapter; the base stays frozen and the trainable (LoRA) parameters are cast to fp32 (for fp16 autocast + GradScaler)."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:  # pragma: no cover
        raise ImportError("peft is not installed (requirements.txt: peft==0.12.0).") from e

    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(LORA_TARGETS), bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, cfg)
    for p in peft_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()  # peft creates fp16 LoRA weights on fp16 base layers; GradScaler rejects fp16 gradients
    return peft_model


def _hf_model(peft_model: nn.Module) -> nn.Module:
    return peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model


def _lora_weight(module: nn.Module, which: str) -> Optional[torch.Tensor]:
    holder = getattr(module, which, None)  # peft: ModuleDict {"default": nn.Linear}
    if holder is None or "default" not in holder:
        return None
    return holder["default"].weight


def head_slices(heads: HeadList, head_dim: int) -> Dict[int, torch.Tensor]:
    """{layer: q_proj row / o_proj column indices of the pruned heads (LongTensor)}."""
    by_layer: Dict[int, List[int]] = {}
    for layer, h in heads:
        by_layer.setdefault(int(layer), []).extend(range(int(h) * head_dim, (int(h) + 1) * head_dim))
    return {l: torch.tensor(sorted(idx), dtype=torch.long) for l, idx in by_layer.items()}


def apply_lora_masks(peft_model: nn.Module, heads: HeadList) -> Dict[str, Any]:
    """
    Zero the pruned-head slices in LoRA and register gradient-mask hooks: q_proj.lora_B[rows, :] and o_proj.lora_A[:, columns].
    The "project" entry of the returned dict must be called AFTER every optimizer step (numerical safety net).
    """
    hf = _hf_model(peft_model)
    layers = compressor._decoder_layers(hf)
    slices = head_slices(heads, compressor._head_dim(hf))
    targets: List[Tuple[torch.Tensor, torch.Tensor, int]] = []  # (parameter, indices, axis)
    hooks = []
    for layer_idx, idx in slices.items():
        attn = layers[layer_idx].self_attn
        for weight, axis in ((_lora_weight(attn.q_proj, "lora_B"), 0), (_lora_weight(attn.o_proj, "lora_A"), 1)):
            if weight is None:
                raise ValueError(f"layer_{layer_idx}: no LoRA on q_proj/o_proj (target modules: {LORA_TARGETS})")
            index = idx.to(weight.device)
            targets.append((weight, index, axis))

            def hook(grad: torch.Tensor, index: torch.Tensor = index, axis: int = axis) -> torch.Tensor:
                grad = grad.clone()
                grad.index_fill_(axis, index.to(grad.device), 0.0)
                return grad

            hooks.append(weight.register_hook(hook))

    @torch.no_grad()
    def project() -> None:
        for weight, index, axis in targets:
            weight.index_fill_(axis, index, 0.0)

    project()
    return {"project": project, "hooks": hooks, "n_masked_tensors": len(targets), "n_layers": len(slices)}


@torch.no_grad()
def pruned_slice_check(peft_model: nn.Module, heads: HeadList) -> Dict[str, Any]:
    """Is the effective LoRA delta ΔW = (alpha/r)·B·A exactly 0 on the pruned slices, and are the non-quantized base slices still 0?"""
    hf = _hf_model(peft_model)
    layers = compressor._decoder_layers(hf)
    out = {"max_abs_delta_q_rows": 0.0, "max_abs_delta_o_cols": 0.0, "max_abs_base_q_rows": 0.0, "max_abs_base_o_cols": 0.0,
           "max_abs_delta_elsewhere": 0.0}
    for layer_idx, idx in head_slices(heads, compressor._head_dim(hf)).items():
        attn = layers[layer_idx].self_attn
        for name, axis, key in (("q_proj", 0, "q_rows"), ("o_proj", 1, "o_cols")):
            mod = getattr(attn, name)
            a, b = _lora_weight(mod, "lora_A"), _lora_weight(mod, "lora_B")
            delta = b.float() @ a.float()
            sel = delta.index_select(axis, idx.to(delta.device))
            out[f"max_abs_delta_{key}"] = max(out[f"max_abs_delta_{key}"], float(sel.abs().max()))
            out["max_abs_delta_elsewhere"] = max(out["max_abs_delta_elsewhere"], float(delta.abs().max()))
            base = getattr(mod, "base_layer", mod).weight
            if base.is_floating_point() and base.dim() == 2 and base.shape == delta.shape:  # skip bnb 4-bit packed weights
                out[f"max_abs_base_{key}"] = max(out[f"max_abs_base_{key}"], float(base.index_select(axis, idx.to(base.device)).abs().max()))
    out["ok"] = out["max_abs_delta_q_rows"] == 0.0 and out["max_abs_delta_o_cols"] == 0.0 and \
        out["max_abs_base_q_rows"] == 0.0 and out["max_abs_base_o_cols"] == 0.0
    return out


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_lora(peft_model: nn.Module, batches: Iterable[torch.Tensor], heads: HeadList, *, steps: int, lr: float, warmup: int,
               max_grad_norm: float, use_amp: bool, log=None, log_every: int = 10) -> Dict[str, Any]:
    """
    LoRA training with next-token cross-entropy (requires_grad parameters only). `batches`: an iterable yielding [B, T] LongTensors
    (at least `steps` items). Mask: apply_lora_masks (gradient hook + post-step projection).
    """
    device = next(peft_model.parameters()).device
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    mask = apply_lora_masks(peft_model, heads)
    peft_model.train()
    losses: List[float] = []
    n_tokens = 0
    t0 = time.time()
    it = iter(batches)
    try:
        for step in range(steps):
            ids = next(it).to(device)
            for g in optimizer.param_groups:
                g["lr"] = cosine_lr(step, steps, lr, warmup)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = peft_model(input_ids=ids, labels=ids, use_cache=False)
            loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"step {step}: loss is not finite ({float(loss)})")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            mask["project"]()
            losses.append(float(loss))
            n_tokens += int(ids.numel())
            if log and (step % log_every == 0 or step == steps - 1):
                log(f"  step {step + 1}/{steps}: loss {losses[-1]:.4f}, lr {optimizer.param_groups[0]['lr']:.2e}, |g| {grad_norm:.3f}, "
                    f"{time.time() - t0:.0f} s")
    finally:
        for h in mask["hooks"]:
            h.remove()
        peft_model.eval()
    k = max(1, min(10, len(losses)))
    return {"steps": len(losses), "n_tokens": n_tokens, "seconds": time.time() - t0, "loss_first": losses[0], "loss_last": losses[-1],
            "loss_mean_first10": sum(losses[:k]) / k, "loss_mean_last10": sum(losses[-k:]) / k, "losses": losses,
            "n_trainable_params": sum(p.numel() for p in params), "n_masked_tensors": mask["n_masked_tensors"]}


def build_train_batches(tokenizer, text: str, *, steps: int, batch_size: int, seq_len: int, seed: int) -> List[torch.Tensor]:
    """Tokenize the WikiText-2 TRAIN text once, split it into seq_len blocks, shuffle with the seed and take steps × batch_size blocks."""
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_blocks = ids.numel() // seq_len
    need = steps * batch_size
    if n_blocks < need:
        raise ValueError(f"not enough training text: {n_blocks} blocks < {need}")
    order = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(seed))[:need]
    blocks = ids[: n_blocks * seq_len].view(n_blocks, seq_len)[order]
    return [blocks[i * batch_size:(i + 1) * batch_size].clone() for i in range(steps)]


def load_wikitext_train_text() -> str:
    from datasets import load_dataset  # lazy import

    return "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])


def adapter_stats(peft_model: nn.Module) -> Dict[str, Any]:
    params = [p for n, p in peft_model.named_parameters() if "lora_" in n]
    n = sum(p.numel() for p in params)
    return {"n_params": n, "bytes_fp32": 4 * n, "bytes_fp16": 2 * n, "gb_fp16": 2 * n / 1e9}


# --------------------------------------------------------------------------- #
# Single fraction
# --------------------------------------------------------------------------- #
def run_fraction(entry: Dict[str, Any], ctx: Dict[str, Any], rep: Dict[str, Any], store: ResultStore, log: Logger) -> None:
    args, dry = ctx["args"], ctx["dry"]
    fk = entry["fraction"]
    heads: HeadList = entry["heads"]
    plan: Dict[int, LayerPlan] = entry["plan"]
    rep.update({"status": "running", "fraction": fk, "config": entry["config"], "source": entry["source"],
                "started_at": datetime.now().isoformat(timespec="seconds")})
    store.save(f"f={fk} started")
    t0 = time.time()
    set_seed(args.seed)
    rep["cuda_allocated_before_load_gb"] = ensure_clean_gpu(log, f"f={fk}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    box: Dict[str, Any] = {}
    try:
        box["tokenizer"], box["model"], rep["model_load_seconds"] = dry.load_model(log) if dry else load_model(log)
        model, tok = box["model"], box["tokenizer"]
        rep["model_bytes_before"] = module_bytes(model)
        rep["pruning"] = apply_structural_pruning(model, prune_only(plan), verbose=False)
        rep["quantization"] = quantize_with_fallback(model, plan, log, _StoreView(store, rep), "cfg")
        rep["n_pruned_heads"], rep["int4_modules"] = rep["pruning"]["pruned_heads"], rep["quantization"]["int4_modules"]
        src = entry["source"]
        if rep["n_pruned_heads"] != len(heads) or (src.get("int4_modules") is not None and rep["int4_modules"] != src["int4_modules"]):
            raise RuntimeError(f"f={fk}: applied plan does not match the source (heads {rep['n_pruned_heads']}/{len(heads)}, "
                               f"INT4 {rep['int4_modules']}/{src.get('int4_modules')})")
        rep["model_bytes_compressed"] = module_bytes(model)
        log(f"f={fk}: plan applied: {rep['n_pruned_heads']} heads, {rep['int4_modules']} INT4 modules, "
            f"{rep['model_bytes_compressed']['gb']:.3f} GB", tag="LORA")

        def ppl() -> float:
            if dry:
                return dry.perplexity(box["peft"] if "peft" in box else model)
            m = box.get("peft", model)
            return ctx["compute_perplexity"](m, tok, ctx["text_test"], next(m.parameters()).device)

        t1 = time.time()
        rep["ppl_before"] = ppl()
        rep["ppl_before_seconds"] = time.time() - t1
        src_ppl = src.get("perplexity")
        rep["ppl_before_minus_source"] = rep["ppl_before"] - src_ppl if isinstance(src_ppl, (int, float)) else None
        log(f"f={fk}: perplexity BEFORE = {rep['ppl_before']:.4f} (source {src_ppl}, difference {rep['ppl_before_minus_source']})", tag="LORA")
        store.save(f"f={fk} ppl before")

        if args.grad_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        box["peft"] = peft_model = attach_lora(model, args.rank, args.alpha, args.dropout)
        rep["adapter"] = adapter_stats(peft_model)
        if dry:
            g = torch.Generator().manual_seed(args.seed)
            batches = [torch.randint(1, model.config.vocab_size, (args.batch_size, args.seq_len), generator=g) for _ in range(args.steps)]
        else:
            batches = ctx["train_batches"]
        use_amp = torch.cuda.is_available() and not dry
        rep["train"] = train_lora(peft_model, batches, heads, steps=args.steps, lr=args.lr, warmup=args.warmup_steps,
                                  max_grad_norm=args.max_grad_norm, use_amp=use_amp, log=lambda m: log(m, tag="LORA"))
        rep["train"]["amp_fp16"] = use_amp
        rep["mask_check"] = pruned_slice_check(peft_model, heads)
        if not rep["mask_check"]["ok"]:
            raise RuntimeError(f"f={fk}: pruning mask violated: {rep['mask_check']}")
        log(f"f={fk}: training {rep['train']['steps']} steps / {rep['train']['n_tokens']} tokens, {rep['train']['seconds']:.0f} s; loss "
            f"{rep['train']['loss_mean_first10']:.4f} -> {rep['train']['loss_mean_last10']:.4f} (mean of first/last 10 steps); mask OK", tag="LORA")
        store.save(f"f={fk} training")

        t1 = time.time()
        rep["ppl_after"] = ppl()
        rep["ppl_after_seconds"] = time.time() - t1
        rep["ppl_recovered"] = rep["ppl_before"] - rep["ppl_after"]
        fp16 = ctx.get("fp16_ppl")
        if isinstance(fp16, (int, float)) and rep["ppl_before"] > fp16:
            rep["recovered_share_of_loss"] = rep["ppl_recovered"] / (rep["ppl_before"] - fp16)
        log(f"f={fk}: perplexity AFTER = {rep['ppl_after']:.4f} (before {rep['ppl_before']:.4f}, recovered {rep['ppl_recovered']:+.4f}, "
            f"share of loss {rep.get('recovered_share_of_loss')})", tag="LORA")
        store.save(f"f={fk} ppl after")
        if args.mmlu:
            import eval_mmlu

            mlog = lambda m: log(m, tag="MMLU")  # noqa: E731
            if dry:
                rep["mmlu_after"] = eval_mmlu.evaluate_mmlu_dry(peft_model, args.seed, log=mlog, record_questions=True)
            else:
                rep["mmlu_after"] = eval_mmlu.evaluate_mmlu(peft_model, tok, subset=ctx["mmlu_subset"], prompt_style="plain",
                                                            batch_size=args.mmlu_batch_size, log=mlog, record_questions=True)
            rep["mmlu_after_acc"] = rep["mmlu_after"]["mmlu_subset_acc"]
            src_acc = src.get("mmlu_subset_acc")
            rep["mmlu_after_minus_before"] = rep["mmlu_after_acc"] - src_acc if isinstance(src_acc, (int, float)) else None
        if args.with_tasks:  # second out-of-domain check (at 60% MMLU can be uninformative due to answer-letter collapse)
            import eval_tasks

            tlog = lambda m: log(m, tag="TASK")  # noqa: E731
            if dry:
                rep["tasks_after"] = eval_tasks.evaluate_tasks_dry(peft_model, args.seed, log=tlog)
            else:
                rep["tasks_after"] = eval_tasks.evaluate_tasks(peft_model, tok, subsets=ctx["task_subsets"],
                                                               batch_size=args.tasks_batch_size, log=tlog)
            rep["tasks_after_summary"] = eval_tasks.summarize_tasks(rep["tasks_after"])
            log(f"f={fk}: tasks AFTER = " + ", ".join(f"{t} acc {s['acc']:.3f} / acc_norm {s['acc_norm']:.3f}"
                                                        for t, s in rep["tasks_after_summary"].items()), tag="LORA")
            store.save(f"f={fk} tasks after")
        if not args.no_save_adapter:
            path = os.path.join(args.adapter_dir, f"f{fk}")
            for p in peft_model.parameters():  # the adapter is saved in fp16 (half the size; training is finished)
                if p.requires_grad:
                    p.data = p.data.half() if p.is_cuda else p.data
            peft_model.save_pretrained(path, safe_serialization=True)
            rep["adapter"]["path"] = path
            rep["adapter"]["files_bytes"] = {f: os.path.getsize(os.path.join(path, f)) for f in sorted(os.listdir(path))}
        rep["status"] = "completed"
    finally:
        rep["peak_vram_gb"] = cuda_gb("peak")
        box.clear()
        rep["cuda_allocated_after_free_gb"] = release_gpu()
        rep["seconds"] = time.time() - t0
        store.save(f"f={fk} finished")


def format_summary_table(fractions: Dict[str, Any], fp16: Optional[float]) -> str:
    def f(v: Any, spec: str) -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    lines = [f"{'frac':<6}{'ppl before':>10}{'ppl after':>11}{'recovered':>13}{'loss share':>13}{'mmlu bef.':>11}{'mmlu after':>12}"
             f"{'loss first→last':>18}{'VRAM':>7}{'s':>7}  status" + (f"   (FP16 ppl {fp16:.4f})" if isinstance(fp16, (int, float)) else "")]
    for fk, r in fractions.items():
        tr = r.get("train") or {}
        lines.append(f"{fk:<6}{f(r.get('ppl_before'), '.4f'):>10}{f(r.get('ppl_after'), '.4f'):>11}{f(r.get('ppl_recovered'), '+.4f'):>13}"
                     f"{f(r.get('recovered_share_of_loss'), '.3f'):>13}{f((r.get('source') or {}).get('mmlu_subset_acc'), '.3f'):>11}"
                     f"{f(r.get('mmlu_after_acc'), '.3f'):>12}{f(tr.get('loss_mean_first10'), '.3f') + '→' + f(tr.get('loss_mean_last10'), '.3f'):>18}"
                     f"{f(r.get('peak_vram_gb'), '.1f'):>7}{f(r.get('seconds'), '.0f'):>7}  {r.get('status')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log = Logger(args.log, to_file=should_log_to_file(args, LOG_FILE))
    store = ResultStore(args.output, log)
    t_start = time.time()
    dry: Optional[DryRun] = DryRun(args.seed) if args.dry_run else None
    if dry:  # mini model: plan from fake scores (same source as measure_speed --dry-run); the real plan has 7B dimensions
        plan = build_compression_plan(dry.tiers)
        entries = [{"key": "fdry", "fraction": "dry", "config": "dry", "seed": None, "heads": heads_from_tiers(dry.tiers), "plan": plan,
                    "source": {"perplexity": None, "mmlu_subset_acc": None, "int4_modules": planned_int4_modules(plan)}}]
        plan_json: Dict[str, Any] = {}
    else:
        if not os.path.exists(args.plan):
            raise SystemExit(f"plan file not found: {args.plan}")
        with open(args.plan, "r", encoding="utf-8") as f:
            plan_json = json.load(f)
        fractions = [x.strip() for x in args.fractions.split(",") if x.strip()] if args.fractions else None
        entries = collect_entries(plan_json, fractions, [args.config], include_fp16=False)
    fp16_ppl = (plan_json.get("fp16_rerun") or {}).get("perplexity")
    est_per = 1.0 + 1.2 + args.steps * 2.5 / 60.0 + 1.3 + (3.0 if args.mmlu else 0.0) + (3.0 if args.with_tasks else 0.0)
    log(f"===== LoRA recovery starting: plan={args.plan} config={args.config} fractions={[e['fraction'] for e in entries]} "
        f"r={args.rank} alpha={args.alpha} dropout={args.dropout} steps={args.steps} batch={args.batch_size}×{args.seq_len} lr={args.lr} "
        f"seed={args.seed} mmlu={args.mmlu} dry_run={args.dry_run} =====", tag="LORA")
    log(f"Time estimate: ~{est_per:.0f} min/fraction × {len(entries)} = ~{est_per * len(entries):.0f} min", tag="LORA")

    previous = None
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            previous = json.load(f)
        log(f"--resume: read {args.output}; completed fractions will be skipped", tag="LORA")

    store.data = {
        "run": {"started_at": datetime.now().isoformat(timespec="seconds"), "status": "running", "args": vars(args),
                "model": "mini (dry-run)" if dry else MODEL_NAME, "env": env_info(),
                "time_estimate": {"minutes_per_fraction": est_per, "total_minutes": est_per * len(entries)}},
        "reference": {"plan_file": None if dry else args.plan, "plan_config": args.config, "fp16_perplexity": fp16_ppl,
                      "fp16_mmlu_subset_acc": (plan_json.get("fp16_rerun") or {}).get("mmlu_subset_acc"),
                      "lora": {"r": args.rank, "alpha": args.alpha, "dropout": args.dropout, "target_modules": list(LORA_TARGETS)},
                      "training": {"dataset": "wikitext-2-raw-v1 (train split)", "steps": args.steps, "batch_size": args.batch_size,
                                   "seq_len": args.seq_len, "tokens": args.steps * args.batch_size * args.seq_len, "lr": args.lr,
                                   "schedule": f"cosine, {args.warmup_steps} steps linear warmup", "optimizer": "AdamW (wd 0)",
                                   "max_grad_norm": args.max_grad_norm, "precision": "fp16 autocast + GradScaler; LoRA fp32",
                                   "seed": args.seed},
                      "mask": "q_proj.lora_B[pruned rows] and o_proj.lora_A[pruned columns]: zero at start, masked via gradient hook, "
                              "projected after every step; verified by pruned_slice_check after training",
                      "notes": ["THE MAIN CLAIM IS TRAINING-FREE; this supplementary experiment measures how much of the loss can be recovered",
                                "training on WikiText-2 train, perplexity on WikiText-2 test: in-domain advantage; MMLU is the out-of-domain check",
                                "mmlu before = value from the plan source run (run_iterative_pruning); this run only measures AFTER training",
                                "the adapter is not merged; its size is reported separately in the 'adapter' field"]},
        "fractions": {e["fraction"]: {"status": "pending"} for e in entries},
    }

    if args.preflight:
        problems: List[str] = []
        if not dry and not torch.cuda.is_available():
            problems.append("no CUDA")
        for pkg in ("peft", "bitsandbytes"):
            try:
                mod = __import__(pkg)
                log(f"{pkg} {getattr(mod, '__version__', '?')}", tag="LORA")
            except Exception as e:
                if not (dry and pkg == "bitsandbytes"):
                    problems.append(f"could not import {pkg}: {e!r}")
        if not entries:
            problems.append(f"no completed {args.config} config in the plan")
        for e in entries:
            log(f"  fraction {e['fraction']}: {len(e['heads'])} heads, {planned_int4_modules(e['plan'])} INT4 modules, source ppl "
                f"{e['source'].get('perplexity')} mmlu {e['source'].get('mmlu_subset_acc')}", tag="LORA")
        if not dry:
            try:
                n_chars = len(load_wikitext_train_text())
                log(f"WikiText-2 train ready: {n_chars:,} characters", tag="LORA")
            except Exception as e:
                problems.append(f"could not load WikiText-2 train: {e!r}")
            if args.mmlu:
                try:
                    import eval_mmlu

                    log(f"MMLU subset ready: {len(eval_mmlu.load_or_build_subset(log=lambda m: log(m, tag='MMLU'))['records'])} questions", tag="MMLU")
                except Exception as e:
                    problems.append(f"could not load the MMLU subset: {e!r}")
            if args.with_tasks:
                try:
                    import eval_tasks

                    for t in eval_tasks.ALL_TASKS:
                        s = eval_tasks.load_or_build_task_subset(t, log=lambda m: log(m, tag="TASK"))
                        log(f"{t} subset ready: {len(s['records'])} examples", tag="TASK")
                except Exception as e:
                    problems.append(f"could not load the HellaSwag/ARC subsets: {e!r}")
        if problems:
            log("Preflight FAILED (model not loaded): " + "; ".join(problems), tag="ERR")
            return 1
        log("Preflight completed (model not loaded). Nothing written to the result file.", tag="LORA")
        return 0
    if not torch.cuda.is_available() and not dry:
        log("WARNING: no CUDA — a 7B model + bitsandbytes does not run on CPU. Run on a GPU machine (or use --dry-run).", tag="LORA")
    store.save("start")

    ctx: Dict[str, Any] = {"args": args, "dry": dry, "fp16_ppl": fp16_ppl}
    exit_code = 0
    try:
        if not dry:
            log("Loading WikiText-2 test + train...", tag="LORA")
            ctx["text_test"] = load_wikitext_text()
            ctx["compute_perplexity"], max_len, stride = perplexity_tools()
            store.data["reference"]["perplexity_settings"] = {"max_length": max_len, "stride": stride, "dataset": "wikitext-2-raw-v1 (test split)"}
            from transformers import AutoTokenizer

            ctx["train_batches"] = build_train_batches(AutoTokenizer.from_pretrained(MODEL_NAME), load_wikitext_train_text(),
                                                       steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len, seed=args.seed)
            if args.mmlu:
                import eval_mmlu

                ctx["mmlu_subset"] = eval_mmlu.load_or_build_subset(log=lambda m: log(m, tag="MMLU"))
            if args.with_tasks:
                import eval_tasks

                ctx["task_subsets"] = {t: eval_tasks.load_or_build_task_subset(t, log=lambda m: log(m, tag="TASK"))
                                       for t in eval_tasks.ALL_TASKS}
        for i, entry in enumerate(entries, 1):
            fk = entry["fraction"]
            prev = (previous or {}).get("fractions", {}).get(fk)
            if prev and prev.get("status") == "completed":
                store.data["fractions"][fk] = {**prev, "resumed": True}
                log(f"FRACTION {i}/{len(entries)} {fk}: --resume, taken from the previous run (ppl {prev.get('ppl_before')} -> {prev.get('ppl_after')})", tag="LORA")
                continue
            log(f"===== FRACTION {i}/{len(entries)}: {fk} ({len(entry['heads'])} heads, {planned_int4_modules(entry['plan'])} INT4 modules) =====", tag="LORA")
            rep = store.data["fractions"][fk]
            try:
                run_fraction(entry, ctx, rep, store, log)
            except Exception as e:  # a failure in one fraction must not abort the others
                rep.update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
                log(f"FRACTION {fk} ERROR: {e!r} — moving on to the next one", tag="ERR")
                log(rep["traceback"], tag="ERR")
                exit_code = 1
                store.save(f"f={fk} error")
        store.data["run"]["status"] = "completed" if exit_code == 0 else "completed_with_failures"
    except BaseException as e:  # including KeyboardInterrupt / SystemExit: keep partial results
        store.data["run"].update({"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
        log(f"ERROR: {e!r} — writing partial results to disk", tag="ERR")
        log(store.data["run"]["traceback"], tag="ERR")
        exit_code = 1
    finally:
        store.data["run"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
        store.data["run"]["total_seconds"] = time.time() - t_start
        store.save("final")
        log("SUMMARY TABLE:\n" + format_summary_table(store.data["fractions"], fp16_ppl), tag="LORA")
        log(f"===== Done: status={store.data['run']['status']}, total {store.data['run']['total_seconds']:.1f} s =====", tag="LORA")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
