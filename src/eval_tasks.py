"""
Additional tasks: HellaSwag and ARC-Challenge (0-shot, choice log-likelihood; acc + acc_norm).

Follows the eval_mmlu pattern (model, tokenizer -> dict; fixed subset file + sha1 verification; --dry-run with the
mini model). Because MMLU no longer discriminates between configurations at pruning ratios of 40% and above, the
evaluation is extended with two multiple-choice tasks:

  hellaswag      Rowan/hellaswag, validation (10,042), 500 fixed examples. Context = "{activity_label}: {ctx_a} {ctx_b.capitalize()}"
                 (lm-eval-harness preprocessing: " [title]" -> ". ", bracketed tags removed); choices = 4 endings.
  arc_challenge  allenai/ai2_arc (ARC-Challenge), test (1,172), 500 fixed examples. Context = "Question: {question}\\nAnswer:";
                 choices = option texts (3–5 choices); gold = position of answerKey in the label list.

Scoring (same definition as lm-eval-harness): for each choice, the SUM of the context-conditioned log-probabilities of
the continuation tokens (" " + choice) (full-vocabulary log_softmax, fp32). acc = argmax(sum); acc_norm = argmax(sum /
BYTE length of the choice text). Continuation tokens = tokenize(context + continuation)[len(tokenize(context)):] (token
merges at the boundary are counted in n_boundary_mismatch). No chat template is used (raw text, as in the perplexity and
MMLU harnesses). Forward passes only, so bnb-quantized models are supported.

Subsets: random.Random(42).sample(range(n), 500), indices sorted ascending; written ONCE to results/<task>_subset_ids.json
(index + dataset id + context sha1); later runs read the file and verify the sha1 hashes (error if the dataset changed).
Per-question records are ALWAYS stored ("per_question": choice log-likelihoods, predictions) for per-example drift analysis.

Usage:
    python src/eval_tasks.py --build-subsets     # download the datasets, write/verify the subset files (no model)
    python src/eval_tasks.py --dry-run           # mini model + fake subsets on CPU (no datasets needed)
    python src/eval_tasks.py [--model M]         # GPU: FP16 model, result in results/tasks_fp16_gun9.json

    from eval_tasks import evaluate_tasks
    res = evaluate_tasks(model, tokenizer)   # res["hellaswag"]["acc_norm"], res["arc_challenge"]["acc"], ...

Estimated cost on an L40S (7B, batch 16): HellaSwag 2000 sequences × ~90 tokens ≈ 30–40 s, ARC ≈ 2000 sequences × ~45 tokens
≈ 15–25 s (FP16); ~1.5–2× in NF4. Extra VRAM < 2 GB (only the logits at continuation positions are cast to fp32).
"""

# HF_HOME must be set BEFORE datasets is imported (see eval_mmlu.py); a value set by the calling script is left untouched.
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

DEFAULT_SEED = 42
DEFAULT_N = 500
OUTPUT_FILE = os.path.join("results", "tasks_fp16_gun9.json")
TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "hellaswag": {"dataset": "Rowan/hellaswag", "config": None, "split": "validation", "id_field": "ind",
                  "ids_file": os.path.join("results", "hellaswag_subset_ids.json"), "chance": 0.25},
    "arc_challenge": {"dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test", "id_field": "id",
                      "ids_file": os.path.join("results", "arc_challenge_subset_ids.json"), "chance": 0.25},
}
ALL_TASKS: Tuple[str, ...] = tuple(TASK_SPECS)

Record = Dict[str, Any]  # {"task", "index", "id", "context", "choices": [str], "gold": int}
RowLoader = Callable[[str], Sequence[Dict[str, Any]]]


# --------------------------------------------------------------------------- #
# Record conversion (same preprocessing as lm-eval-harness)
# --------------------------------------------------------------------------- #
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _to_record(task: str, index: int, row: Dict[str, Any]) -> Record:
    if task == "hellaswag":
        context = _hellaswag_preprocess(f"{row['activity_label']}: {row['ctx_a']} {str(row['ctx_b']).capitalize()}")
        choices = [_hellaswag_preprocess(str(e)) for e in row["endings"]]
        gold = int(row["label"])
    elif task == "arc_challenge":
        context = f"Question: {row['question']}\nAnswer:"
        choices = [str(t) for t in row["choices"]["text"]]
        labels = [str(x) for x in row["choices"]["label"]]
        gold = labels.index(str(row["answerKey"]))
    else:
        raise ValueError(f"unknown task: {task!r}; valid: {ALL_TASKS}")
    if len(choices) < 2 or not 0 <= gold < len(choices):
        raise ValueError(f"{task}[{index}]: unexpected record (choices={len(choices)}, gold={gold})")
    return {"task": task, "index": int(index), "id": str(row.get(TASK_SPECS[task]["id_field"], index)),
            "context": context, "choices": choices, "gold": gold}


def _hf_row_loader(task: str) -> Sequence[Dict[str, Any]]:
    from datasets import load_dataset  # lazy import: datasets is optional locally and not needed for --dry-run

    spec = TASK_SPECS[task]
    args = (spec["dataset"],) if spec["config"] is None else (spec["dataset"], spec["config"])
    return load_dataset(*args, split=spec["split"])


# --------------------------------------------------------------------------- #
# Subset: selection, persistent id file, verification
# --------------------------------------------------------------------------- #
def load_or_build_task_subset(task: str, ids_path: Optional[str] = None, *, n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
                              row_loader: Optional[RowLoader] = None, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """
    Fixed task subset: {"meta": {...}, "records": [Record, ...]} (same pattern as eval_mmlu.load_or_build_subset).
    If ids_path EXISTS, indices are read from the file (the n/seed arguments are ignored; a warning is logged if they
    differ) and each example's dataset id and context sha1 are verified. OTHERWISE random.Random(seed).sample(range(N),
    min(n, N)) is drawn, sorted ascending and written atomically.
    """
    if task not in TASK_SPECS:
        raise ValueError(f"unknown task: {task!r}; valid: {ALL_TASKS}")
    spec = TASK_SPECS[task]
    ids_path = ids_path or spec["ids_file"]
    rows = (row_loader or _hf_row_loader)(task)
    if os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("task") != task:
            raise ValueError(f"{ids_path}: task {meta.get('task')!r}, expected {task!r}")
        if (meta.get("n"), meta.get("seed")) != (n, seed):
            log(f"{task}: {ids_path} exists; using the stored subset instead of the requested n={n}/seed={seed} "
                f"(n={meta.get('n')}, seed={meta.get('seed')})")
        records = [_to_record(task, i, rows[int(i)]) for i in meta["indices"]]
        bad = [r["index"] for r, rid, h in zip(records, meta["ids"], meta["context_sha1"])
               if r["id"] != rid or _sha1(r["context"]) != h]
        if bad:
            raise ValueError(f"{task}: id/sha1 of {len(bad)} example(s) do not match {ids_path} (first: {bad[:3]}); the dataset version "
                             "may have changed. Investigate the cause before deleting the subset file.")
        return {"meta": meta, "records": records}

    n_avail = len(rows)
    indices = sorted(random.Random(seed).sample(range(n_avail), min(n, n_avail)))
    records = [_to_record(task, i, rows[i]) for i in indices]
    meta = {"task": task, "dataset": spec["dataset"], "config": spec["config"], "split": spec["split"], "seed": seed,
            "n": len(records), "n_available": n_avail, "created_at": datetime.now().isoformat(timespec="seconds"),
            "selection": "random.Random(seed).sample(range(n_available), min(n, n_available)); indices sorted ascending",
            "indices": indices, "ids": [r["id"] for r in records], "context_sha1": [_sha1(r["context"]) for r in records]}
    os.makedirs(os.path.dirname(ids_path) or ".", exist_ok=True)
    tmp = ids_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    os.replace(tmp, ids_path)
    log(f"{task}: subset built and written: {ids_path} ({len(records)} examples / {n_avail})")
    return {"meta": meta, "records": records}


def build_fake_task_subset(task: str, seed: int = DEFAULT_SEED, n: int = 8) -> Dict[str, Any]:
    """For --dry-run / tests: short fake examples in the real record schema (NOT written to disk); 3–5 choices for ARC."""
    rng = random.Random(f"{task}-{seed}")
    words = ["alpha", "beta", "gamma", "delta", "kappa", "sigma", "omega", "theta", "lambda", "zeta"]
    records: List[Record] = []
    for i in range(n):
        n_choices = 4 if task == "hellaswag" else rng.choice((3, 4, 5))
        ctx_words = " ".join(rng.choice(words) for _ in range(rng.randint(4, 9)))
        context = f"Activity: {ctx_words}" if task == "hellaswag" else f"Question: {ctx_words} ?\nAnswer:"
        choices = [" ".join(rng.choice(words) for _ in range(rng.randint(1, 5))) + f" {k}" for k in range(n_choices)]
        records.append({"task": task, "index": i, "id": f"fake-{task}-{i}", "context": context, "choices": choices,
                        "gold": rng.randrange(n_choices)})
    meta = {"task": task, "dataset": "fake (dry-run)", "config": None, "split": None, "seed": seed, "n": len(records)}
    return {"meta": meta, "records": records}


# --------------------------------------------------------------------------- #
# Tokenization and evaluation
# --------------------------------------------------------------------------- #
def _encode(tokenizer, text: str, add_special_tokens: bool) -> List[int]:
    return list(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def encode_pair(tokenizer, context: str, continuation: str) -> Tuple[List[int], List[int], bool]:
    """
    (context tokens, continuation tokens, boundary_stable). lm-eval-harness method: the full text is tokenized and the first
    as many tokens as the context's token COUNT are taken as context; if the context prefix is not preserved exactly (a merge
    at the boundary), boundary_stable=False.
    """
    ctx = _encode(tokenizer, context, True)
    whole = _encode(tokenizer, context + continuation, True)
    cont = whole[len(ctx):]
    stable = whole[: len(ctx)] == ctx
    if not cont:  # edge case: the continuation merged entirely into the context -> encode it separately
        cont, stable = _encode(tokenizer, continuation, False), False
        whole = ctx + cont
    return whole[: len(whole) - len(cont)], cont, stable


@torch.no_grad()
def evaluate_task(model: nn.Module, tokenizer, task: str, *, subset: Optional[Dict[str, Any]] = None, ids_path: Optional[str] = None,
                  batch_size: int = 16, max_length: int = 1024, device: Optional[torch.device] = None,
                  log: Callable[[str], None] = print) -> Dict[str, Any]:
    """
    0-shot multiple-choice accuracy on the fixed subset: acc (raw log-likelihood sum) and acc_norm (divided by byte length).

    Returns: {"task", "acc", "acc_norm", "n", "n_correct", "n_correct_norm", "chance", "seconds", "per_question": [{"index", "id",
      "gold", "pred", "pred_norm", "correct", "correct_norm", "ll": [per-choice log-likelihood sums], "cont_bytes": [...]}],
      "mean_gold_prob_norm", "prompt_format", "subset", "batch_size", "max_length", "n_sequences", "n_truncated", "n_boundary_mismatch"}
    """
    t0 = time.time()
    if subset is None:
        subset = load_or_build_task_subset(task, ids_path, log=log)
    records: List[Record] = subset["records"]
    if not records:
        raise ValueError(f"{task}: subset is empty.")
    if device is None:
        device = next(model.parameters()).device
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0

    # (example, choice) pairs -> sequences; batches sorted by length (less padding waste), results written back in place
    seqs: List[Tuple[int, int, List[int], int]] = []  # (example idx, choice idx, tokens, continuation length)
    n_truncated = n_mismatch = 0
    for qi, r in enumerate(records):
        for ci, choice in enumerate(r["choices"]):
            ctx, cont, stable = encode_pair(tokenizer, r["context"], " " + choice)
            n_mismatch += int(not stable)
            ids = ctx + cont
            if len(ids) > max_length:  # truncate from the left (first token kept); the continuation always stays intact
                ids = ids[:1] + ids[-(max_length - 1):]
                n_truncated += 1
            seqs.append((qi, ci, ids, len(cont)))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][2]))
    ll: List[List[float]] = [[0.0] * len(r["choices"]) for r in records]

    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(order), batch_size):
            chunk = [seqs[i] for i in order[start:start + batch_size]]
            width = max(len(s[2]) for s in chunk)
            input_ids = torch.full((len(chunk), width), int(pad_id), dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, (_, _, ids, _) in enumerate(chunk):  # right padding, independent of tokenizer.padding_side
                input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                mask[j, : len(ids)] = 1
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device), use_cache=False).logits
            for j, (qi, ci, ids, n_cont) in enumerate(chunk):
                end = len(ids)
                pos = logits[j, end - n_cont - 1:end - 1].float()  # positions that predict the continuation tokens
                logp = torch.log_softmax(pos, dim=-1)
                target = torch.tensor(ids[end - n_cont:], dtype=torch.long, device=logp.device)
                total = float(logp.gather(-1, target.unsqueeze(-1)).sum())
                if total != total or total in (float("inf"), float("-inf")):
                    raise RuntimeError(f"{task}: NaN/Inf in the log-likelihood of example {qi}, choice {ci}.")
                ll[qi][ci] = total
    finally:
        model.train(was_training)

    per_question: List[Dict[str, Any]] = []
    n_correct = n_correct_norm = 0
    gold_probs: List[float] = []
    for r, scores in zip(records, ll):
        nbytes = [max(len(c.encode("utf-8")), 1) for c in r["choices"]]
        norm = [s / b for s, b in zip(scores, nbytes)]
        pred = max(range(len(scores)), key=lambda k: scores[k])
        pred_norm = max(range(len(norm)), key=lambda k: norm[k])
        n_correct += int(pred == r["gold"])
        n_correct_norm += int(pred_norm == r["gold"])
        gold_probs.append(float(torch.softmax(torch.tensor(norm), dim=-1)[r["gold"]]))
        per_question.append({"index": r["index"], "id": r["id"], "gold": r["gold"], "pred": pred, "pred_norm": pred_norm,
                             "correct": pred == r["gold"], "correct_norm": pred_norm == r["gold"], "ll": scores, "cont_bytes": nbytes})
    meta = subset["meta"]
    n = len(records)
    result = {
        "task": task, "acc": n_correct / n, "acc_norm": n_correct_norm / n, "n": n, "n_correct": n_correct,
        "n_correct_norm": n_correct_norm, "chance": TASK_SPECS[task]["chance"], "seconds": time.time() - t0,
        "mean_gold_prob_norm": sum(gold_probs) / n, "per_question": per_question,
        "prompt_format": {"n_shot": 0, "chat_template_used": False,
                          "scoring": "context-conditioned log-likelihood sum of the choice continuation (' ' + choice) (full-vocabulary log_softmax, fp32); "
                                     "acc = argmax(sum), acc_norm = argmax(sum / choice byte length); lm-eval-harness definition"},
        "subset": {k: meta.get(k) for k in ("task", "dataset", "config", "split", "seed", "n", "created_at")},
        "batch_size": batch_size, "max_length": max_length, "n_sequences": len(seqs), "n_truncated": n_truncated,
        "n_boundary_mismatch": n_mismatch,
    }
    result["subset"]["ids_file"] = (ids_path or TASK_SPECS[task]["ids_file"]) if meta.get("dataset") == TASK_SPECS[task]["dataset"] else None
    log(f"{task} (0-shot): acc {result['acc']:.4f}, acc_norm {result['acc_norm']:.4f} ({n} examples, {len(seqs)} sequences), "
        f"{result['seconds']:.1f} s; boundary mismatches {n_mismatch}, truncated {n_truncated}")
    return result


def evaluate_tasks(model: nn.Module, tokenizer, tasks: Sequence[str] = ALL_TASKS, *, subsets: Optional[Dict[str, Dict[str, Any]]] = None,
                   batch_size: int = 16, max_length: int = 1024, log: Callable[[str], None] = print) -> Dict[str, Dict[str, Any]]:
    """Several tasks: {task: evaluate_task result}; if subsets are given the datasets are not reloaded."""
    return {t: evaluate_task(model, tokenizer, t, subset=(subsets or {}).get(t), batch_size=batch_size, max_length=max_length, log=log)
            for t in tasks}


def evaluate_tasks_dry(model: nn.Module, seed: int = DEFAULT_SEED, tasks: Sequence[str] = ALL_TASKS,
                       log: Callable[[str], None] = print) -> Dict[str, Dict[str, Any]]:
    """Mini model + eval_mmlu.FakeTokenizer + fake subsets (code-path check; accuracy is meaningless)."""
    from eval_mmlu import FakeTokenizer

    tok = FakeTokenizer(model.config.vocab_size)
    max_len = getattr(model.config, "max_position_embeddings", 64)
    return {t: evaluate_task(model, tok, t, subset=build_fake_task_subset(t, seed), batch_size=4, max_length=max_len, log=log)
            for t in tasks}


def summarize_tasks(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Short summary without per-question records (for tables/logs)."""
    return {t: {"acc": r["acc"], "acc_norm": r["acc_norm"], "n": r["n"], "seconds": r["seconds"]} for t, r in results.items()}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HellaSwag + ARC-Challenge subsets (0-shot, choice log-likelihood; acc + acc_norm)")
    p.add_argument("--build-subsets", action="store_true", help="only build/verify the subset files (no model)")
    p.add_argument("--dry-run", action="store_true", help="mini Mistral + fake subsets on CPU")
    p.add_argument("--tasks", default=",".join(ALL_TASKS))
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--model", default=None, help="HF model id (default: Mistral-7B-Instruct-v0.3)")
    p.add_argument("--output", default=OUTPUT_FILE)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASK_SPECS]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}; valid: {list(ALL_TASKS)}")

    if args.dry_run:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        from test_xai_engine_mini import build_mini_model

        res = evaluate_tasks_dry(build_mini_model(), args.seed, tasks)
        print(json.dumps(summarize_tasks(res), indent=2, ensure_ascii=False))
        return 0

    subsets = {t: load_or_build_task_subset(t, n=args.n, seed=args.seed) for t in tasks}
    for t, s in subsets.items():
        print(f"{t}: {len(s['records'])} examples ({TASK_SPECS[t]['ids_file']})")
    if args.build_subsets:
        return 0

    from run_e2e_pipeline import MODEL_NAME, Logger, load_model

    model_name = args.model or MODEL_NAME
    tokenizer, model, load_s = load_model(Logger("log_tasks_gun9.txt"), model_name=model_name)
    res = evaluate_tasks(model, tokenizer, tasks, subsets=subsets, batch_size=args.batch_size)
    payload = {"model": model_name, "precision": "fp16", "model_load_seconds": load_s, "summary": summarize_tasks(res), "tasks": res}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Result saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
