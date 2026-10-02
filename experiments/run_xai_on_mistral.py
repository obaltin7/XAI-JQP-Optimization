"""
Structural importance scores on the full Mistral-7B model.

Runs xai_engine.calculate_importance_scores on the real model with a WikiText-2
calibration sample. The default setting keeps the cost bounded:
few passages (max_batches=4), short window (256 tokens), n_steps=8.

Usage (single L40S GPU):
    python experiments/run_xai_on_mistral.py
    python experiments/run_xai_on_mistral.py --n-steps 16 --max-batches 8 --dtype bf16
    python experiments/run_xai_on_mistral.py --calib-dataset c4      # out-of-domain calibration -> results/importance_scores_c4.json

Output:
    results/importance_scores_gun3.json  (scores + run settings + timing)
"""

# HF_HOME must be set BEFORE transformers/datasets are imported; otherwise the
# model and dataset caches are written to the pod's non-persistent disk
# (/root/.cache) and Mistral-7B (~14GB) is re-downloaded after every restart.
# /workspace is the persistent volume.
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import json
import time
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xai_engine import calculate_importance_scores, format_importance_report

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "importance_scores_gun3.json")
# Second (out-of-domain) calibration set: allenai/c4, config "en", validation split, streaming.
CALIB_DATASETS = ("wikitext2", "c4")
C4_DATASET = {"path": "allenai/c4", "name": "en", "split": "validation"}
C4_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "importance_scores_c4.json")
DATASET_LABELS = {"wikitext2": "wikitext-2-raw-v1 (test split)", "c4": "allenai/c4 en (validation split, streaming)"}

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LIG structural importance scores on Mistral-7B")
    p.add_argument("--dtype", choices=DTYPES.keys(), default="fp16",
                   help="Model loading precision (default fp16; try bf16 if IG gradients overflow)")
    p.add_argument("--n-passages", type=int, default=16,
                   help="Number of short passages taken from the WikiText-2 test split")
    p.add_argument("--max-length", type=int, default=256,
                   help="Maximum number of tokens per passage (truncation)")
    p.add_argument("--min-chars", type=int, default=200,
                   help="Minimum number of characters for a passage to enter the calibration set")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Number of passages per batch (B)")
    p.add_argument("--max-batches", type=int, default=4,
                   help="Maximum number of batches passed to the engine (cost bound)")
    p.add_argument("--n-steps", type=int, default=8,
                   help="Number of IG Riemann steps (n_steps forward+backward passes per block)")
    p.add_argument("--internal-batch-size", type=int, default=2,
                   help="Chunk size in which Captum processes the n_steps*B samples (VRAM bound)")
    p.add_argument("--top-k", type=int, default=40,
                   help="Number of highest-scoring blocks printed to the console")
    p.add_argument("--output", default=OUTPUT_FILE, help="Output JSON path")
    p.add_argument("--seed", type=int, default=None,
                   help="torch/cuda/numpy/random seed (default None = unseeded, as in the reference run). "
                        "Does NOT change passage selection (deterministic first N paragraphs); use --passage-offset for a second sample")
    p.add_argument("--passage-offset", type=int, default=0,
                   help="Skip the first N eligible paragraphs: a second calibration sample disjoint from the default one (e.g. 16)")
    p.add_argument("--calib-dataset", choices=CALIB_DATASETS, default="wikitext2",
                   help="Calibration source: wikitext2 (default) | c4 (allenai/c4 en, validation, "
                        "streaming; first N passages with >= --max-length tokens; default output results/importance_scores_c4.json)")
    args = p.parse_args(argv)
    if args.calib_dataset == "c4" and args.output == OUTPUT_FILE:
        args.output = C4_OUTPUT_FILE
    return args


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Calibration data
# --------------------------------------------------------------------------- #
def select_passages(texts: List[str], n_passages: int, min_chars: int, offset: int = 0) -> List[str]:
    """
    Select the first n_passages 'real' paragraphs from WikiText-2 lines.
    Empty lines and section headings of the form ' = Heading = ' are skipped;
    very short lines are skipped as well (to provide enough context).
    offset: the first `offset` eligible paragraphs are skipped -> a second calibration sample
    disjoint from the default one (selection is deterministic, so --seed does NOT change the passages).
    """
    if offset < 0:
        raise ValueError("offset must be >= 0")
    selected: List[str] = []
    skipped = 0
    for line in texts:
        s = line.strip()
        if not s or s.startswith("="):
            continue
        if len(s) < min_chars:
            continue
        if skipped < offset:
            skipped += 1
            continue
        selected.append(s)
        if len(selected) >= n_passages:
            break
    if len(selected) < n_passages:
        print(f"[XAI RUN] Warning: only {len(selected)} passages found (requested {n_passages})")
    return selected


def select_c4_passages(stream, tokenizer, n_passages: int, min_tokens: int, offset: int = 0):
    """
    Select the first n_passages 'eligible' documents from the C4 stream: >= min_tokens tokens under the tokenizer
    (special tokens included). The stream order is fixed (no shuffling), so selection is deterministic and seed-free;
    offset skips the first eligible documents. With min_tokens = --max-length every passage fills the window
    completely (padding-free [B, T] batches).

    Returns: (texts, ids); id = {"stream_index", "url", "sha1" (first 16 hex chars of the text), "n_tokens", "n_chars"}
    -- C4 has no record-ID field, so stream index + url + sha1 are stored together (for reproducibility checks).
    """
    import hashlib

    if offset < 0:
        raise ValueError("offset must be >= 0")
    selected: List[str] = []
    ids: List[Dict[str, object]] = []
    skipped = 0
    for idx, row in enumerate(stream):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        n_tok = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        if n_tok < min_tokens:
            continue
        if skipped < offset:
            skipped += 1
            continue
        selected.append(text)
        ids.append({"stream_index": idx, "url": row.get("url"), "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
                    "n_tokens": n_tok, "n_chars": len(text)})
        if len(selected) >= n_passages:
            break
    if len(selected) < n_passages:
        print(f"[XAI RUN] Warning: only {len(selected)} eligible passages found in the C4 stream (requested {n_passages})")
    return selected, ids


def load_calibration_passages(tokenizer, calib_dataset: str, n_passages: int, min_chars: int, max_length: int, offset: int = 0):
    """
    Calibration passages: (texts, c4 ids | None). wikitext2 = the default path (WikiText-2 test,
    select_passages); c4 = the first n_passages documents with >= max_length tokens from the allenai/c4 en validation stream.
    """
    if calib_dataset == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return select_passages(dataset["text"], n_passages, min_chars, offset), None
    if calib_dataset == "c4":
        stream = load_dataset(C4_DATASET["path"], C4_DATASET["name"], split=C4_DATASET["split"], streaming=True)
        return select_c4_passages(stream, tokenizer, n_passages, max_length, offset)
    raise ValueError(f"calib_dataset={calib_dataset!r} is invalid; must be one of {CALIB_DATASETS}.")


def build_calibration_batches(
    tokenizer, passages: List[str], batch_size: int, max_length: int
) -> List[Dict[str, torch.Tensor]]:
    """
    Tokenize the passages and return a list of right-padded [B, T] batches.
    Format expected by the engine: {"input_ids", "attention_mask"} dicts.
    Tensors stay on the CPU; the engine moves them to the model's device.
    """
    batches: List[Dict[str, torch.Tensor]] = []
    for i in range(0, len(passages), batch_size):
        chunk = passages[i:i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batches.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return batches


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[XAI RUN] Seed: {args.seed}")

    if not torch.cuda.is_available():
        print("[XAI RUN] Warning: CUDA not available; a 7B model is impractical on CPU. Run on a GPU machine.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPES[args.dtype]
    print(f"[XAI RUN] Device: {device}, dtype: {args.dtype}")

    # --- model ---
    print(f"[XAI RUN] Loading model: {MODEL_NAME}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        # The Mistral tokenizer has no pad token; EOS is used for in-batch padding.
        # Pad positions are masked by attention_mask, so they do not enter the target function.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    load_time = time.time() - t0
    print(f"[XAI RUN] Model loaded ({load_time:.1f} s)")

    # --- calibration data ---
    print(f"[XAI RUN] Loading calibration set: {DATASET_LABELS[args.calib_dataset]}...")
    passages, passage_ids = load_calibration_passages(tokenizer, args.calib_dataset, args.n_passages, args.min_chars,
                                                      args.max_length, args.passage_offset)
    batches = build_calibration_batches(tokenizer, passages, args.batch_size, args.max_length)
    # For the ms/token metric, count only the tokens of the batches actually passed to the engine.
    used_batches = batches[: args.max_batches]
    n_tokens = sum(int(b["attention_mask"].sum()) for b in used_batches)
    print(
        f"[XAI RUN] Calibration: {len(passages)} passages, {len(batches)} batches built "
        f"(B={args.batch_size}, T<={args.max_length}); {len(used_batches)} batches, "
        f"{n_tokens} valid tokens will be passed to the engine"
    )

    # --- attribution ---
    n_layers = model.config.num_hidden_layers
    n_blocks = 2 * n_layers
    print(
        f"[XAI RUN] Expected work: {n_blocks} blocks × {len(used_batches)} batches "
        f"× n_steps={args.n_steps} forward+backward passes (internal_batch_size={args.internal_batch_size})"
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    scores = calculate_importance_scores(
        model,
        batches,
        n_steps=args.n_steps,
        internal_batch_size=args.internal_batch_size,
        max_batches=args.max_batches,
        verbose=True,
    )
    attr_time = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None

    # --- report ---
    print()
    print(f"[XAI RUN] Top {args.top_k} blocks by score:")
    print(format_importance_report(scores, top_k=args.top_k))
    print()
    mlp_scores = {k: v for k, v in scores.items() if ".mlp" in k}
    print("[XAI RUN] MLP blocks (descending):")
    print(format_importance_report(mlp_scores))
    print()
    print(f"[XAI RUN] Attribution time: {attr_time:.1f} s "
          f"({attr_time / n_blocks:.2f} s/block, {attr_time / max(n_tokens, 1) * 1000:.1f} ms/token)")
    if peak_vram_gb is not None:
        print(f"[XAI RUN] Peak VRAM: {peak_vram_gb:.1f} GB")

    result = {
        "model": MODEL_NAME,
        "precision": args.dtype,
        "dataset": DATASET_LABELS[args.calib_dataset],
        "calibration": {
            "calib_dataset": args.calib_dataset,
            **({"passages": passage_ids, "min_tokens": args.max_length} if passage_ids is not None else {}),
            "n_passages": len(passages),
            "n_batches_built": len(batches),
            "n_batches_used": len(used_batches),
            "max_batches": args.max_batches,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "min_chars": args.min_chars,
            "passage_offset": args.passage_offset,
            "n_valid_tokens": n_tokens,
        },
        "attribution": {
            "method": "captum.LayerIntegratedGradients (attribute_to_layer_input)",
            "n_steps": args.n_steps,
            "internal_batch_size": args.internal_batch_size,
            "baseline_token_id": model.config.pad_token_id if model.config.pad_token_id is not None else 0,
        },
        "timing": {
            "model_load_seconds": load_time,
            "attribution_seconds": attr_time,
            "seconds_per_block": attr_time / n_blocks,
        },
        "peak_vram_gb": peak_vram_gb,
        "seed": args.seed,
        "n_scores": len(scores),
        "scores": scores,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[XAI RUN] Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
