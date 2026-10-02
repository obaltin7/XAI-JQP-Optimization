"""
FP16 baseline perplexity measurement.
Reference point for all post-compression comparisons in XAI-JQP.

Usage:
    python tools/evaluation/baseline_eval.py
"""

# HF_HOME must be set BEFORE transformers/datasets are imported; otherwise the
# model and dataset caches are written to the pod's non-persistent disk
# (/root/.cache) and Mistral-7B (~14GB) is re-downloaded after every restart.
# /workspace is the persistent volume.
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import json
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_fp16.json")

# Sliding-window size and stride (tokens) for WikiText-2 perplexity
MAX_LENGTH = 1024
STRIDE = 512


def compute_perplexity(model, tokenizer, text, device):
    """Sliding-window perplexity (standard Hugging Face recipe)."""
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, STRIDE):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).sum() / end_loc).item()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BASELINE] Device: {device}")

    print(f"[BASELINE] Downloading model: {MODEL_NAME} (may take a while on first run)")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"[BASELINE] Model loaded ({time.time() - t0:.1f} s)")

    print("[BASELINE] Downloading WikiText-2 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    print("[BASELINE] Computing perplexity...")
    t0 = time.time()
    ppl = compute_perplexity(model, tokenizer, text, device)
    elapsed = time.time() - t0
    print(f"[BASELINE] FP16 Perplexity: {ppl:.4f} (took {elapsed:.1f} s)")

    result = {
        "model": MODEL_NAME,
        "precision": "fp16",
        "dataset": "wikitext-2-raw-v1 (test split)",
        "perplexity": ppl,
        "eval_time_seconds": elapsed,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[BASELINE] Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
