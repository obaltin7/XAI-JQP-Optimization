"""
MMLU subset evaluation: 0-shot, argmax over A/B/C/D log-probabilities.

Perplexity alone is not a sufficient metric (Jaiswal et al., 2024): compressed
models can preserve perplexity while degrading on knowledge and reasoning tasks.
This module measures accuracy on a small, FIXED MMLU subset with the same interface
as the perplexity harness (model, tokenizer -> dict).

Subset: `cais/mmlu` (HF datasets, test split), 10 subjects (mixed STEM / social
sciences / humanities) × 50 questions = 500 questions. Selection is deterministic
(random.Random(42), in subject order; indices sorted ascending) and is written ONCE
to `results/mmlu_subset_ids.json`; later runs read the same file and verify the
question-text sha1 hashes, so a changed dataset raises an error instead of silently
evaluating different questions.

Scoring: log-probabilities of the " A"/" B"/" C"/" D" tokens from the logits at the
last prompt position (log_softmax over the full vocabulary, fp32), followed by argmax.
The letter tokens are derived from the tokenizer in the context "Answer: A" ("▁A" in
SentencePiece); their ids and the prompt format are recorded in the result dict. The
default format is `plain` (the original MMLU / lm-eval-harness 0-shot template; no chat
template and no system prompt, consistent with the raw-text perplexity harness). The
`chat` option uses the tokenizer's chat template. Works with quantized (bitsandbytes)
models since only forward passes are required.

Usage:
    python src/eval_mmlu.py --build-subset      # download the dataset, write/verify the subset file (no model)
    python src/eval_mmlu.py --dry-run           # mini model + fake subset on CPU (no datasets needed)
    python src/eval_mmlu.py                     # GPU: FP16 Mistral-7B, result in results/mmlu_fp16_gun7.json

    from eval_mmlu import evaluate_mmlu
    res = evaluate_mmlu(model, tokenizer)   # res["mmlu_subset_acc"], res["per_subject"], ...

Estimated cost on an L40S: 500 questions, batch 8, ~150–300 tokens/prompt -> ~1–2 min
in FP16, ~2–3 min in NF4; extra VRAM <1.5 GB (batch logits).
"""

# HF_HOME must be set BEFORE datasets is imported (see run_e2e_pipeline.py).
# setdefault: a value already set by the calling script or the local environment is left untouched.
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import argparse
import hashlib
import json
import random
import re
import sys
import time
import zlib
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Make the repository root, src/ and experiments/ importable regardless of the working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

DATASET = "cais/mmlu"
SPLIT = "test"
SUBSET_IDS_FILE = os.path.join("results", "mmlu_subset_ids.json")
OUTPUT_FILE = os.path.join("results", "mmlu_fp16_gun7.json")
DEFAULT_SEED = 42
DEFAULT_PER_SUBJECT = 50
# STEM: abstract_algebra, anatomy, astronomy, college_computer_science, high_school_mathematics
# social sciences: us_foreign_policy, econometrics | humanities: philosophy, world_religions, moral_scenarios
DEFAULT_SUBJECTS: Tuple[str, ...] = (
    "abstract_algebra", "anatomy", "astronomy", "college_computer_science", "high_school_mathematics",
    "philosophy", "world_religions", "us_foreign_policy", "econometrics", "moral_scenarios",
)
LETTERS = ("A", "B", "C", "D")
PROMPT_STYLES = ("plain", "chat")
PLAIN_TEMPLATE = (
    "The following are multiple choice questions (with answers) about {subject}.\n\n"
    "{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
)
CHAT_USER_TEMPLATE = (
    "The following is a multiple choice question about {subject}.\n\n"
    "{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer with the letter only."
)
CHAT_ASSISTANT_PREFIX = "Answer:"

Record = Dict[str, Any]  # {"subject", "index", "question", "choices": [4 str], "answer": int 0-3}
RowLoader = Callable[[str], Sequence[Dict[str, Any]]]


# --------------------------------------------------------------------------- #
# Subset: selection, persistent id file, verification
# --------------------------------------------------------------------------- #
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _hf_row_loader(subject: str) -> Sequence[Dict[str, Any]]:
    from datasets import load_dataset  # lazy import: datasets is optional locally and not needed for --dry-run

    return load_dataset(DATASET, subject, split=SPLIT)


def _records(subject: str, rows: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Record]:
    out: List[Record] = []
    for i in indices:
        row = rows[int(i)]
        choices = [str(c) for c in row["choices"]]
        answer = int(row["answer"])
        if len(choices) != 4 or not 0 <= answer < 4:
            raise ValueError(f"{subject}[{i}]: unexpected record (choices={len(choices)}, answer={row['answer']!r})")
        out.append({"subject": subject, "index": int(i), "question": str(row["question"]),
                    "choices": choices, "answer": answer})
    return out


def load_or_build_subset(
    ids_path: str = SUBSET_IDS_FILE,
    *,
    subjects: Sequence[str] = DEFAULT_SUBJECTS,
    per_subject: int = DEFAULT_PER_SUBJECT,
    seed: int = DEFAULT_SEED,
    row_loader: Optional[RowLoader] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    Return the fixed MMLU subset: {"meta": {...}, "records": [Record, ...]}.

    If ids_path EXISTS: subjects, indices and order are read from the file (the
    subjects/per_subject/seed arguments are ignored; a warning is logged if they differ)
    and the sha1 of every question is verified.
    OTHERWISE: `sample(range(n), min(per_subject, n))` is drawn per subject, in subject
    order, with random.Random(seed); indices are sorted ascending and the file is written
    atomically once.
    Evaluation order: subject order in the file, ascending index within a subject.
    """
    loader = row_loader or _hf_row_loader
    if os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        asked = {"subjects": list(subjects), "per_subject": per_subject, "seed": seed}
        stored = {k: meta.get(k) for k in asked}
        if asked != stored:
            log(f"MMLU: {ids_path} exists; using the stored subset {stored} instead of the requested {asked}")
        records: List[Record] = []
        for subject in meta["subjects"]:
            item = meta["items"][subject]
            recs = _records(subject, loader(subject), item["indices"])
            bad = [r["index"] for r, h in zip(recs, item["question_sha1"]) if _sha1(r["question"]) != h]
            if bad:
                raise ValueError(f"MMLU {subject}: sha1 of {len(bad)} questions does not match {ids_path} (first: {bad[:3]}); "
                                 "the dataset version may have changed; investigate before deleting the subset file.")
            records.extend(recs)
        return {"meta": meta, "records": records}

    rng = random.Random(seed)
    items: Dict[str, Any] = {}
    records = []
    for subject in subjects:
        rows = loader(subject)
        n = len(rows)
        indices = sorted(rng.sample(range(n), min(per_subject, n)))
        recs = _records(subject, rows, indices)
        items[subject] = {"n_available": n, "indices": indices, "question_sha1": [_sha1(r["question"]) for r in recs]}
        records.extend(recs)
    meta = {
        "dataset": DATASET, "split": SPLIT, "seed": seed, "per_subject": per_subject, "subjects": list(subjects),
        "created_at": datetime.now().isoformat(timespec="seconds"), "n_questions": len(records),
        "selection": "random.Random(seed); sample(range(n), min(per_subject, n)) in subject order; indices sorted ascending",
        "items": items,
    }
    os.makedirs(os.path.dirname(ids_path) or ".", exist_ok=True)
    tmp = ids_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    os.replace(tmp, ids_path)
    log(f"MMLU: subset built and written: {ids_path} ({len(records)} questions, {len(items)} subjects)")
    return {"meta": meta, "records": records}


def build_fake_subset(seed: int = DEFAULT_SEED, n_subjects: int = 3, per_subject: int = 6) -> Dict[str, Any]:
    """For --dry-run / tests: short fake questions in the real record schema (NOT written to disk)."""
    rng = random.Random(seed)
    subjects = [f"fake_subject_{s}" for s in range(n_subjects)]
    records: List[Record] = []
    for subject in subjects:
        for i in range(per_subject):
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            answer = rng.randrange(4)
            choices = [str(a + b + k - answer) for k in range(4)]  # correct choice sits at position `answer`
            records.append({"subject": subject, "index": i, "question": f"What is {a} plus {b} ?",
                            "choices": choices, "answer": answer})
    meta = {"dataset": "fake (dry-run)", "split": None, "seed": seed, "per_subject": per_subject,
            "subjects": subjects, "n_questions": len(records)}
    return {"meta": meta, "records": records}


# --------------------------------------------------------------------------- #
# Prompt and tokenization
# --------------------------------------------------------------------------- #
def _encode(tokenizer, text: str, add_special_tokens: bool) -> List[int]:
    return list(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def format_prompt(record: Record, style: str = "plain") -> str:
    """Question text shown to the model (for chat, only the user message; the assistant prefix is appended separately)."""
    a, b, c, d = record["choices"]
    template = PLAIN_TEMPLATE if style == "plain" else CHAT_USER_TEMPLATE
    return template.format(subject=record["subject"].replace("_", " "), question=record["question"].strip(),
                           a=a, b=b, c=c, d=d)


def encode_prompt(tokenizer, record: Record, style: str = "plain") -> List[int]:
    if style == "plain":
        return _encode(tokenizer, format_prompt(record, "plain"), add_special_tokens=True)
    if style == "chat":
        text = tokenizer.apply_chat_template([{"role": "user", "content": format_prompt(record, "chat")}],
                                             tokenize=False, add_generation_prompt=True)
        return _encode(tokenizer, text + CHAT_ASSISTANT_PREFIX, add_special_tokens=False)  # the template already contains BOS
    raise ValueError(f"invalid prompt_style={style!r}; must be one of {PROMPT_STYLES}.")


def letter_token_ids(tokenizer) -> Tuple[List[int], Dict[str, Any]]:
    """
    Ids of the letter tokens that follow "Answer:": the last token of the encoding of
    "Answer: A" ("▁A" in SentencePiece). `context_stable` is True if exactly one token was
    appended without changing the prefix encoding; if False, the last token is still used
    but the result is flagged.
    """
    prefix = _encode(tokenizer, "Answer:", add_special_tokens=False)
    ids: List[int] = []
    stable = True
    for letter in LETTERS:
        full = _encode(tokenizer, f"Answer: {letter}", add_special_tokens=False)
        if full[: len(prefix)] != prefix or len(full) != len(prefix) + 1:
            stable = False
        ids.append(int(full[-1]))
    if len(set(ids)) != len(LETTERS):
        raise ValueError(f"A/B/C/D token ids are not distinct: {ids}")
    info: Dict[str, Any] = {"ids": ids, "context_stable": stable}
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        info["tokens"] = [str(t) for t in tokenizer.convert_ids_to_tokens(ids)]
    return ids, info


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_mmlu(
    model: nn.Module,
    tokenizer,
    *,
    subset: Optional[Dict[str, Any]] = None,
    ids_path: str = SUBSET_IDS_FILE,
    prompt_style: str = "plain",
    batch_size: int = 8,
    max_length: int = 2048,
    device: Optional[torch.device] = None,
    log: Callable[[str], None] = print,
    record_questions: bool = False,
) -> Dict[str, Any]:
    """
    0-shot accuracy on the fixed MMLU subset (same interface as the perplexity harness).

    record_questions (default False, which leaves the output schema unchanged): if True, a
    per-question record is added to the result:
    "per_question": [{"subject", "index", "gold", "pred", "correct", "logp": [4 letter log-probabilities]}]
    (for per-subject drift analysis; run_qwen_experiments / run_eval_from_plans always enable it).

    Args:
        model, tokenizer: HF causal LM (fp16 or bitsandbytes-quantized) and its tokenizer.
        subset: output of load_or_build_subset / build_fake_subset; None = load/build from ids_path.
        prompt_style: "plain" (default; no chat template) | "chat" (tokenizer chat template).
        batch_size, max_length: batch size; prompts longer than max_length are truncated from the left (BOS kept).

    Returns:
        {"mmlu_subset_acc" (= "accuracy"), "n_questions", "n_correct", "per_subject":
         {subject: {"accuracy", "n", "correct"}}, "seconds", "predicted_letter_counts",
         "gold_letter_counts", "mean_correct_prob" (normalised over the 4 choices), "prompt_format",
         "letter_tokens", "subset", "batch_size", "max_length", "n_truncated"}
    """
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"invalid prompt_style={prompt_style!r}; must be one of {PROMPT_STYLES}.")
    t0 = time.time()
    if subset is None:
        subset = load_or_build_subset(ids_path, log=log)
    records: List[Record] = subset["records"]
    if not records:
        raise ValueError("MMLU subset is empty.")
    if device is None:
        device = next(model.parameters()).device
    letter_ids, letter_info = letter_token_ids(tokenizer)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0

    encoded: List[List[int]] = []
    n_truncated = 0
    for r in records:
        ids = encode_prompt(tokenizer, r, prompt_style)
        if len(ids) > max_length:
            ids = ids[:1] + ids[-(max_length - 1):]
            n_truncated += 1
        encoded.append(ids)

    was_training = model.training
    model.eval()
    preds: List[int] = []
    correct_probs: List[float] = []
    all_logp: List[List[float]] = []
    try:
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start:start + batch_size]
            width = max(len(x) for x in chunk)
            input_ids = torch.full((len(chunk), width), int(pad_id), dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, x in enumerate(chunk):  # right padding, independent of tokenizer.padding_side
                input_ids[j, : len(x)] = torch.tensor(x, dtype=torch.long)
                mask[j, : len(x)] = 1
            last_pos = mask.sum(dim=1) - 1
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device), use_cache=False).logits
            last = logits[torch.arange(len(chunk)), last_pos].float().cpu()  # [B, V]
            logp = torch.log_softmax(last, dim=-1)[:, letter_ids]  # [B, 4]
            if not bool(torch.isfinite(logp).all()):
                raise RuntimeError(f"MMLU: NaN/Inf in the letter log-probabilities of batch {start // batch_size}.")
            preds.extend(logp.argmax(dim=-1).tolist())
            all_logp.extend(logp.tolist())
            gold = torch.tensor([r["answer"] for r in records[start:start + batch_size]])
            correct_probs.extend(torch.softmax(logp, dim=-1)[torch.arange(len(chunk)), gold].tolist())
    finally:
        model.train(was_training)

    per_subject: Dict[str, Dict[str, Any]] = {}
    for r, p in zip(records, preds):
        s = per_subject.setdefault(r["subject"], {"accuracy": None, "n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += int(p == r["answer"])
    for s in per_subject.values():
        s["accuracy"] = s["correct"] / s["n"]
    n_correct = sum(s["correct"] for s in per_subject.values())
    acc = n_correct / len(records)
    meta = subset["meta"]
    result = {
        "mmlu_subset_acc": acc,
        "accuracy": acc,
        "n_questions": len(records),
        "n_correct": n_correct,
        "per_subject": per_subject,
        "seconds": time.time() - t0,
        "predicted_letter_counts": {L: preds.count(i) for i, L in enumerate(LETTERS)},
        "gold_letter_counts": {L: sum(1 for r in records if r["answer"] == i) for i, L in enumerate(LETTERS)},
        "mean_correct_prob": sum(correct_probs) / len(correct_probs),
        "prompt_format": {
            "style": prompt_style, "n_shot": 0, "chat_template_used": prompt_style == "chat", "system_prompt": None,
            "template": PLAIN_TEMPLATE if prompt_style == "plain" else CHAT_USER_TEMPLATE,
            "assistant_prefix": CHAT_ASSISTANT_PREFIX if prompt_style == "chat" else None,
            "scoring": "A/B/C/D token log-probabilities at the last position (full-vocabulary log_softmax, fp32), argmax",
        },
        "letter_tokens": letter_info,
        "subset": {k: meta.get(k) for k in ("dataset", "split", "seed", "per_subject", "subjects", "created_at")},
        "batch_size": batch_size,
        "max_length": max_length,
        "n_truncated": n_truncated,
    }
    result["subset"]["ids_file"] = ids_path if meta.get("dataset") == DATASET else None
    if record_questions:
        result["per_question"] = [{"subject": r["subject"], "index": r["index"], "gold": r["answer"], "pred": int(p),
                                   "correct": bool(p == r["answer"]), "logp": [float(x) for x in lp]}
                                  for r, p, lp in zip(records, preds, all_logp)]
    log(f"MMLU ({prompt_style}, 0-shot): accuracy {acc:.4f} ({n_correct}/{len(records)}), {result['seconds']:.1f} s; "
        f"prediction distribution {result['predicted_letter_counts']}")
    return result


# --------------------------------------------------------------------------- #
# Dry-run support (for the mini model, which has no tokenizer)
# --------------------------------------------------------------------------- #
class FakeTokenizer:
    """
    Word-level fake tokenizer for the mini model (which has no tokenizer): 0=pad, 1-4=" A".." D",
    5=BOS, other words mapped into [6, vocab_size) via crc32. For --dry-run / tests only.
    """

    pad_token_id = 0
    bos_token_id = 5

    def __init__(self, vocab_size: int):
        if vocab_size < 8:
            raise ValueError("vocab_size must be >= 8")
        self.vocab_size = vocab_size
        self._letters = {f" {L}": i + 1 for i, L in enumerate(LETTERS)}

    def __call__(self, text: str, add_special_tokens: bool = True) -> Dict[str, List[int]]:
        ids = [self.bos_token_id] if add_special_tokens else []
        for tok in re.findall(r"\s*\S+", text):
            tok = " " + tok.strip() if tok[0].isspace() else tok
            ids.append(self._letters.get(tok, 6 + zlib.crc32(tok.encode("utf-8")) % (self.vocab_size - 6)))
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        rev = {v: k for k, v in self._letters.items()}
        return [rev.get(int(i), f"<{int(i)}>") for i in ids]


def evaluate_mmlu_dry(model: nn.Module, seed: int = DEFAULT_SEED, log: Callable[[str], None] = print,
                      record_questions: bool = False) -> Dict[str, Any]:
    """evaluate_mmlu with the mini model, FakeTokenizer and a fake subset (code-path check; accuracy is meaningless)."""
    tok = FakeTokenizer(model.config.vocab_size)
    return evaluate_mmlu(model, tok, subset=build_fake_subset(seed), batch_size=4,
                         max_length=getattr(model.config, "max_position_embeddings", 64), log=log,
                         record_questions=record_questions)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MMLU subset (0-shot, argmax over A/B/C/D log-probabilities)")
    p.add_argument("--build-subset", action="store_true", help="only build/verify the subset file (no model)")
    p.add_argument("--dry-run", action="store_true", help="mini Mistral + fake subset on CPU")
    p.add_argument("--ids", default=SUBSET_IDS_FILE)
    p.add_argument("--per-subject", type=int, default=DEFAULT_PER_SUBJECT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--prompt-style", choices=PROMPT_STYLES, default="plain")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--model", default=None, help="HF model id; default None = Mistral-7B-Instruct-v0.3")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.dry_run:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))
        from test_xai_engine_mini import build_mini_model

        res = evaluate_mmlu_dry(build_mini_model(), args.seed)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    subset = load_or_build_subset(args.ids, per_subject=args.per_subject, seed=args.seed)
    counts = {s: len(v["indices"]) for s, v in subset["meta"]["items"].items()}
    print(f"MMLU subset: {len(subset['records'])} questions, per subject {counts}")
    if args.build_subset:
        return 0

    from run_e2e_pipeline import MODEL_NAME, Logger, load_model

    model_name = args.model or MODEL_NAME
    tokenizer, model, load_s = load_model(Logger("log_mmlu_gun7.txt"), model_name=model_name)
    res = evaluate_mmlu(model, tokenizer, subset=subset, ids_path=args.ids, prompt_style=args.prompt_style,
                        batch_size=args.batch_size)
    payload = {"model": model_name, "precision": "fp16", "model_load_seconds": load_s, "mmlu": res}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Result saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
