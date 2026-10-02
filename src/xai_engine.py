# xai_engine.py
"""
XAI-JQP, stage 1: structural explainability analysis.

This module measures the contribution of the *structural* components of a
causal LM (target: Mistral-7B-Instruct-v0.3) to the model output with Captum
LayerIntegratedGradients. Macro blocks are scored rather than individual weights:

  * every **attention head** in every decoder layer
  * every **MLP block** in every decoder layer (optionally at neuron-group
    granularity by splitting the intermediate neurons into `mlp_n_groups` parts)

Method (a deliberate engineering choice rather than exact SHAP):
  Integrated Gradients scales a block's activation along a straight-line path
  from an "empty" baseline to its actual value and integrates the output
  gradient along that path. This yields an axiomatic (completeness-satisfying)
  attribution without the exponential cost of exact SHAP.

Model independence: the number of layers / heads and head_dim are read from the
model config, and attribution points are located by module NAME
(`self_attn.o_proj`, `mlp.down_proj`). Any decoder that uses these names
(Mistral, Llama, Qwen2; q/k/v biases do not affect the attribution point)
therefore works unchanged, and the score formula is identical across models.
The IG baseline token must be chosen per model (`baseline_token_id`; see
`_resolve_baseline_token` and run_qwen_experiments.py).

Attribution points (Mistral/Llama/Qwen2 architecture):
  * attention: the INPUT of `layer.self_attn.o_proj`. This tensor has shape
    [B, T, n_heads * head_dim] and holds the per-head context vectors side by
    side; summing over head_dim slices gives one score per head.
  * MLP: the INPUT of `layer.mlp.down_proj`. This tensor has shape
    [B, T, intermediate_size] and contains the act(gate(x)) * up(x) "neuron"
    activations; summing per group gives the block / neuron-group score.

Target (output) function:
  Sum of the teacher-forced log-probabilities of the correct next tokens
  (masked by attention_mask). The score thus answers "how much does this block
  contribute to the model's ability to predict correctly"; it is a derivative
  of the same quantity that perplexity measures.

Usage:
    scores = calculate_importance_scores(model, dataloader, n_steps=16)
    # {"layer_0.attn.head_0": 0.31, ..., "layer_0.mlp": 4.7, ...}
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients


# --------------------------------------------------------------------------- #
# Discovery of structural targets
# --------------------------------------------------------------------------- #
@dataclass
class StructuralTarget:
    """A single structural block to attribute (one layer's attention or MLP)."""

    layer_idx: int
    kind: str  # "attn" | "mlp"
    module: nn.Module  # o_proj (attn) or down_proj (mlp); attribution is to its input
    n_groups: int  # attn: number of heads, mlp: number of neuron groups
    group_size: int  # attn: head_dim, mlp: intermediate_size // n_groups

    @property
    def key(self) -> str:
        return f"{self.kind}_{self.layer_idx}"

    def group_names(self) -> List[str]:
        if self.kind == "attn":
            return [f"layer_{self.layer_idx}.attn.head_{h}" for h in range(self.n_groups)]
        if self.n_groups == 1:
            return [f"layer_{self.layer_idx}.mlp"]
        return [f"layer_{self.layer_idx}.mlp.group_{g}" for g in range(self.n_groups)]


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Layer list for an HF causal LM (model.model.layers) or a bare decoder (model.layers)."""
    for path in ("model.layers", "layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(
        "Decoder layers not found: expected model.model.layers or model.layers "
        "(a Mistral/Llama-style HF model is required)."
    )


def discover_structural_targets(model: nn.Module, mlp_n_groups: int = 1) -> List[StructuralTarget]:
    """
    Return all attention heads and MLP blocks of the model as a list of StructuralTarget.
    Layer order is preserved: layer_0.attn, layer_0.mlp, layer_1.attn, ...
    """
    config = model.config
    n_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_heads

    targets: List[StructuralTarget] = []
    for i, layer in enumerate(_get_decoder_layers(model)):
        o_proj = layer.self_attn.o_proj
        if o_proj.in_features != n_heads * head_dim:
            raise ValueError(
                f"layer_{i}: o_proj.in_features={o_proj.in_features} does not match "
                f"n_heads*head_dim={n_heads * head_dim}."
            )
        targets.append(StructuralTarget(i, "attn", o_proj, n_heads, head_dim))

        down_proj = layer.mlp.down_proj
        inter = down_proj.in_features
        if inter % mlp_n_groups != 0:
            raise ValueError(
                f"layer_{i}: intermediate_size={inter} is not divisible by mlp_n_groups={mlp_n_groups}."
            )
        targets.append(StructuralTarget(i, "mlp", down_proj, mlp_n_groups, inter // mlp_n_groups))
    return targets


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_forward_func(model: nn.Module):
    """Scalar target passed to Captum: per-example sum of correct-token log-probabilities, shape [B]."""

    def forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, :-1, :].float()  # position t predicts token t+1
        targets = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = attention_mask[:, 1:].to(logp.dtype)
        return (logp * mask).sum(dim=1)

    return forward_func


def _unpack_batch(batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalise a batch given as a dict, an (input_ids, attention_mask) pair or a single tensor."""
    if isinstance(batch, dict):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
    elif isinstance(batch, (tuple, list)):
        input_ids = batch[0]
        attention_mask = batch[1] if len(batch) > 1 else None
    else:
        input_ids, attention_mask = batch, None

    input_ids = torch.as_tensor(input_ids)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids.to(device), torch.as_tensor(attention_mask).to(device)


def _resolve_baseline_token(model: nn.Module, baseline_token_id: Optional[int]) -> int:
    """'Empty' token for the IG baseline: the pad token if defined, otherwise 0 (<unk> in Mistral)."""
    if baseline_token_id is not None:
        return baseline_token_id
    pad = getattr(model.config, "pad_token_id", None)
    return pad if pad is not None else 0


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def calculate_importance_scores(
    model: nn.Module,
    dataloader: Iterable,
    *,
    n_steps: int = 16,
    internal_batch_size: Optional[int] = None,
    max_batches: Optional[int] = None,
    mlp_n_groups: int = 1,
    baseline_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute Captum LayerIntegratedGradients importance scores for the structural
    blocks of the model (attention heads, MLP blocks / neuron groups).

    Args:
        model: HF causal LM (Mistral/Llama architecture). For tests this may be
            a small randomly initialised MistralForCausalLM.
        dataloader: Iterable (calibration data) whose elements are
            {"input_ids", "attention_mask"} dicts, (input_ids, attention_mask)
            tuples or [B, T] input_ids tensors.
        n_steps: Number of Riemann steps of the IG integral. Larger values
            increase accuracy and cost (n_steps forward+backward passes per block).
        internal_batch_size: Chunk size in which Captum processes the n_steps*B
            examples (to bound VRAM). None = all at once.
        max_batches: Maximum number of batches taken from the dataloader.
        mlp_n_groups: Number of groups the MLP intermediate neurons are split
            into. 1 = block level (default); >1 = neuron-group level.
        baseline_token_id: "Empty" token used to build the IG baseline sequence.
            None = pad_token_id, falling back to 0.
        device: If None, the device holding the model parameters.
        verbose: Print progress.

    Returns:
        {"layer_{i}.attn.head_{h}": score, "layer_{i}.mlp": score, ...}
        Score = sum of |attribution| divided by the number of valid tokens
        (mean absolute contribution per token); always >= 0.
        Order: structural (layer_0.attn.head_0, ..., layer_0.mlp, layer_1...).

    Note: a SEPARATE LayerIntegratedGradients call is made for each structural
    block. Captum's multi-layer (list) mode scales all blocks simultaneously and
    thereby cuts the gradient path between blocks; separate calls are more
    expensive but correct. Cost ≈ (2 * n_layers) * n_steps forward+backward passes.
    """
    if device is None:
        device = next(model.parameters()).device

    # --- attribution mode: dropout off, parameter gradients not needed ---
    was_training = model.training
    model.eval()
    prev_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    targets = discover_structural_targets(model, mlp_n_groups=mlp_n_groups)
    forward_func = _make_forward_func(model)
    baseline_id = _resolve_baseline_token(model, baseline_token_id)

    if verbose:
        n_attn = sum(t.n_groups for t in targets if t.kind == "attn")
        n_mlp = sum(t.n_groups for t in targets if t.kind == "mlp")
        print(
            f"[XAI ENGINE] Starting LayerIntegratedGradients — "
            f"{len(targets) // 2} layers, {n_attn} attention heads, {n_mlp} MLP blocks/groups, "
            f"n_steps={n_steps}, baseline_token={baseline_id}, device={device}"
        )

    sums: Dict[str, torch.Tensor] = {t.key: torch.zeros(t.n_groups, dtype=torch.float64) for t in targets}
    total_tokens = 0
    t_start = time.time()

    try:
        with torch.enable_grad():
            for b_idx, batch in enumerate(dataloader):
                if max_batches is not None and b_idx >= max_batches:
                    break
                input_ids, attention_mask = _unpack_batch(batch, device)
                baselines = torch.full_like(input_ids, baseline_id)
                B, T = input_ids.shape
                total_tokens += int(attention_mask[:, 1:].sum().item())

                for t in targets:
                    lig = LayerIntegratedGradients(forward_func, t.module)
                    attr = lig.attribute(
                        inputs=input_ids,
                        baselines=baselines,
                        additional_forward_args=(attention_mask,),
                        n_steps=n_steps,
                        internal_batch_size=internal_batch_size,
                        attribute_to_layer_input=True,
                    )
                    if isinstance(attr, (tuple, list)):
                        attr = attr[0]
                    # [B, T, n_groups*group_size] -> per-group sum of |attribution| -> [n_groups]
                    grouped = attr.detach().abs().reshape(B, T, t.n_groups, t.group_size).sum(dim=(0, 1, 3))
                    sums[t.key] += grouped.double().cpu()

                if verbose:
                    print(f"[XAI ENGINE]   batch {b_idx + 1} done (B={B}, T={T}, {time.time() - t_start:.1f} s)")
    finally:
        # restore the model's original state
        for p, rg in zip(model.parameters(), prev_requires_grad):
            p.requires_grad_(rg)
        if was_training:
            model.train()

    if total_tokens == 0:
        raise ValueError("Dataloader is empty or contains no valid tokens; scores cannot be computed.")

    importance_scores: Dict[str, float] = {}
    for t in targets:
        vals = sums[t.key] / total_tokens
        for name, v in zip(t.group_names(), vals.tolist()):
            importance_scores[name] = float(v)

    if verbose:
        print(
            f"[XAI ENGINE] Computation finished: {len(importance_scores)} structural blocks, "
            f"{total_tokens} tokens, {time.time() - t_start:.1f} s"
        )
    return importance_scores


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_importance_report(scores: Dict[str, float], top_k: Optional[int] = None) -> str:
    """Return the scores as a readable table sorted in descending order."""
    if not scores:
        return "(empty score dictionary)"
    max_score = max(scores.values()) or 1.0
    rows = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        rows = rows[:top_k]
    width = max(len(k) for k, _ in rows)
    lines = [f"{'block'.ljust(width)}  {'score':>12}  {'norm':>6}  bar"]
    for name, v in rows:
        bar = "#" * int(round(30 * v / max_score))
        lines.append(f"{name.ljust(width)}  {v:12.6f}  {v / max_score:6.3f}  {bar}")
    return "\n".join(lines)
