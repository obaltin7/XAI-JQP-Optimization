# XAI-JQP — Results

This document summarises the findings of the XAI-JQP experiments, from the general result to the specific checks, and
states the negative results and limitations explicitly. Every number is read from a JSON file under `results/`; the file
(and, where useful, the field) is given next to each number. The figures referenced in the [README](README.md) and all
tables are regenerated from the same files by `tools/visualization/make_figures.py`.

**Conventions.**
- ppl = perplexity on the WikiText-2 test split (window 1024, stride 512) or on a fixed 256-document C4 subset.
- HellaSwag and ARC-Challenge report `acc_norm`, MMLU reports accuracy; each task uses a fixed 500-item subset (independent
  95 % CI about ±4.4 points), zero-shot, one prompt template; MMLU covers 10 subjects (50 questions each).
- "20 % / 40 % / 60 %" is the fraction of attention heads (blocks) pruned, not of parameters (20 % of heads ≈ 3.08 % of the
  parameters of Mistral-7B, 2.21 % of Qwen2.5-7B).
- Size (GB) = parameters + buffers. For bitsandbytes rows the `quant_state` is not counted, while GPTQ / AWQ rows count
  `qweight`, scales and zeros and HQQ rows count scale and zero-point tensors; the comparison is therefore slightly biased
  in favour of the bitsandbytes rows.
- "Significant" means an exact binomial McNemar test per random seed with Holm correction inside a pre-registered
  family: Mistral primary family 45 tests (111 with the secondary family), Qwen 20 % 15 tests (36 with secondary), Qwen
  10 % 12 tests. Results reported under an **extra** family (E-3: 6 tests, four-tier C-3: 6, sub-4-bit D-2: 30) were
  added after pre-registration and are descriptive.
- Perplexity differences use a paired window bootstrap (seed 42, percentile 95 % CI; 653 / 331 WikiText-2 / C4 windows for
  Mistral, 584 / 295 for Qwen). The bootstrap does not include calibration-sample or seed uncertainty.
- Random controls use three seeds (42 / 43 / 44) and are reported as mean ± sample std (n − 1).

**Models.** Mistral-7B-Instruct-v0.3 in FP16 (32 × 32 = 1024 heads + 32 MLP blocks; `results/importance_scores_gun3.json`)
and Qwen2.5-7B-Instruct in BF16 (28 × 28 = 784 heads, 4 KV heads per layer, q/k/v bias; `results/model2_qwen_gun9.json`).
Qwen needs BF16: its FP16 forward pass is fine (ppl 7.0874, MMLU 0.688) but full Integrated-Gradients attribution
overflows to NaN / Inf in 1 of 812 scores (`layer_0.mlp`, from the run log; the JSON records the aborted run), which a
short smoke test did not catch (`results/model2_qwen_gun9_fp16_nan.json`, `results/model2_qwen_gun9_smoke.json`). Each model is compared with its own
baseline; Mistral was not re-run in BF16. Hardware: one NVIDIA L40S 48 GB; torch 2.4.1+cu124, transformers 4.44.2
(`results/iterative_gun7.json`, `run.env`).

---

## 1. Method details that shape the results

- **Scoring.** Captum `LayerIntegratedGradients`, 8 steps, baseline token 0, on 16 WikiText-2 passages (4 batches × 4,
  max_length 256, 2 678 valid tokens); 345.3 s and 19.97 GB peak VRAM for Mistral-7B
  (`results/importance_scores_gun3.json`, `calibration.*`, `attribution`, `timing.attribution_seconds`).
- **Score ranges.** Heads 0.0103–0.712 (median 0.086), MLP blocks 7.17–20.31 (median 9.03). The two types live on scales
  about 100× apart, so tiers are assigned by **within-type** percentiles (`results/importance_scores_gun3.json`).
- **The number of tiers is a design choice.** k-means silhouette on log10 scores prefers k = 2 for both types
  (heads 0.603 / 0.540 / 0.521 for k = 2 / 3 / 4; MLP 0.810 / 0.564 / 0.582); the scores do not form three or four
  natural clusters (`results/tier_analysis_gun4.json`).
- **20 % plan (Mistral).** Heads 205 prune / 410 INT4 / 409 FP16; MLP blocks 0 / 19 / 13; 129 INT4 modules; nominal size
  ratio 0.549 (8.78 bits per parameter) (`results/tier_analysis_gun4.json`, `results/iterative_gun7.json`,
  `fractions.0.2.budget`).
- **Median rule.** bitsandbytes quantizes whole `Linear` modules, so per-head precision is impossible; a layer's attention
  projections take the lower median of its head tiers. Of the 819 unpruned heads, 104 move to a lower tier and 119 to a
  higher one (about 27 % change tier) (`results/ablation_gun6.json`, `median_rule`).
- **Qwen 20 % plan.** 157 heads (rounds 53 + 52 + 52) and 119 INT4 modules (`results/model2_qwen_gun9.json`); head scores
  0.0135–8.82, MLP 10.9–452.1 (`results/gun9_scores/importance_scores_qwen_qwen2_5_7b_instruct.json`, `scores`).
- **Why the final configuration keeps the round-0 INT4 plan (`xai_iter_fixedq`).** Re-deriving the INT4 plan from the
  last round (`xai_iter`) gives 5.8743 / 10.7660 / 20.4104 ppl at 20 / 40 / 60 %, and the fixed plan is better by −0.0089 /
  −0.4461 / −0.0538: the iterative gain comes from head selection, not from the INT4 plan (`results/iterative_gun7.json`,
  `fractions.<f>.comparison`). The iterative method costs three attribution runs (1035 s of attribution, about
  1170 s wall time per configuration, vs about 480 s for single-shot) and is a single deterministic run per ratio.

---

## 2. Main findings (general → specific)

### 2.1 The final model is 43.4 % smaller with no measurable task loss

The final configuration, iterative XAI-JQP at 20 % with physical head removal and INT4 (`xai_iter_fixedq`), shrinks
Mistral-7B from 14.496 GB to 8.199 GB; physical removal alone gives 14.066 GB (214.96 M parameters, 3.0 %), and INT4
does the rest (`results/speed_gun8_fixedq.json`, `summary.physical`, `summary.physical_int4.model_gb`).

| | GB | ppl WikiText-2 | ppl C4 | HellaSwag | ARC-C | MMLU |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 14.496 | 5.2200 | 7.8264 | 0.832 | 0.592 | 0.546 |
| XAI-JQP iterative 20 % | 8.199 | 5.8654 | 8.6431 | 0.828 | 0.578 | 0.552 |

Sources: `results/tasks_gun15_e5_wikitext.json` (`entries.*.ppl`), `results/tasks_gun9.json` (`entries.fp16`,
`entries.f0.2/xai_iter_fixedq`), `results/iterative_gun7.json` (`fractions.0.2.configs.xai_iter_fixedq`). The masked
iterative models are 8.343 / 7.097 / 6.493 GB at 20 / 40 / 60 %; in masked models the saving comes from INT4 only.
Against FP16 the discordant pairs of the final iterative model are HellaSwag 12 / 10, ARC-C 22 / 15 and MMLU 22 / 25
(single-shot: 11 / 9, 23 / 15, 21 / 22), all with Holm p = 1.0, so the task difference is not distinguishable (`results/tasks_gun9_paired.json`, secondary
family).

### 2.2 xAI head selection beats random, magnitude and reverse selection at the same budget

- **Tasks (Mistral).** At 20 % xAI selection reaches HellaSwag / ARC-C / MMLU 0.828 / 0.576 / 0.548, against random
  0.766 ± 0.011 / 0.499 ± 0.008 / 0.453 ± 0.059 (`results/tasks_gun9.json`, `results/tasks_gun9_seed43.json`,
  `results/tasks_gun9_seed44.json`). On HellaSwag and ARC-C xAI selection is significant against **each** of the three
  random seeds at all three ratios: **18 / 18** comparisons, largest Holm p = 0.0059
  (`results/tasks_gun9_paired.json`, `pairs[].p_adjusted`). On MMLU this holds only at 40 %.
- **Higher ratios (Mistral).** xAI single-shot HellaSwag / ARC-C / MMLU at 40 %: 0.808 / 0.550 / 0.492 vs random
  0.606 ± 0.062 / 0.351 ± 0.039 / 0.254 ± 0.031; at 60 %: 0.714 / 0.446 / 0.280 vs 0.335 ± 0.008 / 0.236 ± 0.015 /
  0.235 ± 0.001 (same files). Random perplexity with the same INT4 plan: 7.5476 ± 1.3096 / 33.1136 ± 11.1053 /
  386.8383 ± 82.4263 (`results/iterative_gun7.json`, `…prune_random.perplexity_values`).
- **Perplexity, pruning only (Mistral, 20 %, 205-head budget).** xAI 6.2250; random 7.9068 ± 2.1461 (seeds 6.84 / 6.50 /
  10.38, so the gap is in the mean, not against every seed); magnitude 50.8672; reverse (pruning the highest-scored heads)
  138.6760; FP16 5.2200. Overlap with the xAI set: random 37 / 43 / 36 heads, magnitude 12, reverse 0
  (`results/ablation_gun6.json`, `configs.*`). Magnitude and reverse were run only at 20 %, pruning only, without INT4
  and without tasks.

### 2.3 Almost all of the loss comes from pruning, not quantization — in both models

| model (20 %) | pruning | quantization | both | interaction | pruning share |
|---|---:|---:|---:|---:|---:|
| Mistral-7B | +1.0050 | +0.0475 | +1.0971 | +0.0446 | 95.5 % |
| Qwen2.5-7B | +5.8174 | +0.1490 | +6.1985 | +0.2322 | 97.5 % |

Pruning share = prune / (prune + quant); relative to the combined loss it is 91.6 % (Mistral) and 93.9 % (Qwen)
(`results/ablation_gun6.json`, `decomposition`; `results/model2_qwen_gun9.json`, `comparison`). Bias correction by mean
replacement recovers only −0.0260 ppl, 2.4 % of the combined loss, in a single pass (`decomposition.biascorr_gain`). The
decomposition was not measured at 40 / 60 % (no pruning-only / quantization-only runs) or for Qwen 10 % (no
quantization-only run).

### 2.4 At the same size, pruning less and quantizing more is better

With the same 129 INT4 modules and the same 8.343 GB, pruning only 102 heads instead of 205 gives ppl 5.5018 and
HellaSwag / ARC-C / MMLU 0.826 / 0.586 / 0.564, better in perplexity than both single-shot (6.3171) and iterative
(5.8654) XAI-JQP at 20 %. On tasks it differs from them by at most 1.6 points and, like them, is not distinguishable
from FP16 (Holm p = 1.0, secondary family); no direct paired test against them was run (`results/iterative_gun7.json`,
`fractions.0.2.configs.az_buda_cok_kuantize`; `results/tasks_gun9.json`; `results/tasks_gun9_paired.json`).

### 2.5 Iterative re-scoring improves perplexity, not task accuracy

| ratio | single-shot ppl | iterative ppl | gain | share of loss recovered | shared pruned heads |
|---|---:|---:|---:|---:|---:|
| 20 % | 6.3171 | 5.8654 | −0.4517 | 41 % | 196 / 205 |
| 40 % | 12.0849 | 10.3199 | −1.7650 | 26 % | 390 / 410 |
| 60 % | 58.9569 | 20.3566 | −38.6003 | 72 % | 576 / 614 |

Source: `results/iterative_gun7.json` (`fractions.<f>.comparison.xai_iter_fixedq_minus_xai_single_ppl`,
`…xai_iter_fixedq.overlap_with_xai_heads`); share of loss = gain / (single-shot ppl − FP16 ppl). At 20 % the gain is
significant in both domains: WikiText-2 −0.4517 [−0.4736, −0.4302], C4 −0.4945 [−0.5726, −0.4228]
(`results/ppl_bootstrap_gun15.json`). It repeats on Qwen: −0.7863 at 20 % (146 / 157 shared heads;
`results/model2_qwen_gun9.json`) and −0.3913 [−0.4131, −0.3707] / −0.3658 [−0.4026, −0.3315] at 10 %
(`results/ppl_bootstrap_gun16_qwen_f010.json`). None of the nine Mistral task comparisons between iterative and
single-shot survives Holm correction (60 % HellaSwag 13 / 31, raw p 0.0096 → Holm 0.20; `results/tasks_gun9_paired.json`).

### 2.6 Perplexity alone is misleading (perplexity–task disconnect)

Six independent cases show perplexity and task accuracy moving apart:

1. **Within-layer z-scored Wanda** is competitive in perplexity (6.0693 / 13.0842 / 33.8460 at 20 / 40 / 60 %; at 60 % it
   beats single-shot xAI) but not on tasks: HellaSwag 0.796 / 0.648 / 0.468, MMLU 0.488 / 0.208 / 0.238. Against xAI the
   HellaSwag discordant pairs at 40 / 60 % are 86 / 6 and 133 / 10 (Holm p < 1e-16, secondary family)
   (`results/iterative_gun7_wanda_ln.json`, `results/tasks_gun9_iterative_gun7_wanda_ln.json`,
   `results/tasks_gun9_paired.json`).
2. **Qwen2.5-7B at 20 %:** pruning only raises perplexity from 7.0868 to 12.9042 (+82.1 %; Mistral +19.3 %) while MMLU
   moves from 0.686 to 0.678 (`results/model2_qwen_gun9.json`).
3. **C4 calibration:** the C4-calibrated single-shot plan has WikiText-2 perplexity 11.6401 (vs 6.3171,
   `results/tasks_gun15_e5_wikitext.json`) but the same task accuracy (HellaSwag 0.828, ARC-C 0.572, MMLU 0.550)
   (`results/tasks_gun15_e5_c4.json`, `results/tasks_gun9_calib_c4_gun9.json`).
4. **LoRA recovery** lowers perplexity at 20 % from 5.8654 to 5.0227 (below FP16) while HellaSwag / ARC-C / MMLU go from
   0.828 / 0.578 / 0.552 to 0.816 / 0.580 / 0.528 (`results/lora_recovery_gun10.json`, `results/tasks_gun9.json`).
5. **Qwen2.5-7B at 10 %:** random pruning has lower perplexity than xAI pruning while xAI is numerically ahead on all
   three tasks, though no task difference is significant (see 2.10).
6. **Sub-4-bit HQQ allocation:** uniform allocation leads xAI allocation in perplexity, but on tasks the two are
   indistinguishable (0 / 6 significant, extra family; see 2.11).

A seventh case strengthens the point: the C4-calibrated iterative model at 60 % has WikiText-2 perplexity 33.8914 (vs
20.3566 for the WikiText-2-calibrated one) yet higher HellaSwag, 0.726 vs 0.678, Holm p = 0.0056 in the extra E-3 family
(`results/calib_c4_gun16_f06.json`, `results/tasks_gun9_calib_c4_gun16_f06.json`, `results/tasks_gun9_paired_e3.json`).

### 2.7 At high pruning ratios MMLU measures letter bias, not knowledge

Collapsed models lock onto one answer letter, and their MMLU accuracy then approximately equals that letter's share of correct answers (A 0.226, C 0.282).
At 60 %: iterative XAI-JQP answers "A" to 421 / 500 questions (accuracy 0.238), Taylor 450 / 500 "A" (0.244), single-shot
XAI-JQP 299 / 500 "C" (0.280); within-layer Wanda answers "A" to 410 / 500 (40 %) and 397 / 500 (60 %); raw Wanda at 20 %
answers "A" to 499 / 500 (`results/iterative_gun7.json`, `results/iterative_gun7_wanda_ln.json`,
`repeats[0].mmlu.predicted_letter_counts`). MMLU values at 60 % (e.g. "single-shot beats iterative") are therefore not
interpretable; HellaSwag and ARC-C keep separating the methods. At 20 % xAI selection stays within ±8 points of FP16 in
every MMLU subject (per-subject 95 % CI about ±14 points, n = 50) (`results/iterative_gun7.json`, `mmlu.per_subject`).

### 2.8 The calibration domain biases in-domain perplexity

In the 2 × 2 cross-domain design each calibration is best in its own domain (`results/tasks_gun15_e5_wikitext.json`,
`results/tasks_gun15_e5_c4.json`, `entries.*.ppl.{wikitext2,c4}.perplexity`):

| calibration | method | ppl WikiText-2 | ppl C4 |
|---|---|---:|---:|
| WikiText-2 | single-shot | 6.3171 | 9.1376 |
| WikiText-2 | iterative | 5.8654 | 8.6431 |
| WikiText-2 | random (seed 42) | 6.9856 | 9.8779 |
| C4 | single-shot | 11.6401 | 8.8588 |
| C4 | iterative | 11.4755 | 8.6789 |
| C4 | random (seed 42) | 6.9984 | 9.8859 |

- **Pre-registered criterion** (C4 Δ ≤ 0 and WikiText-2 Δ > 0, both CIs excluding zero) is met: C4 minus WikiText-2
  calibration changes single-shot WikiText-2 perplexity by **+5.3230 [+5.1500, +5.4954]** and C4 perplexity by
  **−0.2788 [−0.3413, −0.2229]** (`results/ppl_bootstrap_gun15.json`). C4-calibrated xAI is worse than random on
  WikiText-2 (+4.6417) but better on C4 (−1.0271); WikiText-2-calibrated xAI beats random in both domains (−0.6686 /
  −0.7403); C4-calibrated iterative beats C4-calibrated single-shot by −0.1646 / −0.1799 (same file). The random control
  in this design is a single seed.
- **The plans differ in heads and INT4 modules.** The two calibrations share 158 of 205 pruned heads (Jaccard 0.627) and
  their scores agree moderately (head Spearman 0.927, MLP 0.897, top-100 Jaccard 0.600); the C4 plan has 137 INT4 modules
  instead of 129 (`results/compare_wikitext_vs_c4_gun9.json`, `results/diagnose_calib_domain_gun15.json`).
- **The 47 C4-specific heads are not the "most important" heads.** In the WikiText-2 ranking their median rank is 279
  (threshold 205, maximum 641); none is in the reverse (top-205) set; 35 of 47 sit in layers 21–31
  (`results/diagnose_calib_domain_gun15.json`).
- **Group ablation (FP16, masked, no quantization).** Pruning only those 47 heads raises WikiText-2 perplexity by +1.4206
  (+27.2 %) and C4 by +0.0919 (+1.2 %). The largest layer groups, L21–26 and L27–31, add +0.2690 and +0.2209; the sum
  of all five layer groups (+0.518) is 2.7× smaller than the joint effect, so the effect is super-additive. The 47 WikiText-2-specific heads cost
  little in either domain (+0.0829 / +0.0988) (`results/group_ablation_gun15.json`).

### 2.9 Taylor is competitive at low ratios and wins in perplexity at 60 %

Head-level Taylor importance (an adaptation; the LLM-Pruner package itself was not run) gives ppl 8.2164 / 16.0451 /
28.4648 at 20 / 40 / 60 %, against xAI single-shot 6.3171 / 12.0849 / 58.9569 and iterative 5.8654 / 10.3199 / 20.3566:
Taylor beats single-shot xAI at 60 % but not iterative. On tasks Taylor reaches HellaSwag 0.818 / 0.798 / 0.620 and
ARC-C 0.594 / 0.542 / 0.404; the single-shot xAI–Taylor difference is Holm-significant only at 60 % HellaSwag (0.714
vs 0.620, p 4.0e-07; primary family), and iterative xAI also beats Taylor there (Holm p 0.0034, secondary family). Overlap with the xAI set: 170 / 205, 361 / 410, 556 / 614 heads (`results/iterative_gun7.json`,
`…prune_taylor`; `results/tasks_gun9.json`; `results/tasks_gun9_paired.json`).

### 2.10 Generalisation to Qwen2.5-7B holds at 20 %, not at 10 %

- **20 %, pruning only (157 heads, no INT4).** xAI selection 12.9042 ppl vs random 13.7083 ± 0.6814 (a gap of about one
  standard deviation); perplexity barely separates the selectors, MMLU does: xAI 0.678 vs random 0.531 ± 0.034
  (`results/model2_qwen_gun9.json`, `configs.*`). With the same INT4 plan, Taylor reaches 13.3419 ppl / MMLU 0.544
  against single-shot xAI 13.2853 / 0.666 (Holm p 1.2e-06). HellaSwag / ARC-C: xAI 0.782 / 0.558,
  random 0.683 ± 0.069 / 0.443 ± 0.068 (`results/tasks_gun9_model2_qwen_gun9_mmlu.json` and its seed files). Per-seed
  Holm p (family 15): MMLU 7.0e-07, 2.5e-11, 4.0e-06; HellaSwag 0.039, 6.7e-16, 1.0e-04; ARC-C not significant against
  seed 42 (0.56) (`results/tasks_gun9_paired_model2_qwen_gun9.json`, `pairs[].p_adjusted`).
- **20 %, full pipeline.** Single-shot pruning + INT4 13.2853 ppl / HellaSwag 0.774 / ARC-C 0.538 / MMLU 0.666; iterative
  12.4991 / 0.788 / 0.526 / 0.678 (BF16 7.0868 / 0.802 / 0.540 / 0.686); quantization only 7.2358 / 0.798 / 0.536 / 0.682;
  reverse 219 094.9 ppl with tasks at chance (0.280 / 0.254 / 0.226) (`results/model2_qwen_gun9.json`,
  `results/tasks_gun9_model2_qwen_gun9_mmlu.json`). Iterative vs BF16 is not significant (HellaSwag 18 / 11, p 0.26, Holm 1.0; secondary family).
- **10 % (negative).** Pruning-only xAI perplexity is 8.9530 (WikiText-2) / 13.6608 (C4) against random 8.0602 ± 0.2269 /
  12.7171 ± 0.4938; all six bootstrap CIs exclude zero in favour of random (WikiText-2 +0.63 [+0.54, +0.73], +1.01, +1.04;
  C4 +0.41, +1.38, +1.04). Task accuracy is numerically higher for xAI (HellaSwag / ARC-C / MMLU 0.800 / 0.566 / 0.672
  vs 0.776 ± 0.003 / 0.529 ± 0.011 / 0.639 ± 0.005) but none of the 12 primary-family tests (9 against random) is significant (min p 0.26)
  (`results/model2_qwen_gun9_f010.json`, `results/model2_qwen_gun9_f010_seeds.json`,
  `results/tasks_gun9_model2_qwen_gun9_f010.json` and its seed files, `results/ppl_bootstrap_gun16_qwen_f010.json`,
  `results/tasks_gun9_paired_model2_qwen_gun9_f010.json`).
- Qwen was not swept to 40 / 60 %; its C4 perplexity and physical + INT4 size were not measured at 20 %.

### 2.11 Below 4 bits, xAI-guided bit allocation beats random and magnitude allocation, but not uniform allocation

HQQ (0.2.8.post1, group size 64, axis 1) allocation without pruning on Mistral, tiers 8 / 4 / 3 / 2 bit
(`results/mixed_bits_gun11.json`, `configs.hqq_*`, `reference.budgets`, `configs.*.avg_bits_effective`):

| nominal budget | allocation | effective bits | ppl WikiText-2 | ppl C4 | MMLU | GB |
|---|---|---:|---:|---:|---:|---:|
| B = 3.0 | uniform 3-bit | 3.70 | 5.9801 | 8.8653 | 0.472 | 3.765 |
| B = 3.0 | xAI-guided | 3.56 | 6.9787 | 10.0184 | 0.484 | 3.645 |
| B = 3.0 | random (3 seeds) | 3.56 | 26.51 ± 15.61 | – | 0.378 | – |
| B = 3.0 | magnitude | 3.56 | 28.4752 | – | 0.352 | – |
| B = 3.5 | xAI-guided | 4.08 | 6.1748 | 8.9766 | 0.508 | 4.093 |
| B = 3.5 | random (3 seeds) | 4.08 | 11.23 ± 8.28 | – | 0.481 | – |
| B = 3.5 | magnitude | 4.08 | 22.7012 | – | 0.424 | – |
| B = 4.0 (reference) | uniform 4-bit | 4.50 | 5.3170 | 7.9388 | 0.516 | 4.463 |

Tier counts per type (8 / 4 / 3 / 2 bit): B = 3.0 → 0 / 11 / 10 / 11; B = 3.5 → 2 / 12 / 12 / 6. Uniform 4-bit is **not**
a same-budget control for B = 3.5; there the only same-budget controls are random and magnitude allocation.

- **Perplexity.** All 20 bootstrap CIs exclude zero (`results/ppl_bootstrap_gun16_d2.json`); uniform 3-bit leads xAI by
  +0.9986 [+0.9584, +1.0396] (B = 3.0, WikiText-2), although xAI is the smaller model (3.645 vs 3.765 GB).
- **Tasks (extra family, descriptive).** At B = 3.0, xAI allocation (HellaSwag / ARC-C / MMLU 0.802 / 0.524 / 0.484)
  beats random (0.719 ± 0.028 / 0.448 ± 0.036 / 0.378 ± 0.004) in 8 / 9 and magnitude (0.724 / 0.414 / 0.352) in 3 / 3
  comparisons and is indistinguishable from uniform 3-bit (0.812 / 0.520 / 0.472; 0 / 3). At B = 3.5, xAI (0.820 / 0.550 /
  0.508) beats random (0.792 ± 0.012 / 0.507 ± 0.010 / 0.481 ± 0.011) in only 1 / 9 and magnitude (0.772 / 0.488 / 0.424)
  in 3 / 3; against uniform 4-bit (0.838 / 0.570 / 0.516) 0 / 3 (`results/tasks_gun9_mixed_bits_gun11.json` and its seed
  files, `results/tasks_gun9_paired_mixed_bits_gun11.json`).
- **Allocation matters less as the budget relaxes.** xAI − random perplexity per seed shrinks from −30.78 / −1.70 / −26.11
  (B = 3.0) to −14.62 / −0.38 / −0.17 (B = 3.5) and to +0.0046 / +0.0001 / +0.0045 for the FP16 / NF4 mix of 2.12 (about 4 bits)
  (`results/ppl_bootstrap_gun16_d2.json`, `results/ppl_bootstrap_gun16_d1.json`). This is a qualitative trend over three
  points and two quantizers.

### 2.12 Mixed FP16 / NF4 precision without pruning: the choice does not matter

Keeping k MLP / attention blocks in FP16 and the rest in NF4 (Mistral; `results/mixed_precision_gun11.json`,
`fractions.0.2.configs.mixed_*`): NF4 alone costs only +0.087 ppl (5.3065 / C4 7.9314 / MMLU 0.530 / 4.027 GB). At k = 10
xAI selection gives 5.2995 / 7.9240 / 0.536 / 5.009 GB, random 5.2964 ± 0.0026 / 7.9222 ± 0.0022 / 0.540 ± 0.003,
magnitude 5.2913 / 7.9087 / 0.524; k = 20 gives 5.2895 / MMLU 0.556 / 5.990 GB. Bootstrap xAI − control: random
+0.0046 [+0.0026, +0.0066], +0.0001 (not significant), +0.0045; magnitude +0.0082 [+0.0059, +0.0104]; k = 0 −0.0070
[−0.0088, −0.0053]; the pre-registered criterion is not met (`results/ppl_bootstrap_gun16_d1.json`). Task accuracy was
not measured for this experiment (effect ≤ 0.01 ppl on WikiText-2, ≤ 0.02 on C4).

### 2.13 xAI pruning concentrates in the last layers; random pruning does not

Gini coefficient of the pruned heads per layer against a 97.5 % random reference: Qwen 10 % 0.721 (reference 0.386), Qwen
20 % 0.590 (0.260), Mistral 20 % 0.590 (0.242); random sets ≤ 0.331. All six xAI sets are concentrated and none of the
seven random sets is. At Qwen 10 %, 39 of 78 pruned heads are in layers 25–27 and 15 of 28 layers are untouched; at
Mistral 20 %, layer 25 loses 72 % of its heads (`results/layer_concentration_gun16.json`). This is descriptive; a causal
link to the Qwen 10 % result is a hypothesis.

### 2.14 Explanations stay stable at 20 % and drift at 60 %

Spearman ρ between round-0 and post-pruning scores of the surviving heads (`results/drift_gun7.json`,
`entries[].metrics.spearman.heads_surviving`): at 20 % xAI single-shot 0.984, iterative after round 3 0.984, Taylor 0.931,
Wanda (z-scored) 0.968, raw Wanda 0.838, attention confidence 0.868, random 0.961 ± 0.021; at 40 % xAI single-shot 0.887,
iterative 0.890, attention confidence 0.664; at 60 % xAI single-shot 0.675, iterative 0.769, attention confidence 0.205.
Top-100 Jaccard at 20 %: xAI 0.869, Taylor 0.587, attention confidence 0.361. The numbers are the surviving-head view
(`heads_all`, `all_blocks` and `all_surviving` views are also stored; e.g. attention confidence top-100 Jaccard is 0.205
over all heads); the drift of `xai_iter_fixedq` equals that of `xai_iter` (same pruned heads); the link between drift and task accuracy was
not measured.

### 2.15 Score stability: insensitive to IG steps, moderately sensitive to the calibration sample

- IG steps 8 → 16: head Spearman 0.9989, MLP 1.0000, top-100 Jaccard 1.000, 6 / 1024 heads change tier
  (`results/compare_importance_scores_gun3_vs_importance_scores_gun3_n16.json`).
- Different calibration passages (16–31): head Spearman 0.9811, MLP 0.9322, top-100 Jaccard 0.724; pruned-set overlap
  188 / 205 (Jaccard 0.847); 82 / 1024 heads change tier and the INT4 plan shrinks from 129 to 125 modules
  (`results/compare_importance_scores_gun3_vs_importance_scores_gun3_offset16.json`). Sensitivity was measured on scores
  and plans, not on perplexity.

### 2.16 Reproducibility checks

- **Stored plans re-apply bit-identically.** Re-applying the saved plans reproduces perplexity (6.317065 vs 6.317065) and
  MMLU (`mmlu_minus_source` = 0.0 in all 17 entries of `results/tasks_gun9.json`), with identical HellaSwag / ARC-C
  per-question predictions (`results/tasks_gun9_verify.json`).
  This holds on the same pod and library versions; it is not guaranteed on another GPU or bitsandbytes version.
- **Masked and physical pruning are numerically equivalent:** Mistral 5.8174 vs 5.8173 (`results/speed_gun8_fixedq.json`),
  Qwen 12.9042 vs 12.9047 (`results/speed_gun12_qwen_physical.json`).
- **Attention kernel.** The small Qwen masked–physical gap comes from an sdpa–eager kernel mismatch amplified at low
  precision (0.5B test: logit gaps FP16 0.40 / 1.99, BF16 2.67 / 12.08; with matched kernels FP16 0.088 / 0.000)
  (`results/diagnose_qwen_kernel_gun12.json`). The default attention kernel stays eager so that Mistral numbers remain
  bit-identical; the `auto` kernel is used only for validation and one extra speed row.

---

## 3. Claim table

Status: **robust** = holds with the stated significance; **narrowed** = holds only in the stated scope; **negative** = not
supported; **limitation** = accepted weakness.

| # | claim | evidence | significance | scope | status |
|---|---|---|---|---|---|
| 1 | xAI selection beats random selection at the same budget | Mistral 20 % HellaSwag 0.828 vs 0.766 ± 0.011, ARC-C 0.576 vs 0.499 ± 0.008; Qwen 20 % (pruning only) MMLU 0.678 vs 0.531 ± 0.034 | Mistral HellaSwag + ARC-C 18 / 18 (Holm, per seed); Qwen HellaSwag + MMLU against all three seeds | two models; Qwen 10 % excluded | **robust** (Mistral HellaSwag + ARC-C, Qwen HellaSwag + MMLU); narrowed for Mistral MMLU and Qwen ARC-C |
| 2 | xAI selection beats Taylor | Mistral 60 % HellaSwag 0.714 vs 0.620; Qwen MMLU 0.666 (xAI + INT4) vs 0.544 (Taylor + INT4); ppl 20 % 6.3171 vs 8.2164 | Holm-significant only at Mistral 60 % HellaSwag and on Qwen MMLU | ratio-dependent; Taylor wins in ppl at 60 % vs single-shot | **narrowed** |
| 3 | Iterative re-scoring beats single-shot | ppl 6.3171 → 5.8654, 12.0849 → 10.3199, 58.9569 → 20.3566; Qwen −0.7863 (20 %), −0.3913 (10 %) | bootstrap CIs exclude zero (Mistral 20 %, Qwen 10 %, both domains); no task difference after Holm | perplexity only | **narrowed** |
| 4 | Pruning the most important heads collapses the model | reverse: ppl 138.6760 (Mistral), 219 094.9 (Qwen, tasks at chance) | deterministic, large effect | two models | **robust** |
| 5 | Masked and physical pruning are equivalent | 5.8174 / 5.8173 (Mistral); 12.9042 / 12.9047 (Qwen) | difference ≤ 0.0005 ppl | two architectures | **robust** |
| 6 | Loss comes from pruning, quantization is cheap | pruning share 95.5 % (Mistral), 97.5 % (Qwen) | deterministic | two models, 20 % | **robust** |
| 7 | Perplexity alone is misleading | six cases in 2.6 | Holm on tasks where applicable | two models | **robust** |
| 8 | Calibration domain biases in-domain perplexity | +5.3230 [+5.1500, +5.4954] / −0.2788 [−0.3413, −0.2229] | pre-registered bootstrap criterion met; E-3 HellaSwag Holm p 0.0056 (extra family) | one model, 20 % and 60 % | **robust**; the 47 C4-specific heads explain ~27 % of the effect |
| 9 | Results generalise to Qwen2.5-7B | ordering, iterative gain, pruning share and task retention repeat at 20 % | per-seed Holm (family 15) | 20 % only; no ratio sweep | **robust** at 20 %; **narrowed** at 10 % |
| 10 | XAI-JQP is the best compressor | NF4 5.3065 / 4.027 GB vs XAI-JQP 5.8654 / 8.199 GB | – | perplexity vs size | **limitation** (uniform 4-bit is ahead) |
| 11 | Sub-4-bit xAI allocation beats random / magnitude | B = 3.0 ppl 6.9787 vs 26.51 ± 15.61 / 28.4752 | bootstrap CIs; tasks 8 / 9 vs random, 3 / 3 vs magnitude (extra family) | one model, HQQ | **robust** at B = 3.0; **narrowed** at B = 3.5 |
| 12 | Sub-4-bit xAI allocation beats uniform allocation | 6.9787 vs 5.9801 at B = 3.0 (uniform 4-bit at B = 4.0 is only a reference) | CIs in favour of uniform; tasks 0 / 3 at B = 3.0 | one model | **negative** in perplexity, equal on tasks |
| 13 | Mixed FP16 / NF4 precision without pruning: xAI beats random / magnitude | k = 10: xAI 5.2995, random 5.2964 ± 0.0026, magnitude 5.2913 | pre-registered criterion not met | one model, one budget | **negative** (effect ≤ 0.02 ppl) |
| 14 | LoRA recovers the pruning loss | ppl recovered 131 % / 95 % / 89 %; tasks at 20 % do not recover | no task test | one model, in-domain training | **narrowed** (perplexity only) |
| 15 | Attention confidence is an alternative to IG | ppl 13.7359 / 77.7639 / 841.71, worse than random | single run, large effect | one model, three ratios; not Voita's full method | **negative** |
| 16 | A fourth INT8 tier adds compression for free | 7.726 GB (−7.4 %); tasks unchanged | Holm 0 / 6 (extra family) | one model, 20 %, single run | **robust** (latency not measured) |
| 17 | Physical head removal reduces latency | 0.995× (eager), 0.957× (sdpa), 1.024× (auto) | 3 repeats | one GPU, small batch | **negative** (gain is in size only) |

Sources: Section 2, Section 4, and `results/mixed_precision_gun11.json`, `results/ppl_bootstrap_gun16_d1.json`
(claim 13), `results/lora_recovery_gun10.json` (14), `results/iterative_gun7_attnconf.json` (15),
`results/iterative_gun7_4tier.json`, `results/tasks_gun9_paired_iterative_gun7_4tier.json` (16),
`results/speed_gun8_eager.json`, `results/speed_gun8_sdpa.json`, `results/speed_gun8_sdpa_auto.json` (17),
`results/baselines_gun7.json`, `results/iterative_gun7.json`, `results/speed_gun8_fixedq.json` (10).

---

## 4. Negative results and supplementary experiments

- **Uniform 4-bit quantization is ahead in absolute compression.** NF4 5.3065 ppl / MMLU 0.530 / 4.027 GB, GPTQ 5.3194 /
  0.530 / 4.168 GB, AWQ 5.3242 / 0.540 / 4.163 GB (`results/baselines_gun7.json`, `summary.configs.*`), against
  5.8654 / 0.552 / 8.199 GB for XAI-JQP (`results/iterative_gun7.json`, `fractions.0.2.configs.xai_iter_fixedq`;
  `results/speed_gun8_fixedq.json`). GPTQ and AWQ were calibrated on WikiText-2 train, which favours them on the WikiText-2
  test. The XAI-JQP peak VRAM of 19.97 GB (`xai_iter_fixedq.peak_vram_gb`) includes re-scoring and is not an inference
  peak.
- **No latency gain from physical head removal.** Measured on the single-shot 20 % plan (L40S, 8 prompts × 64 new tokens,
  greedy, KV cache, 1 warm-up, 3 repeats): eager 28.64 ± 0.08 → 28.78 ± 0.03 ms/token (0.995×); sdpa 26.82 ± 0.05 →
  28.02 ± 0.79 (0.957×); auto kernel 26.08 ± 0.16 → 25.45 ± 0.12 (1.024×) (`results/speed_gun8_eager.json`,
  `results/speed_gun8_sdpa.json`, `results/speed_gun8_sdpa_auto.json`). The physically pruned layers run through a wrapper
  that uses the eager kernel by default, so the sdpa row also contains a kernel difference; each speed-up is relative to
  its own FP16 row. With INT4 (bitsandbytes NF4) the final model runs at 32.33 vs 25.88 ms/token (0.801×, single run;
  `results/speed_gun8_fixedq.json`); peak generation VRAM is 9.87 GB.
- **Attention confidence fails at head level.** The selector used here is the mean maximum attention probability (lowest
  pruned), not Voita et al.'s full method (LRP + learned gates), so the result shows that confidence alone is insufficient.
  ppl 13.7359 / 77.7639 / 841.71 at 20 / 40 / 60 %, worse than random at every ratio; MMLU 0.280 / 0.260 / 0.224, at or
  below random level; overlap with the xAI set 2 / 30 / 248 heads; tasks other than MMLU were not measured; single run
  (`results/iterative_gun7_attnconf.json`).
- **Raw Wanda collapses at head level.** Raw Wanda is an unstructured criterion and our head-level use is an unverified
  adaptation. ppl 60.0156 / 208.8197 / 348.5138 with MMLU at chance (0.226 / 0.220 / 0.226)
  (`results/iterative_gun7.json`, `…prune_wanda`). The cause is layer scale, not a bug: the score correlates with layer
  depth (Spearman +0.874), raw scores span 34.08–38 209.09, and 181 of 205 pruned heads sit in layers 0–5
  (`results/diagnose_wanda_gun8_pod.json`). Within-layer z-scoring fixes perplexity but not tasks (2.6, case 1).
- **Mixed precision without pruning does not beat random or magnitude selection** (2.12).
- **Qwen2.5-7B at 10 %: random pruning wins in perplexity** and no task difference is significant (2.10).
- **LoRA recovery (supplementary; the method itself needs no training).** QLoRA r 16, α 32, dropout 0.05 on
  q / k / v / o / gate / up / down, 200 steps × 4 × 512 WikiText-2 train tokens (409 600 tokens), lr 2e-4 cosine, 10 warm-up
  steps; adapter 0.084 GB, not merged; pruned slices stay zero; one seed. Perplexity before → after: 5.8654 → 5.0227
  (131 % of the loss), 10.3199 → 5.4705 (95 %), 20.3566 → 6.8099 (89 %). Tasks after: 0.816 / 0.580 / 0.528 at 20 %,
  0.782 / 0.536 / 0.508 at 40 %, 0.742 / 0.484 / 0.260 at 60 % (before 0.678 / 0.444 / 0.238); no paired test. Training and
  test on WikiText-2 give LoRA an in-domain advantage (`results/lora_recovery_gun10.json`; task values before LoRA from
  `results/tasks_gun9.json`, `entries.f<f>/xai_iter_fixedq`).
- **Fourth INT8 tier (supplementary).** Head tiers (0.2, 0.3, 0.3, 0.2), MLP (0.4, 0.3, 0.3); 91 INT4 + 102 INT8 modules;
  7.726 GB (−7.4 %); single-shot 6.3048 / MMLU 0.554, iterative 5.8711 / 0.566; tasks unchanged (0 / 6 Holm, extra family);
  LLM.int8 latency not measured; single run (`results/iterative_gun7_4tier.json`,
  `results/tasks_gun9_iterative_gun7_4tier.json`, `results/tasks_gun9_paired_iterative_gun7_4tier.json`).
- **Iterative re-scoring has no significant task effect** (2.5), and **sub-4-bit xAI allocation stays behind uniform
  allocation in perplexity** (2.11).

## 5. Limitations

1. **Not the best compressor.** Uniform NF4 / GPTQ / AWQ / HQQ give lower perplexity at the same or smaller size; the
   contribution is a single explainable importance spectrum that drives both pruning and precision and beats the other
   selectors. On tasks, uniform and xAI allocation are indistinguishable (0 / 6).
2. **Domain-specific heads explain only part of the domain effect.** The 47 C4-specific heads account for +1.42 of the
   +5.32 WikiText-2 perplexity shift; the rest (158 shared heads + INT4 interaction) was not measured, and what the heads
   compute was not analysed.
3. **E-3 is a single run with a BF16 fallback.** The C4-calibrated iterative 60 % result (138 INT4 modules, 6.682 GB, ARC-C
   0.468, MMLU 0.254; threshold HellaSwag ≥ 0.694 fixed in the study plan before the run) rests on one run whose round-3 re-scoring fell back to
   BF16 after a non-finite score (`results/calib_c4_gun16_f06.json`,
   `fractions.0.6.configs.xai_iter_fixedq.repeats[0].rounds[2].rescore_fallback`); it is indistinguishable from single-shot
   (25 / 19, p 1.0) and the cause of the NaN was not verified.
4. **Narrow evaluation.** 500 items per task, one prompt template, zero-shot, MMLU over 10 subjects; generation, code, long
   context and multilingual ability were not measured. Power for small differences is low (Qwen 10 %: 9 / 9 numerical
   wins, 0 significant). MMLU at 60 % is dominated by letter bias (2.7).
5. **Two dense 7B models on one GPU type.** Qwen has no ratio sweep (10 % and 20 % only); why Qwen is far more
   perplexity-sensitive than Mistral (+82.1 % vs +19.3 % with fewer pruned parameters) was not measured. Other scales, MoE
   models and other hardware were not tested.
6. **MLP blocks are never pruned.** Pruning is at attention-head level only; MLP blocks receive precision tiers.
7. **No speed-up.** The gain is in model size only (Section 4).
8. **Moderate sensitivity to the calibration sample** (2.15); sensitivity to the IG baseline was not measured; calibration
   is 16 passages / 2 678 tokens.
9. **Three random seeds.** Some random controls have a very large spread (e.g. HQQ B = 3.0: 26.51 ± 15.61), so
   "significant against all three seeds" is a small-sample rule; the 2 × 2 domain design uses one random seed.
10. **Statistical scope.** The window bootstrap does not include calibration-sample or seed uncertainty; the McNemar tests
    depend on a single prompt template; E-3, C-3 and D-2 task tests are extra families and descriptive.
11. **Masked pruning does not reduce size.** The 8.343 GB of masked configurations comes from INT4 alone; only physical
    removal gives 8.199 GB. LoRA adapter weights are not included in the repository.
12. **Measurement caveats.** GB definitions differ slightly between quantizers (see Conventions); HQQ nominal and effective
    bits differ; the kernel comparison was measured on the single-shot plan; bit-identical re-application is guaranteed only on the same
    software stack.

## 6. Future work

1. **MoE and larger scale:** the same IG spectrum extends to expert pruning; nothing outside 7B dense models was tested.
2. **MLP neuron / intermediate-dimension pruning:** MLPs hold about 65 % of the parameters but only receive precision tiers.
3. **Mixed-domain calibration** to reduce the in-domain perplexity bias (2.8).
4. **A cross-domain evaluation protocol** as the default: report in-domain and out-of-domain perplexity plus tasks.
5. **Re-scoring domain at 20 / 40 %:** test whether C4 re-scoring also changes task results at lower ratios (E-3 was 60 %).
6. **Bit allocation with Hessian information** or a minimum of 3 bits per module (2.11).
7. **Task-based LoRA recovery** instead of in-domain language modelling (Section 4).
8. **Head mechanism analysis** (attention patterns) for the domain-specific heads.
9. **Fused INT4 kernels** to turn the size gain into a latency gain.
10. **A powered Qwen 10 % study** (2 000+ items, 5 seeds).

## 7. Experiment inventory

45 recorded GPU runs, 13.95 GPU-hours on one L40S: 40 run records (`run.total_seconds`, including the 0.48 h
`results/gun7_smoke.json` run) plus 5 attribution runs (`results/importance_scores*.json` and
`results/gun9_scores/importance_scores_qwen_qwen2_5_7b_instruct.json`, `timing.model_load_seconds + attribution_seconds`).
The six speed runs (about 0.33 h, no `total_seconds`), preflights and idle time are not included. The longest runs are
`results/iterative_gun7.json` (4.53 h) and `results/mixed_bits_gun11.json` (1.36 h); `results/calib_c4_gun14_f06.json`
ended as `completed_with_failures` and is superseded by `results/calib_c4_gun16_f06.json`. The table
`kapanis_deney_envanteri` of `make_figures.py` regenerates this inventory.
