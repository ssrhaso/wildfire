# Paper data extract for `UWE.tex`

All numbers below are computed from the existing `results/{model}/{config}/seed_*.json`
files. No retraining was performed.

Reproduce with `PYTHONIOENCODING=utf-8 python scripts/paper_extract.py`.

**Stale-CSV warning.** `results/analysis/hybrid/summary.csv` and `statistics.csv`
contain older numbers for `freeze_backbone` (mean 0.9882, std 0.00231). The current
seed JSONs give mean 0.9878, std 0.00151. The seed JSONs are the source of truth;
all hybrid numbers below are recomputed from them. Re-running
`python src/analyse_results.py --model hybrid` will refresh the CSVs.

---

## 1. Missing ViT rows (Table 5 gaps)

ViT-B/16, mean ± std (n=5 seeds, ddof=1) from `results/vit/{config}/seed_*.json`.
All five seeds {0, 5, 10, 15, 20} ran successfully for both configs; no missing seeds.

| Config | Test accuracy | F1 (fire) |
|---|---|---|
| `freeze_patch_blocks0-3` | **0.9869 ± 0.0022** | **0.9874 ± 0.0022** |
| `freeze_patch_blocks0-5` | 0.9910 ± **0.0012** | 0.9913 ± **0.0012** |

Per-seed test accuracy:
- `freeze_patch_blocks0-3`: 0.9877 (s0), 0.9864 (s5), 0.9843 (s10), 0.9902 (s15), 0.9860 (s20)
- `freeze_patch_blocks0-5`: 0.9902 (s0), 0.9911 (s5), 0.9894 (s10), 0.9919 (s15), 0.9924 (s20)

Per-seed F1 (fire):
- `freeze_patch_blocks0-3`: 0.9881, 0.9869, 0.9848, 0.9906, 0.9865
- `freeze_patch_blocks0-5`: 0.9906, 0.9914, 0.9897, 0.9922, 0.9926

Notes:
1. Means for `freeze_patch_blocks0-5` (0.9910 / 0.9913) match the values already in the paper.
2. `freeze_patch_blocks0-3` row needs both mean and std added — values above.

---

## 2. Compute hours

Source: `total_train_time_seconds` field in each seed JSON (n=165 runs total).
W&B log dump and Colab job logs are not present in the repo, so the per-run device
record comes from each JSON's `gpu_name` field — there is **no T4 entry**, so the
"RTX 4070 vs Colab T4" split requested cannot be produced; actual device mix below.

**Total: 260,751 s = 72.43 GPU-h across 165 runs.**

By architecture:

| Architecture | n runs | GPU-h | Seconds |
|---|---:|---:|---:|
| ViT (6 cfgs × 5) | 30 | 12.60 | 45,356 |
| ResNet-50 (6 cfgs × 5) | 30 | 37.15 | 133,750 |
| Hybrid (21 cfgs × 5) | 105 | 22.68 | 81,644 |

By GPU device (no Colab T4 records present):

| Device | n runs | GPU-h |
|---|---:|---:|
| NVIDIA GTX 1060 3GB | 30 | 37.15 |
| NVIDIA RTX 4090 | 106 | 22.88 |
| NVIDIA RTX 4070 | 14 | 9.46 |
| NVIDIA RTX 5080 | 10 | 2.35 |
| NVIDIA RTX 5090 | 5 | 0.59 |

1. Total wall-clock = **72.43 GPU-h** for the full 165-run sweep.
2. ResNet on GTX 1060 dominates at 37.15 GPU-h (≈ 51% of total) — slow card, not the architecture itself.
3. The full 30-run ViT sweep took **12.60 GPU-h** mostly on RTX 4090 / 5080 / 5090.
4. The 105-run Hybrid sweep took **22.68 GPU-h** entirely on RTX 4090.
5. **No T4 was used** — paper's 4070-vs-T4 framing should be replaced with 4090 / 4070 / 1060 mix or generalised to "consumer GPUs". Reproduce with `scripts/paper_extract.py` (`section2()`).

---

## 3. Statistical significance — three headline pairs

Welch's two-sample t (`scipy.stats.ttest_ind(equal_var=False)`, n=5 each, df via Welch–Satterthwaite). Cohen's d uses pooled SD with `ddof=1`.

| Pair | Δ acc (pp) | p-value | Cohen's d | df | Match |
|---|---:|---:|---:|---:|---|
| ViT `freeze_patch` → `freeze_patch_b0-8` | -2.9117 | 3.2e-07 | -13.10 | 6.55 | matches paper (Δ=2.91, p=3.2e-7, d=-13.1) |
| RN50 `freeze_conv1_l1-3` vs `_l1-4` | +2.0968 | 1.7e-05 | +6.40 | 7.14 | matches paper (Δ=2.10, p=1.7e-5, d=6.40) |
| Hybrid `freeze_blocks0-8` vs `freeze_backbone_proj_blocks0-8` | +6.2903 | 1.9e-05 | +11.77 | 4.53 | matches paper (Δ=6.29, p<1e-4, d=11.77) |

All three claims verified, none flagged at the 5% relative-error threshold.

1. ViT pair: Δ=-2.91 pp (sign flip per direction), p=3.2e-07, d=-13.10, df=6.55.
2. RN50 pair: Δ=+2.10 pp, p=1.7e-05, d=+6.40, df=7.14.
3. Hybrid pair: Δ=+6.29 pp, p=1.9e-05, d=+11.77, df=4.53. (Paper says "p<0.0001"; exact value is 1.9e-05.)

---

## 4. Better Hybrid headline comparison

All four candidate Hybrid headline pairs (Welch's t, n=5 each):

| Pair | m₁ | m₂ | Δ (pp) | p | Cohen's d | df | Δ/\|d\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| `freeze_backbone` vs `freeze_blocks0-8_bnfrozen` (current) | 0.9878 | 0.5171 | +47.06 | 1.6e-15 | +229.96 | 6.63 | 0.20 |
| `freeze_backbone` vs `freeze_none` (best vs full FT) | 0.9878 | 0.9761 | +1.16 | 0.10 | +1.32 | 4.12 | 0.88 |
| `freeze_blocks0-8` vs `freeze_backbone_proj_blocks0-8` (proj-layer) | 0.9876 | 0.9247 | +6.29 | 1.9e-05 | +11.77 | 4.53 | 0.53 |
| `freeze_backbone` vs `freeze_transformer_only` (component) | 0.9878 | 0.9828 | +0.50 | 0.39 | +0.60 | 4.14 | 0.83 |

Note: the current-headline values differ from the paper's claim (Δ=47.11, p=1.5e-17, d=197) because the underlying `freeze_backbone` JSONs have been re-run since `summary.csv` was generated — fresh mean is 0.9878 not 0.9882. The result is still extreme; only the precise digits change.

1. **Highest Δ/\|d\| ratio**: `freeze_backbone` vs `freeze_none` (0.88) — but **p=0.10**, not significant. Same caveat for `freeze_backbone` vs `freeze_transformer_only` (0.83, p=0.39).
2. Among **significant** pairs: `freeze_blocks0-8` vs `freeze_backbone_proj_blocks0-8` has Δ/\|d\|=0.53 — highest of the significant set, with a comprehensible 6.3 pp gap, d=11.77 instead of 230, and the cleanest mechanistic story (isolates the projection-layer effect with all other freezes held constant).
3. **Recommendation: replace the headline with the `freeze_blocks0-8` vs `freeze_backbone_proj_blocks0-8` pair.** It is the only candidate that is both significant and has an interpretable effect size; the existing best-vs-failure framing inflates d to ~230 by including a divergent BN-frozen failure mode.

---

## 5. ResNet-50 plateau verification

All `C(6,2)=15` pairwise Welch's t-tests across the 6 ResNet-50 configs. Pairs with **p<0.05** below (sorted by p):

| Config 1 | Config 2 | Δ (pp) | p | Cohen's d |
|---|---|---:|---:|---:|
| `freeze_conv1_layer1-3` | `freeze_conv1_layer1-4` | +2.0968 | 1.7e-05 | +6.40 |
| `freeze_conv1_layer1` | `freeze_conv1_layer1-4` | +1.9355 | 3.9e-05 | +6.18 |
| `freeze_conv1` | `freeze_conv1_layer1-4` | +1.9100 | 6.9e-05 | +6.39 |
| `freeze_conv1_layer1-2` | `freeze_conv1_layer1-4` | +2.0713 | 1.1e-04 | +7.33 |

The remaining 11 pairs all have p > 0.05 (smallest non-significant p ≈ 0.15 between `freeze_conv1` and `freeze_conv1_layer1-2`).

1. **Exactly 4 pairs reach p<0.05** — paper's "four significant comparisons" wording is correct.
2. **All four involve `freeze_conv1_layer1-4`** as one member of the pair — paper's statement holds.
3. **No other ResNet pair differs significantly** (p > 0.05 for all 11 non-L1-4 pairs) — plateau claim confirmed verbatim.

---

## 6. Macro metrics for best-of-architecture (n=5 seeds, mean ± std, ddof=1)

Macro F1 = mean of `test_f1_fire` and `test_f1_nofire` per run, then averaged across seeds.

| Architecture | Best config | Test acc | Macro F1 | Precision (fire) | Recall (fire) | F1 (fire) |
|---|---|---:|---:|---:|---:|---:|
| ViT-B/16 | `freeze_patch_blocks0-8` | 0.9932 ± 0.0016 | **0.9932 ± 0.0016** | **0.9956 ± 0.0012** | **0.9913 ± 0.0031** | 0.9934 ± 0.0016 |
| ResNet-50 | `freeze_conv1_layer1-3` | 0.9873 ± 0.0027 | **0.9873 ± 0.0026** | **0.9935 ± 0.0018** | **0.9818 ± 0.0057** | 0.9876 ± 0.0026 |
| Hybrid | `freeze_backbone` | 0.9878 ± 0.0015 | **0.9878 ± 0.0015** | **0.9919 ± 0.0022** | **0.9845 ± 0.0020** | 0.9882 ± 0.0015 |

1. ViT precision-fire = 0.9956, the highest of the three.
2. ResNet has the lowest fire-recall (0.9818) — most missed fires of the three best configs.
3. Hybrid sits between ViT and ResNet on every metric except std (it has the tightest variance across seeds).

---

## 7. Available figures inventory

Source: `find results -type f \( -iname "*.png" -o -iname "*.pdf" \)`.

**Boxplots** (one per architecture, both PNG and PDF):
- `results/analysis/vit/boxplot_accuracy.{png,pdf}`
- `results/analysis/resnet/boxplot_accuracy.{png,pdf}`
- `results/analysis/hybrid/boxplot_accuracy.{png,pdf}`

**Grad-CAM**: **[unavailable]** — no Grad-CAM PNGs exist in the repo. `src/gradcam.py` and `src/compare_models.py` are present but `results/analysis/*/eval_cache/` is empty and no `compare_*.png` files are on disk. The specific `compare_0000_resnet.png` referenced in the paper draft does not exist; cannot confirm the `freeze_conv1_l1-3` (best) vs `freeze_none` (worst) overlay or the ~100% / ~94.2% confidence values. Re-running `python src/compare_models.py` (or `src/gradcam.py`) would be required to regenerate them.

**Cross-model**:
- `results/analysis/cross_model_comparison.{png,pdf}` — single-figure comparison across the three architectures.

**Per-architecture supplementary** (PNG and PDF, in `results/analysis/{vit,resnet,hybrid}/`):
- `confusion_matrices.{png,pdf}` — best vs worst config per architecture.
- `train_val_curves.{png,pdf}` — averaged train/val accuracy curves, best and worst per architecture.
- `val_curves.{png,pdf}` — per-config validation accuracy curves (one line per freeze).
- `val_f1_fire_curves.{png,pdf}` — per-config validation F1-fire curves.
- `lr_schedule.{png,pdf}` — cosine LR schedule trace per freeze (first seed).

**Other**: no per-config attention rollout, training-loss-only, or per-seed plots are saved. `venv/images/imagehash.png` is a third-party library asset, not project output.

1. Boxplots, confusion matrices, val curves, F1-fire curves, LR schedules and train/val curves all exist for all three architectures (PNG + PDF).
2. Cross-model comparison figure exists at `results/analysis/cross_model_comparison.{png,pdf}`.
3. **Grad-CAM figures are not present** — paper references must either be regenerated via `src/gradcam.py` / `src/compare_models.py`, or removed.

---

## 8. Sanity check totals

Source: directory walk of `results/{vit,resnet,hybrid}/*/seed_*.json`; param counts and dataset sizes from any seed JSON.

| Quantity | Paper claim | Observed | Match |
|---|---|---|---|
| Total runs | 165 (33 × 5) | 165 (33 × 5) | OK |
| ViT configs | 6 | 6 (30 runs) | OK |
| ResNet configs | 6 | 6 (30 runs) | OK |
| Hybrid configs | 21 | 21 (105 runs) | OK |
| Train split | 18,847 | 18,847 | OK |
| Val split | 2,356 | 2,356 | OK |
| Test split | 2,356 | 2,356 | OK |
| ViT-B/16 total params | ≈86M | 85,800,194 | OK (85.8M) |
| Hybrid total params | ≈94.5M | 94,540,098 | OK (94.5M) |
| ResNet-50 total params | ≈25M | 23,512,130 | **flag**: actual is 23.5M, not 25M |

1. 33 configs × 5 seeds = 165 runs confirmed (6 ViT + 6 ResNet + 21 Hybrid).
2. Train/val/test = 18847 / 2356 / 2356 confirmed (total 23,559).
3. ViT-B/16 ≈ 85.8M and Hybrid ≈ 94.5M confirmed; **ResNet-50 reports 23.5M, not 25M** — the binary classifier head trims the standard 25.6M slightly. Replace "≈25M" with "≈23.5M" (or "≈24M") in Table 1.
