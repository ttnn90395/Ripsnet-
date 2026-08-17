# TFN Robustness to Corrupted Point Clouds — Findings

## Setup

- Full-scale 2D circles dataset: 600 points/cloud, 3 classes (1-, 2-, 3-circle
  configurations), 900 train / 300 test. "Noisy" test corrupts 200 of 600
  points (33%) with uniform noise inside each circle's bounding box.
- 3D shapes (`shapes3d_*`): 600 points, Gaussian jitter (`noise_sigma=0.15`)
  at evaluation.
- Flag suffix order in result files follows source flag order
  (e.g. `--noise-aug --robust-knn` → `_noise-aug_robust-knn`).

## Headline numbers (GTTensorFieldNetworkV2, full-scale circles)

| Setting | clean | noisy |
|---|---:|---:|
| baseline (3 trials, 100 ep) | 1.000 | 0.667 |
| PersNet baseline (non-TFN reference) | 0.863 | 0.749 |
| robust-knn (eval geometry) | 1.000 | 0.570 |
| DTM pre-filter, keep 0.7–0.9 (eval) | 0.60–0.92 | 0.667 |
| DTM-softcap readout, eval-only (`--dtm-readout`) | 0.703 | 0.667 |
| noise-aug + robust-knn (es@17) | 0.733 | 0.670 |
| matched noise-aug (r=0.33, s=1.0) + train-robust-knn | 0.737 | 0.673 |
| mixed-batch noise-aug, frac=0.5 (`--aug-frac 0.5`) | 1.000 | 0.667 |
| max-pool readout (`--pool max`) | 0.993 | 0.333 |
| norm-readout (`--norm-readout`) | 1.000 | 0.667 |
| trained directly on `circles_noisy` (matched dist.) | 0.999 | 0.999 |
| multiscale (`--multiscale`, 2 trials) | 1.000 | 0.667 |
| denoise (`--denoise`, 2 trials) | 1.000 | 0.667 |
| geom-reg (`--geom-reg`, 2 trials) | 1.000 | 0.667 |
| train-dtm-readout (`--train-dtm-readout`, 2 trials) | 1.000 | 0.667 |
| cov-feat (`--cov-feat`, 2 trials) | 1.000 | 0.667 |
| feat-pd (`--feat-pd`, 2 trials) | 1.000 | 0.553 |
| consistency (`--consistency`, 2 trials) | 1.000 | 0.440 |

All newly-tested flag variants preserve clean accuracy (1.000) but none
escape the 0.667 noisy plateau on circles; `feat-pd` (0.553) and
`consistency` (0.440) are *worse* than baseline noisy. On `circles_noisy`
(matched corruption distribution) the same variants reach 0.993–1.000
noisy, consistent with the distribution-shift conclusion.

3-class chance is 0.333, so 0.667 is *above* chance but notably below
PersNet (0.749). Earlier reports of a "collapse to chance" were incorrect.

## Why DTM-based fixes fail at high corruption

The motivating hypothesis was that outliers corrupt precomputed k-NN geometry,
so DTM-based outlier handling should recover accuracy. At 33% uniform
corruption all three DTM variants fail:

1. **Robust k-NN re-ranking** (`dtm_rank_distance`, `--robust-knn`): only
   changes neighbor selection. Outliers still enter the global descriptor
   through the readout `node_inv.sum(0)`, which sums over *every* point.
2. **DTM point pre-filtering** (`--dtm-filter`): removes low-density points,
   but at 33% corruption the uniform noise floods the density field and the
   filter also removes many legitimate boundary/inlier points (clean acc drops
   to 0.60–0.92; noisy unchanged).
3. **DTM-weighted readout** (`--dtm-readout`): soft outlier suppression in the
   readout. The descriptor is scale-sensitive (trained on sums over ~600
   points), so reweighting distorts the classifier input (clean 1.0 → 0.70).
   Even a mild soft-cap form breaks clean accuracy.

A quick-mode (100 pt, 20% corruption) DTM-filter experiment that recovered
0.67 → 0.89 does **not** transfer to full scale. (Early quick-mode numbers were
partly unreliable: one evaluation harness misaligned cloud labels with their
targets, producing spurious near-chance results. All headline numbers above
were re-verified with the CLI eval path / correctly aligned harness.)

## Noisy accuracy is insensitive to k-NN geometry at eval

Eval-only sweep of `k_neighbors` on a full-scale 100-epoch GTTFNv2 checkpoint
(trained with k=16), clean and noisy test sets generated identically to the
training script:

| eval k | clean | noisy |
|---|---:|---:|
| 6  | 0.633 | 0.667 |
| 8  | 0.660 | 0.667 |
| 12 | 0.670 | 0.667 |
| 16 (trained) | 1.000 | 0.667 |
| 24 | 0.667 | 0.667 |
| 32 | 0.667 | 0.613 |
| 48 | 0.583 | 0.333 |

Noisy accuracy is pinned at exactly 0.667 for k ∈ [6, 24] and only degrades
once neighborhoods grow large enough that noise points dominate. Corrupted-cloud
decisions therefore do not depend on fine local k-NN structure at all — the
classifier operates on global/radial statistics that survive the flood. Clean
accuracy, by contrast, is peaked at the trained k=16 and drops at any other k
(an eval-time distribution mismatch for clean, where local geometry still
matters). Re-running this sweep with DTM-robust k-NN cannot help: the 0.667
plateau is k-independent.

## Which class is lost under corruption

Per-class analysis on noisy clouds shows GTTFNv2 predicts classes 1 and 3
perfectly but **never predicts class 2** (two-circle clouds): `[1.00, 0.00,
1.00]` with prediction distribution `[100, 0, 200]`. The uniform in-circle flood
merges the two-circle radial signature into the three-circle one, so noisy
accuracy sits at exactly 2/3 — one class always misclassified. This explains
why every aggregation, geometry, and augmentation intervention lands at 0.667.

## Why augmentation fails at full scale

Training on corrupted clouds (`--noise-aug`, rates 0.2 and 0.33, scales 1.0
and 2.0) overfits the corrupted distribution: train acc reaches ~96–99% while
validation on clean clouds collapses (val 33–67%), triggering early stopping.
The model never preserves the clean-cloud representation, so neither clean
eval nor the *actual* noisy eval (33% corruption, uniform in-circle noise —
distributionally different from the augmentation) improves (noisy 0.667–0.673).

## Conclusion

- The clean→noisy gap on circles is a **distribution-shift** problem, not a
  k-NN-corruption problem. Evidence: GTTFNv2 reaches 0.999 noisy when trained
  directly on the corrupted distribution, i.e. the architecture can represent
  corrupted clouds — it just does not transfer across this shift.
- On **3D shapes** GTTFNv2 already transfers well under Gaussian jitter
  (topology 0.926, 8way 0.873 vs PersNet 0.887 / 0.659), so the weak spot
  is dataset-specific (2D multi-circle clouds with uniform flood corruption).

## Code changes (all additive, default-off)

- `gt_tfn_layer.py`:
  - `dtm_rank_distance(dist, m, alpha, eps)` — neighbor-side DTM scale for
    k-NN ranking (`robust_alpha` / `robust_m` kwargs on `knn_geometry`).
  - `dtm_readout_weights(pos, m, gamma, thr)` — soft DTM mask for the readout
    (`robust_readout` / `robust_m` / `robust_gamma` on base
    `GTTensorFieldNetwork`, covering GTTFN and GTTFNv2).
- `shape/train_shape.py`:
  - `--robust-knn`, `--robust-alpha=` — robust k-NN geometry everywhere.
  - `--train-robust-knn` — robust k-NN geometry in augmented training only
    (eval stays faithful to actual-cloud geometry).
  - `--dtm-readout`, `--dtm-gamma=` — soft DTM readout at final inference
    (trained unweighted).
  - `--noise-rate=`, `--noise-scale=` — matched corruption augmentation.
  - TFN models now receive noise-augmented clouds with on-the-fly geometry
    recompute (previously skipped).
- `shape/consolidate_results.py` — variant keys for `robust-knn`,
  `train-robust-knn`, `dtm-readout`, `readout_pool`, `norm_readout`,
  `aug_frac` in `results_summary.csv`.
