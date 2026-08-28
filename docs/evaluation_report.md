# Model Evaluation Report

Train samples: **1400**  |  Test samples: **198**

Feature vector: **20** engineered features

Test samples are generated from a **held-out base image never seen during training**, so this measures generalization to new visual content, not just new degradation noise on familiar scenes.

## Per-issue classifier performance

| Issue | Accuracy | Precision | Recall | F1 | ROC-AUC | Threshold |
|---|---|---|---|---|---|---|
| blur | 0.783 | 0.585 | 0.932 | 0.719 | 0.866 | 0.56 |
| underexposure | 0.944 | 1.000 | 0.738 | 0.849 | 0.948 | 0.58 |
| overexposure | 0.904 | 0.717 | 0.846 | 0.776 | 0.928 | 0.42 |
| noise | 0.985 | 1.000 | 0.930 | 0.964 | 0.992 | 0.40 |
| corruption | 0.980 | 0.833 | 0.833 | 0.833 | 0.996 | 0.80 |

### Confusion matrices ([[TN, FP], [FN, TP]])

- **blur**: [[100, 39], [4, 55]]
- **underexposure**: [[156, 0], [11, 31]]
- **overexposure**: [[146, 13], [6, 33]]
- **noise**: [[155, 0], [3, 40]]
- **corruption**: [[184, 2], [2, 10]]

## Overall quality_score

- Replaced regressor with a transparent weighted formula based on classifier probabilities.

## Anomaly / generic defect detector (IsolationForest)

- Accuracy vs. 'any labeled issue present': **0.783**
- Precision: **0.820**  Recall: **0.913**
- IsolationForest is trained ONLY on clean-image features and used at inference as a generic novelty/defect signal, independent of the 5 explicit supervised issue classifiers. Compared here against 'any labeled issue present' purely as a sanity check, not as its primary evaluation target.

## Feature importance (averaged across issue classifiers)

| Feature | Importance |
|---|---|
| blockiness | 0.1043 |
| brightness_mean | 0.0941 |
| noise_estimate | 0.0848 |
| hist_low_mass | 0.0689 |
| sharpness_norm | 0.0685 |
| sharpness_lap_var | 0.0676 |
| noise_high_freq_energy | 0.0632 |
| bright_pixel_ratio | 0.0626 |
| hist_high_mass | 0.0556 |
| dark_pixel_ratio | 0.0499 |
| noise_local_var | 0.0437 |
| sharpness_tenengrad | 0.0374 |
| edge_density | 0.0372 |
| entropy | 0.0287 |
| contrast_rms | 0.0251 |
| colorfulness | 0.0251 |
| brightness_std | 0.0233 |
| saturation_mean | 0.0232 |
| channel_mean_asymmetry | 0.0224 |
| gradient_coherence | 0.0144 |

## Known limitations & failure modes

- Training/test data is **synthetically degraded** from a small set of clean base photos. Real-world camera defects (lens smudges, sensor dust, motion blur streaks, rolling-shutter artifacts, chromatic aberration) are only approximated by the blur/noise/exposure/corruption simulations used here.
- The `corruption` class is the least separable from heavy `noise`/`blur`, since our synthetic corruption pipeline (low-quality JPEG re-encode + block dropout) shares low-level statistics with those classes; see its confusion matrix above for the resulting false positive/negative rate.
- Compound degradations (two or three issues at once) are included in training but are harder for single-label classifiers to separate cleanly; severity of each is likely to be under-estimated when both are present.
- Only 10 base scenes (7 train / 3 held-out test) are used due to no external dataset access; a production system should validate against a larger, more diverse, real-photo benchmark (e.g. KADID-10k, TID2013, or an in-domain dataset).