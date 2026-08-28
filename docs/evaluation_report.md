# Model Evaluation Report

Train samples: **1400**  |  Test samples: **198**

Feature vector: **20** engineered features

Test samples are generated from a **held-out base image never seen during training**, so this measures generalization to new visual content, not just new degradation noise on familiar scenes.

## Per-issue classifier performance

| Issue | Accuracy | Precision | Recall | F1 | ROC-AUC | Threshold |
|---|---|---|---|---|---|---|
| blur | 0.793 | 0.592 | 0.983 | 0.739 | 0.877 | 0.40 |
| underexposure | 0.949 | 0.944 | 0.810 | 0.872 | 0.948 | 0.48 |
| overexposure | 0.955 | 1.000 | 0.769 | 0.870 | 0.935 | 0.64 |
| noise | 0.985 | 1.000 | 0.930 | 0.964 | 0.988 | 0.46 |
| corruption | 0.985 | 0.909 | 0.833 | 0.870 | 0.996 | 0.84 |

### Confusion matrices ([[TN, FP], [FN, TP]])

- **blur**: [[99, 40], [1, 58]]
- **underexposure**: [[154, 2], [8, 34]]
- **overexposure**: [[159, 0], [9, 30]]
- **noise**: [[155, 0], [3, 40]]
- **corruption**: [[185, 1], [2, 10]]

## Overall quality_score

- Replaced regressor with a transparent weighted formula based on classifier probabilities.

## Anomaly / generic defect detector (IsolationForest)

- Accuracy vs. 'any labeled issue present': **0.783**
- Precision: **0.820**  Recall: **0.913**
- IsolationForest is trained ONLY on clean-image features and used at inference as a generic novelty/defect signal, independent of the 5 explicit supervised issue classifiers. Compared here against 'any labeled issue present' purely as a sanity check, not as its primary evaluation target.

## Feature importance (averaged across issue classifiers)

| Feature | Importance |
|---|---|
| blockiness | 0.1040 |
| brightness_mean | 0.1009 |
| noise_estimate | 0.0842 |
| sharpness_norm | 0.0730 |
| hist_low_mass | 0.0673 |
| sharpness_lap_var | 0.0663 |
| noise_high_freq_energy | 0.0639 |
| bright_pixel_ratio | 0.0600 |
| hist_high_mass | 0.0552 |
| dark_pixel_ratio | 0.0496 |
| noise_local_var | 0.0429 |
| sharpness_tenengrad | 0.0372 |
| edge_density | 0.0345 |
| entropy | 0.0293 |
| contrast_rms | 0.0256 |
| colorfulness | 0.0250 |
| saturation_mean | 0.0236 |
| brightness_std | 0.0222 |
| channel_mean_asymmetry | 0.0217 |
| gradient_coherence | 0.0134 |

## Known limitations & failure modes

- Training/test data is **synthetically degraded** from a small set of clean base photos. Real-world camera defects (lens smudges, sensor dust, motion blur streaks, rolling-shutter artifacts, chromatic aberration) are only approximated by the blur/noise/exposure/corruption simulations used here.
- The `corruption` class is the least separable from heavy `noise`/`blur`, since our synthetic corruption pipeline (low-quality JPEG re-encode + block dropout) shares low-level statistics with those classes; see its confusion matrix above for the resulting false positive/negative rate.
- Compound degradations (two or three issues at once) are included in training but are harder for single-label classifiers to separate cleanly; severity of each is likely to be under-estimated when both are present.
- Only 10 base scenes (7 train / 3 held-out test) are used due to no external dataset access; a production system should validate against a larger, more diverse, real-photo benchmark (e.g. KADID-10k, TID2013, or an in-domain dataset).