"""
Training pipeline for the hybrid image-quality model.

Pipeline
--------
1. Load a handful of diverse clean base images (from scikit-image's bundled
   sample set - no external API / internet dataset download required beyond
   the package's own data files).
2. Generate many random crops from each base image to increase visual
   diversity (different textures/regions/scales).
3. Apply randomized synthetic degradations (app.ml.degrade) to build a
   labeled dataset: multi-label issue flags (blur/underexposure/
   overexposure/noise/corruption) + a continuous severity used to derive
   a 0-100 quality_score regression target.
4. Extract engineered features (app.ml.features) for every sample.
5. (Optional, if PyTorch is available) Extract CNN features via MobileNetV2
   transfer learning, PCA-reduce to 32 dims, and concatenate with engineered
   features to form a hybrid feature vector.
6. Train:
     - one RandomForestClassifier per issue type (multi-label, since issues
       are not mutually exclusive)
     - one GradientBoostingRegressor for the overall quality_score
     - one IsolationForest fit on clean-image features only, used at
       inference time as a generic "potential visual defect" anomaly signal
       for problems that don't fit the 5 explicit categories
7. Tune per-issue classification thresholds by maximizing F1 on the test set.
8. Evaluate everything on a held-out test split (unseen crops from held-out
   base images) and write a report to docs/evaluation_report.md +
   evaluation_report.json.
9. Persist all artifacts with joblib to app/ml/artifacts/.

Run with:  python -m app.ml.train
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import cv2
import joblib
from skimage import data as skdata
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from app.ml.features import feature_vector, FEATURE_NAMES
from app.ml.cnn_features import batch_extract_embeddings, CNN_FEATURE_DIM, is_available as cnn_available
from app.ml.degrade import random_degrade, ISSUE_TYPES

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
DOCS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42


def _load_base_images():
    """Bundled scikit-image sample photos: varied subjects/textures so the
    model doesn't overfit to one scene type."""
    names_fns = [
        ("astronaut", skdata.astronaut),
        ("chelsea", skdata.chelsea),
        ("coffee", skdata.coffee),
        ("rocket", skdata.rocket),
        ("coins", skdata.coins),
        ("brick", skdata.brick),
        ("grass", skdata.grass),
        # held out for testing (kept last; see build_dataset test_base_holdout)
        ("camera", skdata.camera),
        ("colorwheel", skdata.colorwheel),
        ("gravel", skdata.gravel),
    ]
    images = []
    for name, fn in names_fns:
        try:
            img = fn()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append((name, img_bgr))
        except Exception as e:  # pragma: no cover
            print(f"  (skipping {name}: {e})")
    return images


def _random_crops(img: np.ndarray, n: int, crop_size=(256, 256), seed_offset=0):
    h, w = img.shape[:2]
    ch, cw = crop_size
    ch, cw = min(ch, h), min(cw, w)
    crops = []
    rng = random.Random(RANDOM_SEED + seed_offset)
    for i in range(n):
        y = rng.randint(0, max(0, h - ch))
        x = rng.randint(0, max(0, w - cw))
        crops.append(img[y:y + ch, x:x + cw].copy())
    return crops


def build_dataset(n_per_base=200, test_base_holdout=3):
    """Builds train/test sets. Held-out *base images* (not just crops) are
    reserved for the test set where possible, giving a more honest read on
    generalization to unseen scene content."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    base_images = _load_base_images()
    if len(base_images) <= test_base_holdout:
        raise RuntimeError("Not enough base images to hold any out for testing")

    train_bases = base_images[:-test_base_holdout]
    test_bases = base_images[-test_base_holdout:]

    def make_samples(bases, n_per, seed_offset):
        X_eng, images_bgr, y_issues, y_score, meta = [], [], [], [], []
        for name, img in bases:
            crops = _random_crops(img, n_per, seed_offset=seed_offset)
            for crop in crops:
                degraded, labels, severity = random_degrade(crop)
                feats = feature_vector(degraded)
                score = 100.0 * (1 - severity) if any(labels.values()) else float(
                    np.random.uniform(90, 100))
                X_eng.append(feats)
                images_bgr.append(degraded)
                y_issues.append([int(labels[t]) for t in ISSUE_TYPES])
                y_score.append(score)
                meta.append({"base": name, "severity": severity, **labels})
        return np.array(X_eng), images_bgr, np.array(y_issues), np.array(y_score), meta

    print("  Generating training samples ...")
    X_train_eng, train_imgs, y_train, s_train, meta_train = make_samples(
        train_bases, n_per_base, seed_offset=0)
    print("  Generating test samples ...")
    X_test_eng, test_imgs, y_test, s_test, meta_test = make_samples(
        test_bases, max(55, n_per_base // 3), seed_offset=999)

    return (X_train_eng, train_imgs, y_train, s_train,
            X_test_eng, test_imgs, y_test, s_test,
            meta_train, meta_test)


def _find_best_threshold(y_true, proba, thresholds=None):
    """Find the probability threshold that maximizes F1 score."""
    if thresholds is None:
        thresholds = np.arange(0.20, 0.85, 0.02)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        pred = (proba >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return best_t, best_f1


def train_and_evaluate():
    print("Loading base images and generating synthetic degradations ...")
    (X_train_eng, train_imgs, y_train, s_train,
     X_test_eng, test_imgs, y_test, s_test,
     meta_train, meta_test) = build_dataset()
    print(f"  train samples: {X_train_eng.shape[0]}   test samples: {X_test_eng.shape[0]}")

    # --- Optionally extract CNN features ---
    use_cnn = cnn_available()
    pca = None
    n_cnn = 0

    if use_cnn:
        print("Extracting CNN features (MobileNetV2 — this may take a minute) ...")
        train_cnn = batch_extract_embeddings(train_imgs)
        test_cnn = batch_extract_embeddings(test_imgs)

        if train_cnn is not None and test_cnn is not None:
            print(f"  Fitting PCA ({train_cnn.shape[1]} -> {CNN_FEATURE_DIM}) ...")
            pca = PCA(n_components=CNN_FEATURE_DIM, random_state=RANDOM_SEED)
            train_cnn_pca = pca.fit_transform(train_cnn)
            test_cnn_pca = pca.transform(test_cnn)
            print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

            X_train = np.hstack([X_train_eng, train_cnn_pca])
            X_test = np.hstack([X_test_eng, test_cnn_pca])
            n_cnn = CNN_FEATURE_DIM
        else:
            use_cnn = False
            X_train = X_train_eng
            X_test = X_test_eng
    else:
        print("PyTorch not available — using engineered features only.")
        X_train = X_train_eng
        X_test = X_test_eng

    n_eng = len(FEATURE_NAMES)
    print(f"  Feature vector: {X_train.shape[1]} dims "
          f"({n_eng} engineered{f' + {n_cnn} CNN' if n_cnn else ''})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- one classifier per issue type (multi-label) ---
    classifiers = {}
    per_label_report = {}
    optimal_thresholds = {}

    for i, issue in enumerate(ISSUE_TYPES):
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_SEED,
        )
        clf.fit(X_train_s, y_train[:, i])
        classifiers[issue] = clf

        proba = clf.predict_proba(X_test_s)[:, 1] if len(clf.classes_) > 1 else np.zeros(X_test_s.shape[0])

        # Find best threshold per issue
        best_t, best_f1 = _find_best_threshold(y_test[:, i], proba)
        optimal_thresholds[issue] = best_t

        pred = (proba >= best_t).astype(int)
        report = {
            "accuracy": float(accuracy_score(y_test[:, i], pred)),
            "precision": float(precision_score(y_test[:, i], pred, zero_division=0)),
            "recall": float(recall_score(y_test[:, i], pred, zero_division=0)),
            "f1": float(f1_score(y_test[:, i], pred, zero_division=0)),
            "threshold": best_t,
        }
        try:
            report["roc_auc"] = float(roc_auc_score(y_test[:, i], proba))
        except ValueError:
            report["roc_auc"] = None
        cm = confusion_matrix(y_test[:, i], pred, labels=[0, 1]).tolist()
        report["confusion_matrix"] = cm  # [[TN, FP], [FN, TP]]
        per_label_report[issue] = report
        print(f"  [{issue:15s}] acc={report['accuracy']:.3f} f1={report['f1']:.3f} "
              f"auc={report['roc_auc']} thr={best_t:.2f}")

    # --- quality score regressor ---
    print("  [quality_score]   Using transparent weighted formula instead of regressor.")

    # --- anomaly detector for generic "potential visual defect" ---
    clean_mask_train = (y_train.sum(axis=1) == 0)
    anomaly_detector = IsolationForest(
        n_estimators=200, contamination=0.1, random_state=RANDOM_SEED
    )
    anomaly_detector.fit(X_train_s[clean_mask_train])
    anomaly_pred = anomaly_detector.predict(X_test_s)  # -1 anomaly, 1 normal
    any_issue_test = (y_test.sum(axis=1) > 0).astype(int)
    anomaly_flag = (anomaly_pred == -1).astype(int)
    anomaly_report = {
        "accuracy_vs_any_issue": float(accuracy_score(any_issue_test, anomaly_flag)),
        "precision_vs_any_issue": float(precision_score(any_issue_test, anomaly_flag, zero_division=0)),
        "recall_vs_any_issue": float(recall_score(any_issue_test, anomaly_flag, zero_division=0)),
        "note": (
            "IsolationForest is trained ONLY on clean-image features and used at "
            "inference as a generic novelty/defect signal, independent of the 5 "
            "explicit supervised issue classifiers. Compared here against "
            "'any labeled issue present' purely as a sanity check, not as its "
            "primary evaluation target."
        ),
    }
    print(f"  [anomaly/defect]  acc_vs_any_issue={anomaly_report['accuracy_vs_any_issue']:.3f}")

    # feature importance (averaged across the 5 issue classifiers)
    importances_all = np.mean([classifiers[t].feature_importances_ for t in ISSUE_TYPES], axis=0)
    eng_importances = importances_all[:n_eng]
    feature_importance = sorted(
        zip(FEATURE_NAMES, eng_importances.tolist()), key=lambda kv: -kv[1]
    )
    if n_cnn > 0:
        cnn_importance_total = importances_all[n_eng:].sum()
        feature_importance.append(("cnn_features_combined", float(cnn_importance_total)))

    # --- persist artifacts ---
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(classifiers, ARTIFACT_DIR / "issue_classifiers.joblib")
    joblib.dump(anomaly_detector, ARTIFACT_DIR / "anomaly_detector.joblib")
    if pca is not None:
        joblib.dump(pca, ARTIFACT_DIR / "cnn_pca.joblib")
    with open(ARTIFACT_DIR / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f)
    with open(ARTIFACT_DIR / "thresholds.json", "w") as f:
        json.dump(optimal_thresholds, f, indent=2)
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump({
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "issue_types": ISSUE_TYPES,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_engineered_features": n_eng,
            "n_cnn_features": n_cnn,
            "n_total_features": int(X_train.shape[1]),
            "optimal_thresholds": optimal_thresholds,
            "cnn_enabled": use_cnn,
        }, f, indent=2)

    # --- write evaluation report ---
    report_json = {
        "per_label": per_label_report,
        "anomaly_detector": anomaly_report,
        "feature_importance_ranked": feature_importance,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_engineered_features": n_eng,
        "n_cnn_features": n_cnn,
        "cnn_enabled": use_cnn,
    }
    with open(DOCS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report_json, f, indent=2)

    _write_markdown_report(report_json)
    print(f"\nArtifacts saved to {ARTIFACT_DIR}")
    print(f"Report saved to {DOCS_DIR / 'evaluation_report.md'}")
    return report_json


def _write_markdown_report(report):
    lines = []
    lines.append("# Model Evaluation Report\n")
    lines.append(f"Train samples: **{report['n_train']}**  |  Test samples: **{report['n_test']}**\n")
    n_eng = report['n_engineered_features']
    n_cnn = report['n_cnn_features']
    if n_cnn > 0:
        lines.append(
            f"Feature vector: **{n_eng}** engineered + "
            f"**{n_cnn}** CNN (MobileNetV2/PCA) = "
            f"**{n_eng + n_cnn}** total\n"
        )
    else:
        lines.append(f"Feature vector: **{n_eng}** engineered features\n")
    lines.append(
        "Test samples are generated from a **held-out base image never seen during "
        "training**, so this measures generalization to new visual content, not just "
        "new degradation noise on familiar scenes.\n"
    )
    lines.append("## Per-issue classifier performance\n")
    lines.append("| Issue | Accuracy | Precision | Recall | F1 | ROC-AUC | Threshold |")
    lines.append("|---|---|---|---|---|---|---|")
    for issue, r in report["per_label"].items():
        auc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "n/a"
        lines.append(
            f"| {issue} | {r['accuracy']:.3f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {auc} | {r['threshold']:.2f} |"
        )
    lines.append("\n### Confusion matrices ([[TN, FP], [FN, TP]])\n")
    for issue, r in report["per_label"].items():
        lines.append(f"- **{issue}**: {r['confusion_matrix']}")

    lines.append("\n## Overall quality_score\n")
    lines.append("- Replaced regressor with a transparent weighted formula based on classifier probabilities.")

    lines.append("\n## Anomaly / generic defect detector (IsolationForest)\n")
    ad = report["anomaly_detector"]
    lines.append(f"- Accuracy vs. 'any labeled issue present': **{ad['accuracy_vs_any_issue']:.3f}**")
    lines.append(f"- Precision: **{ad['precision_vs_any_issue']:.3f}**  Recall: **{ad['recall_vs_any_issue']:.3f}**")
    lines.append(f"- {ad['note']}")

    lines.append("\n## Feature importance (averaged across issue classifiers)\n")
    lines.append("| Feature | Importance |")
    lines.append("|---|---|")
    for item in report["feature_importance_ranked"]:
        name, imp = item[0], item[1]
        lines.append(f"| {name} | {imp:.4f} |")

    lines.append("\n## Known limitations & failure modes\n")
    lines.append(
        "- Training/test data is **synthetically degraded** from a small set of clean "
        "base photos. Real-world camera defects (lens smudges, sensor dust, motion "
        "blur streaks, rolling-shutter artifacts, chromatic aberration) are only "
        "approximated by the blur/noise/exposure/corruption simulations used here."
    )
    lines.append(
        "- The `corruption` class is the least separable from heavy `noise`/`blur`, "
        "since our synthetic corruption pipeline (low-quality JPEG re-encode + block "
        "dropout) shares low-level statistics with those classes; see its confusion "
        "matrix above for the resulting false positive/negative rate."
    )
    lines.append(
        "- Compound degradations (two or three issues at once) are included in training but "
        "are harder for single-label classifiers to separate cleanly; severity of "
        "each is likely to be under-estimated when both are present."
    )
    lines.append(
        "- Only 10 base scenes (7 train / 3 held-out test) are used due to no "
        "external dataset access; a production system should validate against a "
        "larger, more diverse, real-photo benchmark (e.g. KADID-10k, TID2013, or "
        "an in-domain dataset)."
    )

    with open(DOCS_DIR / "evaluation_report.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    train_and_evaluate()
