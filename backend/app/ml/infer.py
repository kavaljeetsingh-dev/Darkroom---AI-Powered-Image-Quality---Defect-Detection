"""
Inference: given a decoded image, run the full hybrid pipeline (engineered
features + optional CNN features -> trained classifiers/anomaly
detector) and produce a structured, explainable analysis result.

Quality score is computed via a transparent weighted-probability formula
rather than a black-box regressor, making it easier to explain and defend.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from app.ml.features import feature_vector, extract_features, sharpness_heatmap, FEATURE_NAMES
from app.ml.cnn_features import extract_cnn_features, is_available as cnn_available
from app.ml.degrade import ISSUE_TYPES

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# Human-readable descriptions + which raw feature(s) best explain each issue.
ISSUE_EXPLANATIONS = {
    "blur": {
        "description": "Image lacks sharp edges / fine detail, consistent with defocus or motion blur.",
        "driving_features": ["sharpness_lap_var", "sharpness_norm", "sharpness_tenengrad", "edge_density", "gradient_coherence"],
    },
    "underexposure": {
        "description": "Image is too dark; shadow detail is likely clipped/lost.",
        "driving_features": ["brightness_mean", "dark_pixel_ratio", "hist_low_mass", "contrast_rms"],
    },
    "overexposure": {
        "description": "Image is too bright; highlight detail is likely clipped/blown out.",
        "driving_features": ["brightness_mean", "bright_pixel_ratio", "hist_high_mass", "contrast_rms"],
    },
    "noise": {
        "description": "Visible sensor/compression noise (graininess) detected, especially in flat regions.",
        "driving_features": ["noise_estimate", "noise_local_var", "noise_high_freq_energy"],
    },
    "corruption": {
        "description": "Severe visual degradation consistent with heavy compression artifacts, block loss, or file damage.",
        "driving_features": ["blockiness", "channel_mean_asymmetry", "entropy"],
    },
}

# Weights for the transparent quality-score formula.  Each classifier's
# predicted probability is multiplied by its weight and subtracted from
# 100 to obtain the final score.  Corruption is weighted most heavily.
QUALITY_WEIGHTS = {
    "blur": 0.25,
    "noise": 0.20,
    "underexposure": 0.20,
    "overexposure": 0.20,
    "corruption": 0.35,
}


class QualityModel:
    """Loads all trained artifacts once and exposes a single `.analyze()`
    entry point used by the API layer."""

    def __init__(self):
        self.scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
        self.classifiers = joblib.load(ARTIFACT_DIR / "issue_classifiers.joblib")
        self.anomaly_detector = joblib.load(ARTIFACT_DIR / "anomaly_detector.joblib")

        # Regressor is optional — we use the formula by default but keep
        # backward compatibility if the artifact exists.
        regressor_path = ARTIFACT_DIR / "quality_regressor.joblib"
        self.regressor = joblib.load(regressor_path) if regressor_path.exists() else None

        # CNN PCA is optional
        pca_path = ARTIFACT_DIR / "cnn_pca.joblib"
        self.pca = joblib.load(pca_path) if pca_path.exists() else None

        # Check if model was trained with CNN features
        meta_path = ARTIFACT_DIR / "metadata.json"
        self.use_cnn = False
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.use_cnn = meta.get("cnn_enabled", False)

        # Load per-issue calibrated thresholds
        thresholds_path = ARTIFACT_DIR / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                self.thresholds = json.load(f)
        else:
            # Fallback to 0.5 if thresholds file not found
            self.thresholds = {issue: 0.5 for issue in ISSUE_TYPES}

    @classmethod
    def artifacts_available(cls) -> bool:
        required = ["scaler.joblib", "issue_classifiers.joblib",
                     "anomaly_detector.joblib"]
        return all((ARTIFACT_DIR / r).exists() for r in required)

    # ------------------------------------------------------------------ #
    #  E3 — severity from feature magnitude, NOT from confidence          #
    # ------------------------------------------------------------------ #
    def _issue_severity(self, issue_type: str, feats: dict) -> str:
        """Determine severity based on actual image feature magnitudes."""
        if issue_type == "blur":
            s = feats.get("sharpness_norm", 1.0)
            if s < 0.02:
                return "high"
            if s < 0.06:
                return "medium"
            return "low"
        if issue_type == "noise":
            n = feats.get("noise_estimate", 0.0)
            if n > 15.0:
                return "high"
            if n > 7.0:
                return "medium"
            return "low"
        if issue_type == "underexposure":
            d = feats.get("dark_pixel_ratio", 0.0)
            if d > 0.70:
                return "high"
            if d > 0.40:
                return "medium"
            return "low"
        if issue_type == "overexposure":
            b = feats.get("bright_pixel_ratio", 0.0)
            if b > 0.70:
                return "high"
            if b > 0.40:
                return "medium"
            return "low"
        if issue_type == "corruption":
            bk = feats.get("blockiness", 0.0)
            if bk > 8.0:
                return "high"
            if bk > 3.0:
                return "medium"
            return "low"
        return "medium"  # fallback

    # ------------------------------------------------------------------ #
    #  E7 — transparent quality score from classifier probabilities       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_quality_score(probas: dict[str, float]) -> float:
        """score = 100 × (1 − Σ wᵢ × pᵢ), clamped to [0, 100]."""
        penalty = sum(
            QUALITY_WEIGHTS.get(issue, 0.0) * prob
            for issue, prob in probas.items()
        )
        return max(0.0, min(100.0, round(100.0 * (1.0 - penalty), 1)))

    def _quality_label(self, score: float, issues: list) -> str:
        has_corruption = any(i["type"] == "corruption" for i in issues)
        has_high_severity = any(i["severity"] == "high" for i in issues)

        if has_corruption or score < 50:
            return "DEFECTIVE"
        if score < 78 or has_high_severity:
            return "DEGRADED"
        return "ACCEPTABLE"

    # ------------------------------------------------------------------ #
    #  E2 — recommended action (PASS / REVIEW / REJECT)                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _recommended_action(score: float, issues: list) -> str:
        if any(
            issue["type"] == "corruption" or issue["severity"] == "high"
            for issue in issues
        ):
            return "REJECT"
        if any(issue["type"] == "potential_defect" for issue in issues):
            return "REVIEW"
        if score < 78 or any(issue["severity"] == "medium" for issue in issues):
            return "REVIEW"
        return "PASS"

    def _build_feature_vector(self, img_bgr: np.ndarray, feats_dict: dict) -> np.ndarray:
        """Build the feature vector: engineered only, or hybrid (eng + CNN)."""
        eng = np.array([[feats_dict[k] for k in FEATURE_NAMES]])

        if self.use_cnn and self.pca is not None and cnn_available():
            cnn = extract_cnn_features(img_bgr, pca=self.pca)
            if cnn is not None:
                return np.hstack([eng, cnn.reshape(1, -1)])

        return eng

    # ------------------------------------------------------------------ #
    #  E5 — top-N z-score deviations for anomaly evidence                 #
    # ------------------------------------------------------------------ #
    def _anomaly_evidence(self, x_scaled: np.ndarray, feats: dict,
                          top_n: int = 3) -> dict:
        """Return the top-N features that deviate most from the clean
        distribution (by absolute z-score)."""
        z = np.abs(x_scaled[0])
        # Only use engineered feature indices (first len(FEATURE_NAMES))
        n_eng = min(len(FEATURE_NAMES), len(z))
        top_indices = np.argsort(z[:n_eng])[-top_n:][::-1]
        evidence = {}
        for idx in top_indices:
            fname = FEATURE_NAMES[idx]
            evidence[fname] = round(float(feats[fname]), 4)
        return evidence

    def analyze(self, img_bgr: np.ndarray, include_heatmap: bool = True) -> dict:
        feats = extract_features(img_bgr)
        x = self._build_feature_vector(img_bgr, feats)
        x_scaled = self.scaler.transform(x)

        issues = []
        probas = {}  # collect all classifier probabilities for quality score

        for issue in ISSUE_TYPES:
            clf = self.classifiers[issue]
            proba = clf.predict_proba(x_scaled)[0]
            # proba indexed by clf.classes_; find prob of class==1
            classes = list(clf.classes_)
            confidence = float(proba[classes.index(1)]) if 1 in classes else 0.0
            probas[issue] = confidence

            threshold = self.thresholds.get(issue, 0.5)
            predicted = confidence >= threshold

            if predicted:
                info = ISSUE_EXPLANATIONS[issue]
                issues.append({
                    "type": issue,
                    "severity": self._issue_severity(issue, feats),  # E3
                    "confidence": round(confidence, 3),
                    "description": info["description"],
                    "evidence": {k: round(feats[k], 4) for k in info["driving_features"]},
                })

        # ---- anomaly detector (E4, E5) ---- #
        anomaly_pred = self.anomaly_detector.predict(x_scaled)[0]  # -1 anomaly, 1 normal
        anomaly_score = float(-self.anomaly_detector.score_samples(x_scaled)[0])  # higher = more anomalous
        if anomaly_pred == -1 and not any(i["type"] in ("corruption", "noise", "blur") for i in issues):
            # E4: report raw anomaly_score, not a fake confidence
            issues.append({
                "type": "potential_defect",
                "severity": "medium" if anomaly_score < 0.15 else "high",
                "confidence": None,  # E4: not a calibrated probability
                "anomaly_score": round(anomaly_score, 4),
                "description": (
                    "Image statistics are unusual relative to typical clean images, "
                    "but don't clearly match a specific known issue category. "
                    "Flagged by an anomaly detector for manual review."
                ),
                "evidence": self._anomaly_evidence(x_scaled, feats),  # E5
            })

        # ---- quality score (E7) — transparent weighted formula ---- #
        quality_score = self._compute_quality_score(probas)
        quality_label = self._quality_label(quality_score, issues)
        recommended_action = self._recommended_action(quality_score, issues)  # E2

        result = {
            "quality_score": quality_score,
            "quality_label": quality_label,
            "recommended_action": recommended_action,  # E2
            "issues": sorted(issues, key=lambda i: -(i["confidence"] or 0)),
            "image_stats": {k: round(v, 4) for k, v in feats.items()},
        }

        if include_heatmap:
            result["blur_heatmap_png_base64"] = self._render_blur_heatmap(img_bgr)

        return result

    def _render_blur_heatmap(self, img_bgr: np.ndarray) -> str:
        """Render a small overlay PNG (as base64) showing per-tile sharpness,
        for the frontend to display as a localization heatmap (bonus:
        'quality heatmaps / localization of problematic regions')."""
        heat = sharpness_heatmap(img_bgr, block=24)
        heat_norm = np.clip(heat, 0, np.percentile(heat, 95) + 1e-6)
        heat_norm = (heat_norm / (heat_norm.max() + 1e-9) * 255).astype(np.uint8)
        heat_resized = cv2.resize(heat_norm, (img_bgr.shape[1], img_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        heat_color = cv2.applyColorMap(255 - heat_resized, cv2.COLORMAP_JET)  # red = blurry
        overlay = cv2.addWeighted(img_bgr, 0.55, heat_color, 0.45, 0)
        ok, buf = cv2.imencode(".png", overlay)
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")


_model_singleton: QualityModel | None = None


def reset_model():
    """Clear the cached model singleton (useful for server reloads)."""
    global _model_singleton
    _model_singleton = None


def get_model() -> QualityModel:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = QualityModel()
    return _model_singleton
