"""
Synthetic image-quality degradation generator.

We don't have access to a labeled real-world defect dataset (and are not
permitted to call external AI services), so training/evaluation data is
built by taking clean base images and applying controlled, parametrized
degradations. This is a standard and well-accepted strategy for image
quality assessment (IQA) research (e.g. LIVE, TID2013, KADID-10k all use
synthetically-degraded reference images).

Each function returns (degraded_image_bgr, labels_dict) where labels_dict
has boolean flags for each of the issue types the assessment asks us to
detect, plus a continuous 'severity' in [0, 1] used to build a
quality_score regression target.
"""
from __future__ import annotations

import random
import numpy as np
import cv2


ISSUE_TYPES = ["blur", "underexposure", "overexposure", "noise", "corruption"]


def clean(img: np.ndarray):
    return img.copy(), {t: False for t in ISSUE_TYPES}, 0.0


def blur(img: np.ndarray, severity: float | None = None):
    severity = random.uniform(0.25, 1.0) if severity is None else severity
    # Randomly choose between Gaussian blur and motion blur
    if random.random() < 0.5:
        # Gaussian blur (original)
        ksize = int(3 + severity * 18)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        out = cv2.GaussianBlur(img, (ksize, ksize), 0)
    else:
        # Motion blur — directional kernel simulating camera/subject motion
        ksize = int(5 + severity * 25)
        angle = random.uniform(0, 180)
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        # Draw a line through the kernel center at the chosen angle
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        for i in range(ksize):
            offset = i - center
            y = int(round(center + offset * sin_a))
            x = int(round(center + offset * cos_a))
            if 0 <= y < ksize and 0 <= x < ksize:
                kernel[y, x] = 1.0
        kernel /= kernel.sum() + 1e-9
        out = cv2.filter2D(img, -1, kernel)
    labels = {t: False for t in ISSUE_TYPES}
    labels["blur"] = severity > 0.15
    return out, labels, severity


def underexpose(img: np.ndarray, severity: float | None = None):
    severity = random.uniform(0.3, 1.0) if severity is None else severity
    gamma = 1.0 + severity * 3.5
    scale = 1.0 - 0.5 * severity
    out = img.astype(np.float32) * scale
    out = 255.0 * (out / 255.0) ** gamma
    out = np.clip(out, 0, 255).astype(np.uint8)
    labels = {t: False for t in ISSUE_TYPES}
    labels["underexposure"] = severity > 0.15
    return out, labels, severity


def overexpose(img: np.ndarray, severity: float | None = None):
    severity = random.uniform(0.3, 1.0) if severity is None else severity
    gain = 1.0 + severity * 2.2
    offset = severity * 90
    out = img.astype(np.float32) * gain + offset
    out = np.clip(out, 0, 255).astype(np.uint8)
    labels = {t: False for t in ISSUE_TYPES}
    labels["overexposure"] = severity > 0.15
    return out, labels, severity


def add_noise(img: np.ndarray, severity: float | None = None):
    severity = random.uniform(0.2, 1.0) if severity is None else severity
    sigma = severity * 45
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    labels = {t: False for t in ISSUE_TYPES}
    labels["noise"] = severity > 0.12
    return out, labels, severity


def corrupt(img: np.ndarray, severity: float | None = None):
    """Simulate severe degradation / corruption: heavy JPEG requantization,
    block loss, and channel shuffling artifacts."""
    severity = random.uniform(0.4, 1.0) if severity is None else severity
    out = img.copy()

    # heavy jpeg re-encode at very low quality -> blocking + ringing
    quality = int(max(2, 25 - severity * 20))
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # random block dropout ("corrupted" missing data blocks)
    h, w = out.shape[:2]
    n_blocks = int(severity * 10)
    for _ in range(n_blocks):
        bs = random.randint(10, max(11, min(h, w) // 6))
        y = random.randint(0, max(0, h - bs))
        x = random.randint(0, max(0, w - bs))
        out[y:y + bs, x:x + bs] = random.randint(0, 255)

    labels = {t: False for t in ISSUE_TYPES}
    labels["corruption"] = severity > 0.2
    return out, labels, severity


def mild_jpeg(img: np.ndarray, severity: float | None = None):
    """Mild-to-moderate JPEG compression artifacts (quality 30-70).
    Distinct from the extreme corruption path — represents typical
    over-compressed web images."""
    severity = random.uniform(0.2, 0.7) if severity is None else severity
    quality = int(max(20, 70 - severity * 55))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    else:
        out = img.copy()
    labels = {t: False for t in ISSUE_TYPES}
    # Only flag as corruption at higher compression levels
    labels["corruption"] = severity > 0.45
    return out, labels, severity * 0.6  # lower overall severity than full corruption


def compound(img: np.ndarray):
    """Combine two or three mild degradations, since real photos often
    have more than one simultaneous issue (e.g. dark AND noisy)."""
    fns = [blur, underexpose, overexpose, add_noise]
    n_issues = random.choices([2, 3], weights=[0.7, 0.3], k=1)[0]
    selected = random.sample(fns, min(n_issues, len(fns)))
    out = img.copy()
    all_labels = {t: False for t in ISSUE_TYPES}
    max_sev = 0.0
    for fn in selected:
        out, labels, sev = fn(out, severity=random.uniform(0.2, 0.55))
        for t in ISSUE_TYPES:
            all_labels[t] = all_labels[t] or labels[t]
        max_sev = max(max_sev, sev)
    return out, all_labels, max_sev


DEGRADATIONS = [clean, blur, underexpose, overexpose, add_noise, corrupt, mild_jpeg, compound]
DEGRADATION_WEIGHTS = [0.18, 0.15, 0.13, 0.13, 0.13, 0.08, 0.06, 0.14]


def random_degrade(img: np.ndarray):
    fn = random.choices(DEGRADATIONS, weights=DEGRADATION_WEIGHTS, k=1)[0]
    return fn(img)
