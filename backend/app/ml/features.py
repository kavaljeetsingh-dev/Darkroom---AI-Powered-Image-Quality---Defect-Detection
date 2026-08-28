"""
Engineered image-quality feature extraction.

All features are computed from a decoded image (BGR, uint8, as returned by
cv2.imread / cv2.imdecode) and are the inputs to the downstream learned
classifiers in model.py. Every feature is individually interpretable, which
is what lets us build human-readable explanations for each prediction later
(see infer.py::explain).
"""
from __future__ import annotations

import numpy as np
import cv2


FEATURE_NAMES = [
    "sharpness_lap_var",      # focus / blur measure (Laplacian variance)
    "sharpness_norm",         # laplacian variance normalized by image energy
    "sharpness_tenengrad",    # Sobel gradient magnitude — noise-resilient sharpness
    "brightness_mean",        # mean luminance (0-255)
    "brightness_std",         # contrast proxy
    "contrast_rms",           # RMS contrast — better exposure separation
    "dark_pixel_ratio",       # fraction of near-black pixels
    "bright_pixel_ratio",     # fraction of near-white (clipped) pixels
    "hist_low_mass",          # fraction of luminance histogram mass in [0,32)
    "hist_high_mass",         # fraction of luminance histogram mass in [223,256)
    "noise_estimate",         # Immerkaer fast noise-sigma estimator
    "noise_local_var",        # mean local variance in flat (low-gradient) patches
    "noise_high_freq_energy", # FFT high-frequency energy ratio
    "edge_density",           # fraction of pixels that are strong edges (Canny)
    "gradient_coherence",     # directional gradient consistency (blur vs texture)
    "colorfulness",           # Hasler-Susstrunk colorfulness metric
    "saturation_mean",        # mean HSV saturation
    "entropy",                # Shannon entropy of luminance histogram
    "blockiness",             # JPEG-style 8x8 block boundary artifact score
    "channel_mean_asymmetry", # abs diff between max/min channel means (color cast / corruption)
]


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _tenengrad(gray: np.ndarray) -> float:
    """Sobel-based gradient magnitude sharpness measure (Tenengrad).

    Less sensitive to noise than Laplacian variance because Sobel smooths
    along the perpendicular axis.  Good at distinguishing genuinely sharp
    images from noisy ones that happen to have high Laplacian variance.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def _contrast_rms(gray: np.ndarray) -> float:
    """RMS contrast — standard deviation of luminance normalized to [0,1]
    range.  Better at separating under/over-exposure from normal images
    than raw brightness_std alone."""
    return float(gray.astype(np.float64).std() / 255.0)


def _noise_immerkaer(gray: np.ndarray) -> float:
    """Fast noise-sigma estimator (Immerkaer, 1996).

    Convolves with a Laplacian-like kernel designed to cancel out edges/flat
    regions and isolate additive noise, then rescales to a sigma estimate.
    """
    h, w = gray.shape
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2) + 1e-9)
    return float(sigma)


def _noise_high_freq_energy(gray: np.ndarray) -> float:
    """Ratio of high-frequency energy in the DFT spectrum.

    Noise adds energy uniformly across all frequencies, so a high ratio
    of high-freq to total energy is a strong noise indicator — and this
    separates noise from legitimately sharp/detailed images whose energy
    concentrates in mid-frequency edges.
    """
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    # Define "high frequency" as the outer 25% of the spectrum radius
    radius = min(cy, cx)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    high_freq_mask = dist > (radius * 0.75)
    total_energy = np.sum(magnitude ** 2) + 1e-9
    high_freq_energy = np.sum(magnitude[high_freq_mask] ** 2)
    return float(high_freq_energy / total_energy)


def _local_flat_variance(gray: np.ndarray, patch: int = 16) -> float:
    """Mean variance inside low-gradient ('flat') patches only.

    Flat regions should be near-constant in a clean image, so any residual
    variance there is a strong, gradient-independent noise signal.
    """
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    h, w = gray.shape
    variances = []
    for y in range(0, h - patch, patch):
        for x in range(0, w - patch, patch):
            g_patch = grad_mag[y:y + patch, x:x + patch]
            if g_patch.mean() < 8.0:  # "flat" threshold
                variances.append(gray[y:y + patch, x:x + patch].var())
    if not variances:
        return 0.0
    return float(np.mean(variances))


def _gradient_coherence(gray: np.ndarray) -> float:
    """Directional consistency of image gradients.

    In a motion-blurred image, gradients are strongly aligned along one
    direction.  In a naturally textured image (e.g. gravel, grass), gradients
    point in many directions.  This feature helps the blur classifier avoid
    false-positiving on textures.

    Returns a value in [0, 1]: 1 = all gradients point the same way (blur),
    0 = uniformly random directions (clean texture).
    """
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # Only consider pixels with non-trivial gradient (suppress noise)
    mask = mag > np.percentile(mag, 50)
    if mask.sum() < 10:
        return 0.0
    # Normalize gradient directions and compute mean direction vector
    gx_n = gx[mask] / (mag[mask] + 1e-9)
    gy_n = gy[mask] / (mag[mask] + 1e-9)
    # Coherence = magnitude of the mean direction vector
    mean_gx = np.mean(gx_n)
    mean_gy = np.mean(gy_n)
    coherence = np.sqrt(mean_gx ** 2 + mean_gy ** 2)
    return float(coherence)


def _colorfulness(img_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, mean_rg = rg.std(), rg.mean()
    std_yb, mean_yb = yb.std(), yb.mean()
    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    return float(std_root + 0.3 * mean_root)


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _blockiness(gray: np.ndarray) -> float:
    """Approximate JPEG blocking-artifact score.

    Measures how much stronger the intensity discontinuity is exactly at
    8-pixel block boundaries vs. one pixel away from them. Heavy compression
    / corruption raises this score.
    """
    gray = gray.astype(np.float32)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    diffs_at_boundary = []
    diffs_off_boundary = []
    for x in range(8, w - 1, 8):
        diffs_at_boundary.append(np.mean(np.abs(gray[:, x] - gray[:, x - 1])))
        diffs_off_boundary.append(np.mean(np.abs(gray[:, x - 4] - gray[:, x - 5])))
    for y in range(8, h - 1, 8):
        diffs_at_boundary.append(np.mean(np.abs(gray[y, :] - gray[y - 1, :])))
        diffs_off_boundary.append(np.mean(np.abs(gray[y - 4, :] - gray[y - 5, :])))
    if not diffs_at_boundary:
        return 0.0
    boundary = np.mean(diffs_at_boundary)
    off = np.mean(diffs_off_boundary) + 1e-6
    return float(max(0.0, (boundary - off) / off))


def extract_features(img_bgr: np.ndarray) -> dict:
    """Compute the full engineered feature vector for one image.

    Parameters
    ----------
    img_bgr: np.ndarray, shape (H, W, 3), dtype uint8, BGR order (OpenCV default)

    Returns
    -------
    dict mapping feature name -> float value, keys ordered per FEATURE_NAMES.
    """
    gray = _to_gray(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lap_var = _laplacian_variance(gray)
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())

    dark_ratio = float(np.mean(gray < 16))
    bright_ratio = float(np.mean(gray > 240))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-9)
    hist_low_mass = float(hist_norm[:32].sum())
    hist_high_mass = float(hist_norm[223:].sum())

    noise_est = _noise_immerkaer(gray)
    noise_local_var = _local_flat_variance(gray)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    colorfulness = _colorfulness(img_bgr)
    saturation_mean = float(hsv[:, :, 1].mean())

    entropy = _entropy(gray)
    blockiness = _blockiness(gray)

    b_mean, g_mean, r_mean = [float(img_bgr[:, :, i].mean()) for i in range(3)]
    channel_means = [b_mean, g_mean, r_mean]
    channel_mean_asymmetry = float(max(channel_means) - min(channel_means))

    # normalize sharpness by overall image variance so a low-detail (e.g. flat
    # sky) sharp image and a high-detail sharp image are comparable
    sharpness_norm = float(lap_var / (brightness_std ** 2 + 1e-6))

    # New features
    tenengrad = _tenengrad(gray)
    contrast_rms = _contrast_rms(gray)
    gradient_coherence = _gradient_coherence(gray)
    noise_hf = _noise_high_freq_energy(gray)

    values = {
        "sharpness_lap_var": lap_var,
        "sharpness_norm": sharpness_norm,
        "sharpness_tenengrad": tenengrad,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "contrast_rms": contrast_rms,
        "dark_pixel_ratio": dark_ratio,
        "bright_pixel_ratio": bright_ratio,
        "hist_low_mass": hist_low_mass,
        "hist_high_mass": hist_high_mass,
        "noise_estimate": noise_est,
        "noise_local_var": noise_local_var,
        "noise_high_freq_energy": noise_hf,
        "edge_density": edge_density,
        "gradient_coherence": gradient_coherence,
        "colorfulness": colorfulness,
        "saturation_mean": saturation_mean,
        "entropy": entropy,
        "blockiness": blockiness,
        "channel_mean_asymmetry": channel_mean_asymmetry,
    }
    return {k: values[k] for k in FEATURE_NAMES}


def feature_vector(img_bgr: np.ndarray) -> np.ndarray:
    feats = extract_features(img_bgr)
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64)


def sharpness_heatmap(img_bgr: np.ndarray, block: int = 24) -> np.ndarray:
    """Localize blur by tiling the image and computing per-tile Laplacian
    variance. Returns a small (h//block, w//block) map, low values = blurrier
    regions. Used to render a quality heatmap overlay (see infer.py).
    """
    gray = _to_gray(img_bgr)
    h, w = gray.shape
    rows = max(1, h // block)
    cols = max(1, w // block)
    heat = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            y0, y1 = i * block, min(h, (i + 1) * block)
            x0, x1 = j * block, min(w, (j + 1) * block)
            tile = gray[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            heat[i, j] = cv2.Laplacian(tile, cv2.CV_64F).var()
    return heat
