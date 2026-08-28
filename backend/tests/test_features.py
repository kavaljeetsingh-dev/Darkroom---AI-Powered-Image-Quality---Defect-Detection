"""Tests for feature extraction."""
import numpy as np
import pytest
from app.ml.features import extract_features, FEATURE_NAMES


@pytest.fixture
def random_image():
    """A random 128x128 BGR image."""
    return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def black_image():
    """A pure-black 128x128 BGR image (underexposed)."""
    return np.zeros((128, 128, 3), dtype=np.uint8)


@pytest.fixture
def white_image():
    """A pure-white 128x128 BGR image (overexposed)."""
    return np.full((128, 128, 3), 255, dtype=np.uint8)


def test_extract_features_returns_all_names(random_image):
    feats = extract_features(random_image)
    for name in FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"


def test_feature_count_is_20(random_image):
    feats = extract_features(random_image)
    assert len(feats) == 20


def test_features_are_numeric(random_image):
    feats = extract_features(random_image)
    for name, val in feats.items():
        assert isinstance(val, (int, float, np.floating)), f"{name} is not numeric: {type(val)}"
        assert np.isfinite(val), f"{name} is not finite: {val}"


def test_black_image_has_low_brightness(black_image):
    feats = extract_features(black_image)
    assert feats["brightness_mean"] < 5.0


def test_white_image_has_high_brightness(white_image):
    feats = extract_features(white_image)
    assert feats["brightness_mean"] > 250.0


def test_black_image_high_dark_pixel_ratio(black_image):
    feats = extract_features(black_image)
    assert feats["dark_pixel_ratio"] > 0.9


def test_white_image_high_bright_pixel_ratio(white_image):
    feats = extract_features(white_image)
    assert feats["bright_pixel_ratio"] > 0.9
