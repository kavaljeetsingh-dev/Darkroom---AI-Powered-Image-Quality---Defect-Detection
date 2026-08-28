"""Tests for image validation logic."""
import pytest
from app.utils.image_validation import validate_and_decode, ImageValidationError


def test_empty_file_raises():
    with pytest.raises(ImageValidationError, match="empty"):
        validate_and_decode(b"", "image/jpeg", "test.jpg")


def test_too_large_file_raises():
    # 16 MB of zeros — exceeds 15 MB limit
    big = b"\x00" * (16 * 1024 * 1024)
    with pytest.raises(ImageValidationError, match="too large"):
        validate_and_decode(big, "image/jpeg", "big.jpg")


def test_invalid_non_image_raises():
    with pytest.raises(ImageValidationError, match="not a valid"):
        validate_and_decode(b"this is plain text", "text/plain", "test.txt")


def test_valid_jpeg_decodes():
    """Create a tiny valid JPEG in memory and verify it decodes."""
    from PIL import Image
    import io

    img = Image.new("RGB", (32, 32), color=(128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw = buf.getvalue()

    bgr = validate_and_decode(raw, "image/jpeg", "test.jpg")
    assert bgr is not None
    assert bgr.shape[0] == 32
    assert bgr.shape[1] == 32
    assert bgr.shape[2] == 3


def test_valid_png_decodes():
    from PIL import Image
    import io

    img = Image.new("RGB", (64, 48), color=(10, 200, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()

    bgr = validate_and_decode(raw, "image/png", "test.png")
    assert bgr.shape == (48, 64, 3)


def test_too_small_image_raises():
    """An 8x8 image should be rejected (minimum is 16x16)."""
    from PIL import Image
    import io

    img = Image.new("RGB", (8, 8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw = buf.getvalue()

    with pytest.raises(ImageValidationError, match="too small"):
        validate_and_decode(raw, "image/jpeg", "tiny.jpg")
