"""Validation and safe decoding of uploaded image files."""
from __future__ import annotations

import io
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}  # E9
MIN_DIMENSION = 16
MAX_DIMENSION = 8000


class ImageValidationError(ValueError):
    """Raised for any invalid/unreadable/unsupported upload."""


def validate_and_decode(file_bytes: bytes, content_type: str | None, filename: str | None) -> np.ndarray:
    """Validates an uploaded file and returns a decoded BGR uint8 numpy array.

    Raises ImageValidationError with a clear, user-facing message on any
    problem (empty file, too large, wrong type, unreadable/corrupt data,
    degenerate dimensions). The caller maps this to HTTP 400.
    """
    if not file_bytes:
        raise ImageValidationError("Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError(
            f"File too large ({len(file_bytes) / 1e6:.1f} MB). Max allowed is "
            f"{MAX_FILE_SIZE_BYTES / 1e6:.0f} MB."
        )

    # Try Pillow first: gives a much clearer error for genuinely corrupt /
    # non-image files, and verifies structural integrity.
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()  # raises if structurally corrupt
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ImageValidationError(f"File is not a valid or is a corrupted image: {e}")

    # Re-open after verify() (which leaves the file unusable for further ops)
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise ImageValidationError(f"Could not decode image data: {e}")

    # E9: Strict format validation based on actual detected format
    detected_format = pil_img.format
    if detected_format not in ALLOWED_PIL_FORMATS:
        raise ImageValidationError(
            f"Unsupported image format: {detected_format}. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_PIL_FORMATS))}."
        )

    pil_img = pil_img.convert("RGB")

    w, h = pil_img.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions too small ({w}x{h}). Minimum is "
            f"{MIN_DIMENSION}x{MIN_DIMENSION}px."
        )
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions too large ({w}x{h}). Maximum is "
            f"{MAX_DIMENSION}x{MAX_DIMENSION}px."
        )

    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if bgr is None or bgr.size == 0:
        raise ImageValidationError("Image decoded to empty data.")

    return bgr
