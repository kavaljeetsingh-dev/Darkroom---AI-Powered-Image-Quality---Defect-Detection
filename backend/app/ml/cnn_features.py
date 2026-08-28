"""
CNN-based feature extraction using MobileNetV2 (transfer learning).

Uses a pretrained MobileNetV2 (ImageNet weights) as a frozen feature extractor.
The penultimate layer produces a 1280-dimensional embedding which is then
reduced to a compact representation via PCA.  These learned features capture
high-level texture/structure patterns that complement the hand-crafted IQA
features in features.py.

This module is OPTIONAL — if PyTorch is not available, all functions
gracefully return None/empty, and the training pipeline falls back to
engineered-features-only mode.
"""
from __future__ import annotations

import numpy as np
import cv2
import joblib
from pathlib import Path

_CNN_DIM = 32  # PCA output dimensions
ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# Check if PyTorch is available
_TORCH_AVAILABLE = False
try:
    import torch
    from torchvision import models, transforms
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Return True if CNN features can be extracted (PyTorch installed)."""
    return _TORCH_AVAILABLE


def _get_model():
    """Lazy-load the MobileNetV2 model (CPU, eval mode, frozen)."""
    if not _TORCH_AVAILABLE:
        return None, None

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()  # strip final FC → 1280-dim output
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


# Singleton to avoid reloading the model on every call
_model_cache = None


def _ensure_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = _get_model()
    return _model_cache


def extract_cnn_embedding(img_bgr: np.ndarray) -> np.ndarray | None:
    """Extract a raw 1280-dim MobileNetV2 embedding from a BGR image.

    Returns a 1-D float64 array of shape (1280,), or None if PyTorch unavailable.
    """
    if not _TORCH_AVAILABLE:
        return None

    model, preprocess = _ensure_model()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(img_rgb).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        embedding = model(tensor).squeeze(0).numpy()
    return embedding.astype(np.float64)


def extract_cnn_features(img_bgr: np.ndarray, pca=None) -> np.ndarray | None:
    """Extract PCA-reduced CNN features from a BGR image.

    Returns np.ndarray of shape (_CNN_DIM,) if pca is provided and PyTorch
    is available, else None.
    """
    if not _TORCH_AVAILABLE:
        return None

    emb = extract_cnn_embedding(img_bgr)
    if emb is None:
        return None
    if pca is not None:
        return pca.transform(emb.reshape(1, -1)).flatten()
    return emb


def batch_extract_embeddings(images_bgr: list[np.ndarray]) -> np.ndarray | None:
    """Extract raw embeddings for a batch of images.

    Returns array of shape (N, 1280), or None if PyTorch unavailable.
    """
    if not _TORCH_AVAILABLE:
        return None

    model, preprocess = _ensure_model()
    tensors = []
    for img in images_bgr:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors.append(preprocess(img_rgb))
    batch = torch.stack(tensors)

    # Process in mini-batches to limit memory
    embeddings = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(batch), bs):
            out = model(batch[i:i + bs])
            embeddings.append(out.numpy())
    return np.vstack(embeddings).astype(np.float64)


CNN_FEATURE_DIM = _CNN_DIM
