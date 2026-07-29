from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.core.logger import get_logger

_model: SentenceTransformer | None = None
logger = get_logger(__name__)


def load_model() -> None:
    global _model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Qwen3-VL-Embedding-2B on %s...", device)
    _model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device=device)
    logger.info("Qwen3-VL-Embedding-2B loaded — dim=2048")


def embed_image(image: Image.Image) -> np.ndarray:
    if _model is None:
        raise RuntimeError("Embedding model not loaded. Call load_model() first.")
    return np.asarray(_model.encode([image])[0])
