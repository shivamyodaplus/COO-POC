from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def load_model() -> None:
    global _model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen3-VL-Embedding-2B on {device}...")
    _model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device=device)
    print("Qwen3-VL-Embedding-2B loaded — dim=2048")


def embed_image(image: Image.Image) -> np.ndarray:
    if _model is None:
        raise RuntimeError("Embedding model not loaded. Call load_model() first.")
    return np.asarray(_model.encode([image])[0])
