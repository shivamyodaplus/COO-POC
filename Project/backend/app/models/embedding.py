from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logger import get_logger

# ---------------------------------------------------------------------------
# Image embedding model (Qwen3-VL — 2048-dim, used for template matching)
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None

# ---------------------------------------------------------------------------
# Text embedding model (BGE-M3 — 1024-dim, used for PACD/COO cross-reference)
# ---------------------------------------------------------------------------
_bge_model = None  # BGEM3EmbeddingFunction — typed as Any to avoid import-time loading
BGE_DIM = 1024

logger = get_logger(__name__)


def load_model() -> None:
    """Load Qwen3-VL-Embedding-2B image embedding model."""
    global _model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Qwen3-VL-Embedding-2B on %s...", device)
    _model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B", device=device)
    logger.info("Qwen3-VL-Embedding-2B loaded — dim=2048")


def load_bge_model() -> None:
    """Load BAAI/bge-m3 text embedding model for PACD/COO cross-reference."""
    global _bge_model
    try:
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "pymilvus[model] is required for BGE-M3 embeddings. "
            "Install with: pip install 'pymilvus[model]'"
        ) from exc

    # Honour explicit device override first — useful on ARM64/Jetson where
    # torch.cuda.is_available() returns False even though the GPU is exposed
    # by the NVIDIA Container Toolkit.
    if settings.BGE_M3_DEVICE:
        device = settings.BGE_M3_DEVICE
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_fp16 = settings.BGE_M3_USE_FP16 and device == "cuda"
    logger.info("Loading BGE-M3 (%s) on %s fp16=%s...", settings.BGE_M3_MODEL, device, use_fp16)
    _bge_model = BGEM3EmbeddingFunction(
        model_name=settings.BGE_M3_MODEL,
        device=device,
        use_fp16=use_fp16,
    )
    logger.info("BGE-M3 loaded — dim=%d", BGE_DIM)


def embed_image(image: Image.Image) -> np.ndarray:
    """Return a 2048-dim embedding for a PIL image (template matching)."""
    if _model is None:
        raise RuntimeError("Embedding model not loaded. Call load_model() first.")
    return np.asarray(_model.encode([image])[0])


def _to_sparse_dict(sv: object) -> dict[int, float]:
    """Normalise a sparse vector to {int: float} dict for Milvus insert."""
    if isinstance(sv, dict):
        return {int(k): float(v) for k, v in sv.items()}
    # scipy sparse matrix (csr / coo)
    try:
        cx = sv.tocoo()  # type: ignore[union-attr]
        return {int(i): float(d) for i, d in zip(cx.col, cx.data)}
    except AttributeError:
        return {}


def embed_text_chunks(
    texts: list[str],
) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Embed a list of text chunks with BGE-M3.

    Returns
    -------
    dense_vecs : list of 1024-dim float lists (one per text)
    sparse_vecs : list of {token_id: weight} dicts for SPARSE_FLOAT_VECTOR
    """
    if _bge_model is None:
        raise RuntimeError("BGE-M3 model not loaded. Call load_bge_model() first.")
    embeddings = _bge_model(texts)
    dense_vecs = [v.tolist() for v in embeddings["dense"]]
    sparse_vecs = [_to_sparse_dict(sv) for sv in embeddings["sparse"]]
    return dense_vecs, sparse_vecs


def embed_text_queries(
    texts: list[str],
) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Encode query texts with BGE-M3 (query-optimised encoding)."""
    if _bge_model is None:
        raise RuntimeError("BGE-M3 model not loaded. Call load_bge_model() first.")
    embeddings = _bge_model.encode_queries(texts)
    dense_vecs = [v.tolist() for v in embeddings["dense"]]
    sparse_vecs = [_to_sparse_dict(sv) for sv in embeddings["sparse"]]
    return dense_vecs, sparse_vecs
