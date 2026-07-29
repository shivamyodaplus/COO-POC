from __future__ import annotations

from typing import list

import numpy as np
import torch
from PIL import Image

from app.core.logger import get_logger

try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False

_ocr = None
logger = get_logger(__name__)


def load_ocr() -> None:
    global _ocr
    if not _PADDLE_AVAILABLE:
        logger.warning("PaddleOCR not installed — OCR disabled. Text component of hybrid search will be skipped.")
        return
    device = "gpu:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading PaddleOCR on %s...", device)
    _ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        engine="transformers",
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    logger.info("PaddleOCR loaded")


def extract_text(image: Image.Image) -> str:
    if _ocr is None:
        return ""
    try:
        result = _ocr.predict(np.array(image))
        chunks: list[str] = []
        for res in result:
            rec_texts = getattr(res, "rec_texts", None)
            if rec_texts is None and isinstance(res, dict):
                rec_texts = res.get("rec_texts", [])
            if rec_texts:
                chunks.extend([t for t in rec_texts if isinstance(t, str)])
        return " ".join(" ".join(chunks).split())
    except Exception:
        return ""
