from __future__ import annotations

from pathlib import Path
from typing import list

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}


def render_pdf_pages(pdf_path: Path, scale: float = 2.0) -> list[Image.Image]:
    document = pdfium.PdfDocument(str(pdf_path))
    return [page.render(scale=scale).to_pil().convert("RGB") for page in document]


def load_document_pages(file_path: Path) -> list[Image.Image]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return render_pdf_pages(file_path)
    if suffix in SUPPORTED_EXTENSIONS:
        return [Image.open(file_path).convert("RGB")]
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def preprocess_image(image: Image.Image, target_size: tuple = (896, 896)) -> Image.Image:
    # 1. Grayscale for CLAHE
    img_np = np.array(image.convert("L"))

    # 2. CLAHE — adaptive contrast (same params as notebook)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_np = clahe.apply(img_np)

    # 3. Back to RGB for VLM
    enhanced_img = Image.fromarray(enhanced_np).convert("RGB")

    # 4. Resize preserving aspect ratio
    enhanced_img.thumbnail(target_size, Image.Resampling.LANCZOS)

    # 5. White-pad to exact target_size and center
    new_img = Image.new("RGB", target_size, (255, 255, 255))
    x_off = (target_size[0] - enhanced_img.size[0]) // 2
    y_off = (target_size[1] - enhanced_img.size[1]) // 2
    new_img.paste(enhanced_img, (x_off, y_off))
    return new_img
