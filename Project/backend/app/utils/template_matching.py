from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Union

import cv2
import numpy as np
import torch

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── LightGlue singletons ─────────────────────────────────────────────────────
_extractor = None
_matcher   = None
_device: torch.device | None = None


def _ensure_lightglue_installed() -> None:
    """Install lightglue via pip if it is not already importable."""
    try:
        import lightglue  # noqa: F401
    except ImportError:
        import subprocess
        import sys

        logger.info("lightglue not found — installing from GitHub via pip...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/cvg/LightGlue.git",
            ]
        )
        logger.info("lightglue installed successfully")


def load_lightglue_models() -> None:
    """Ensure lightglue is installed, then load SuperPoint + LightGlue into singletons."""
    global _extractor, _matcher, _device

    _ensure_lightglue_installed()

    from lightglue import LightGlue, SuperPoint  # noqa: PLC0415

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading SuperPoint + LightGlue on %s...", _device)
    _extractor = SuperPoint(max_num_keypoints=2048).eval().to(_device)
    _matcher   = LightGlue(features="superpoint").eval().to(_device)
    logger.info("LightGlue models loaded on %s", _device)


# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TemplateMatchResult:
    template_index: int
    found: bool
    score: float
    scale: float
    bounding_box: tuple[int, int, int, int]  # (startX, startY, endX, endY)
    # Phase 2 fields — defaults so existing callers are unaffected
    p2_n_matches: int       = 0
    p2_n_inliers: int       = 0
    p2_verdict: str         = "NOT_RUN"
    # "TRUE_POSITIVE" | "FALSE_POSITIVE" | "NO_MATCH" | "HOMOGRAPHY_FAILED" | "NOT_RUN"
    p2_homography_vis_jpeg_b64: str | None = None
    p2_refined_jpeg_b64:        str | None = None


@dataclass
class TemplateMatchingOutput:
    results: list[TemplateMatchResult] = field(default_factory=list)

    @property
    def any_found(self) -> bool:
        return any(r.found for r in self.results)

    def get_found(self) -> list[TemplateMatchResult]:
        return [r for r in self.results if r.found]


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR ndarray → grayscale float32 tensor [1, H, W] in [0, 1] on _device."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).to(_device)


def _encode_jpeg_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _run_phase2(
    crop_bgr: np.ndarray,
    template_bgr: np.ndarray,
    p2_threshold: int,
    min_match_count: int,
) -> tuple[str, int, int, str | None, str | None]:
    """
    Run SuperPoint + LightGlue + MAGSAC++ on the Phase-1 crop vs template.

    Returns
    -------
    (verdict, n_matches, n_inliers, homography_vis_b64, refined_b64)
    """
    from lightglue.utils import rbd  # noqa: PLC0415

    t_crop = _to_tensor(crop_bgr)
    t_tmpl = _to_tensor(template_bgr)

    with torch.no_grad():
        feats_crop = _extractor.extract(t_crop)
        feats_tmpl = _extractor.extract(t_tmpl)
        result_lg  = _matcher({"image0": feats_crop, "image1": feats_tmpl})

    feats_crop, feats_tmpl, result_lg = [rbd(x) for x in [feats_crop, feats_tmpl, result_lg]]

    match_idx = result_lg["matches"].cpu().numpy()   # [N, 2]
    n_matches = len(match_idx)

    if n_matches < min_match_count:
        return "NO_MATCH", n_matches, 0, None, None

    kp_crop_all = feats_crop["keypoints"].cpu().numpy()
    kp_tmpl_all = feats_tmpl["keypoints"].cpu().numpy()
    pts_crop = kp_crop_all[match_idx[:, 0]].astype(np.float32)
    pts_tmpl = kp_tmpl_all[match_idx[:, 1]].astype(np.float32)

    H, mask = cv2.findHomography(
        pts_tmpl.reshape(-1, 1, 2),
        pts_crop.reshape(-1, 1, 2),
        cv2.USAC_MAGSAC,
        5.0,
    )
    if H is None:
        return "HOMOGRAPHY_FAILED", n_matches, 0, None, None

    inlier_mask = mask.ravel().astype(bool)
    n_inliers   = int(inlier_mask.sum())
    verdict     = "TRUE_POSITIVE" if n_inliers >= p2_threshold else "FALSE_POSITIVE"

    # ── Build homography visualisation ────────────────────────────────────────
    h_t, w_t       = template_bgr.shape[:2]
    corners_tmpl   = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
    corners_crop   = cv2.perspectiveTransform(corners_tmpl, H).reshape(-1, 2)

    cH_c, cW_c = crop_bgr.shape[:2]
    x0c = max(0,    int(corners_crop[:, 0].min()))
    y0c = max(0,    int(corners_crop[:, 1].min()))
    x1c = min(cW_c, int(corners_crop[:, 0].max()))
    y1c = min(cH_c, int(corners_crop[:, 1].max()))

    ann_crop = crop_bgr.copy()
    cv2.polylines(ann_crop, [corners_crop.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    cv2.rectangle(ann_crop, (x0c, y0c), (x1c, y1c), (0, 165, 255), 2)
    for pt in pts_crop[inlier_mask]:
        cv2.circle(ann_crop, tuple(np.round(pt).astype(int)), 4, (255, 255, 0), -1)

    homography_vis_b64 = _encode_jpeg_b64(ann_crop)

    # ── Refined region ────────────────────────────────────────────────────────
    refined = crop_bgr[y0c:y1c, x0c:x1c]
    if refined.size == 0:
        refined = np.zeros((10, 10, 3), dtype=np.uint8)
    refined_b64 = _encode_jpeg_b64(refined)

    return verdict, n_matches, n_inliers, homography_vis_b64, refined_b64


def match_templates(
    image: np.ndarray,
    templates: Union[np.ndarray, list[np.ndarray]],
    threshold: float = 0.2,
    scales: np.ndarray = None,
    canny_low: int = 30,
    canny_high: int = 100,
    p2_threshold: int = 10,
    min_match_count: int = 4,
) -> TemplateMatchingOutput:
    """
    Match one or more templates against an image using multi-scale Canny edge matching
    (Phase 1) followed by SuperPoint + LightGlue + MAGSAC++ verification (Phase 2).

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale).
    templates : np.ndarray or list of np.ndarray
        A single template or a list of templates (BGR or grayscale).
    threshold : float
        Minimum normalised cross-correlation score for Phase 1 to pass.
    scales : np.ndarray, optional
        Array of scale factors to search over. Defaults to 40 steps from 0.1 to 1.0.
    canny_low : int
        Lower hysteresis threshold for Canny edge detection.
    canny_high : int
        Upper hysteresis threshold for Canny edge detection.
    p2_threshold : int
        Minimum LightGlue + MAGSAC++ inliers required for TRUE_POSITIVE verdict.
    min_match_count : int
        Minimum LightGlue matches needed to attempt homography estimation.

    Returns
    -------
    TemplateMatchingOutput
        Contains a TemplateMatchResult per template with:
          - found        : bool — TRUE only when Phase 2 verdict == "TRUE_POSITIVE"
          - score        : float — Phase 1 normalised cross-correlation score (for sorting)
          - scale        : float — image scale at which Phase 1 best score was achieved
          - bounding_box : (startX, startY, endX, endY) in original image coordinates
          - p2_verdict   : str  — "TRUE_POSITIVE" | "FALSE_POSITIVE" | "NO_MATCH" |
                                  "HOMOGRAPHY_FAILED" | "NOT_RUN"
          - p2_n_matches : int  — total LightGlue matches
          - p2_n_inliers : int  — MAGSAC++ inlier count
          - p2_homography_vis_jpeg_b64 : JPEG base64 of Phase-1 crop with projected
                                         quad (green), bbox (orange), inlier kps (cyan)
          - p2_refined_jpeg_b64        : JPEG base64 of homography-refined region
    """
    if scales is None:
        scales = np.linspace(0.1, 1.0, 40)[::-1]

    if isinstance(templates, np.ndarray) and templates.ndim in (2, 3):
        templates = [templates]

    # image kept in colour for Phase 2 crop extraction
    gray = _to_gray(image)
    output = TemplateMatchingOutput()

    for idx, template in enumerate(templates):
        tmpl_gray  = _to_gray(template)
        tH, tW     = tmpl_gray.shape[:2]
        tmpl_edges = cv2.Canny(cv2.GaussianBlur(tmpl_gray, (5, 5), 0), canny_low, canny_high)

        best: tuple | None = None  # (score, loc, ratio, scale)

        for scale in scales:
            rw      = max(1, int(gray.shape[1] * scale))
            rh      = max(1, int(gray.shape[0] * scale))
            resized = cv2.resize(gray, (rw, rh))
            ratio   = gray.shape[1] / float(resized.shape[1])

            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            img_edges = cv2.Canny(cv2.GaussianBlur(resized, (5, 5), 0), canny_low, canny_high)
            result           = cv2.matchTemplate(img_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

            if best is None or maxVal > best[0]:
                best = (maxVal, maxLoc, ratio, float(scale))

        if best is None:
            output.results.append(
                TemplateMatchResult(
                    template_index=idx,
                    found=False,
                    score=0.0,
                    scale=0.0,
                    bounding_box=(0, 0, 0, 0),
                    p2_verdict="NOT_RUN",
                )
            )
            continue

        best_val, best_loc, ratio, best_scale = best
        startX = int(best_loc[0] * ratio)
        startY = int(best_loc[1] * ratio)
        endX   = int((best_loc[0] + tW) * ratio)
        endY   = int((best_loc[1] + tH) * ratio)

        # ── Phase 2: LightGlue verification ───────────────────────────────────
        if best_val >= threshold:
            # image must be BGR for Phase 2; convert if grayscale was passed
            image_bgr = (
                cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if len(image.shape) == 2
                else image
            )
            template_bgr = (
                cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
                if len(template.shape) == 2
                else template
            )
            crop_bgr = image_bgr[startY:endY, startX:endX]
            if crop_bgr.size == 0:
                crop_bgr = image_bgr  # fallback: use full image if crop is degenerate

            verdict, n_matches, n_inliers, hom_vis_b64, ref_b64 = _run_phase2(
                crop_bgr, template_bgr, p2_threshold, min_match_count
            )
        else:
            verdict, n_matches, n_inliers = "NOT_RUN", 0, 0
            hom_vis_b64, ref_b64 = None, None

        output.results.append(
            TemplateMatchResult(
                template_index=idx,
                found=(verdict == "TRUE_POSITIVE"),
                score=round(float(best_val), 6),
                scale=best_scale,
                bounding_box=(startX, startY, endX, endY),
                p2_n_matches=n_matches,
                p2_n_inliers=n_inliers,
                p2_verdict=verdict,
                p2_homography_vis_jpeg_b64=hom_vis_b64,
                p2_refined_jpeg_b64=ref_b64,
            )
        )

    return output
