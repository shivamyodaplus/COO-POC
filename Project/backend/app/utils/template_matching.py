import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Union


@dataclass
class TemplateMatchResult:
    template_index: int
    found: bool
    score: float
    scale: float
    bounding_box: tuple[int, int, int, int]  # (startX, startY, endX, endY)


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


def match_templates(
    image: np.ndarray,
    templates: Union[np.ndarray, list[np.ndarray]],
    threshold: float = 0.2,
    scales: np.ndarray = None,
    canny_low: int = 30,
    canny_high: int = 100,
) -> TemplateMatchingOutput:
    """
    Match one or more templates against an image using multi-scale Canny edge matching.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or grayscale).
    templates : np.ndarray or list of np.ndarray
        A single template or a list of templates (BGR or grayscale).
    threshold : float
        Minimum normalised cross-correlation score to consider a match found.
    scales : np.ndarray, optional
        Array of scale factors to search over. Defaults to 40 steps from 0.1 to 1.0.
    canny_low : int
        Lower hysteresis threshold for Canny edge detection.
    canny_high : int
        Upper hysteresis threshold for Canny edge detection.

    Returns
    -------
    TemplateMatchingOutput
        Contains a TemplateMatchResult per template with:
          - found        : bool — whether score >= threshold
          - score        : float — best normalised cross-correlation score
          - scale        : float — image scale at which best score was achieved
          - bounding_box : (startX, startY, endX, endY) in original image coordinates
    """
    if scales is None:
        scales = np.linspace(0.1, 1.0, 40)[::-1]

    if isinstance(templates, np.ndarray) and templates.ndim in (2, 3):
        templates = [templates]

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
                )
            )
            continue

        best_val, best_loc, ratio, best_scale = best
        startX = int(best_loc[0] * ratio)
        startY = int(best_loc[1] * ratio)
        endX   = int((best_loc[0] + tW) * ratio)
        endY   = int((best_loc[1] + tH) * ratio)

        output.results.append(
            TemplateMatchResult(
                template_index=idx,
                found=best_val >= threshold,
                score=round(float(best_val), 6),
                scale=best_scale,
                bounding_box=(startX, startY, endX, endY),
            )
        )

    return output
