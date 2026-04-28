from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from art_recognition.preprocessing import preprocess_gallery_image


@dataclass
class CropResult:
    image_bgr: np.ndarray
    method: str
    confidence: float
    polygon: np.ndarray | None = None


def remove_border(image_bgr: np.ndarray, ratio: float = 0.05) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    y_margin = int(height * ratio)
    x_margin = int(width * ratio)
    if y_margin * 2 >= height or x_margin * 2 >= width:
        return image_bgr
    return image_bgr[y_margin : height - y_margin, x_margin : width - x_margin]


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        raise ValueError("at least four points are required")

    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def perspective_correct(image_bgr: np.ndarray, polygon: np.ndarray | None) -> np.ndarray:
    if polygon is None or len(polygon) < 4:
        return image_bgr

    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(contour.astype(np.float32))
    corners = cv2.boxPoints(rect)
    ordered = order_points(corners)

    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    width = max(int(top_width), int(bottom_width), 16)
    height = max(int(left_height), int(right_height), 16)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image_bgr, matrix, (width, height))


class PaintingCropper:
    def __init__(
        self,
        yolo_model_path: str | Path | None = "data/models/painting_yolo_seg.pt",
        border_ratio: float = 0.05,
    ) -> None:
        self.yolo_model_path = Path(yolo_model_path) if yolo_model_path else None
        self.border_ratio = border_ratio
        self._model = None

    def _load_yolo(self):
        if self._model is not None:
            return self._model
        if self.yolo_model_path is None or not self.yolo_model_path.exists():
            return None
        try:
            from ultralytics import YOLO
        except ImportError:
            return None
        self._model = YOLO(str(self.yolo_model_path))
        return self._model

    def _crop_with_yolo(self, image_bgr: np.ndarray) -> CropResult | None:
        model = self._load_yolo()
        if model is None:
            return None

        results = model.predict(image_bgr, verbose=False)
        if not results:
            return None
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        confidences = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(boxes))
        best_index = int(np.argmax(confidences))
        confidence = float(confidences[best_index])

        polygon = None
        masks = getattr(result, "masks", None)
        if masks is not None and masks.xy:
            polygon = np.asarray(masks.xy[best_index], dtype=np.float32)

        if polygon is not None and len(polygon) >= 4:
            crop = perspective_correct(image_bgr, polygon)
        else:
            xyxy = boxes.xyxy[best_index].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            crop = image_bgr[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]

        if crop.size == 0:
            return None
        return CropResult(remove_border(crop, self.border_ratio), "yolo_seg", confidence, polygon)

    def _crop_with_legacy_contours(self, image_bgr: np.ndarray) -> CropResult:
        result = preprocess_gallery_image(image_bgr)
        candidates = result.get("candidates") or []
        if not candidates:
            return CropResult(remove_border(image_bgr, self.border_ratio), "full_image_border_removed", 0.0)

        best = max(candidates, key=lambda candidate: candidate.crop.shape[0] * candidate.crop.shape[1])
        polygon = getattr(best, "painting_corners", None)
        crop = perspective_correct(best.crop, polygon) if polygon is not None and len(polygon) >= 4 else best.crop
        return CropResult(remove_border(crop, self.border_ratio), "legacy_contour", 0.5, polygon)

    def crop(self, image_bgr: np.ndarray) -> CropResult:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("image_bgr must be a non-empty image")

        yolo_result = self._crop_with_yolo(image_bgr)
        if yolo_result is not None:
            return yolo_result
        return self._crop_with_legacy_contours(image_bgr)
