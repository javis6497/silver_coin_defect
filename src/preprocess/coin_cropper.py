from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CropResult:
    crop_xyxy: Tuple[int, int, int, int]
    circle: Tuple[int, int, int]


class CoinAutoCropper:
    def __init__(self, pad_ratio: float = 0.08):
        self.pad_ratio = pad_ratio

    def detect_circle(self, image: np.ndarray) -> Tuple[int, int, int]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = gray.shape[:2]
        min_r = int(min(h, w) * 0.25)
        max_r = int(min(h, w) * 0.52)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(h, w) // 2,
            param1=120,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None and len(circles[0]) > 0:
            x, y, r = circles[0][0]
            return int(x), int(y), int(r)
        return self._fallback_contour(gray)

    def _fallback_contour(self, gray: np.ndarray) -> Tuple[int, int, int]:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        if not contours:
            r = int(min(h, w) * 0.45)
            return w // 2, h // 2, r
        cnt = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < min(h, w) * 0.1:
            r = int(min(h, w) * 0.45)
            x, y = w // 2, h // 2
        return int(x), int(y), int(r)

    def build_crop(self, image: np.ndarray) -> CropResult:
        h, w = image.shape[:2]
        x, y, r = self.detect_circle(image)
        half = int(r * (1.0 + self.pad_ratio))
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half)
        y2 = min(h, y + half)
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        return CropResult(crop_xyxy=(x1, y1, x2, y2), circle=(x, y, r))

    def crop_image(self, image: np.ndarray) -> tuple[np.ndarray, CropResult]:
        result = self.build_crop(image)
        x1, y1, x2, y2 = result.crop_xyxy
        return image[y1:y2, x1:x2].copy(), result


def enhance_image(image: np.ndarray, clahe: bool = True, sharpen: bool = True) -> np.ndarray:
    out = image.copy()
    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)
    return out


def remap_bbox_to_crop(bbox_xywh, crop_xyxy, src_shape, dst_shape=None):
    x, y, w, h = bbox_xywh
    x1c, y1c, x2c, y2c = crop_xyxy
    bx1 = max(x, x1c)
    by1 = max(y, y1c)
    bx2 = min(x + w, x2c)
    by2 = min(y + h, y2c)
    if bx2 <= bx1 or by2 <= by1:
        return None
    nx = bx1 - x1c
    ny = by1 - y1c
    nw = bx2 - bx1
    nh = by2 - by1
    if dst_shape is not None:
        src_h = max(1, y2c - y1c)
        src_w = max(1, x2c - x1c)
        dst_h, dst_w = dst_shape[:2]
        sx = dst_w / src_w
        sy = dst_h / src_h
        nx, ny, nw, nh = nx * sx, ny * sy, nw * sx, nh * sy
    return [float(nx), float(ny), float(nw), float(nh)]
