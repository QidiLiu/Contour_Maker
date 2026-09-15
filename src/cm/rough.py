"""粗糙轮廓提取: 由目标框 ROI 内 Otsu 自适应阈值分割 -> 最大轮廓。

这是方案 A 的第一步 (与 YOLO26n 检测框配合), 也是 contour-refiner 训练数据对的
"粗糙端" 生成器。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RoughResult:
    mask: np.ndarray          # 框内分割得到的二值 mask (全图尺寸, uint8 0/1)
    contour: np.ndarray       # 最大轮廓点集 (N,2) float32, 全图坐标
    bbox: tuple[int, int, int, int]   # 实际使用的 ROI 框 x1,y1,x2,y2
    threshold: float          # 使用的 Otsu 阈值
    ok: bool                  # 是否成功提取到轮廓


def otsu_threshold(roi: np.ndarray) -> float:
    """在 ROI 上计算 Otsu 阈值 (ROI 为 uint8 灰度)。"""
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256]).ravel()
    total = roi.size
    if total == 0:
        return 128.0
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 1e-12] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.argmax(sigma_b))
    return float(k)


def otsu_adaptive_rough_contour(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    polarity: str = "auto",
    pad_ratio: float = 0.0,
    min_area: int = 30,
    morph_kernel: int = 5,
    blur_ksize: int = 5,
) -> RoughResult:
    """由检测框得到 Otsu 自适应分割的粗糙轮廓。

    Args:
        image: 全图灰度 uint8。
        bbox: (x1, y1, x2, y2) 浮点像素坐标, 目标框。
        polarity: 'dark' 目标低回声(暗), 'bright' 目标高回声, 'auto' 依 Otsu 后前景占比自动判定。
        pad_ratio: 在框外扩展的比例。**默认 0**: 实测外扩会把低回声病灶与周围暗背景
            连成一片, 使"最大暗连通域"退化为背景 (见 EXPERIMENT.md §7), 因此默认不外扩。
        min_area: 最小连通域面积。
        morph_kernel: 形态学开闭运算核尺寸。
        blur_ksize: 高斯模糊核尺寸 (去斑点噪声)。
    """
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = bw * pad_ratio, bh * pad_ratio
    xi1 = int(np.clip(np.floor(x1 - px), 0, W - 1))
    yi1 = int(np.clip(np.floor(y1 - py), 0, H - 1))
    xi2 = int(np.clip(np.ceil(x2 + px), 0, W))
    yi2 = int(np.clip(np.ceil(y2 + py), 0, H))
    if xi2 - xi1 < 4 or yi2 - yi1 < 4:
        return RoughResult(np.zeros((H, W), np.uint8), np.zeros((0, 2), np.float32),
                           (xi1, yi1, xi2, yi2), 128.0, False)

    roi = image[yi1:yi2, xi1:xi2]
    roi_blur = cv2.GaussianBlur(roi, (blur_ksize | 1, blur_ksize | 1), 0)
    thr = otsu_threshold(roi_blur)

    if polarity == "auto":
        # Otsu 后较亮一侧占多数 -> 目标应为暗区
        bright_ratio = float((roi_blur > thr).mean())
        polarity = "dark" if bright_ratio > 0.5 else "bright"

    if polarity == "dark":
        fg = (roi_blur <= thr).astype(np.uint8)
    else:
        fg = (roi_blur > thr).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    mask_roi = np.zeros_like(fg)
    if n > 1:
        # 取最大连通域 (排除背景)
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        # 排除"边界背景"式连通域: 若最大域贴满 ROI 四边则视为阈值失败后的背景
        comp = (labels == idx).astype(np.uint8)
        if comp.sum() < min_area:
            return RoughResult(np.zeros((H, W), np.uint8), np.zeros((0, 2), np.float32),
                               (xi1, yi1, xi2, yi2), thr, False)
        mask_roi = comp * 255

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros((H, W), np.uint8)
    if not contours:
        return RoughResult(mask, np.zeros((0, 2), np.float32), (xi1, yi1, xi2, yi2), thr, False)

    c = max(contours, key=cv2.contourArea)
    mask[yi1:yi2, xi1:xi2] = (mask_roi > 0).astype(np.uint8)
    pts = c.reshape(-1, 2).astype(np.float32) + np.array([xi1, yi1], np.float32)
    return RoughResult(mask, pts, (xi1, yi1, xi2, yi2), thr, True)


def bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_rough_contour(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    shrink: float = 0.0,
) -> RoughResult:
    """回退策略: 直接用检测框矩形作为粗糙轮廓 (Otsu 未能分割出前景时使用)。

    shrink: 矩形相对框中心收缩比例 (0 = 正好是框)。
    """
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * (1 - shrink), (y2 - y1) * (1 - shrink)
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return RoughResult(mask, pts, (int(x1), int(y1), int(x2), int(y2)), float("nan"), True)


def rough_contour_with_fallback(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    polarity: str = "dark",
    shrink: float = 0.0,
) -> RoughResult:
    """先做 Otsu 自适应分割; 失败或前景面积异常时退回检测框矩形。"""
    rr = otsu_adaptive_rough_contour(image, bbox, polarity=polarity)
    x1, y1, x2, y2 = bbox
    box_area = max(1.0, (x2 - x1) * (y2 - y1))
    if not rr.ok or len(rr.contour) < 3:
        return box_rough_contour(image, bbox, shrink=shrink)
    # 前景几乎填满整个框 -> 阈值把背景也算了进来, 同样回退
    if rr.mask.sum() / box_area > 0.92:
        return box_rough_contour(image, bbox, shrink=shrink)
    return rr
