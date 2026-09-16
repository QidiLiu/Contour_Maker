"""针对"框初始化"分支的数据增强 (不改 refiner 结构)。

动机: 推理时 A 方案的初始轮廓 = 检测框矩形, 但 YOLO 框在真实数据上会
退化 —— 框只覆盖病灶一部分(合并/遮挡)、框偏大偏小、框是任意四边形状。
原 A1/A2 只作用于 Otsu 分支, 框分支的几何形状是固定的完美矩形, 属于
"过理想"的训练分布, 因此这里补上三类形状级增强:

  C1 per-side jitter : 四条边独立抖动 -> 模拟"只框住一部分"
  C2 scale/shift     : 宽高独立缩放 + 整体平移
  C3 interior offset : 在框内部按随机内缩比例取轮廓 (不再是贴边的矩形)
  C4 corner jitter   : 四角独立抖动 -> 任意四边形
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import AugConfig


def sample_box_shape(
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
    acfg: AugConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """按增强配置合成一个"退化的检测框"多边形 (4 点, 逆时针, 全图像素坐标)。

    acfg.box_aug=False 时退化为原始矩形。
    """
    H, W = shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    if not getattr(acfg, "box_aug", False):
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    else:
        # ---- C1: 每条边独立抖动 ----
        j = acfg.box_side_jitter
        if j > 0:
            x1 += rng.uniform(-j, j) * bw
            x2 += rng.uniform(-j, j) * bw
            y1 += rng.uniform(-j, j) * bh
            y2 += rng.uniform(-j, j) * bh
            if x2 - x1 < 4:
                x2 = x1 + 4
            if y2 - y1 < 4:
                y2 = y1 + 4

        # ---- C2: 宽高独立缩放 (回到中心) + 整体平移 ----
        lo, hi = 1.0 - acfg.box_scale_jitter, 1.0 + acfg.box_scale_jitter
        sw, sh = rng.uniform(lo, hi), rng.uniform(lo, hi)
        w2, h2 = (x2 - x1) * sw, (y2 - y1) * sh
        ccx = cx + rng.uniform(-acfg.box_shift_jitter, acfg.box_shift_jitter) * bw
        ccy = cy + rng.uniform(-acfg.box_shift_jitter, acfg.box_shift_jitter) * bh
        x1, x2 = ccx - w2 / 2, ccx + w2 / 2
        y1, y2 = ccy - h2 / 2, ccy + h2 / 2

        # ---- C3: 内部内缩 (轮廓位于框内, 不再贴边) ----
        if acfg.box_interior_offset > 0:
            k = rng.uniform(0.0, acfg.box_interior_offset)
            dx, dy = (x2 - x1) * k / 2, (y2 - y1) * k / 2
            x1, x2, y1, y2 = x1 + dx, x2 - dx, y1 + dy, y2 - dy

        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)

        # ---- C4: 四角独立抖动 -> 任意四边形 ----
        c = acfg.box_corner_jitter
        if c > 0:
            noise = rng.uniform(-c, c, pts.shape).astype(np.float32) * np.array([bw, bh], np.float32)
            pts = pts + noise

    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    # 保证有面积
    if abs(cv2.contourArea(pts.astype(np.float32))) < 16:
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts


def polygon_mask(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m
