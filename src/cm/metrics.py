"""分割质量评估指标: Dice / IoU / HD95 / ASSD / 面积误差 / 周长误差。"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def _surface(mask: np.ndarray) -> np.ndarray:
    """mask 的边界像素坐标 (N,2)。"""
    m = mask.astype(bool)
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)
    return np.argwhere(m & ~er)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0
    return float(2.0 * (p & g).sum() / s)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    u = (p | g).sum()
    if u == 0:
        return 1.0
    return float((p & g).sum() / u)


def hd95(pred: np.ndarray, gt: np.ndarray, percentile: float = 95.0) -> float:
    """95% Hausdorff 距离 (像素)。"""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    if len(sp) == 0 or len(sg) == 0:
        return float("nan")
    d1 = ndimage.distance_transform_edt(~_mask_from_points(sp, gt.shape))[tuple(sg.T)]
    d2 = ndimage.distance_transform_edt(~_mask_from_points(sg, pred.shape))[tuple(sp.T)]
    d = np.concatenate([d1, d2])
    return float(np.percentile(d, percentile))


def _mask_from_points(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, bool)
    if len(pts):
        m[pts[:, 0], pts[:, 1]] = True
    return m


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """平均对称表面距离 (像素)。"""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    if len(sp) == 0 or len(sg) == 0:
        return float("nan")
    d1 = ndimage.distance_transform_edt(~_mask_from_points(sp, gt.shape))[tuple(sg.T)]
    d2 = ndimage.distance_transform_edt(~_mask_from_points(sg, pred.shape))[tuple(sp.T)]
    return float((d1.mean() + d2.mean()) / 2.0)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    """容差 tol 像素下的边界 F1。"""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~_mask_from_points(sg, gt.shape))
    dt_p = ndimage.distance_transform_edt(~_mask_from_points(sp, pred.shape))
    d_p2g = dt_g[tuple(sp.T)]
    d_g2p = dt_p[tuple(sg.T)]
    precision = float((d_p2g <= tol).mean())
    recall = float((d_g2p <= tol).mean())
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def contour_length(pts: np.ndarray) -> float:
    if pts is None or len(pts) < 2:
        return 0.0
    d = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    return float(d.sum())


def evaluate_pair(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """返回单个 (预测 mask, GT mask) 的完整指标字典。"""
    p = pred_mask.astype(np.uint8)
    g = gt_mask.astype(np.uint8)
    out = dict(dice=dice(p, g), iou=iou(p, g), hd95=hd95(p, g), assd=assd(p, g),
               bf1=boundary_f1(p, g))
    ga = float(g.sum())
    out["area_err"] = float(abs(float(p.sum()) - ga) / ga) if ga > 0 else float("nan")
    return out


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def polygon_from_mask(mask: np.ndarray) -> np.ndarray | None:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if len(c) < 3:
        return None
    return c.reshape(-1, 2).astype(np.float32)
