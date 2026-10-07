"""分割评估指标：Dice / IoU / HD95 / ASSD / 边界F1 / 面积误差。

全部以像素为单位，在原图分辨率下计算。
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------- 基础
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


def _surface(mask: np.ndarray) -> np.ndarray:
    """mask 的边界像素坐标 (N,2)，行优先。"""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros((0, 2), int)
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)
    return np.argwhere(m & ~er)


def _points_to_mask(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, bool)
    if len(pts):
        m[pts[:, 0], pts[:, 1]] = True
    return m


def _surface_distances(pred: np.ndarray, gt: np.ndarray):
    """返回 (pred->gt 距离数组, gt->pred 距离数组)。"""
    sp, sg = _surface(pred), _surface(gt)
    if len(sp) == 0 or len(sg) == 0:
        return None, None
    dt_g = ndimage.distance_transform_edt(~_points_to_mask(sg, gt.shape))
    dt_p = ndimage.distance_transform_edt(~_points_to_mask(sp, pred.shape))
    return dt_g[tuple(sp.T)], dt_p[tuple(sg.T)]


def hd95(pred: np.ndarray, gt: np.ndarray, percentile: float = 95.0) -> float:
    """95% Hausdorff 距离 (px)。"""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    d_pg, d_gp = _surface_distances(pred, gt)
    if d_pg is None:
        return float("nan")
    return float(np.percentile(np.concatenate([d_pg, d_gp]), percentile))


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """平均对称表面距离 (px)。"""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    d_pg, d_gp = _surface_distances(pred, gt)
    if d_pg is None:
        return float("nan")
    return float((d_pg.mean() + d_gp.mean()) / 2.0)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> float:
    """容差 tol px 下的边界 F1。"""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    d_pg, d_gp = _surface_distances(pred, gt)
    if d_pg is None:
        return float("nan")
    precision = float((d_pg <= tol).mean())
    recall = float((d_gp <= tol).mean())
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def area_error(pred: np.ndarray, gt: np.ndarray) -> float:
    ga = float(gt.astype(bool).sum())
    if ga <= 0:
        return float("nan")
    return float(abs(float(pred.astype(bool).sum()) - ga) / ga)


def evaluate_pair(pred: np.ndarray, gt: np.ndarray, boundary_tol: float = 2.0) -> dict:
    """单个 (pred_mask, gt_mask) 的完整指标。"""
    p = (pred.astype(bool)).astype(np.uint8)
    g = (gt.astype(bool)).astype(np.uint8)
    return dict(
        dice=dice(p, g), iou=iou(p, g),
        hd95=hd95(p, g), assd=assd(p, g),
        bf1=boundary_f1(p, g, boundary_tol),
        area_err=area_error(p, g),
    )


# ---------------------------------------------------------------- 掩码 <-> 轮廓
def mask_to_contour(mask: np.ndarray) -> np.ndarray | None:
    """二值 mask -> 最大外轮廓点集 (N,2) float32。"""
    m = (mask > 0).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if len(c) < 3 or cv2.contourArea(c) < 1:
        return None
    return c.reshape(-1, 2).astype(np.float32)


def contour_to_mask(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """多边形填充为二值 mask (0/1 uint8)。"""
    m = np.zeros(shape, np.uint8)
    if pts is None or len(pts) < 3:
        return m
    cv2.fillPoly(m, [np.round(np.asarray(pts, np.float64)).astype(np.int32)], 1)
    return m


def mask_bbox(mask: np.ndarray):
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


# ---------------------------------------------------------------- 匹配
def greedy_match(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray],
                 thresh: float = 0.5):
    """按 mask IoU 贪心匹配（先匹配合适度最高的）。

    返回 (pairs, n_miss, n_fp)：
      pairs    = [(pred_idx, gt_idx, iou), ...]
      n_miss   = 未匹配的 GT 数
      n_fp     = 未匹配的预测数
    """
    if not pred_masks or not gt_masks:
        return [], len(gt_masks), len(pred_masks)

    iou_mat = np.zeros((len(pred_masks), len(gt_masks)), np.float32)
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            iou_mat[i, j] = iou(p, g)

    order = np.dstack(np.unravel_index(np.argsort(-iou_mat, axis=None), iou_mat.shape))[0]
    used_p, used_g, pairs = set(), set(), []
    for i, j in order:
        if i in used_p or j in used_g:
            continue
        if iou_mat[i, j] < thresh:
            break  # 已按 IoU 降序, 后面的都低于阈值
        used_p.add(int(i))
        used_g.add(int(j))
        pairs.append((int(i), int(j), float(iou_mat[i, j])))
    return pairs, len(gt_masks) - len(used_g), len(pred_masks) - len(used_p)


def gt_components(mask: np.ndarray, min_area: int = 50):
    """GT mask -> 连通域列表，每项含 mask/bbox/contour。"""
    from .metrics import contour_to_mask  # 自身引用保持显式

    m = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (labels == lab).astype(np.uint8)
        bb = mask_bbox(comp)
        ct = mask_to_contour(comp)
        if bb is None or ct is None:
            continue
        out.append(dict(mask=comp, bbox=bb, contour=ct, area=int(stats[lab, cv2.CC_STAT_AREA])))
    return out


def prf(n_tp: int, n_fp: int, n_fn: int) -> dict:
    p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    r = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=p, recall=r, f1=f)


def paired_bootstrap_ci(deltas: list[float], n_boot: int = 5000, seed: int = 42,
                        alpha: float = 0.05):
    """按样本配对 bootstrap：返回 (观测均值, CI下界, CI上界, 双侧p值)。

    deltas 必须已按同一单位(图像或目标)配对，长度一致。
    """
    import random

    d = list(deltas)
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    obs = sum(d) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    p = 2.0 * min(sum(1 for b in boots if b <= 0), sum(1 for b in boots if b >= 0)) / n_boot
    return obs, lo, hi, min(1.0, p)