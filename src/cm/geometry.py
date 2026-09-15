"""轮廓几何工具: 重采样、点对应、方向统一、mask<->polygon。"""
from __future__ import annotations

import cv2
import numpy as np


# ------------------------------------------------------------ basic ops
def signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_ccw(pts: np.ndarray) -> np.ndarray:
    """保证逆时针 (图像坐标系 y 向下, signed area > 0 记为逆时针)。"""
    return pts if signed_area(pts) > 0 else pts[::-1].copy()


def ensure_cw(pts: np.ndarray) -> np.ndarray:
    return pts if signed_area(pts) < 0 else pts[::-1].copy()


def resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """把闭合轮廓按弧长等间隔重采样为 n 个点 (不含重复首点)。"""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 3:
        return np.repeat(pts[:1], n, axis=0)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2), np.float64)
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)
    t = (targets - cum[idx]) / np.maximum(seg[idx], 1e-9)
    out[:] = closed[idx] + t[:, None] * (closed[idx + 1] - closed[idx])
    return out.astype(np.float32)


def mask_to_contour(mask: np.ndarray) -> np.ndarray | None:
    """二值 mask -> 最大外轮廓点集 (N,2) float32。"""
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 1:
        return None
    return c.reshape(-1, 2).astype(np.float32)


def contour_to_mask(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """多边形填充为二值 mask。"""
    m = np.zeros(shape, np.uint8)
    if pts is None or len(pts) < 3:
        return m
    cv2.fillPoly(m, [np.round(np.asarray(pts, np.float64)).astype(np.int32)], 1)
    return m


def contour_area(pts: np.ndarray) -> float:
    return abs(signed_area(np.asarray(pts, np.float64)))


# ------------------------------------------------------------ reparametrisation
def closest_param(ref: np.ndarray, p: np.ndarray) -> tuple[int, float]:
    """ref 上离点 p 最近的顶点下标及距离。"""
    d = np.linalg.norm(ref - p[None, :], axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def rotate_to_start(pts: np.ndarray, start_idx: int) -> np.ndarray:
    return np.roll(pts, -int(start_idx), axis=0)


def canonical_start(pts: np.ndarray) -> np.ndarray:
    """取一个确定性的起始点 (最高点, 并列取最左), 保证 GT 轮廓序号稳定。"""
    order = np.lexsort((pts[:, 0], pts[:, 1]))   # 先按 y, 再按 x
    return rotate_to_start(pts, int(order[0]))


def arc_correspondence(rough: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """弧长对应: 把 exact 的起始点对齐到 rough[0] 的最近点, 再等弧长采样到 len(rough)。

    返回与 rough 一一对应的 exact 点 (同序、同向、同起点)。
    """
    n = len(rough)
    ex = resample_closed(exact, 4 * n)              # 上采样提高对齐精度
    i, _ = closest_param(ex, rough[0])
    ex = rotate_to_start(ex, i)
    ex = resample_closed(ex, n)
    # 方向一致性: 两者 signed area 同号
    if signed_area(np.asarray(rough, np.float64)) * signed_area(np.asarray(ex, np.float64)) < 0:
        ex = ex[::-1].copy()
        ex = rotate_to_start(ex, closest_param(ex, rough[0])[0])
    return ex.astype(np.float32)


def radial_correspondence(rough: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """径向对应: rough 点沿质心射线与 exact 轮廓求交, 作为对应精准点。

    对强凹凸形状可能失败, 失败时退化为 arc_correspondence。
    """
    n = len(rough)
    ex = resample_closed(exact, 512)
    c = rough.mean(axis=0)
    out = np.empty((n, 2), np.float32)
    ok = 0
    for i, p in enumerate(rough):
        d = p - c
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            out[i] = p
            continue
        d = d / nd
        # 每条线段与射线求交 (射线: c + t d, t>0)
        a = ex
        b = np.roll(ex, -1, axis=0)
        e = b - a
        denom = d[0] * e[:, 1] - d[1] * e[:, 0]
        valid = np.abs(denom) > 1e-9
        t = np.full(len(a), np.nan)
        u = np.full(len(a), np.nan)
        ap = a - c
        t[valid] = (ap[valid, 0] * e[valid, 1] - ap[valid, 1] * e[valid, 0]) / denom[valid]
        u[valid] = (ap[valid, 0] * d[1] - ap[valid, 1] * d[0]) / -denom[valid]
        m = valid & (t > 0) & (u >= 0) & (u <= 1)
        if not np.any(m):
            out[i] = p
            continue
        ts = t[m]
        pts = c[None, :] + ts[:, None] * d[None, :]
        # 取离 rough 点最近的那个交点, 避免穿过目标打到对侧
        j = int(np.argmin(np.linalg.norm(pts - p[None, :], axis=1)))
        out[i] = pts[j]
        ok += 1
    if ok < n * 0.9:
        return arc_correspondence(rough, exact)
    return out


def compute_correspondence(rough: np.ndarray, exact: np.ndarray, method: str = "arc") -> np.ndarray:
    if method == "radial":
        return radial_correspondence(rough, exact)
    return arc_correspondence(rough, exact)


# ------------------------------------------------------------ smoothing
def smooth_closed(pts: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    """闭合轮廓的循环滑动平均 (轻微去抖)。"""
    if k <= 1 or len(pts) < k:
        return pts
    out = np.asarray(pts, np.float32).copy()
    for _ in range(iters):
        acc = np.zeros_like(out)
        for j in range(-(k // 2), k // 2 + 1):
            acc += np.roll(out, j, axis=0)
        out = acc / k
    return out
