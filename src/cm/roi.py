"""ROI 采样 + 数据增强 A1-A5 + 检测框抖动。

核心数据对生成流程 (对应用户给的算法描述):
    GT mask -> 精准轮廓 (n_points 个等距点, 逆时针)
    YOLO 检测框 (+抖动) -> ROI 内 Otsu -> 最大轮廓 -> 粗糙轮廓 (n_points)
    对每个粗糙点: 以该点为中心取边长为 window 的正方形 ROI, resize 到 32x32 灰度
    -> 64 x 32x32 输入矩阵 (逐图 Z-score), 输出 64 个精准点在 ROI 内的归一化坐标 (-1~1)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as G
from .config import AugConfig, RefinerConfig
from .rough import bbox_from_mask, box_rough_contour, otsu_adaptive_rough_contour


# ---------------------------------------------------------------- data classes
@dataclass
class FrameInfo:
    """粗糙点/精准点所在的局部 ROI 窗口 (可映射回全图坐标)。"""

    scale: float                 # px = (patch_norm01 - 0.5) * scale + center
    center: np.ndarray           # (2,) float32 全图像素坐标
    patch_px: int                # resize 前的采样边长
    center_norm: np.ndarray      # (2,) 该窗口中心在图内的归一化坐标 (0~1)


@dataclass
class Sample:
    """一个 (粗糙 -> 精准) 训练/评估样本。"""

    patches: np.ndarray              # (P, S, S) float32, Z-score 后
    rough_norm: np.ndarray           # (P, 2) 粗糙点在自身窗口内的归一化坐标 (-1~1)
    target_norm: np.ndarray          # (P, 2) 精准点坐标 (-1~1)  <- 回归目标
    rough_global: np.ndarray         # (P, 2) 粗糙点全图坐标 (供 mask 重建)
    frames: list[FrameInfo]          # 每个点的窗口信息 (供 mask 重建)
    aux: np.ndarray                  # (P, 3) 额外几何特征: 相对检测框位置 + 切向角
    shape: tuple[int, int] = (0, 0)  # 原图 (H, W)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers
def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def jitter_bbox(bbox, rng: np.random.Generator, strength: float, shape) -> tuple[float, float, float, float]:
    """在框中心/尺度上加相对抖动, 模拟 YOLO 检测框误差。"""
    if strength <= 0:
        return tuple(float(v) for v in bbox)  # type: ignore[return-value]
    H, W = shape
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # 中心抖动: 相对框半尺寸的高斯噪声
    cx += rng.normal(0, strength * w / 2)
    cy += rng.normal(0, strength * h / 2)
    # 尺度抖动
    sx = 1.0 + rng.normal(0, strength)
    sy = 1.0 + rng.normal(0, strength)
    sx = float(np.clip(sx, 0.7, 1.4))
    sy = float(np.clip(sy, 0.7, 1.4))
    w2, h2 = w * sx, h * sy
    nx1 = clamp01((cx - w2 / 2) / W) * W
    ny1 = clamp01((cy - h2 / 2) / H) * H
    nx2 = clamp01((cx + w2 / 2) / W) * W
    ny2 = clamp01((cy + h2 / 2) / H) * H
    if nx2 - nx1 < 4:
        nx2 = min(W, nx1 + 4)
    if ny2 - ny1 < 4:
        ny2 = min(H, ny1 + 4)
    return nx1, ny1, nx2, ny2


def augment_image(
    image: np.ndarray,
    rng: np.random.Generator,
    a3: float = 0.0,
    a4: float = 0.0,
    rot_center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A3/A4 图像级扰动。返回 (新图, 2x3 仿射矩阵)。"""
    H, W = image.shape[:2]
    M = np.eye(2, 3, dtype=np.float64)
    out = image
    if a4 > 0 and rng.random() < min(1.0, 2 * a4):
        ang = float(rng.uniform(-180, 180) * a4 * 3)     # a4=1/3 -> ±180°
        c = rot_center if rot_center is not None else np.array([W / 2, H / 2])
        Mr = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
        out = cv2.warpAffine(out, Mr, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        M = Mr
    if a3 > 0:
        # 亮暗 / 对比度
        alpha = 1.0 + float(rng.uniform(-1, 1) * a3 * 0.9)
        beta = float(rng.uniform(-1, 1) * a3 * 90)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta) if out.dtype == np.uint8 else np.clip(
            out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        # 图像噪声
        sigma = float(abs(rng.normal(0, 1)) * a3 * 12)
        if sigma > 0.2:
            noise = rng.normal(0, sigma, out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, M


def apply_affine(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    p = np.asarray(pts, np.float64)
    return (p @ M[:, :2].T + M[:, 2][None, :]).astype(np.float32)


# ---------------------------------------------------------------- ROI extraction
def extract_patches(
    image: np.ndarray,
    centers: np.ndarray,
    window: float | np.ndarray,
    size: int,
    normalize: bool = True,
) -> tuple[np.ndarray, list[FrameInfo]]:
    """对每个中心点取边长 window 的正方形 ROI 并 resize 到 size x size。

    window 可为标量或每点不同的数组 (自适应窗口)。
    """
    H, W = image.shape[:2]
    imgf = image.astype(np.float32)
    windows = np.full(len(centers), float(window), np.float32) if np.isscalar(window) else np.asarray(window, np.float32)
    patches = np.zeros((len(centers), size, size), np.float32)
    infos: list[FrameInfo] = []
    for i, c in enumerate(centers):
        win = float(windows[i])
        half = win / 2.0
        cx, cy = float(c[0]), float(c[1])
        x1 = int(round(cx - half)); y1 = int(round(cy - half))
        x2 = x1 + max(2, int(round(win))); y2 = y1 + max(2, int(round(win)))
        pad_l, pad_t = max(0, -x1), max(0, -y1)
        pad_r, pad_b = max(0, x2 - W), max(0, y2 - H)
        xa, xb = max(0, x1), min(W, x2)
        ya, yb = max(0, y1), min(H, y2)
        patch = imgf[ya:yb, xa:xb]
        if pad_l or pad_t or pad_r or pad_b:
            patch = np.pad(patch, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
        if normalize:
            mu, sd = float(patch.mean()), float(patch.std())
            patch = (patch - mu) / (sd + 1e-6)
        patches[i] = patch
        infos.append(FrameInfo(scale=float(x2 - x1), center=np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32),
                               patch_px=int(x2 - x1),
                               center_norm=np.array([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H], np.float32)))
    return patches, infos


def norm_from_frame(pts: np.ndarray, infos: list[FrameInfo]) -> np.ndarray:
    """全图坐标 -> 各窗口内的归一化坐标 (-1~1)。"""
    out = np.zeros((len(infos), 2), np.float32)
    for i, fi in enumerate(infos):
        out[i] = (np.asarray(pts[i], np.float32) - fi.center) / (fi.scale / 2.0)
    return out


def frame_from_norm(norm: np.ndarray, infos: list[FrameInfo]) -> np.ndarray:
    out = np.zeros((len(infos), 2), np.float32)
    for i, fi in enumerate(infos):
        out[i] = np.asarray(norm[i], np.float32) * (fi.scale / 2.0) + fi.center
    return out


def aux_features(pts: np.ndarray, bbox, image_shape) -> np.ndarray:
    """每个点的辅助几何特征: 相对检测框归一化位置 (2) + 轮廓切向角 (1)。"""
    H, W = image_shape
    x1, y1, x2, y2 = bbox
    bw, bh = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    rel = np.stack([(pts[:, 0] - x1) / bw, (pts[:, 1] - y1) / bh], axis=1)
    nxt = np.roll(pts, -1, axis=0)
    d = nxt - pts
    ang = np.arctan2(d[:, 1], d[:, 0])
    return np.concatenate([rel, ang[:, None]], axis=1).astype(np.float32)


# ---------------------------------------------------------------- A1/A2
def gaussian_noise_contour(pts: np.ndarray, rng: np.random.Generator, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return pts.copy()
    return (pts + rng.normal(0, sigma_px, pts.shape)).astype(np.float32)


def otsu_snap_contour(
    image: np.ndarray,
    rough_pts: np.ndarray,
    rng: np.random.Generator,
    sigma_px: float,
    search: int = 6,
) -> np.ndarray:
    """A2: 对每个粗糙点在局部 ROI 内用 Otsu 求最近边缘点, 再加微量噪声。"""
    H, W = image.shape[:2]
    out = rough_pts.copy()
    for i, (x, y) in enumerate(rough_pts):
        cx, cy = int(round(x)), int(round(y))
        x1, x2 = max(0, cx - search), min(W, cx + search + 1)
        y1, y2 = max(0, cy - search), min(H, cy + search + 1)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        patch = cv2.GaussianBlur(image[y1:y2, x1:x2], (3, 3), 0)
        thr = _otsu(patch)
        bright = float((patch > thr).mean())
        fg = (patch <= thr).astype(np.uint8) if bright > 0.5 else (patch > thr).astype(np.uint8)
        edge = cv2.Canny((fg * 255).astype(np.uint8), 40, 120)
        ys, xs = np.nonzero(edge)
        if xs.size == 0:
            continue
        d = (xs + x1 - x) ** 2 + (ys + y1 - y) ** 2
        j = int(np.argmin(d))
        out[i] = (xs[j] + x1, ys[j] + y1)
    if sigma_px > 0:
        out = out + rng.normal(0, sigma_px, out.shape)
    return out.astype(np.float32)


def _otsu(patch: np.ndarray) -> float:
    hist = np.bincount(patch.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = max(1.0, hist.sum())
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = np.maximum(omega * (1 - omega), 1e-12)
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(np.argmax(sigma_b))


# ---------------------------------------------------------------- main builder
@dataclass
class TargetSpec:
    """单个目标的生成参数。"""

    image: np.ndarray             # 灰度 uint8
    gt_mask: np.ndarray           # 全图二值 mask
    bbox: tuple[float, float, float, float]
    polarity: str = "dark"


def build_sample(
    spec: TargetSpec,
    rcfg: RefinerConfig,
    acfg: AugConfig,
    rng: np.random.Generator,
    level: int | None = None,
    jitter: bool = True,
    max_expand: float = 4.0,
) -> Sample | None:
    """生成一组 (64 张 ROI -> 64 个精准点) 样本。失败返回 None。"""
    lvl = acfg.level if level is None else level
    image = spec.image
    H, W = image.shape[:2]

    # ---- 精准轮廓 (GT)
    gt_contour = G.mask_to_contour(spec.gt_mask)
    if gt_contour is None:
        return None
    exact_pts = G.resample_closed(G.canonical_start(G.ensure_ccw(gt_contour)), rcfg.n_points)

    # ---- 粗糙轮廓 (检测框 -> Otsu)
    bbox_used = spec.bbox
    if jitter and acfg.bbox_jitter > 0:
        bbox_used = jitter_bbox(spec.bbox, rng, acfg.bbox_jitter, (H, W))
    # box 比例: 部分样本直接用检测框矩形作为初始轮廓, 让模型学会"从框出发"的精细化,
    # 从而在 Otsu 失败时仍可工作 (推理端有同样机制, 见 rough_contour_with_fallback)
    use_box = acfg.box_init_prob > 0 and rng.random() < acfg.box_init_prob
    if use_box:
        rough = box_rough_contour(image, bbox_used, shrink=0.0)
    else:
        rough = otsu_adaptive_rough_contour(image, bbox_used, polarity=spec.polarity)
        if not rough.ok or len(rough.contour) < 3:
            if acfg.box_fallback:
                rough = box_rough_contour(image, bbox_used, shrink=0.0)
    if not rough.ok or len(rough.contour) < 3:
        return None

    # ---- 图像级增强 A3/A4 (对全图做, 几何随之变换)
    a3 = acfg.a3_strength if acfg.enabled(3) else 0.0
    a4 = acfg.a4_strength if acfg.enabled(4) else 0.0
    if a3 > 0 or a4 > 0:
        center = np.array([(bbox_used[0] + bbox_used[2]) / 2, (bbox_used[1] + bbox_used[3]) / 2])
        image, M = augment_image(image, rng, a3=a3, a4=a4, rot_center=center)
        exact_pts = apply_affine(exact_pts, M)
        rough_pts_src = apply_affine(rough.contour, M)
        x1, y1, x2, y2 = [float(v) for v in bbox_used]
        corners = apply_affine(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32), M)
        bx1, by1 = float(corners[:, 0].min()), float(corners[:, 1].min())
        bx2, by2 = float(corners[:, 0].max()), float(corners[:, 1].max())
        bbox_used = (bx1, by1, bx2, by2)
        bbox_used = (clamp01(bbox_used[0] / W) * W, clamp01(bbox_used[1] / H) * H,
                     clamp01(bbox_used[2] / W) * W, clamp01(bbox_used[3] / H) * H)
        # 变换后重新提取粗糙轮廓 (保证粗糙轮廓始终来自图像本身的 Otsu 分割)
        if use_box:
            rough_pts_src = box_rough_contour(image, bbox_used, shrink=0.0).contour
        else:
            rough2 = otsu_adaptive_rough_contour(image, bbox_used, polarity=spec.polarity)
            if rough2.ok and len(rough2.contour) >= 3:
                rough_pts_src = rough2.contour
            elif acfg.box_fallback:
                rough_pts_src = box_rough_contour(image, bbox_used, shrink=0.0).contour
        # 旋转变换后 GT 轮廓方向可能翻转, 重新规范化
        exact_pts = G.canonical_start(G.ensure_ccw(G.resample_closed(exact_pts, 4 * rcfg.n_points)))
        exact_pts = G.resample_closed(exact_pts, rcfg.n_points)
    else:
        rough_pts_src = rough.contour

    rough_pts = G.resample_closed(G.ensure_ccw(rough_pts_src), rcfg.n_points)

    # ---- A1/A2: 粗糙点扰动
    diag = float(np.hypot(bbox_used[2] - bbox_used[0], bbox_used[3] - bbox_used[1]))
    window = max(6.0, rcfg.window_ratio * diag)
    if acfg.enabled(1) and lvl >= 1 and not use_box:
        sigma_px = acfg.a1_sigma * window
        if acfg.enabled(2) and lvl >= 2:
            # A2: Otsu 最近边缘点 + A1 的 [0, 1/3] 噪声
            sub = float(rng.uniform(0, acfg.a2_ratio)) * sigma_px
            mode = rng.random()
            if mode < 0.34:      # 纯 A2
                rough_pts = otsu_snap_contour(image, rough_pts, rng, sub)
            elif mode < 0.67:    # A2 + 少量 A1
                rough_pts = otsu_snap_contour(image, rough_pts, rng, sub)
                rough_pts = gaussian_noise_contour(rough_pts, rng, sigma_px * acfg.a2_ratio)
            else:                # 纯 A1
                rough_pts = gaussian_noise_contour(rough_pts, rng, sigma_px)
        else:
            rough_pts = gaussian_noise_contour(rough_pts, rng, sigma_px)

    # ---- 点对应: 粗糙 -> 精准
    target_pts = G.compute_correspondence(G.ensure_ccw(rough_pts), G.ensure_ccw(exact_pts), method="arc")

    # ---- A5: 点序偏移 (整体平移序号起点, 保持对应关系与逆时针顺序)
    if acfg.enabled(5) and lvl >= 5:
        shift = int(rng.integers(0, rcfg.n_points))
        if shift:
            rough_pts = np.roll(rough_pts, -shift, axis=0)
            target_pts = np.roll(target_pts, -shift, axis=0)

    # ---- 采样 64 张 ROI 图 (自适应窗口: 保证对应的精准点落在窗口内, 最多放大 max_expand 倍)
    H_, W_ = image.shape[:2]
    win = np.full(rcfg.n_points, window, np.float32)
    if rcfg.adaptive_window:
        for _ in range(3):
            _p, frames0 = extract_patches(image, rough_pts, win, rcfg.roi_size, normalize=False)
            tgt_norm0 = norm_from_frame(target_pts, frames0)
            need = np.max(np.abs(tgt_norm0), axis=1) > 0.98
            if not np.any(need):
                break
            win[need] = np.minimum(win[need] * 1.6, max_expand * window)
    patches, frames = extract_patches(image, rough_pts, win, rcfg.roi_size, normalize=rcfg.zscore)
    rough_norm = norm_from_frame(rough_pts, frames)
    target_norm = norm_from_frame(target_pts, frames)
    # 统计落在窗口外的精准点比例 (诊断用)
    out_ratio = float((np.max(np.abs(target_norm), axis=1) > 1.0).mean())
    aux = aux_features(rough_pts, bbox_used, (H, W))

    return Sample(patches=patches, rough_norm=rough_norm, target_norm=target_norm,
                  rough_global=rough_pts.astype(np.float32), frames=frames, aux=aux,
                  shape=(H, W),
                  meta=dict(bbox=np.asarray(bbox_used, np.float32), window=window,
                            windows=win, out_ratio=out_ratio, box_init=bool(use_box),
                            exact_global=target_pts.astype(np.float32)))


def sample_to_mask(sample: Sample, shape: tuple[int, int], use_target: bool = True) -> np.ndarray:
    """由样本中的 (预测) 归一化点重建全图 mask。"""
    norm = sample.target_norm if use_target else sample.rough_norm
    pts = frame_from_norm(norm, sample.frames)
    return G.contour_to_mask(pts, shape)
