"""ROI 几何工具：letterbox、框裁剪、掩码贴回、mask<->轮廓。"""
from __future__ import annotations

import cv2
import numpy as np

# letterbox 填充值：Ultralytics 默认 114
PAD_VALUE = 114


# ---------------------------------------------------------------- letterbox
def letterbox(img: np.ndarray, new_shape=(640, 640), color=PAD_VALUE,
              auto: bool = False, scale_fill: bool = False):
    """缩放并填充到目标尺寸，保持宽高比。

    返回 (out, ratio, (dw, dh))，其中 ratio 为缩放比，(dw,dh) 为左右/上下总填充。
    与 Ultralytics 的 letterbox 行为一致（stride 无关，不做 stride 对齐）。
    """
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    if not scale_fill and r != 1:
        # auto=False 时强制拉伸到目标(等价于直接 resize);此处采用等比缩放+填充
        pass
    new_unpad = (max(1, int(round(w * r))), max(1, int(round(h * r))))
    if (new_unpad[0] != w or new_unpad[1] != h) and not scale_fill:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw, dh = nw - new_unpad[0], nh - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    dw, dh = dw / 2, dh / 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def unletterbox_boxes(boxes_xyxy: np.ndarray, orig_hw, ratio: float, pad) -> np.ndarray:
    """letterbox 空间的 xyxy -> 原图坐标。"""
    h, w = orig_hw
    left, top = pad
    out = boxes_xyxy.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / ratio
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def unletterbox_mask(mask_lb: np.ndarray, orig_hw, ratio: float, pad) -> np.ndarray:
    """letterbox 空间的掩码 -> 原图分辨率掩码。

    步骤：裁掉填充区 -> 反缩放 -> 最近邻回原尺寸。
    """
    h, w = orig_hw
    left, top = pad
    nh, nw = mask_lb.shape[:2]
    m = mask_lb[top:nh - top, left:nw - left]
    if m.size == 0:
        m = mask_lb
    out = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8)


# ---------------------------------------------------------------- ROI 裁剪
def crop_roi(img: np.ndarray, bbox, pad_ratio: float = 0.0):
    """按框裁剪 ROI，可选向外扩 pad_ratio。

    返回 (roi, (x1,y1,x2,y2))，坐标已 clip 到图像范围。
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    x1i = int(max(0, np.floor(x1 - px)))
    y1i = int(max(0, np.floor(y1 - py)))
    x2i = int(min(w, np.ceil(x2 + px)))
    y2i = int(min(h, np.ceil(y2 + py)))
    if x2i - x1i < 2 or y2i - y1i < 2:
        return None, None
    return img[y1i:y2i, x1i:x2i], (x1i, y1i, x2i, y2i)


def resize_roi(roi: np.ndarray, size: int, square_pad: bool = True) -> np.ndarray:
    """ROI resize 到 size×size。

    square_pad=True 时等比缩放后居中填充（保留形状，避免拉伸失真）。
    """
    if roi is None:
        return None
    if square_pad:
        h, w = roi.shape[:2]
        s = size / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        r = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = np.full((size, size, *roi.shape[2:]), PAD_VALUE, np.uint8)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        out[y0:y0 + nh, x0:x0 + nw] = r
        return out
    return cv2.resize(roi, (size, size), interpolation=cv2.INTER_LINEAR)


def paste_roi_mask(roi_mask: np.ndarray, roi_hw, size: int, bbox_orig,
                   orig_shape, square_pad: bool = True) -> np.ndarray:
    """把 size×size 的 ROI 掩码还原到原图坐标系的整图掩码。

    步骤：去填充 -> 反缩放到 roi_hw -> 平移贴到 bbox_orig 位置。
    """
    out = np.zeros(orig_shape, np.uint8)
    if roi_mask is None:
        return out
    m = roi_mask
    if square_pad:
        # 计算 letterbox/square_pad 的填充量（与 resize_roi 一致）
        rh, rw = roi_hw
        s = size / max(rh, rw)
        nw, nh = max(1, int(round(rw * s))), max(1, int(round(rh * s)))
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        m = m[y0:y0 + nh, x0:x0 + nw]
    m = cv2.resize(m, (roi_hw[1], roi_hw[0]), interpolation=cv2.INTER_NEAREST)
    x1, y1, x2, y2 = bbox_orig
    rh, rw = roi_hw
    # roi_hw 是裁剪尺寸，贴回时需与裁剪框对齐
    ix1, iy1 = int(x1), int(y1)
    ih, iw = out.shape[:2]
    ex2, ey2 = min(iw, ix1 + rw), min(ih, iy1 + rh)
    ex1, ey1 = max(0, ix1), max(0, iy1)
    if ex2 > ex1 and ey2 > ey1:
        out[ey1:ey2, ex1:ex2] = m[ey1 - iy1:ey2 - iy1, ex1 - ix1:ex2 - ix1]
    return out


# ---------------------------------------------------------------- mask <-> contour
def mask_to_contour(mask: np.ndarray) -> np.ndarray | None:
    """二值掩码 -> 最大外轮廓点集 (N,2) float32。"""
    m = (mask > 0).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if len(c) < 3 or cv2.contourArea(c) < 1:
        return None
    return c.reshape(-1, 2).astype(np.float32)


def contour_to_mask(pts: np.ndarray, shape) -> np.ndarray:
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


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.7) -> list[int]:
    """标准 NMS（纯 numpy，无 ultralytics 依赖），返回保留的下标。"""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = (boxes_xyxy[:, i] for i in range(4))
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        iw = (xx2 - xx1).clip(0)
        ih = (yy2 - yy1).clip(0)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return keep