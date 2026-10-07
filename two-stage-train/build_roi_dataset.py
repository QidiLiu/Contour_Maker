"""构建 MK-UNet 的 ROI 训练集。

对每个目标裁剪 ROI（可含框抖动）并 resize 到 MK-UNet 输入尺寸，
生成 (roi_image, roi_mask) 对，写成目录结构供 train_mkunet.py 直接读取。

    <out>/<split>/<uid>_t<k>.png    # ROI 图像 (灰度)
    <out>/<split>/<uid>_t<k>_mask.png
    <out>/index.csv                 # uid, target_idx, src, box, ...

用法:
    # 训练集用 GT 框（干净初始化）
    python build_roi_dataset.py --box-source gt --splits train val --out ../data/roi_cache/train_gt

    # 或用检测框（匹配推理分布）
    python build_roi_dataset.py --det ../runs/det_yolo26n/weights/best.pt \
        --box-source det --splits train --out ../data/roi_cache/train_det
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    MKUNET_ROI_SIZE,
    MIN_MASK_AREA,
    ROI_CACHE,
    SPLITS,
    UNIFIED,
    YOLO_CONF,
    YOLO_IMGSZ,
)
from common.roi import crop_roi, resize_roi  # noqa: E402


def jitter_box(box, rng, strength: float, shape):
    """检测框抖动：中心与尺度加相对噪声，模拟 YOLO 定位误差。"""
    if strength <= 0:
        return tuple(float(v) for v in box)
    h, w = shape
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cx += rng.normal(0, strength * bw / 2)
    cy += rng.normal(0, strength * bh / 2)
    sx = float(np.clip(1.0 + rng.normal(0, strength), 0.7, 1.4))
    sy = float(np.clip(1.0 + rng.normal(0, strength), 0.7, 1.4))
    nx1, ny1 = cx - bw * sx / 2, cy - bh * sy / 2
    nx2, ny2 = cx + bw * sx / 2, cy + bh * sy / 2
    nx1 = float(np.clip(nx1, 0, w)); nx2 = float(np.clip(nx2, 0, w))
    ny1 = float(np.clip(ny1, 0, h)); ny2 = float(np.clip(ny2, 0, h))
    if nx2 - nx1 < 4:
        nx2 = min(w, nx1 + 4)
    if ny2 - ny1 < 4:
        ny2 = min(h, ny1 + 4)
    return (nx1, ny1, nx2, ny2)


def expand_box(box, ratio: float):
    """向外扩 ratio（给边界留上下文，避免贴边目标的边界被裁掉）。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    return (x1 - bw * ratio, y1 - bh * ratio, x2 + bw * ratio, y2 + bh * ratio)


def gt_targets(mask: np.ndarray, min_area: int = MIN_MASK_AREA):
    """GT mask -> [(comp_mask, bbox), ...]"""
    m = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (labels == lab).astype(np.uint8)
        ys, xs = np.nonzero(comp)
        if xs.size == 0:
            continue
        box = (float(xs.min()), float(ys.min()),
               float(xs.max()) + 1, float(ys.max()) + 1)
        out.append((comp, box))
    return out


def det_boxes_for_image(model, img_path: Path, conf: float, imgsz: int, device: str):
    """YOLO26n 检测框（原图坐标）。"""
    res = model.predict(str(img_path), conf=conf, imgsz=imgsz, verbose=False, device=device)
    r = res[0]
    if r.boxes is None or not len(r.boxes):
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    return [(tuple(float(v) for v in b), int(c), float(s))
            for b, c, s in zip(xyxy, cls, confs)]


def match_box_to_target(det_box, gt_boxes, shape, min_iou: float = 0.3):
    """把检测框匹配到 GT 目标（按 IoU 最大）。返回 gt 索引或 None。

    gt_boxes: [(x1,y1,x2,y2), ...]
    """
    best, best_iou = None, min_iou
    for i, gb in enumerate(gt_boxes):
        iou = _iou_xyxy(det_box, gb)
        if iou >= best_iou:
            best, best_iou = i, iou
    return best


def _iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 MK-UNet ROI 训练集")
    ap.add_argument("--det", default=None, help="检测权重，--box-source det 时必需")
    ap.add_argument("--box-source", choices=["gt", "det"], default="gt")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--out", default=None, help="输出目录，默认 data/roi_cache/<name>")
    ap.add_argument("--name", default=None)
    ap.add_argument("--roi-size", type=int, default=MKUNET_ROI_SIZE)
    ap.add_argument("--pad-ratio", type=float, default=0.15,
                    help="框向外扩的比例，给边界留上下文")
    ap.add_argument("--jitter", type=float, default=0.10,
                    help="框抖动强度（模拟检测误差）；0 表示不抖动")
    ap.add_argument("--variants", type=int, default=1, help="每目标生成的样本组数")
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROI_CACHE / (
        args.name or f"{'_'.join(args.splits)}_{args.box_source}")
    out.mkdir(parents=True, exist_ok=True)
    for s in args.splits:
        (out / s).mkdir(parents=True, exist_ok=True)

    sp_path = SPLITS / "splits.csv"
    if not sp_path.exists():
        print(f"[error] 未找到 {sp_path}")
        return 1
    df = pd.read_csv(sp_path)
    df = df[df["split"].isin(args.splits)]
    if args.max_images:
        df = df.head(args.max_images)

    det_model = None
    if args.box_source == "det":
        if not args.det:
            print("[error] --box-source det 需要 --det")
            return 1
        from ultralytics import YOLO
        det_model = YOLO(args.det)
        device = 0 if args.device == "cuda" else args.device
    else:
        device = "cpu"

    rng = np.random.default_rng(args.seed)
    rows = []
    n_img = n_ok = n_fail_det = n_fail_geom = 0

    for n, (_, r) in enumerate(df.iterrows(), start=1):
        # 注意: 用方括号取列。r.mask 会命中 pandas Series.mask 方法而非本列。
        img_p = UNIFIED / r["image"]
        msk_p = UNIFIED / r["mask"]
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        n_img += 1
        h, w = img.shape[:2]

        tgts = gt_targets(msk)
        if not tgts:
            continue

        # ---- 决定每个目标用哪个框
        if args.box_source == "gt":
            boxes = [(i, gb, int(r["class_id"])) for i, (_c, gb) in enumerate(tgts)]
        else:
            dets = det_boxes_for_image(det_model, img_p, args.conf, args.imgsz, device)
            boxes = []
            for db, dcls, dconf in dets:
                gi = match_box_to_target(db, [t[1] for t in tgts], (h, w))
                if gi is None:
                    n_fail_det += 1
                    continue
                boxes.append((gi, db, dcls))
        if not boxes:
            continue

        # ---- 逐目标生成 ROI
        for gi, box, cls_id in boxes:
            comp = tgts[gi][0]
            for v in range(args.variants):
                b = jitter_box(box, rng, args.jitter, (h, w)) if args.variants > 1 \
                    else expand_box(box, args.pad_ratio)
                if args.variants > 1:
                    b = expand_box(b, args.pad_ratio)
                roi, rc = crop_roi(img, b)
                if roi is None or rc is None:
                    n_fail_geom += 1
                    continue
                x1, y1, x2, y2 = rc
                sub_mask = comp[y1:y2, x1:x2]
                if sub_mask.sum() < 10:
                    n_fail_geom += 1
                    continue

                roi_img = resize_roi(roi, args.roi_size, square_pad=True)
                roi_msk = resize_roi((sub_mask * 255).astype(np.uint8),
                                     args.roi_size, square_pad=True)
                # square_pad 的居中偏移需要在贴回时还原，这里把原 ROI 尺寸记下来
                key = f'{r["uid"]}_t{gi}_v{v}'
                sub = Path(r["split"])
                dst = out / sub
                dst.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst / f"{key}.png"), roi_img)
                cv2.imwrite(str(dst / f"{key}_mask.png"), roi_msk)
                rows.append(dict(
                    uid=r["uid"], split=r["split"], dataset=r["dataset"],
                    class_id=int(cls_id), target=gi, variant=v,
                    roi_h=int(y2 - y1), roi_w=int(x2 - x1),
                    box_x1=x1, box_y1=y1, box_x2=x2, box_y2=y2,
                    file=f"{key}.png", mask_file=f"{key}_mask.png",
                ))
                n_ok += 1

        if n % 500 == 0:
            print(f"  {n}/{len(df)} images={n_img} samples={n_ok}", flush=True)

    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False)
    print(f"\n[ok  ] images={n_img} samples={len(index)} "
          f"fail_det={n_fail_det} fail_geom={n_fail_geom}")
    print(f"[out ] {out}")
    if len(index):
        print("\n=== 每 split 样本数 ===")
        print(index["split"].value_counts().to_string())
        print("\n=== 类别分布 ===")
        print(index["class_id"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())