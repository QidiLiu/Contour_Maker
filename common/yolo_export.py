"""导出 YOLO 数据集（det 与 seg 共用同一份 images/，靠 labels 软链切换）。

用法:
    python -m common.yolo_export --kind det   --out data/yolo_det
    python -m common.yolo_export --kind seg   --out data/yolo_seg
    python -m common.yolo_export --kind both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import MIN_MASK_AREA, SPLITS, UNIFIED
from .roi import mask_to_contour


def _contours_with_area(mask: np.ndarray, min_area: int = MIN_MASK_AREA):
    """掩码 -> [(contour, area, bbox)], 只保留足够大的连通域。"""
    m = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == lab).astype(np.uint8)
        c = mask_to_contour(comp)
        if c is None:
            continue
        x, y, bw, bh = cv2.boundingRect(comp)
        out.append((c, area, (x, y, x + bw, y + bh)))
    return out


def export_one(df: pd.DataFrame, kind: str, out: Path) -> dict:
    """按 splits.csv 导出 YOLO 数据集。kind ∈ {det, seg}。"""
    labels_dir = out / ("labels_seg" if kind == "seg" else "labels_det")
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_tgt = 0
    n_skip = 0
    for row in df.itertuples():
        img_p = UNIFIED / row.image
        msk_p = UNIFIED / row.mask
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            n_skip += 1
            continue
        if img.shape[:2] != msk.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
        cls = int(row.class_id)

        comps = _contours_with_area(msk)
        if not comps:
            n_skip += 1
            continue

        seg_lines, det_lines = [], []
        for c, _area, (x1, y1, x2, y2) in comps:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            det_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # 128 点等弧长多边形
            poly = _resample_closed(c, 128)
            p01 = np.clip(poly / np.array([w, h], np.float32), 0, 1)
            seg_lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in p01.reshape(-1)))
            n_tgt += 1

        split = row.split
        # 统一写 3 通道 PNG（ultralytics 读取更稳）
        dst = out / "images" / split / f"{row.uid}.png"
        if not dst.exists():
            cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        if kind == "seg":
            (labels_dir / split / f"{row.uid}.txt").write_text("\n".join(seg_lines))
        else:
            (labels_dir / split / f"{row.uid}.txt").write_text("\n".join(det_lines))
        n_img += 1

    # ultralytics 在 images/ 同级找 labels/；写临时 yaml 指向具体标签目录
    yml = out / f"data_{kind}.yaml"
    yml.write_text(
        f"path: {out}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(_class_names()))
    )
    # 直接把 labels 软链到 labels_det / labels_seg，省去训练脚本的临时切换
    link = out / "labels"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        import shutil
        shutil.rmtree(link)
    link.symlink_to(labels_dir.name, target_is_directory=True)

    return dict(kind=kind, n_images=n_img, n_targets=n_tgt, n_skip=n_skip, out=str(out))


def _class_names():
    from .config import CLASS_NAMES
    return CLASS_NAMES


def _resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """闭合轮廓按弧长等间隔重采样为 n 点（不含重复首点）。"""
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
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg) - 1)
    t = (targets - cum[idx]) / np.maximum(seg[idx], 1e-9)
    out = closed[idx] + t[:, None] * (closed[idx + 1] - closed[idx])
    return out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 YOLO 数据集")
    ap.add_argument("--kind", choices=["det", "seg", "both"], default="both")
    ap.add_argument("--splits-csv", default=str(SPLITS / "splits.csv"))
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    sp = Path(args.splits_csv)
    if not sp.exists():
        print(f"[error] 未找到 {sp}, 请先运行 python -m common.splits")
        return 1
    df = pd.read_csv(sp)
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]

    from .config import YOLO_DET, YOLO_SEG
    kinds = ["det", "seg"] if args.kind == "both" else [args.kind]
    for k in kinds:
        out = YOLO_DET if k == "det" else YOLO_SEG
        r = export_one(df, k, out)
        print(f"[{r['kind']:3s}] images={r['n_images']} targets={r['n_targets']} "
              f"skip={r['n_skip']} -> {r['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())