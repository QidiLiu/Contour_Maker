"""导出 YOLO 数据集:

* 检测数据集 (YOLO26n 训练): labels_det/*.txt  -> class cx cy w h (归一化)
* 分割数据集 (YOLO26n-seg 训练): labels_seg/*.txt -> class x1 y1 x2 y2 ... (归一化多边形)

用同一套 data/splits/splits.csv 划分, 保证与方案 A 完全一致的数据条件。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm import geometry as G  # noqa: E402
from cm.config import ROOT, RUNS, SPLITS, UNIFIED  # noqa: E402


def mask_components(mask: np.ndarray, min_area: int = 50):
    m = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (labels == lab).astype(np.uint8)
        out.append(comp)
    return out


def polygon_text(comp: np.ndarray, w: int, h: int, cls: int = 0) -> tuple[str, str]:
    contour = G.mask_to_contour(comp)
    if contour is None:
        return "", ""
    c = G.resample_closed(contour, 128)
    c01 = np.clip(c / np.array([w, h], np.float32), 0, 1)
    seg = f"{cls} " + " ".join(f"{v:.6f}" for v in c01.reshape(-1))
    x1, y1 = c01.min(0)
    x2, y2 = c01.max(0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    det = f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    return seg, det


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "yolo"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    sp = pd.read_csv(SPLITS / "splits.csv")
    for sub in ("images/train", "images/val", "images/test",
                "labels_det/train", "labels_det/val", "labels_det/test",
                "labels_seg/train", "labels_seg/val", "labels_seg/test"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    n_targets = 0
    for _, row in sp.iterrows():
        split = row["split"]
        uid = row["uid"]
        img = cv2.imread(str(UNIFIED / row["image"]), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(UNIFIED / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        H, W = img.shape[:2]
        comps = mask_components(msk)
        if not comps:
            continue
        # 统一写成 3 通道 PNG (YOLO/ultralytics 读取更稳)
        cv2.imwrite(str(out / f"images/{split}/{uid}.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        seg_lines, det_lines = [], []
        for comp in comps:
            s, d = polygon_text(comp, W, H)
            if s:
                seg_lines.append(s)
                det_lines.append(d)
                n_targets += 1
        (out / f"labels_seg/{split}/{uid}.txt").write_text("\n".join(seg_lines))
        (out / f"labels_det/{split}/{uid}.txt").write_text("\n".join(det_lines))

    for kind in ("det", "seg"):
        for split in ("train", "val", "test"):
            yml = out / f"data_{kind}_{split}.yaml"
            yml.write_text(
                f"path: {out}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"test: images/test\n"
                f"names:\n  0: lesion\n"
                f"# 当前为 {split} 视图, 标签目录 labels_{kind}/\n")
    (out / "EXPORT_NOTE.md").write_text(
        "ultralytics 默认在 images/ 同级查找 labels/ 目录。为使 det/seg 共用同一份 images,\n"
        "训练脚本会临时把 labels_det 或 labels_seg 软链接为 labels/,\n"
        "并写出 train/val 分别指向正确标签目录的临时 data yaml。\n")
    print(f"[export] {len(sp)} 张图, {n_targets} 个目标 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
