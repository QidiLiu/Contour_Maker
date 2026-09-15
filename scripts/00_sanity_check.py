"""数据管线自检: 检查粗糙轮廓、对应关系、ROI 采样、A1-A5 增强是否正常。

    PYTHONPATH=src ./.conda/bin/python scripts/00_sanity_check.py --n 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm import geometry as G  # noqa: E402
from cm import metrics as M  # noqa: E402
from cm import roi as R  # noqa: E402
from cm.config import REPORTS, SPLITS, UNIFIED, AugConfig, RefinerConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--window-ratio", type=float, default=0.25)
    ap.add_argument("--out", default=str(REPORTS / "sanity"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rcfg = RefinerConfig(window_ratio=args.window_ratio)
    sp = pd.read_csv(SPLITS / "splits.csv")
    sp = sp.sample(n=min(args.n, len(sp)), random_state=0)

    print("level | rough_dice | rough->gt_px | out_of_window | rough_pts_in_bbox | ok")
    for level in [0, 1, 2, 3, 4, 5]:
        acfg = AugConfig(level=level)
        rows = []
        for _, row in sp.iterrows():
            img = cv2.imread(str(UNIFIED / row["image"]), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(str(UNIFIED / row["mask"]), cv2.IMREAD_GRAYSCALE)
            m = (gt > 127).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            if n < 2:
                continue
            lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            comp = (labels == lab).astype(np.uint8)
            bbox = R.bbox_from_mask(comp)
            spec = R.TargetSpec(image=img, gt_mask=comp, bbox=bbox, polarity="dark")
            for k in range(3):
                rng = np.random.default_rng(1234 + k)
                s = R.build_sample(spec, rcfg, acfg, rng, jitter=(level > 0))
                if s is None:
                    continue
                shape = img.shape[:2]
                gm = G.contour_to_mask(s.meta["exact_global"], shape)
                rough_mask = G.contour_to_mask(s.rough_global, shape)
                # 粗糙轮廓与 GT 的逐点偏差 (像素), 即 refiner 需要修正的平均幅度
                d_px = float(np.linalg.norm(s.meta["exact_global"] - s.rough_global, axis=1).mean())
                rm = M.mask_bbox(comp)
                inside = float(np.mean([(rm[0] <= p[0] <= rm[2]) and (rm[1] <= p[1] <= rm[3])
                                        for p in s.rough_global]))
                rows.append((M.dice(rough_mask, gm), d_px, s.meta["out_ratio"], inside))
                if level == 5 and k == 0 and len(rows) <= 3:
                    vis = np.concatenate([s.patches[i] for i in range(8)], axis=1)
                    vis = ((vis - vis.min()) / (float(np.ptp(vis)) + 1e-6) * 255).astype(np.uint8)
                    cv2.imwrite(str(out / f"patches_{row['dataset']}_{row['uid']}_L{level}.png"),
                                cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        if rows:
            arr = np.array(rows)
            print(f"{level:5d} | {arr[:,0].mean():10.4f} | {arr[:,1].mean():12.2f} | {arr[:,2].mean():13.4f} | "
                  f"{arr[:,3].mean():17.4f} | {len(rows)}")
        else:
            print(f"{level:5d} | 无有效样本")
    print(f"\n可视化 patch 已存到 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
