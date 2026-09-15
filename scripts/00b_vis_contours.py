"""可视化粗糙轮廓 (Otsu) 与 GT 轮廓, 检查对应质量。

    PYTHONPATH=src ./.conda/bin/python scripts/00b_vis_contours.py --per-dataset 4
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
    ap.add_argument("--per-dataset", type=int, default=4)
    ap.add_argument("--split", default="val")
    ap.add_argument("--window-ratio", type=float, default=0.35)
    ap.add_argument("--out", default=str(REPORTS / "contour_check"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rcfg = RefinerConfig(window_ratio=args.window_ratio)
    acfg = AugConfig(level=0)
    sp = pd.read_csv(SPLITS / "splits.csv")
    sp = sp[sp["split"] == args.split]

    stats = []
    for ds, grp in sp.groupby("dataset"):
        for _, row in grp.head(args.per_dataset * 3).iterrows():
            img = cv2.imread(str(UNIFIED / row["image"]), cv2.IMREAD_GRAYSCALE)
            m = cv2.imread(str(UNIFIED / row["mask"]), cv2.IMREAD_GRAYSCALE)
            mm = (m > 127).astype(np.uint8)
            n, labels, st, _ = cv2.connectedComponentsWithStats(mm, 8)
            if n < 2:
                continue
            lab = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            comp = (labels == lab).astype(np.uint8)
            bbox = R.bbox_from_mask(comp)
            spec = R.TargetSpec(image=img, gt_mask=comp, bbox=bbox, polarity="dark")
            s = R.build_sample(spec, rcfg, acfg, np.random.default_rng(0), level=0, jitter=False)
            if s is None:
                continue
            gt_mask = G.contour_to_mask(s.meta["exact_global"], img.shape[:2])
            rough_mask = G.contour_to_mask(s.rough_global, img.shape[:2])
            stats.append(dict(dataset=ds, uid=row["uid"],
                              rough_dice=M.dice(rough_mask, gt_mask),
                              rough_iou=M.iou(rough_mask, gt_mask),
                              pt_err=float(np.linalg.norm(s.meta["exact_global"] - s.rough_global,
                                                          axis=1).mean()),
                              window=s.meta["window"] / min(img.shape[:2])))
            if len([1 for f in out.glob(f"contour_{ds}_*.png")]) >= args.per_dataset:
                continue
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, [np.round(s.meta["exact_global"]).astype(np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(vis, [np.round(s.rough_global).astype(np.int32)], True, (0, 200, 255), 2)
            for i, (rp, tp) in enumerate(zip(s.rough_global, s.meta["exact_global"])):
                if i % 4 == 0:
                    cv2.line(vis, tuple(np.round(rp).astype(int)), tuple(np.round(tp).astype(int)),
                             (255, 120, 0), 1)
                    cv2.circle(vis, tuple(np.round(rp).astype(int)), 2, (0, 200, 255), -1)
            cv2.imwrite(str(out / f"contour_{ds}_{row['uid']}.png"), vis)

    df = pd.DataFrame(stats)
    df.to_csv(out / "contour_stats.csv", index=False)
    if len(df):
        print(df.groupby("dataset")[["rough_dice", "rough_iou", "pt_err", "window"]].mean().to_string())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
