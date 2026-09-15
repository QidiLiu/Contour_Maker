"""生成对比可视化: 每个数据集若干张图, 叠加

  绿 = GT 轮廓, 橙 = Otsu 粗糙轮廓, 蓝 = contour-refiner 精细化轮廓, 品红 = YOLO26n-seg

    PYTHONPATH=src ./.conda/bin/python scripts/10_make_vis.py --split test --n 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm import geometry as G  # noqa: E402
from cm import metrics as M  # noqa: E402
from cm import roi as R  # noqa: E402
from cm.config import REPORTS, RUNS, SPLITS, UNIFIED  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ev", str(Path(__file__).resolve().parent / "07_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refiner", default="runs/refiner_A5/best.pt")
    ap.add_argument("--det", default="runs/det_yolo26n_sub/weights/best.pt")
    ap.add_argument("--seg", default="runs/seg_yolo26n_sub/weights/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4, help="每个数据集生成多少张")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--out", default=str(REPORTS / "vis_compare"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from cm.config import RefinerConfig
    rcfg = RefinerConfig()
    refiner = None
    if Path(args.refiner).exists():
        refiner, rcfg, _ = ev.load_refiner(Path(args.refiner), device)
        print(f"[vis] refiner={args.refiner} device={device}")
    else:
        print(f"[vis] 未找到 {args.refiner}, 只画 GT/Otsu/YOLO-seg")
    det = seg = None
    from ultralytics import YOLO
    if Path(args.det).exists():
        det = YOLO(args.det)
    if Path(args.seg).exists():
        seg = YOLO(args.seg)

    sp = pd.read_csv(SPLITS / "splits.csv")
    sp = sp[sp["split"] == args.split]
    stats = []
    for ds, grp in sp.groupby("dataset"):
        done = 0
        for _, row in grp.iterrows():
            if done >= args.n:
                break
            img_path = UNIFIED / row["image"]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(str(UNIFIED / row["mask"]), cv2.IMREAD_GRAYSCALE)
            if img is None or gt is None:
                continue
            gts = ev.gt_targets(gt)
            if not gts:
                continue
            H, W = img.shape[:2]
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for g in gts:
                cv2.polylines(vis, [np.round(g["contour"]).astype(np.int32)], True, (0, 255, 0), 2)
            # 方案 B
            if seg is not None:
                _, polys = ev.boxes_from_yolo(seg, img_path, args.conf, str(device), args.imgsz)
                for poly in polys:
                    if poly is not None and len(poly) >= 3:
                        cv2.polylines(vis, [np.round(poly).astype(np.int32)], True, (255, 0, 255), 2)
            # 方案 A: YOLO 框 -> Otsu -> refiner
            boxes = [g["bbox"] for g in gts]
            if det is not None:
                dboxes, _ = ev.boxes_from_yolo(det, img_path, args.conf, str(device), args.imgsz)
                if dboxes:
                    boxes = dboxes
            for b in boxes:
                rr = R.otsu_adaptive_rough_contour(img, b, polarity="dark")
                if not rr.ok or len(rr.contour) < 3:
                    continue
                rough = G.resample_closed(G.ensure_ccw(rr.contour), rcfg.n_points)
                cv2.polylines(vis, [np.round(rough).astype(np.int32)], True, (0, 165, 255), 1)
                if refiner is not None:
                    pts = ev.refine_contour(refiner, rcfg, img, b, rough, device)
                    cv2.polylines(vis, [np.round(pts).astype(np.int32)], True, (255, 0, 0), 2)
                # 指标
                pm = G.contour_to_mask(pts if refiner is not None else rough, (H, W))
                gm = G.contour_to_mask(g["contour"], (H, W))
                stats.append(dict(dataset=ds, uid=row["uid"],
                                  dice_refined=M.dice(pm, gm) if refiner is not None else np.nan,
                                  dice_rough=M.dice(G.contour_to_mask(rough, (H, W)), gm)))
            cv2.putText(vis, f"{ds} | G=GT O=Otsu B=refiner M=seg", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imwrite(str(out / f"{ds}_{row['uid']}.png"), vis)
            done += 1
        print(f"[vis] {ds}: {done} 张")

    pd.DataFrame(stats).to_csv(out / "vis_stats.csv", index=False)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
