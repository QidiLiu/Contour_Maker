"""生成失败案例与成功案例对比图 (按 refiner 相对粗糙轮廓的 Dice 变化排序)。

    PYTHONPATH=src ./.conda/bin/python scripts/11_failure_gallery.py --n 6
"""
from __future__ import annotations

import argparse
import importlib.util
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
from cm.config import REPORTS, SPLITS, UNIFIED  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ev", str(Path(__file__).resolve().parent / "07_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refiner", default="runs/refiner_A5/best.pt")
    ap.add_argument("--det", default="runs/det_yolo26n_sub/weights/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=str(REPORTS / "failure_gallery"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refiner, rcfg, _ = ev.load_refiner(Path(args.refiner), device)
    det = None
    if Path(args.det).exists():
        from ultralytics import YOLO
        det = YOLO(args.det)

    sp = pd.read_csv(SPLITS / "splits.csv")
    sp = sp[sp["split"] == args.split]
    rows = []
    for _, row in sp.iterrows():
        img_path = UNIFIED / row["image"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        gts = ev.gt_targets(gt)
        if not gts:
            continue
        H, W = img.shape[:2]
        boxes = [g["bbox"] for g in gts]
        if det is not None:
            db, _ = ev.boxes_from_yolo(det, img_path, args.conf, str(device), args.imgsz)
            if db:
                boxes = db
        # 只取与 GT 匹配最好的框
        best = None
        for b in boxes:
            bb = M.mask_bbox(G.contour_to_mask(
                np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]), (H, W)))
            i = max((M.bbox_iou(bb, g["bbox"]) for g in gts), default=0.0)
            if best is None or i > best[0]:
                best = (i, b)
        if best is None or best[0] < 0.3:
            continue
        b = best[1]
        rr = R.otsu_adaptive_rough_contour(img, b, polarity="dark")
        if not rr.ok or len(rr.contour) < 3:
            continue
        rough = G.resample_closed(G.ensure_ccw(rr.contour), rcfg.n_points)
        pts = ev.refine_contour(refiner, rcfg, img, b, rough, device)
        g = max(gts, key=lambda gg: M.bbox_iou(M.mask_bbox(G.contour_to_mask(pts, (H, W))), gg["bbox"]))
        gm = G.contour_to_mask(g["contour"], (H, W))
        rm = G.contour_to_mask(rough, (H, W))
        pm = G.contour_to_mask(pts, (H, W))
        rows.append(dict(uid=row["uid"], dataset=row["dataset"],
                         dice_rough=M.dice(rm, gm), dice_refined=M.dice(pm, gm),
                         box_iou=best[0]))
    df = pd.DataFrame(rows).sort_values("dice_refined")
    df.to_csv(out / "cases.csv", index=False)
    if df.empty:
        print("没有可用案例")
        return 1

    picks = pd.concat([df.head(args.n // 2), df.tail(args.n - args.n // 2)])
    sub = sp.set_index("uid")
    for _, r in picks.iterrows():
        row = sub.loc[r["uid"]]
        img_path = UNIFIED / row["image"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / row["mask"]), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        gts = ev.gt_targets(gt)
        boxes = [g["bbox"] for g in gts]
        if det is not None:
            db, _ = ev.boxes_from_yolo(det, img_path, args.conf, str(device), args.imgsz)
            if db:
                boxes = db
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for g in gts:
            cv2.polylines(vis, [np.round(g["contour"]).astype(np.int32)], True, (0, 255, 0), 2)
        for b in boxes:
            rr = R.otsu_adaptive_rough_contour(img, b, polarity="dark")
            if not rr.ok or len(rr.contour) < 3:
                continue
            rough = G.resample_closed(G.ensure_ccw(rr.contour), rcfg.n_points)
            pts = ev.refine_contour(refiner, rcfg, img, b, rough, device)
            cv2.polylines(vis, [np.round(rough).astype(np.int32)], True, (0, 165, 255), 1)
            cv2.polylines(vis, [np.round(pts).astype(np.int32)], True, (255, 0, 0), 2)
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 0), 1)
        txt = f"{r['dataset']} rough={r['dice_rough']:.3f} refined={r['dice_refined']:.3f}"
        cv2.putText(vis, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"{r['dataset']}_{r['uid']}_d{r['dice_refined']:.2f}.png"), vis)
    print(df.describe().to_string())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
