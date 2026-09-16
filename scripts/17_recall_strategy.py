"""漏检修复的最终候选方案评估 (检测命中目标上的端到端轮廓质量)。

对比若干"初始化 + 精细化"组合, 以及多候选打分选择 (MVP: 用边界梯度/内部方差打分)。
    PYTHONPATH=src ./.conda/bin/python scripts/17_recall_strategy.py
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
from cm.config import REPORTS, SPLITS, UNIFIED  # noqa: E402
from cm.rough import box_rough_contour, otsu_adaptive_rough_contour  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ev", str(Path(__file__).resolve().parent / "07_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def edge_score(img: np.ndarray, mask: np.ndarray) -> float:
    """候选质量打分: 边界平均梯度 - 内部灰度标准差 (越大越好)。"""
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 20:
        return -1e9
    band = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    e = float(g[band > 0].mean()) if (band > 0).any() else 0.0
    return e - 0.5 * float(img[m > 0].std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refiner", default="runs/refiner_v3_A5_busi-ddti-tn3k/best.pt")
    ap.add_argument("--det", default="runs/det_yolo26n_sub/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张 (调试)")
    ap.add_argument("--out", default=str(REPORTS / "recall_strategy"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, rcfg, _ = ev.load_refiner(Path(args.refiner), device)
    det = __import__("ultralytics").YOLO(args.det)

    sub = pd.read_csv("data/yolo_subset/_subset_splits.csv")
    sub = sub[sub["split"] == "test"]
    sp = pd.read_csv(SPLITS / "splits.csv").merge(sub[["uid"]], on="uid")
    if args.limit:
        sp = sp.head(args.limit)

    variants = ["otsu", "otsu+ref", "box", "box+ref", "mvp", "mvp+ref"]
    per_target: list[dict] = []
    rec: dict[str, list] = {v: [] for v in variants}
    cnt = dict(gt=0, det_hit=0)

    for n, (_, r) in enumerate(sp.iterrows(), 1):
        img = cv2.imread(str(UNIFIED / r["image"]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / r["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        H, W = img.shape[:2]
        m = (gt > 127).astype(np.uint8)
        ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        comps = [((labels == i).astype(np.uint8), i) for i in range(1, ncomp)
                 if stats[i, cv2.CC_STAT_AREA] >= 50]
        if not comps:
            continue
        res = det.predict(str(UNIFIED / r["image"]), conf=args.conf, imgsz=args.imgsz,
                          verbose=False, device=str(device), max_det=30)[0]
        boxes = ([tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
                 if res.boxes is not None and len(res.boxes) else [])
        for comp, lab in comps:
            cnt["gt"] += 1
            gb = M.mask_bbox(comp)
            cand = [b for b in boxes if M.bbox_iou(tuple(map(int, b)), gb) >= 0.5]
            if not cand:
                continue
            cnt["det_hit"] += 1
            b = cand[0]

            def mask_of(pts):
                return G.contour_to_mask(pts, (H, W))

            cands: dict[str, np.ndarray] = {}
            rr = otsu_adaptive_rough_contour(img, b, polarity="dark")
            otsu_pts = None
            if rr.ok and len(rr.contour) >= 3:
                otsu_pts = G.resample_closed(G.ensure_ccw(rr.contour), rcfg.n_points)
                cands["otsu"] = mask_of(G.resample_closed(otsu_pts, 256))
            bb = box_rough_contour(img, b)
            box_pts = G.resample_closed(G.ensure_ccw(bb.contour), rcfg.n_points)
            cands["box"] = mask_of(G.resample_closed(box_pts, 256))

            # 精细化
            if otsu_pts is not None:
                p_ref = ev.refine_contour(model, rcfg, img, b, otsu_pts, device)
                cands["otsu+ref"] = mask_of(p_ref)
            p_box = ev.refine_contour(model, rcfg, img, b, box_pts, device)
            cands["box+ref"] = mask_of(p_box)
            # MVP: 多候选打分选择 (在 {otsu, otsu+ref, box, box+ref} 中挑)
            pool = {k: v for k, v in cands.items() if k in ("otsu", "otsu+ref", "box", "box+ref")}
            best = max(pool, key=lambda k: edge_score(img, pool[k]))
            cands["mvp"] = pool[best]
            # MVP + 对选中候选的初始轮廓再过一次 refiner
            init_map = {"otsu": otsu_pts, "otsu+ref": otsu_pts, "box": box_pts, "box+ref": box_pts}
            if init_map[best] is not None:
                p2 = ev.refine_contour(model, rcfg, img, b, init_map[best], device)
                cands["mvp+ref"] = mask_of(p2)

            for v in variants:
                if v in cands:
                    rec[v].append(dict(uid=r["uid"], dataset=r["dataset"],
                                       dice=M.dice(cands[v], comp), iou=M.iou(cands[v], comp)))
            per_target.append(dict(uid=r["uid"], dataset=r["dataset"],
                                   **{f"{v}_iou": (M.iou(cands[v], comp) if v in cands else np.nan)
                                      for v in variants}))
        if n % 50 == 0:
            print(f"  {n}/{len(sp)}", flush=True)

    rows = []
    for v in variants:
        if not rec[v]:
            continue
        d = pd.DataFrame(rec[v])
        rows.append(dict(variant=v, n=len(d), hitrate_iou50=float((d.iou >= 0.5).mean()),
                         dice=d["dice"].mean(), iou=d["iou"].mean(),
                         dice_matched=d.loc[d.iou >= 0.5, "dice"].mean() if (d.iou >= 0.5).any() else np.nan))
    df = pd.DataFrame(rows).sort_values("hitrate_iou50", ascending=False)
    df.to_csv(out / "strategy_compare.csv", index=False)
    pt = pd.DataFrame(per_target)
    pt.to_csv(out / "per_target.csv", index=False)
    if len(pt):
        for combo in [("otsu+ref", "box+ref"), ("otsu+ref", "box"), ("otsu", "box+ref")]:
            iou = pt[[f"{c}_iou" for c in combo]]
            best = iou.max(axis=1)
            print(f"  oracle({'+'.join(combo)}): hitrate={float((best>=0.5).mean()):.4f} "
                  f"iou={float(best.mean()):.4f}")
    print(f"\nGT={cnt['gt']} 检测命中={cnt['det_hit']}")
    print(df.round(4).to_string(index=False))
    print(f"\n-> {out/'strategy_compare.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
