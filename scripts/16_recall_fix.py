"""漏检修复方案对比 (检测层面, 目标级 IoU>=0.5 全局贪心匹配)。

变体:
  det@T            YOLO26n 检测框, 置信度阈值 T
  seg@T            YOLO26n-seg 的框, 置信度阈值 T
  pool@T           检测框 + 分割框 候选池化 (按置信度 NMS)
  det+flip@T       检测框 + 水平翻转 TTA 框 池化
  det+multiscale   检测框 + 多尺度 (512/768) 池化

    PYTHONPATH=src ./.conda/bin/python scripts/16_recall_fix.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm import metrics as M  # noqa: E402
from cm.config import REPORTS, SPLITS, UNIFIED  # noqa: E402


def nms(cands: list[tuple[tuple, float]], thr: float = 0.55):
    keep = []
    for b, c in sorted(cands, key=lambda t: -t[1]):
        if all(M.bbox_iou(tuple(map(int, b)), tuple(map(int, k[0]))) < thr for k in keep):
            keep.append((b, c))
    return keep


def global_prf(preds: list[tuple[str, tuple]], gts: list[tuple[str, tuple]]) -> dict:
    """全局贪心匹配: preds/gts 为 (uid, box) 列表。"""
    if not preds or not gts:
        return dict(tp=0, fp=len(preds), fn=len(gts), precision=0.0, recall=0.0, f1=0.0)
    iou = np.zeros((len(preds), len(gts)), np.float32)
    for i, (_, pb) in enumerate(preds):
        for j, (_, gb) in enumerate(gts):
            iou[i, j] = M.bbox_iou(tuple(map(int, pb)), tuple(map(int, gb)))
    used_p, used_g, tp = set(), set(), 0
    for i, j in np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]:
        if i in used_p or j in used_g or iou[i, j] < 0.5:
            continue
        used_p.add(int(i)); used_g.add(int(j)); tp += 1
    P = tp / len(preds); R = tp / len(gts)
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(tp=tp, fp=len(preds) - tp, fn=len(gts) - tp,
                precision=P, recall=R, f1=F)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/det_yolo26n_sub/weights/best.pt")
    ap.add_argument("--seg", default="runs/seg_yolo26n_sub/weights/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--out", default=str(REPORTS / "recall_fix.csv"))
    args = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(args.det)
    seg = YOLO(args.seg)

    sub = pd.read_csv("data/yolo_subset/_subset_splits.csv")
    sub = sub[sub["split"] == args.split]
    sp = pd.read_csv(SPLITS / "splits.csv").merge(sub[["uid"]], on="uid")

    store: dict[str, list[tuple[str, tuple]]] = {}
    gts: list[tuple[str, tuple]] = []
    for n, (_, r) in enumerate(sp.iterrows(), 1):
        img = cv2.imread(str(UNIFIED / r["image"]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / r["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        m = (gt > 127).astype(np.uint8)
        ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for lab in range(1, ncomp):
            if stats[lab, cv2.CC_STAT_AREA] < 50:
                continue
            bb = M.mask_bbox((labels == lab).astype(np.uint8))
            if bb:
                gts.append((r["uid"], bb))
        p = str(UNIFIED / r["image"])
        r_det = det.predict(p, conf=0.001, imgsz=args.imgsz, verbose=False, max_det=50)[0]
        r_seg = seg.predict(p, conf=0.001, imgsz=args.imgsz, verbose=False, max_det=50)[0]
        r_ms = det.predict(p, conf=0.001, imgsz=int(args.imgsz * 1.5), verbose=False, max_det=50)[0]
        r_fl = det.predict(np.fliplr(img), conf=0.001, imgsz=args.imgsz, verbose=False, max_det=50)[0]
        W = img.shape[1]

        def bx(res):
            out = []
            if res.boxes is not None and len(res.boxes):
                for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                    out.append((tuple(float(v) for v in b), float(c)))
            return out

        d_b, s_b, ms_b = bx(r_det), bx(r_seg), bx(r_ms)
        f_b = [((W - b[2], b[1], W - b[0], b[3]), c) for b, c in bx(r_fl)]

        for T in (0.25, 0.10, 0.05, 0.01):
            store.setdefault(f"det@{T}", []).extend((r["uid"], b) for b, c in d_b if c >= T)
            store.setdefault(f"seg@{T}", []).extend((r["uid"], b) for b, c in s_b if c >= T)
            store.setdefault(f"pool@det+seg@{T}", []).extend(
                (r["uid"], b) for b, c in nms([x for x in d_b + s_b if x[1] >= T]))
            store.setdefault(f"pool@det+flip@{T}", []).extend(
                (r["uid"], b) for b, c in nms([x for x in d_b + f_b if x[1] >= T]))
            store.setdefault(f"pool@det+multiscale@{T}", []).extend(
                (r["uid"], b) for b, c in nms([x for x in d_b + ms_b if x[1] >= T]))
        if n % 50 == 0:
            print(f"  {n}/{len(sp)}", flush=True)

    rows = []
    for name, preds in store.items():
        res = global_prf(preds, gts)
        rows.append(dict(variant=name, n_pred=len(preds), n_gt=len(gts), **res))
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    df.to_csv(args.out, index=False)
    print(f"\nGT 目标数 = {len(gts)}")
    print(df.round(4).to_string(index=False))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
