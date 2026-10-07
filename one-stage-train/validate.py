"""单阶段方案（YOLO26n-seg）的 Python 侧验证。

口径与 C++ 侧 `one-stage-test/` 严格一致，用于校验 Python 与 MNN 推理的一致性。

用法:
    python validate.py --seg ../runs/seg_yolo26n/weights/best.pt \
        --split test --out ../reports/one_stage
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    BOUNDARY_F1_TOL,
    CLASS_NAMES,
    IOU_MATCH_THRESHOLD,
    SPLITS,
    UNIFIED,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU_NMS,
)
from common.metrics import evaluate_pair, greedy_match, mask_to_contour, prf  # noqa: E402


def letterbox_simple(img, size):
    """等比缩放 + 居中填充，返回 (画布, ratio, (padx, pady))。"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padx, pady = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size), 114, np.uint8)
    canvas[pady:pady + nh, padx:padx + nw] = img_r
    return canvas, r, (padx, pady)


def _gt_targets(mask: np.ndarray, min_area: int = 50):
    m = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (labels == lab).astype(np.uint8)
        ct = mask_to_contour(comp)
        if ct is None:
            continue
        out.append(dict(mask=comp, contour=ct))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="单阶段方案验证")
    ap.add_argument("--seg", required=True, help="YOLO26n-seg 权重")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--iou", type=float, default=YOLO_IOU_NMS)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-vis", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "reports" / "one_stage")
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.seg)
    device = 0 if args.device == "cuda" else args.device

    sp_path = SPLITS / "splits.csv"
    if not sp_path.exists():
        print(f"[error] 未找到 {sp_path}")
        return 1
    df = pd.read_csv(sp_path)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.max_images:
        df = df.head(args.max_images)
    print(f"[cfg ] split={args.split} imgsz={args.imgsz} conf={args.conf} "
          f"classes={CLASS_NAMES}")
    print(f"[data] {len(df)} 张")

    records, det_records = [], []
    vis_saved = 0
    t_total = 0.0

    for n, (_, r) in enumerate(df.iterrows(), start=1):
        img = cv2.imread(str(UNIFIED / r.image), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / r.mask), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        h, w = img.shape[:2]
        gts = _gt_targets(gt)

        t0 = time.perf_counter()
        canvas, ratio, pad = letterbox_simple(img, args.imgsz)
        res = model.predict(canvas, conf=args.conf, imgsz=args.imgsz,
                            verbose=False, device=device, iou=args.iou,
                            retina_masks=True)
        rr = res[0]
        t_total += time.perf_counter() - t0

        pred_masks, pred_cls, pred_contours = [], [], []
        if rr.masks is not None and rr.boxes is not None and len(rr.boxes):
            cls_list = rr.boxes.cls.cpu().numpy().astype(int).tolist()
            for k, poly in enumerate(rr.masks.xy):
                if poly is None or len(poly) < 3:
                    continue
                mk = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mk, [np.round(np.asarray(poly)).astype(np.int32)], 1)
                if mk.sum() == 0:
                    continue
                pred_masks.append(mk)
                pred_cls.append(cls_list[k] if k < len(cls_list) else 0)
                pred_contours.append(np.asarray(poly, np.float32))

        gt_masks = [g["mask"] for g in gts]
        pairs, n_miss, n_fp = greedy_match(pred_masks, gt_masks, IOU_MATCH_THRESHOLD)
        det_records.append(dict(uid=r.uid, dataset=r.dataset, split=r.split,
                                n_gt=len(gts), n_pred=len(pred_masks),
                                n_miss=n_miss, n_fp=n_fp))
        for pi, gi, miou in pairs:
            rec = dict(uid=r.uid, dataset=r.dataset, split=r.split,
                       modality=r.modality, match_iou=miou,
                       pred_cls=pred_cls[pi], gt_cls=int(r.class_id),
                       cls_ok=int(pred_cls[pi] == int(r.class_id)))
            rec.update(evaluate_pair(pred_masks[pi], gt_masks[gi],
                                     BOUNDARY_F1_TOL))
            records.append(rec)

        if vis_saved < args.save_vis:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for g in gts:
                cv2.polylines(vis, [np.round(g["contour"]).astype(np.int32)],
                              True, (0, 255, 0), 1)
            for ct in pred_contours:
                cv2.polylines(vis, [np.round(ct).astype(np.int32)], True,
                              (255, 0, 255), 1)
            cv2.imwrite(str(out_dir / f"vis_{r.dataset}_{r.uid}.png"), vis)
            vis_saved += 1

        if n % 200 == 0:
            print(f"  {n}/{len(df)} {t_total:.1f}s", flush=True)

    _summarize(records, det_records, out_dir, args, t_total, len(df))
    return 0


def _summarize(records, det_records, out_dir, args, t_total, n_img):
    pd.DataFrame(records).to_csv(out_dir / "per_target.csv", index=False)
    pd.DataFrame(det_records).to_csv(out_dir / "per_image.csv", index=False)

    print("\n" + "=" * 70)
    print("=== 单阶段方案 (YOLO26n-seg) ===")
    print("=" * 70)

    agg = {}
    if records:
        s = pd.DataFrame(records)
        agg = s.agg(n=("dice", "size"), dice=("dice", "mean"),
                    iou=("iou", "mean"), hd95=("hd95", "mean"),
                    assd=("assd", "mean"), bf1=("bf1", "mean"),
                    area_err=("area_err", "mean")).to_dict()
        agg["fail_rate"] = float((s["dice"] < 0.5).mean())
        agg["cls_acc"] = float(s["cls_ok"].mean())
        print("\n[分割指标]")
        for k, v in agg.items():
            print(f"  {k:10s} {v}")
        print("\n[按数据集]")
        by = s.groupby("dataset").agg(
            n=("dice", "size"), dice=("dice", "mean"), iou=("iou", "mean"),
            hd95=("hd95", "mean"), bf1=("bf1", "mean")).round(4)
        print(by.to_string())

    pf = {}
    if det_records:
        d = pd.DataFrame(det_records)
        n_gt, n_miss, n_fp = int(d.n_gt.sum()), int(d.n_miss.sum()), int(d.n_fp.sum())
        tp = n_gt - n_miss
        pf = prf(tp, n_fp, n_miss)
        print("\n[检测环节]")
        print(f"  GT={n_gt} TP={tp} FP={n_fp} FN={n_miss}")
        print(f"  P={pf['precision']:.4f} R={pf['recall']:.4f} F1={pf['f1']:.4f}")

    ms = 1000.0 * t_total / max(1, n_img)
    print("\n[速度]")
    print(f"  端到端 {ms:7.2f} ms/图  ({1000/max(ms,1e-9):.1f} FPS)")

    (out_dir / "summary.json").write_text(json.dumps(
        dict(config=vars(args), segmentation=agg, detection=pf,
             speed=dict(total_ms=ms, fps=1000/max(ms, 1e-9))),
        indent=2, default=str))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())