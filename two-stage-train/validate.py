"""两阶段方案的 Python 侧验证：YOLO26n(det) → ROI → MK-UNet → 带类别的轮廓。

该脚本的口径与 C++ 侧 `two-stage-test/` 严格一致（预处理见 common/roi.py），
用于校验 Python 权重与 MNN 推理结果的一致性。

用法:
    python validate.py --det ../runs/det_yolo26n/weights/best.pt \
        --mkunet ../runs/mkunet_yolo26n/best.pt \
        --split test --out ../reports/two_stage
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    BOUNDARY_F1_TOL,
    CLASS_NAMES,
    IOU_MATCH_THRESHOLD,
    RUNS,
    SPLITS,
    UNIFIED,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU_NMS,
)
from common.metrics import evaluate_pair, greedy_match, prf  # noqa: E402
from common.roi import (  # noqa: E402
    mask_to_contour,
    paste_roi_mask,
    resize_roi,
    unletterbox_boxes,
)
from common.mkunet import build_mkunet  # noqa: E402


# ---------------------------------------------------------------- ROI 贴回
def square_pad_offsets(roi_hw, size: int):
    """与 common.roi.resize_roi(square_pad=True) 一致的居中偏移。"""
    rh, rw = roi_hw
    s = size / max(rh, rw)
    nw, nh = max(1, int(round(rw * s))), max(1, int(round(rh * s)))
    return (size - nh) // 2, (size - nw) // 2, nh, nw


def roi_mask_to_full(roi_mask, roi_hw, bbox, orig_shape, size: int,
                     square_pad: bool = True) -> np.ndarray:
    """把 size×size 的 ROI 掩码还原为原图尺寸掩码。"""
    m = roi_mask
    if square_pad:
        y0, x0, nh, nw = square_pad_offsets(roi_hw, size)
        m = m[y0:y0 + nh, x0:x0 + nw]
    m = cv2.resize(m.astype(np.uint8), (roi_hw[1], roi_hw[0]),
                   interpolation=cv2.INTER_NEAREST)
    out = np.zeros(orig_shape, np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    ih, iw = orig_shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(iw, x2), min(ih, y2)
    if cx2 > cx1 and cy2 > cy1:
        out[cy1:cy2, cx1:cx2] = m[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
    return out


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


# ---------------------------------------------------------------- 模型
def load_mkunet(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    variant = ck.get("variant", "MK_UNet")
    roi_size = int(ck.get("roi_size", 256))
    model = build_mkunet(variant, num_classes=1, in_channels=1).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, roi_size, variant


@torch.no_grad()
def run_mkunet(model, rois, device, roi_size):
    """rois: list of size×size uint8 -> list of size×size {0,1} masks."""
    if not rois:
        return []
    batch = torch.from_numpy(
        np.stack([r.astype(np.float32) / 255.0 for r in rois])[:, None]
    ).to(device)
    logit = model(batch)
    pm = (torch.sigmoid(logit.float()) > 0.5)
    pm = pm[:, 0].cpu().numpy().astype(np.uint8)
    return [pm[i] for i in range(pm.shape[0])]


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="两阶段方案验证")
    ap.add_argument("--det", required=True, help="YOLO26n 检测权重")
    ap.add_argument("--mkunet", required=True, help="MK-UNet 权重")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="输出目录，默认 reports/two_stage")
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--iou", type=float, default=YOLO_IOU_NMS)
    ap.add_argument("--pad-ratio", type=float, default=0.15)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-vis", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available()
                          else "cpu")
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "reports" / "two_stage")
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    det_model = YOLO(args.det)
    mk_model, roi_size, variant = load_mkunet(Path(args.mkunet), device)
    print(f"[cfg ] split={args.split} imgsz={args.imgsz} conf={args.conf} "
          f"roi={roi_size} mkunet={variant} device={device}")

    sp_path = SPLITS / "splits.csv"
    if not sp_path.exists():
        print(f"[error] 未找到 {sp_path}")
        return 1
    df = pd.read_csv(sp_path)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.max_images:
        df = df.head(args.max_images)
    print(f"[data] {len(df)} 张 (split={args.split})")

    records = []
    det_records = []
    vis_saved = 0
    t_det = t_seg = 0.0

    for n, (_, r) in enumerate(df.iterrows(), start=1):
        img = cv2.imread(str(UNIFIED / r["image"]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / r["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        h, w = img.shape[:2]

        # ---- GT 目标
        gts = _gt_targets(gt)

        # ---- 第一级: YOLO26n 检测
        t0 = time.perf_counter()
        canvas, ratio, pad = letterbox_simple(img, args.imgsz)
        res = det_model.predict(canvas, conf=args.conf, imgsz=args.imgsz,
                                verbose=False, device=device, iou=args.iou)
        rr = res[0]
        det_boxes, det_cls, det_conf = [], [], []
        if rr.boxes is not None and len(rr.boxes):
            lb = rr.boxes.xyxy.cpu().numpy()
            det_boxes = unletterbox_boxes(lb, (h, w), ratio, pad)
            det_cls = rr.boxes.cls.cpu().numpy().astype(int).tolist()
            det_conf = rr.boxes.conf.cpu().numpy().tolist()
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_det += time.perf_counter() - t0

        # ---- 第二级: ROI -> MK-UNet
        t0 = time.perf_counter()
        rois, metas = [], []
        for b in det_boxes:
            bw, bh = b[2] - b[0], b[3] - b[1]
            eb = (b[0] - bw * args.pad_ratio, b[1] - bh * args.pad_ratio,
                  b[2] + bw * args.pad_ratio, b[3] + bh * args.pad_ratio)
            x1 = max(0, int(np.floor(eb[0])))
            y1 = max(0, int(np.floor(eb[1])))
            x2 = min(w, int(np.ceil(eb[2])))
            y2 = min(h, int(np.ceil(eb[3])))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            roi = img[y1:y2, x1:x2]
            rois.append(resize_roi(roi, roi_size, square_pad=True))
            metas.append(((x1, y1, x2, y2), (y2 - y1, x2 - x1)))

        seg_masks = run_mkunet(mk_model, rois, device, roi_size)

        pred_masks, pred_cls, pred_contours = [], [], []
        for k, rm in enumerate(seg_masks):
            bbox, roi_hw = metas[k]
            full = roi_mask_to_full(rm, roi_hw, bbox, (h, w), roi_size)
            ct = mask_to_contour(full)
            if ct is None:
                continue
            pred_masks.append(full)
            pred_cls.append(det_cls[k] if k < len(det_cls) else 0)
            pred_contours.append(ct)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_seg += time.perf_counter() - t0

        # ---- 评估（mask IoU 贪心匹配）
        gt_masks = [g["mask"] for g in gts]
        pairs, n_miss, n_fp = greedy_match(pred_masks, gt_masks, IOU_MATCH_THRESHOLD)
        det_records.append(dict(uid=r["uid"], dataset=r["dataset"], split=r["split"],
                                n_gt=len(gts), n_pred=len(pred_masks),
                                n_miss=n_miss, n_fp=n_fp))
        for pi, gi, miou in pairs:
            rec = dict(uid=r["uid"], dataset=r["dataset"], split=r["split"],
                       modality=r["modality"], match_iou=miou,
                       pred_cls=pred_cls[pi], gt_cls=int(r["class_id"]),
                       cls_ok=int(pred_cls[pi] == int(r["class_id"])))
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
            cv2.imwrite(str(out_dir / f"vis_{r['dataset']}_{r['uid']}.png"), vis)
            vis_saved += 1

        if n % 200 == 0:
            print(f"  {n}/{len(df)} det={t_det:.1f}s seg={t_seg:.1f}s", flush=True)

    _summarize(records, det_records, out_dir, args, t_det, t_seg, len(df))
    return 0


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
        out.append(dict(mask=comp, contour=ct,
                        bbox=(float(stats[lab, cv2.CC_STAT_LEFT]),
                              float(stats[lab, cv2.CC_STAT_TOP]),
                              float(stats[lab, cv2.CC_STAT_LEFT] +
                                    stats[lab, cv2.CC_STAT_WIDTH]),
                              float(stats[lab, cv2.CC_STAT_TOP] +
                                    stats[lab, cv2.CC_STAT_HEIGHT]))))
    return out


def _summarize(records, det_records, out_dir, args, t_det, t_seg, n_img):
    pd.DataFrame(records).to_csv(out_dir / "per_target.csv", index=False)
    pd.DataFrame(det_records).to_csv(out_dir / "per_image.csv", index=False)

    print("\n" + "=" * 70)
    print("=== 两阶段方案 (YOLO26n-det + MK-UNet) ===")
    print("=" * 70)

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
    else:
        agg = {}

    if det_records:
        d = pd.DataFrame(det_records)
        n_gt, n_pred = int(d.n_gt.sum()), int(d.n_pred.sum())
        n_miss, n_fp = int(d.n_miss.sum()), int(d.n_fp.sum())
        tp = n_gt - n_miss
        pf = prf(tp, n_fp, n_miss)
        print("\n[检测环节]")
        print(f"  GT={n_gt} pred={n_pred} TP={tp} FP={n_fp} FN={n_miss}")
        print(f"  P={pf['precision']:.4f} R={pf['recall']:.4f} F1={pf['f1']:.4f}")
    else:
        pf = {}

    print("\n[速度]")
    ms = 1000.0 * (t_det + t_seg) / max(1, n_img)
    print(f"  检测       {1000*t_det/max(1,n_img):7.2f} ms/图")
    print(f"  ROI+MKUNet {1000*t_seg/max(1,n_img):7.2f} ms/图")
    print(f"  合计       {ms:7.2f} ms/图  ({1000/max(ms,1e-9):.1f} FPS)")

    summary = dict(config=vars(args), segmentation=agg, detection=pf,
                   speed=dict(det_ms=1000*t_det/max(1, n_img),
                              seg_ms=1000*t_seg/max(1, n_img),
                              total_ms=ms, fps=1000/max(ms, 1e-9)))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                      default=str))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())