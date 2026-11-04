"""Det-UNet scheme: Python validation (YOLO26n-det -> ROI -> MK-UNet).

Pre-processing (letterbox / ROI crop / square-pad) comes from ``common.roi`` and
is byte-for-byte the same geometry as the C++ ``det_unet_test`` pipeline.

Usage::

    python validate.py --det ../runs/det_yolo26n/weights/best.pt \
        --mkunet ../runs/mkunet_yolo26n/best.pt --split test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    ADE20K_CLASS_NAMES,
    ADE20K_SPLITS,
    ADE20K_UNIFIED,
    CLASS_NAMES,
    MKUNET_ROI_SIZE,
    SPLITS,
    UNIFIED,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU_NMS,
)
from common.eval import Prediction, evaluate_split, load_split  # noqa: E402
from common.mkunet import build_mkunet  # noqa: E402
from common.roi import (  # noqa: E402
    crop_roi,
    letterbox,
    mask_to_contour,
    paste_roi_mask,
    resize_roi,
    unletterbox_boxes,
)


def load_mkunet(ckpt_path: Path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    variant = ck.get("variant", "MK_UNet")
    roi_size = int(ck.get("roi_size", MKUNET_ROI_SIZE))
    model = build_mkunet(variant, num_classes=1, in_channels=1).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, roi_size


@torch.no_grad()
def run_mkunet(model, rois: list[np.ndarray], device, thr: float) -> list[np.ndarray]:
    if not rois:
        return []
    batch = torch.from_numpy(np.stack([r.astype(np.float32) / 255.0 for r in rois])[:, None]).to(
        device
    )
    logit = model(batch)
    pm = torch.sigmoid(logit.float()) > thr
    pm = pm[:, 0].cpu().numpy().astype(np.uint8)
    return [pm[i] for i in range(pm.shape[0])]


def make_predict(det_model, mk_model, roi_size, args, device):
    def predict(img: np.ndarray, _row) -> tuple[list[Prediction], dict[str, float]]:
        h, w = img.shape[:2]
        preds: list[Prediction] = []

        t_det = time.perf_counter()
        canvas, ratio, pad = letterbox(img, (args.imgsz, args.imgsz))
        res = det_model.predict(
            canvas, conf=args.conf, imgsz=args.imgsz, verbose=False, device=device, iou=args.iou
        )
        rr = res[0]
        if rr.boxes is None or not len(rr.boxes):
            return preds, {"detection": (time.perf_counter() - t_det) * 1000.0}
        boxes = unletterbox_boxes(rr.boxes.xyxy.cpu().numpy(), (h, w), ratio, pad)
        classes = rr.boxes.cls.cpu().numpy().astype(int).tolist()
        scores = rr.boxes.conf.cpu().numpy().tolist()
        det_ms = (time.perf_counter() - t_det) * 1000.0

        t_roi = time.perf_counter()
        rois, metas = [], []
        for k, b in enumerate(boxes):
            bw, bh = b[2] - b[0], b[3] - b[1]
            eb = (
                b[0] - bw * args.pad_ratio,
                b[1] - bh * args.pad_ratio,
                b[2] + bw * args.pad_ratio,
                b[3] + bh * args.pad_ratio,
            )
            roi, rc = crop_roi(img, eb)
            if roi is None or rc is None:
                continue
            x1, y1, x2, y2 = rc
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            rois.append(resize_roi(roi, roi_size, square_pad=True))
            metas.append(
                (
                    rc,
                    (y2 - y1, x2 - x1),
                    classes[k] if k < len(classes) else 0,
                    scores[k] if k < len(scores) else 0.0,
                )
            )
        roi_ms = (time.perf_counter() - t_roi) * 1000.0

        t_seg = time.perf_counter()
        seg_masks = run_mkunet(mk_model, rois, device, args.thr)
        seg_ms = (time.perf_counter() - t_seg) * 1000.0

        t_post = time.perf_counter()
        for k, rm in enumerate(seg_masks):
            rc, roi_hw, cls, score = metas[k]
            full = paste_roi_mask(rm, roi_hw, roi_size, rc, (h, w), square_pad=True)
            ct = mask_to_contour(full)
            if ct is None:
                continue
            preds.append(Prediction(mask=full, class_id=cls, score=score, contour=ct))
        post_ms = (time.perf_counter() - t_post) * 1000.0
        return preds, {"detection": det_ms, "roi": roi_ms, "mkunet": seg_ms, "post": post_ms}

    return predict


def main() -> int:
    ap = argparse.ArgumentParser(description="Det-UNet Python validation")
    ap.add_argument("--dataset", choices=["ultrasound", "ade20k"], default="ultrasound")
    ap.add_argument("--det", required=True, help="YOLO26n detector weights")
    ap.add_argument("--mkunet", required=True, help="MK-UNet weights")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits-csv", default=None)
    ap.add_argument("--unified", default=None)
    ap.add_argument("--gt-instances", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--iou", type=float, default=YOLO_IOU_NMS)
    ap.add_argument("--roi-size", type=int, default=MKUNET_ROI_SIZE)
    ap.add_argument("--pad-ratio", type=float, default=0.15)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-vis", type=int, default=20)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ade = args.dataset == "ade20k"
    splits_csv = (
        Path(args.splits_csv)
        if args.splits_csv
        else (ADE20K_SPLITS / "splits.csv" if ade else SPLITS / "splits.csv")
    )
    unified = Path(args.unified) if args.unified else (ADE20K_UNIFIED if ade else UNIFIED)
    gt_instances = (
        Path(args.gt_instances)
        if args.gt_instances
        else (ADE20K_UNIFIED / "instances" if ade else None)
    )
    class_names = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes
        else (ADE20K_CLASS_NAMES if ade else CLASS_NAMES)
    )
    out_dir = (
        Path(args.out)
        if args.out
        else (Path(__file__).resolve().parents[1] / "reports" / f"{args.dataset}_det_unet_py")
    )

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    from ultralytics import YOLO

    det_model = YOLO(args.det)
    mk_model, roi_size = load_mkunet(Path(args.mkunet), device)
    df = load_split(args.split, splits_csv, max_images=args.max_images)
    print(
        f"[cfg ] scheme=det-unet dataset={args.dataset} split={args.split} "
        f"imgsz={args.imgsz} roi={roi_size} images={len(df)}"
    )

    evaluate_split(
        make_predict(det_model, mk_model, roi_size, args, device),
        df,
        scheme="Det-UNet",
        split=args.split,
        unified=unified,
        gt_instances_dir=gt_instances,
        out_dir=out_dir,
        class_names=class_names,
        save_vis=args.save_vis,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
