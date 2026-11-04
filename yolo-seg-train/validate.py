"""YOLO-Seg scheme: Python validation (YOLO26n-seg, end-to-end).

The evaluation protocol is shared with the C++ ``yolo_seg_test`` binary via
``common.eval``; only the model call is scheme-specific.

Usage::

    python validate.py --seg ../runs/seg_yolo26n/weights/best.pt --split test
    python validate.py --dataset ade20k --seg ../runs/yolo_seg_ade20k/weights/best.pt \
        --split test --out ../reports/ade20k_yolo_seg_py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    ADE20K_CLASS_NAMES,
    ADE20K_SPLITS,
    ADE20K_UNIFIED,
    CLASS_NAMES,
    SPLITS,
    UNIFIED,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_IOU_NMS,
)
from common.eval import Prediction, evaluate_split, load_split  # noqa: E402


def make_predict(model, args, device):
    def predict(img: np.ndarray, _row) -> tuple[list[Prediction], dict[str, float]]:
        h, w = img.shape[:2]
        img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t0 = time.perf_counter()
        # Hand the original image to ultralytics so it performs its own
        # letterbox/inverse mapping.  retina_masks=True returns masks in the
        # coordinate system of the image that was fed in.
        res = model.predict(
            img3,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
            device=device,
            iou=args.iou,
            retina_masks=True,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        rr = res[0]
        preds: list[Prediction] = []
        if rr.masks is not None and rr.boxes is not None and len(rr.boxes):
            cls_list = rr.boxes.cls.cpu().numpy().astype(int).tolist()
            scores = rr.boxes.conf.cpu().numpy().tolist()
            for k, poly in enumerate(rr.masks.xy):
                if poly is None or len(poly) < 3:
                    continue
                mk = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mk, [np.round(np.asarray(poly)).astype(np.int32)], 1)
                if mk.sum() == 0:
                    continue
                preds.append(
                    Prediction(
                        mask=mk,
                        class_id=cls_list[k] if k < len(cls_list) else 0,
                        score=scores[k] if k < len(scores) else 0.0,
                        contour=np.asarray(poly, np.float32),
                    )
                )
        return preds, {"inference": infer_ms}

    return predict


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO-Seg Python validation")
    ap.add_argument("--dataset", choices=["ultrasound", "ade20k"], default="ultrasound")
    ap.add_argument("--seg", required=True, help="YOLO26n-seg weights")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits-csv", default=None)
    ap.add_argument("--unified", default=None)
    ap.add_argument("--gt-instances", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--iou", type=float, default=YOLO_IOU_NMS)
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
        else (Path(__file__).resolve().parents[1] / "reports" / f"{args.dataset}_yolo_seg_py")
    )

    from ultralytics import YOLO

    model = YOLO(args.seg)
    device = 0 if args.device == "cuda" else args.device
    df = load_split(args.split, splits_csv, max_images=args.max_images)
    print(
        f"[cfg ] scheme=yolo-seg dataset={args.dataset} split={args.split} "
        f"imgsz={args.imgsz} conf={args.conf} images={len(df)}"
    )

    evaluate_split(
        make_predict(model, args, device),
        df,
        scheme="YOLO-Seg",
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
