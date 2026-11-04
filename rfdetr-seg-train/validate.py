"""RF-DETR-Seg scheme: Python validation.

Usage::

    python validate.py --weights ../runs/rfdetr_seg_ultrasound/checkpoint_best_total.pth \
        --split test
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
    RFDETR_RESOLUTION,
    SPLITS,
    UNIFIED,
    YOLO_CONF,
)
from common.eval import Prediction, evaluate_split, load_split  # noqa: E402
from common.roi import mask_to_contour  # noqa: E402


def make_predict(model, args):
    def predict(img: np.ndarray, _row) -> tuple[list[Prediction], dict[str, float]]:
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        t0 = time.perf_counter()
        det = model.predict(rgb, threshold=args.conf)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        preds: list[Prediction] = []
        if det is None or len(det) == 0:
            return preds, {"inference": infer_ms}
        masks = det.mask
        classes = det.class_id
        scores = det.confidence
        for k in range(len(det)):
            mk = (np.asarray(masks[k]) > 0).astype(np.uint8)
            if mk.shape[:2] != (h, w):
                mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
            if mk.sum() == 0:
                continue
            ct = mask_to_contour(mk)
            if ct is None:
                continue
            preds.append(
                Prediction(mask=mk, class_id=int(classes[k]), score=float(scores[k]), contour=ct)
            )
        return preds, {"inference": infer_ms}

    return predict


def main() -> int:
    ap = argparse.ArgumentParser(description="RF-DETR-Seg Python validation")
    ap.add_argument("--dataset", choices=["ultrasound", "ade20k"], default="ultrasound")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits-csv", default=None)
    ap.add_argument("--unified", default=None)
    ap.add_argument("--gt-instances", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--resolution", type=int, default=RFDETR_RESOLUTION)
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
        else (Path(__file__).resolve().parents[1] / "reports" / f"{args.dataset}_rfdetr_seg_py")
    )

    from rfdetr import RFDETRSegNano

    model = RFDETRSegNano(pretrain_weights=args.weights, resolution=args.resolution)
    df = load_split(args.split, splits_csv, max_images=args.max_images)
    print(
        f"[cfg ] scheme=rfdetr-seg dataset={args.dataset} split={args.split} "
        f"conf={args.conf} images={len(df)}"
    )

    evaluate_split(
        make_predict(model, args),
        df,
        scheme="RF-DETR-Seg",
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
