"""Build the MK-UNet ROI cache used by the second stage of Det-UNet.

Each target is cropped with its detection box (or GT box), resized to the
MK-UNet input size and stored together with the binary ROI mask::

    <out>/<split>/<uid>_t<k>_v<v>.png       ROI image (grayscale)
    <out>/<split>/<uid>_t<k>_v<v>_mask.png  ROI mask
    <out>/index.csv                         uid, target, class, box, files, ...

The class of each target comes from the individual GT instance (sidecar) when
available, otherwise from the image row (ultrasound).

Usage::

    python build_roi_dataset.py --box-source det --det ../runs/det_yolo26n/weights/best.pt \
        --splits train --out ../data/roi_cache/train
    python build_roi_dataset.py --box-source gt --splits val --out ../data/roi_cache/val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    ADE20K_ROI_CACHE,
    ADE20K_SPLITS,
    ADE20K_UNIFIED,
    MIN_MASK_AREA,
    MKUNET_ROI_SIZE,
    ROI_CACHE,
    SPLITS,
    UNIFIED,
    YOLO_CONF,
    YOLO_IMGSZ,
)
from common.instances import gt_instances  # noqa: E402
from common.roi import crop_roi, resize_roi  # noqa: E402


def jitter_box(box, rng, strength: float):
    """Jitter the box centre and scale to mimic detector localisation noise."""
    if strength <= 0:
        return tuple(float(v) for v in box)
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cx += rng.normal(0, strength * bw / 2)
    cy += rng.normal(0, strength * bh / 2)
    sx = float(np.clip(1.0 + rng.normal(0, strength), 0.7, 1.4))
    sy = float(np.clip(1.0 + rng.normal(0, strength), 0.7, 1.4))
    return (cx - bw * sx / 2, cy - bh * sy / 2, cx + bw * sx / 2, cy + bh * sy / 2)


def expand_box(box, ratio: float):
    """Expand the box by ``ratio`` of its width/height to keep boundary context."""
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    return (x1 - bw * ratio, y1 - bh * ratio, x2 + bw * ratio, y2 + bh * ratio)


def det_boxes_for_image(model, img_path: Path, conf: float, imgsz: int, device: str):
    """Return detector boxes ``[(xyxy, class_id, score)]`` in original coordinates."""
    res = model.predict(str(img_path), conf=conf, imgsz=imgsz, verbose=False, device=device)
    r = res[0]
    if r.boxes is None or not len(r.boxes):
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    return [
        (tuple(float(v) for v in b), int(c), float(s))
        for b, c, s in zip(xyxy, cls, confs, strict=False)
    ]


def match_box_to_target(det_box, gt_boxes, min_iou: float = 0.3):
    """Return the GT target index with the highest IoU (>= ``min_iou``) or None."""
    best, best_iou = None, min_iou
    for i, gb in enumerate(gt_boxes):
        iou = _iou_xyxy(det_box, gb)
        if iou >= best_iou:
            best, best_iou = i, iou
    return best


def _iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the MK-UNet ROI cache")
    ap.add_argument("--dataset", choices=["ultrasound", "ade20k"], default="ultrasound")
    ap.add_argument("--det", default=None, help="detector weights (required for --box-source det)")
    ap.add_argument("--box-source", choices=["gt", "det"], default="gt")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--splits-csv", default=None)
    ap.add_argument("--unified", default=None)
    ap.add_argument("--gt-instances", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--roi-size", type=int, default=MKUNET_ROI_SIZE)
    ap.add_argument("--pad-ratio", type=float, default=0.15)
    ap.add_argument("--jitter", type=float, default=0.10)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--min-area", type=int, default=MIN_MASK_AREA)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    ade = args.dataset == "ade20k"
    splits_csv = (
        Path(args.splits_csv)
        if args.splits_csv
        else (ADE20K_SPLITS / "splits.csv" if ade else SPLITS / "splits.csv")
    )
    unified = Path(args.unified) if args.unified else (ADE20K_UNIFIED if ade else UNIFIED)
    gt_instances_dir = (
        Path(args.gt_instances)
        if args.gt_instances
        else (ADE20K_UNIFIED / "instances" if ade else None)
    )
    out = (
        Path(args.out)
        if args.out
        else (ADE20K_ROI_CACHE if ade else ROI_CACHE)
        / (args.name or f"{'_'.join(args.splits)}_{args.box_source}")
    )
    out.mkdir(parents=True, exist_ok=True)
    for s in args.splits:
        (out / s).mkdir(parents=True, exist_ok=True)

    if not splits_csv.exists():
        print(f"[error] split file not found: {splits_csv}")
        return 1
    df = pd.read_csv(splits_csv)
    df = df[df["split"].isin(args.splits)]
    if args.max_images:
        df = df.head(args.max_images)

    det_model = None
    if args.box_source == "det":
        if not args.det:
            print("[error] --box-source det requires --det")
            return 1
        from ultralytics import YOLO

        det_model = YOLO(args.det)
        device = 0 if args.device == "cuda" else args.device
    else:
        device = "cpu"

    rng = np.random.default_rng(args.seed)
    rows = []
    n_img = n_ok = n_fail_det = n_fail_geom = 0

    for n, (_, r) in enumerate(df.iterrows(), start=1):
        img_p = unified / r["image"]
        msk_p = unified / r["mask"]
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
        if img is None or msk is None:
            continue
        n_img += 1
        h, w = img.shape[:2]

        sidecar = None
        if gt_instances_dir is not None:
            sidecar = gt_instances_dir / f"{Path(str(r['mask'])).stem}.csv"
        targets = gt_instances(
            msk, sidecar=sidecar, default_class_id=int(r["class_id"]), min_area=args.min_area
        )
        if not targets:
            continue

        if args.box_source == "gt":
            boxes = [(i, t.bbox, t.class_id) for i, t in enumerate(targets)]
        else:
            dets = det_boxes_for_image(det_model, img_p, args.conf, args.imgsz, device)
            boxes = []
            for db, _dcls, _dconf in dets:
                gi = match_box_to_target(db, [t.bbox for t in targets])
                if gi is None:
                    n_fail_det += 1
                    continue
                boxes.append((gi, db, targets[gi].class_id))
        if not boxes:
            continue

        for gi, box, cls_id in boxes:
            comp = targets[gi].mask
            for v in range(args.variants):
                b = jitter_box(box, rng, args.jitter) if args.variants > 1 else box
                b = expand_box(b, args.pad_ratio)
                roi, rc = crop_roi(img, b)
                if roi is None or rc is None:
                    n_fail_geom += 1
                    continue
                x1, y1, x2, y2 = rc
                sub_mask = comp[y1:y2, x1:x2]
                if sub_mask.sum() < 10:
                    n_fail_geom += 1
                    continue

                roi_img = resize_roi(roi, args.roi_size, square_pad=True)
                roi_msk = resize_roi(
                    (sub_mask * 255).astype(np.uint8), args.roi_size, square_pad=True
                )
                key = f"{r['uid']}_t{gi}_v{v}"
                dst = out / str(r["split"])
                dst.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst / f"{key}.png"), roi_img)
                cv2.imwrite(str(dst / f"{key}_mask.png"), roi_msk)
                rows.append(
                    dict(
                        uid=r["uid"],
                        split=r["split"],
                        dataset=r["dataset"],
                        class_id=int(cls_id),
                        target=gi,
                        variant=v,
                        roi_h=int(y2 - y1),
                        roi_w=int(x2 - x1),
                        box_x1=x1,
                        box_y1=y1,
                        box_x2=x2,
                        box_y2=y2,
                        file=f"{key}.png",
                        mask_file=f"{key}_mask.png",
                    )
                )
                n_ok += 1

        if n % 500 == 0:
            print(f"  {n}/{len(df)} images={n_img} samples={n_ok}", flush=True)

    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False)
    print(
        f"\n[ok  ] images={n_img} samples={len(index)} "
        f"fail_det={n_fail_det} fail_geom={n_fail_geom}"
    )
    print(f"[out ] {out}")
    if len(index):
        print("\n=== samples per split ===")
        print(index["split"].value_counts().to_string())
        print("\n=== class distribution ===")
        print(index["class_id"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
