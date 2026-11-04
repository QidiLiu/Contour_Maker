"""Export a unified manifest to COCO instance-segmentation format.

RF-DETR is trained from COCO annotations, so both datasets are exported to the
Roboflow directory layout ``train/ valid/ test/`` with one
``_annotations.coco.json`` per split.  Instance-id masks with a class sidecar
(ADE20K) and binary masks (ultrasound) are both supported.

Usage::

    python -m common.coco_export                     # ultrasound -> data/coco
    python -m common.coco_export --dataset ade20k    # -> data/ade20k/coco
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd

from .config import (
    ADE20K_CLASS_NAMES,
    ADE20K_COCO,
    ADE20K_SPLITS,
    ADE20K_UNIFIED,
    CLASS_NAMES,
    COCO,
    MIN_MASK_AREA,
    SPLITS,
    UNIFIED,
)
from .instances import gt_instances


def export_split(
    df: pd.DataFrame,
    split: str,
    out_root: Path,
    *,
    unified: Path,
    class_names: list[str],
    gt_instances_dir: Path | None,
    min_area: int = MIN_MASK_AREA,
) -> dict:
    split_name = {"val": "valid"}.get(split, split)
    dst = out_root / split_name
    dst.mkdir(parents=True, exist_ok=True)

    images, annotations, skipped = [], [], 0
    ann_id = 1
    for image_id, row in enumerate(df.itertuples(), start=1):
        img_p = unified / row.image
        msk_p = unified / row.mask
        mask = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
        if mask is None:
            skipped += 1
            continue
        h, w = mask.shape[:2]
        sidecar = None
        if gt_instances_dir is not None:
            sidecar = gt_instances_dir / f"{Path(str(row.mask)).stem}.csv"
        insts = gt_instances(
            mask, sidecar=sidecar, default_class_id=int(row.class_id), min_area=min_area
        )

        # Copy or symlink the image next to the JSON.
        name = f"{row.uid}.png"
        target = dst / name
        if not target.exists():
            try:
                target.symlink_to(img_p.resolve())
            except OSError:
                shutil.copy2(img_p, target)

        images.append(dict(id=image_id, file_name=name, width=w, height=h))
        for inst in insts:
            cnts, _ = cv2.findContours(inst.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea).reshape(-1, 2)
            if len(contour) < 3:
                continue
            x1, y1, x2, y2 = inst.bbox
            annotations.append(
                dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=int(inst.class_id),
                    bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    area=float(inst.area),
                    iscrowd=0,
                    segmentation=[contour.astype(float).reshape(-1).tolist()],
                )
            )
            ann_id += 1

    coco = dict(
        info=dict(description="Contour_Maker export", version="1.0"),
        licenses=[],
        images=images,
        annotations=annotations,
        categories=[dict(id=i, name=n, supercategory="none") for i, n in enumerate(class_names)],
    )
    (dst / "_annotations.coco.json").write_text(
        json.dumps(coco, ensure_ascii=False), encoding="utf-8"
    )
    return dict(split=split_name, images=len(images), annotations=len(annotations), skipped=skipped)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export COCO instance-segmentation datasets")
    ap.add_argument("--dataset", choices=["ultrasound", "ade20k"], default="ultrasound")
    ap.add_argument("--splits-csv", default=None)
    ap.add_argument("--unified", default=None)
    ap.add_argument("--gt-instances", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--min-area", type=int, default=MIN_MASK_AREA)
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
    out = Path(args.out) if args.out else (ADE20K_COCO if ade else COCO)
    class_names = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes
        else (ADE20K_CLASS_NAMES if ade else CLASS_NAMES)
    )

    if not splits_csv.exists():
        print(f"[error] split file not found: {splits_csv}")
        return 1
    df_all = pd.read_csv(splits_csv)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df = df_all[df_all["split"] == split]
        if df.empty:
            continue
        r = export_split(
            df,
            split,
            out,
            unified=unified,
            class_names=class_names,
            gt_instances_dir=gt_instances,
            min_area=args.min_area,
        )
        print(
            f"[{r['split']:5s}] images={r['images']} annotations={r['annotations']} "
            f"skip={r['skipped']}"
        )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
