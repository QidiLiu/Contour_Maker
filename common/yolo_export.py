"""Export a unified manifest to the YOLO detection/segmentation layout.

``det`` and ``seg`` share the same ``images/`` tree; the active label directory
is selected through a ``labels`` symlink.  Both the ultrasound dataset (binary
masks, one class per image) and the ADE20K subset (instance-id masks + class
sidecars) are supported.

Usage::

    python -m common.yolo_export --kind both
    python -m common.yolo_export --kind both --dataset ade20k
    python -m common.yolo_export --kind seg --unified data/unified \
        --splits-csv data/splits/splits.csv --out-seg data/yolo_seg
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import (
    ADE20K_CLASS_NAMES,
    ADE20K_SPLITS,
    ADE20K_UNIFIED,
    ADE20K_YOLO_DET,
    ADE20K_YOLO_SEG,
    CLASS_NAMES,
    MIN_MASK_AREA,
    SPLITS,
    UNIFIED,
    YOLO_DET,
    YOLO_SEG,
)
from .instances import read_sidecar
from .roi import mask_to_contour


def _components_with_area(
    mask: np.ndarray,
    *,
    sidecar: Path | None = None,
    default_class_id: int = 0,
    min_area: int = MIN_MASK_AREA,
):
    """Mask -> [(contour, area, bbox, class_id)].

    With a sidecar the mask is an instance-id map: every id is one instance
    (split into connected components if necessary) carrying the id's class.
    Without a sidecar the mask is binary and every component gets
    ``default_class_id``.
    """
    sid = read_sidecar(sidecar) if sidecar else {}
    out = []
    if sid:
        for value in np.unique(mask):
            if value == 0:
                continue
            cls = int(sid.get(int(value), default_class_id))
            comp = (mask == value).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(comp, 8)
            for lab in range(1, n):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                m = (labels == lab).astype(np.uint8)
                c = mask_to_contour(m)
                if c is None:
                    continue
                x, y, bw, bh = cv2.boundingRect(m)
                out.append((c, area, (x, y, x + bw, y + bh), cls))
        return out

    binary = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        m = (labels == lab).astype(np.uint8)
        c = mask_to_contour(m)
        if c is None:
            continue
        x, y, bw, bh = cv2.boundingRect(m)
        out.append((c, area, (x, y, x + bw, y + bh), int(default_class_id)))
    return out


def export_one(
    df: pd.DataFrame,
    kind: str,
    out: Path,
    *,
    unified: Path = UNIFIED,
    class_names: list[str] | None = None,
    gt_instances_dir: Path | None = None,
    min_area: int = MIN_MASK_AREA,
) -> dict:
    """Write one YOLO dataset (kind in {det, seg}) and return its statistics."""
    class_names = class_names or CLASS_NAMES
    labels_dir = out / ("labels_seg" if kind == "seg" else "labels_det")
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    n_img = n_tgt = n_skip = 0
    for row in df.itertuples():
        img_p = unified / row.image
        msk_p = unified / row.mask
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
        if img is None or msk is None:
            n_skip += 1
            continue
        if img.shape[:2] != msk.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]

        sidecar = None
        if gt_instances_dir is not None:
            sidecar = gt_instances_dir / f"{Path(str(row.mask)).stem}.csv"
        comps = _components_with_area(
            msk,
            sidecar=sidecar,
            default_class_id=int(row.class_id),
            min_area=min_area,
        )
        if not comps:
            n_skip += 1
            continue

        seg_lines, det_lines = [], []
        for contour, _area, (x1, y1, x2, y2), cls in comps:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            det_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # 128-point equal-arc-length polygon, normalised to [0, 1]
            poly = _resample_closed(contour, 128)
            p01 = np.clip(poly / np.array([w, h], np.float32), 0, 1)
            seg_lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in p01.reshape(-1)))
            n_tgt += 1

        split = row.split
        dst = out / "images" / split / f"{row.uid}.png"
        if not dst.exists():
            cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        lines = seg_lines if kind == "seg" else det_lines
        (labels_dir / split / f"{row.uid}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_img += 1

    yml = out / f"data_{kind}.yaml"
    yml.write_text(
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names)),
        encoding="utf-8",
    )

    # ultralytics looks for labels/ next to images/; point it at the active kind.
    link = out / "labels"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(labels_dir.name, target_is_directory=True)

    return dict(kind=kind, n_images=n_img, n_targets=n_tgt, n_skip=n_skip, out=str(out))


def _resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed contour to ``n`` equally spaced points (no duplicate end)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 3:
        return np.repeat(pts[:1], n, axis=0)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n, endpoint=False)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg) - 1)
    t = (targets - cum[idx]) / np.maximum(seg[idx], 1e-9)
    out = closed[idx] + t[:, None] * (closed[idx + 1] - closed[idx])
    return out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export YOLO datasets")
    ap.add_argument("--kind", choices=["det", "seg", "both"], default="both")
    ap.add_argument("--dataset", choices=["ultrasound", "ade20k"], default="ultrasound")
    ap.add_argument("--splits-csv", default=None)
    ap.add_argument("--unified", default=None)
    ap.add_argument(
        "--gt-instances", default=None, help="sidecar dir for instance-id masks (ADE20K)"
    )
    ap.add_argument("--out-det", default=None)
    ap.add_argument("--out-seg", default=None)
    ap.add_argument(
        "--classes",
        default=None,
        help="comma-separated class names; defaults to the dataset profile",
    )
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
    out_det = Path(args.out_det) if args.out_det else (ADE20K_YOLO_DET if ade else YOLO_DET)
    out_seg = Path(args.out_seg) if args.out_seg else (ADE20K_YOLO_SEG if ade else YOLO_SEG)
    class_names = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes
        else (ADE20K_CLASS_NAMES if ade else CLASS_NAMES)
    )

    if not splits_csv.exists():
        print(f"[error] split file not found: {splits_csv}; run common.splits first")
        return 1
    df = pd.read_csv(splits_csv)
    kinds = ["det", "seg"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        out = out_det if kind == "det" else out_seg
        r = export_one(
            df,
            kind,
            out,
            unified=unified,
            class_names=class_names,
            gt_instances_dir=gt_instances,
            min_area=args.min_area,
        )
        print(
            f"[{r['kind']:3s}] images={r['n_images']} targets={r['n_targets']} "
            f"skip={r['n_skip']} -> {r['out']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
