"""Ground-truth instance handling shared by every scheme.

Two mask conventions are supported:

* **binary masks** (the ultrasound dataset) -- every connected component of
  ``mask > 127`` is one instance and all instances share the image's
  ``class_id`` from ``splits.csv``.
* **instance-id masks** (the ADE20K subset) -- every non-zero pixel value is one
  instance id; the class of each id is stored in a sidecar CSV
  (``instances/<uid>.csv`` with columns ``instance_id,class_id``).  This keeps
  multi-class, multi-instance images representable without 16-bit PNGs.

The sidecar format is intentionally tiny and is read by both the Python and the
C++ evaluation code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import MASK_THRESHOLD, MIN_MASK_AREA


@dataclass
class GtInstance:
    """One ground-truth instance at the original image resolution."""

    mask: np.ndarray  # (H, W) uint8, values {0, 1}
    class_id: int
    area: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (exclusive)


def read_sidecar(path: str | Path) -> dict[int, int]:
    """Read ``instance_id,class_id`` pairs.  Missing file -> empty mapping."""
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[int, int] = {}
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["instance_id"])] = int(row["class_id"])
    return out


def write_sidecar(path: str | Path, instance_classes: dict[int, int]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "class_id"])
        for iid in sorted(instance_classes):
            w.writerow([iid, instance_classes[iid]])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def gt_instances(
    mask: np.ndarray,
    *,
    sidecar: str | Path | None = None,
    default_class_id: int = 0,
    min_area: int = MIN_MASK_AREA,
) -> list[GtInstance]:
    """Decode a GT mask into instances.

    When ``sidecar`` exists the mask is treated as an instance-id map.  Any
    instance id that splits into multiple connected components is split into
    separate instances (they share the id's class), because the metric matches
    connected components.
    """
    out: list[GtInstance] = []
    sid = read_sidecar(sidecar) if sidecar else {}

    if sid:
        for value in np.unique(mask):
            if value == 0:
                continue
            cls = sid.get(int(value), default_class_id)
            comp = (mask == value).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(comp, 8)
            for lab in range(1, n):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                m = (labels == lab).astype(np.uint8)
                bb = _bbox(m)
                if bb is None:
                    continue
                out.append(GtInstance(mask=m, class_id=int(cls), area=area, bbox=bb))
        return out

    # Binary mask convention.
    binary = (mask > MASK_THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        m = (labels == lab).astype(np.uint8)
        bb = _bbox(m)
        if bb is None:
            continue
        out.append(GtInstance(mask=m, class_id=int(default_class_id), area=area, bbox=bb))
    return out
