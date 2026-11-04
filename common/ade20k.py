"""Build the ADE20K indoor-object subset used as the generalisation benchmark.

ADE20K is the de-facto scene-understanding benchmark in embodied AI / robot
perception.  The official release used here is distributed as
``annotations_instance`` PNGs in which the **R channel is the ADE20K thing-class
index (1..100)** and the **G channel is the instance id**.  We keep a compact set
of ten indoor classes that matter for manipulation and navigation (see
``common.config.ADE20K_CLASSES``), convert the images to grayscale (the embedded
pipeline is single-channel) and store masks as 8-bit instance-id maps with a
class sidecar.  This representation keeps multi-class, multi-instance images
intact and is read by both the Python and the C++ evaluators.

Output layout (``data/ade20k``)::

    unified/images/<uid>.png        grayscale input
    unified/masks/<uid>.png         8-bit instance ids (0 = background)
    unified/instances/<uid>.csv     instance_id,class_id
    unified/manifest.csv
    splits/splits.csv
    splits/splits_meta.json

Usage::

    python -m common.ade20k --download
    python -m common.ade20k --build
    python -m common.ade20k --all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import requests

from .config import (
    ADE20K_ADE_ID_TO_CLASS,
    ADE20K_CLASS_NAMES,
    ADE20K_HF_REPO,
    ADE20K_HF_REVISION,
    ADE20K_INSTANCE_FILES,
    ADE20K_MAX_INSTANCES,
    ADE20K_MIN_INSTANCE_AREA,
    ADE20K_RAW,
    ADE20K_SPLIT_SEED,
    ADE20K_SPLITS,
    ADE20K_TEST_IMAGES,
    ADE20K_TRAIN_IMAGES,
    ADE20K_UNIFIED,
    ADE20K_VAL_IMAGES,
    HF_ENDPOINT,
)
from .instances import write_sidecar

# Official splits used by this benchmark: train/val are carved out of the
# official training set, test is the official validation set.
_POOLS = {
    "train": [("train", f) for f in ADE20K_INSTANCE_FILES["train"]],
    "test": [("validation", f) for f in ADE20K_INSTANCE_FILES["validation"]],
}


def _parquet_path(split: str, rel: str) -> Path:
    # ``rel`` already starts with the official split name (train/0000.parquet);
    # keep the on-disk name stable regardless of which pool points at it.
    return ADE20K_RAW / rel.replace("/", "_")


def download(force: bool = False) -> None:
    """Download the ADE20K instance parquet shards from the HF mirror."""
    ADE20K_RAW.mkdir(parents=True, exist_ok=True)
    for split, files in ADE20K_INSTANCE_FILES.items():
        for rel in files:
            dst = _parquet_path(split, rel)
            if dst.exists() and not force:
                print(f"  [skip] {dst.name} ({dst.stat().st_size / 1e6:.0f} MB)")
                continue
            url = (
                f"{HF_ENDPOINT}/datasets/{ADE20K_HF_REPO}/resolve/"
                f"{ADE20K_HF_REVISION}/instance_segmentation/{split}/{rel}"
            )
            print(f"  [get ] {url}")
            part = dst.with_suffix(".part")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with part.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(
                                f"\r        {100 * done / total:5.1f}% "
                                f"{done / 1e6:7.1f}/{total / 1e6:.1f} MB",
                                end="",
                                flush=True,
                            )
            print()
            part.replace(dst)
            print(f"  [ok  ] {dst.name} ({dst.stat().st_size / 1e6:.0f} MB)")


def _decode(row: dict) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imdecode(np.frombuffer(row["image"], np.uint8), cv2.IMREAD_COLOR)
    ann = cv2.imdecode(np.frombuffer(row["annotation"], np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or ann is None:
        raise ValueError("decode failed")
    if ann.shape[:2] != img.shape[:2]:
        ann = cv2.resize(ann, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, ann


def _extract_instances(ann: np.ndarray, min_area: int):
    """Return [(class_id, component_mask)] for the selected classes.

    ``ann`` is BGR from OpenCV; the R channel is the ADE20K thing-class index,
    the G channel is the instance id.
    """
    b, g, r = cv2.split(ann)
    out = []
    for ade_id, cls_id in ADE20K_ADE_ID_TO_CLASS.items():
        for inst_id in np.unique(g[r == ade_id]):
            if inst_id == 0:
                continue
            m = ((r == ade_id) & (g == inst_id)).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            for lab in range(1, n):
                if int(stats[lab, cv2.CC_STAT_AREA]) < min_area:
                    continue
                out.append((cls_id, (labels == lab).astype(np.uint8)))
    return out


def _scan(pool: str) -> list[tuple[str, str, int, int, str]]:
    """First pass: list images that contain at least one selected instance."""
    found: list[tuple[str, str, int, int, str]] = []  # (split, rel, rg, idx, name)
    for split, rel in _POOLS[pool]:
        path = _parquet_path(split, rel)
        if not path.exists():
            raise FileNotFoundError(f"missing parquet: {path}; run --download")
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["image.path", "annotation"])
            for i in range(t.num_rows):
                name = t["image"][i].as_py()["path"]
                try:
                    ann = cv2.imdecode(
                        np.frombuffer(t["annotation"][i].as_py()["bytes"], np.uint8),
                        cv2.IMREAD_UNCHANGED,
                    )
                except Exception:
                    continue
                if ann is None or ann.ndim != 3:
                    continue
                if _extract_instances(ann, ADE20K_MIN_INSTANCE_AREA):
                    found.append((split, rel, rg, i, name))
    return found


def _write_one(row: dict, stem: str, split: str) -> dict | None:
    img, ann = _decode(row)
    insts = _extract_instances(ann, ADE20K_MIN_INSTANCE_AREA)
    if not insts:
        return None
    if len(insts) > ADE20K_MAX_INSTANCES:
        insts = sorted(insts, key=lambda x: int(x[1].sum()), reverse=True)[:ADE20K_MAX_INSTANCES]

    uid = f"ade20k_{stem}"
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    id_map = np.zeros((h, w), np.uint8)
    mapping: dict[int, int] = {}
    for i, (cls_id, m) in enumerate(insts, start=1):
        id_map[m > 0] = i
        mapping[i] = int(cls_id)

    UNIFIED_IMG = ADE20K_UNIFIED / "images"
    UNIFIED_MSK = ADE20K_UNIFIED / "masks"
    UNIFIED_INS = ADE20K_UNIFIED / "instances"
    for d in (UNIFIED_IMG, UNIFIED_MSK, UNIFIED_INS):
        d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(UNIFIED_IMG / f"{uid}.png"), gray)
    cv2.imwrite(str(UNIFIED_MSK / f"{uid}.png"), id_map)
    write_sidecar(UNIFIED_INS / f"{uid}.csv", mapping)

    cls_counter = Counter(mapping.values())
    primary = cls_counter.most_common(1)[0][0]
    return dict(
        uid=uid,
        dataset="ade20k",
        modality="indoor",
        class_id=int(primary),
        class_source="ade20k_thing",
        image=f"images/{uid}.png",
        mask=f"masks/{uid}.png",
        height=h,
        width=w,
        src_image=stem,
        src_mask=stem,
        n_instances=len(mapping),
        classes=";".join(str(c) for c in sorted(set(mapping.values()))),
        split=split,
    )


def build(seed: int = ADE20K_SPLIT_SEED) -> None:
    """Sample and write the benchmark subset."""
    rng = random.Random(seed)
    print("[scan] listing candidate images (annotation-only pass) ...")
    train_pool = _scan("train")
    test_pool = _scan("test")
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    print(f"  train candidates={len(train_pool)}  test candidates={len(test_pool)}")

    plan: list[tuple[tuple[str, str, int, int, str], str]] = []
    for item in train_pool[:ADE20K_TRAIN_IMAGES]:
        plan.append((item, "train"))
    rest = train_pool[ADE20K_TRAIN_IMAGES:]
    for item in rest[:ADE20K_VAL_IMAGES]:
        plan.append((item, "val"))
    for item in test_pool[:ADE20K_TEST_IMAGES]:
        plan.append((item, "test"))

    # Group by (parquet, row group) so each row group is decoded once.
    by_group: dict[tuple[str, str, int], list[tuple[int, str]]] = {}
    for (split, rel, rg, idx, _name), target in plan:
        by_group.setdefault((split, rel, rg), []).append((idx, target))

    rows: list[dict] = []
    for (split, rel, rg), items in by_group.items():
        pf = pq.ParquetFile(_parquet_path(split, rel))
        t = pf.read_row_group(rg)
        for idx, target in items:
            name = t["image"][idx].as_py()["path"]
            stem = Path(name).stem
            rec = _write_one(
                {
                    "image": t["image"][idx].as_py()["bytes"],
                    "annotation": t["annotation"][idx].as_py()["bytes"],
                },
                stem,
                target,
            )
            if rec is not None:
                rows.append(rec)
    print(f"[write] {len(rows)} images")

    import pandas as pd

    df = pd.DataFrame(rows)
    ADE20K_UNIFIED.mkdir(parents=True, exist_ok=True)
    ADE20K_SPLITS.mkdir(parents=True, exist_ok=True)
    df.to_csv(ADE20K_UNIFIED / "manifest.csv", index=False)
    df.to_csv(ADE20K_SPLITS / "splits.csv", index=False)
    (ADE20K_SPLITS / "splits_meta.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                requested=dict(
                    train=ADE20K_TRAIN_IMAGES, val=ADE20K_VAL_IMAGES, test=ADE20K_TEST_IMAGES
                ),
                counts={s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
                num_classes=len(ADE20K_CLASS_NAMES),
                class_names=ADE20K_CLASS_NAMES,
                source="ADE20K instance_segmentation (validation used as test)",
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== split x class (primary class per image) ===")
    print(pd.crosstab(df["dataset"] + "_" + df["class_id"].astype(str), df["split"]))
    print("\n=== instances ===")
    print(df.groupby("split")["n_instances"].agg(["sum", "mean", "max"]).round(2))
    print(f"\n-> {ADE20K_UNIFIED / 'manifest.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the ADE20K indoor-object subset")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=ADE20K_SPLIT_SEED)
    args = ap.parse_args()
    if not any((args.download, args.build, args.all)):
        ap.error("specify --download / --build / --all")
    if args.download or args.all:
        print("=== ADE20K download ===")
        download(force=args.force)
    if args.build or args.all:
        print("=== ADE20K build ===")
        build(seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
