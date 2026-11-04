"""Scheme-agnostic evaluation loop for the Python validators.

Every scheme only has to provide ``predict(image, row) -> (predictions, timings)``;
the common loop takes care of GT decoding, IoU matching, metrics, per-image CSV
output, visualisation and the summary JSON.  This keeps the Python and C++
evaluation protocols identical across schemes and datasets.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .config import (
    BOUNDARY_F1_TOL,
    IOU_MATCH_THRESHOLD,
    MIN_MASK_AREA,
    SPLITS,
    UNIFIED,
)
from .instances import GtInstance, gt_instances
from .metrics import evaluate_pair, greedy_match, mask_to_contour, prf


@dataclass
class Prediction:
    """One predicted instance at the original image resolution."""

    mask: np.ndarray
    class_id: int
    score: float = 1.0
    contour: np.ndarray | None = None

    def contour_points(self) -> np.ndarray | None:
        if self.contour is not None:
            return self.contour
        return mask_to_contour(self.mask)


PredictFn = Callable[[np.ndarray, pd.Series], tuple[list[Prediction], dict[str, float]]]


@dataclass
class EvalReport:
    scheme: str
    split: str
    n_images: int = 0
    segmentation: dict[str, float] = field(default_factory=dict)
    detection: dict[str, float] = field(default_factory=dict)
    speed: dict[str, float] = field(default_factory=dict)
    per_dataset: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(
            scheme=self.scheme,
            split=self.split,
            n_images=self.n_images,
            segmentation=self.segmentation,
            detection=self.detection,
            speed=self.speed,
            per_dataset=self.per_dataset,
        )


def load_split(
    split: str,
    splits_csv: str | Path | None = None,
    *,
    max_images: int | None = None,
) -> pd.DataFrame:
    path = Path(splits_csv) if splits_csv else SPLITS / "splits.csv"
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    df = pd.read_csv(path)
    df = df[df["split"] == split].reset_index(drop=True)
    if max_images:
        df = df.head(max_images).reset_index(drop=True)
    return df


def evaluate_split(
    predict: PredictFn,
    df: pd.DataFrame,
    *,
    scheme: str,
    split: str,
    unified: str | Path | None = None,
    gt_instances_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    class_names: Sequence[str] | None = None,
    min_area: int = MIN_MASK_AREA,
    match_iou: float = IOU_MATCH_THRESHOLD,
    boundary_tol: float = BOUNDARY_F1_TOL,
    save_vis: int = 20,
    warmup: int = 1,
    verbose: bool = True,
) -> EvalReport:
    """Run ``predict`` over every row of ``df`` and aggregate the metrics."""
    unified_root = Path(unified) if unified else UNIFIED
    gt_root = Path(gt_instances_dir) if gt_instances_dir else None
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    # One discarded prediction removes cold-start / CUDA autotune time from the
    # reported latency (the C++ harness does the same with its warmup loop).
    if warmup > 0 and len(df):
        first = df.iloc[0]
        img0 = cv2.imread(str(unified_root / first["image"]), cv2.IMREAD_GRAYSCALE)
        if img0 is not None:
            for _ in range(warmup):
                predict(img0, first)

    records: list[dict] = []
    det_records: list[dict] = []
    stage_sums: dict[str, float] = {}
    total_ms = 0.0
    vis_saved = 0

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        img_path = unified_root / row["image"]
        gt_path = unified_root / row["mask"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if img is None or gt_mask is None:
            continue
        h, w = img.shape[:2]
        if gt_mask.shape[:2] != (h, w):
            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        sidecar = None
        if gt_root is not None:
            sidecar = gt_root / f"{Path(str(row['mask'])).stem}.csv"
        gts: list[GtInstance] = gt_instances(
            gt_mask,
            sidecar=sidecar,
            default_class_id=int(row["class_id"]),
            min_area=min_area,
        )

        t0 = time.perf_counter()
        preds, timings = predict(img, row)
        elapsed = (time.perf_counter() - t0) * 1000.0
        total_ms += elapsed
        for k, v in timings.items():
            stage_sums[k] = stage_sums.get(k, 0.0) + float(v)

        pred_masks = [p.mask for p in preds]
        gt_masks = [g.mask for g in gts]
        pairs, n_miss, n_fp = greedy_match(pred_masks, gt_masks, match_iou)
        det_records.append(
            dict(
                image=Path(str(row["image"])).name,
                uid=row["uid"],
                dataset=row.get("dataset", ""),
                split=row.get("split", split),
                n_gt=len(gts),
                n_pred=len(preds),
                n_miss=n_miss,
                n_fp=n_fp,
            )
        )

        for pi, gi, miou in pairs:
            rec = dict(
                uid=row["uid"],
                dataset=row.get("dataset", ""),
                split=row.get("split", split),
                modality=row.get("modality", ""),
                match_iou=miou,
                pred_cls=int(preds[pi].class_id),
                gt_cls=int(gts[gi].class_id),
                cls_ok=int(preds[pi].class_id == gts[gi].class_id),
                pred_score=float(preds[pi].score),
            )
            rec.update(evaluate_pair(pred_masks[pi], gt_masks[gi], boundary_tol))
            records.append(rec)

        if out and vis_saved < save_vis:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for g in gts:
                ct = mask_to_contour(g.mask)
                if ct is not None:
                    cv2.polylines(vis, [np.round(ct).astype(np.int32)], True, (0, 255, 0), 1)
            for p in preds:
                ct = p.contour_points()
                if ct is not None:
                    cv2.polylines(vis, [np.round(ct).astype(np.int32)], True, (255, 0, 255), 1)
            cv2.imwrite(str(out / f"vis_{row.get('dataset', '')}_{row['uid']}.png"), vis)
            vis_saved += 1

        if verbose and n % 200 == 0:
            print(f"  {n}/{len(df)}  {total_ms / n:.1f} ms/img", flush=True)

    report = _summarise(
        records,
        det_records,
        stage_sums,
        total_ms,
        scheme=scheme,
        split=split,
        n_images=len(det_records),
        class_names=class_names,
    )
    if out:
        pd.DataFrame(records).to_csv(out / "per_target.csv", index=False)
        pd.DataFrame(det_records).to_csv(out / "per_image.csv", index=False)
        (out / "summary.json").write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        if verbose:
            print(f"\n-> {out}")
    return report


def _summarise(
    records: list[dict],
    det_records: list[dict],
    stage_sums: dict[str, float],
    total_ms: float,
    *,
    scheme: str,
    split: str,
    n_images: int,
    class_names: Sequence[str] | None,
) -> EvalReport:
    rep = EvalReport(scheme=scheme, split=split, n_images=n_images)
    n = max(1, n_images)

    print("\n" + "=" * 72)
    print(f"=== {scheme}  (split={split}, n={n_images}) ===")
    print("=" * 72)

    if records:
        s = pd.DataFrame(records)
        seg = dict(
            n=float(len(s)),
            dice=float(s["dice"].mean()),
            iou=float(s["iou"].mean()),
            hd95=float(s["hd95"].mean()),
            assd=float(s["assd"].mean()),
            bf1=float(s["bf1"].mean()),
            area_err=float(s["area_err"].mean()),
        )
        seg["fail_rate"] = float((s["dice"] < 0.5).mean())
        seg["cls_acc"] = float(s["cls_ok"].mean())
        rep.segmentation = {k: float(v) for k, v in seg.items()}
        print("\n[segmentation]")
        for k, v in rep.segmentation.items():
            print(f"  {k:10s} {v:.4f}")
        print("\n[by dataset]")
        rep.per_dataset = (
            s.groupby("dataset")
            .agg(
                n=("dice", "size"),
                dice=("dice", "mean"),
                iou=("iou", "mean"),
                hd95=("hd95", "mean"),
                bf1=("bf1", "mean"),
                cls_acc=("cls_ok", "mean"),
            )
            .round(4)
            .to_dict(orient="index")
        )
        print(pd.DataFrame(rep.per_dataset).T.to_string())

    if det_records:
        d = pd.DataFrame(det_records)
        n_gt = int(d["n_gt"].sum())
        n_pred = int(d["n_pred"].sum())
        n_miss = int(d["n_miss"].sum())
        n_fp = int(d["n_fp"].sum())
        tp = n_gt - n_miss
        pf = prf(tp, n_fp, n_miss)
        rep.detection = dict(
            gt=n_gt,
            pred=n_pred,
            tp=tp,
            fp=n_fp,
            fn=n_miss,
            precision=pf["precision"],
            recall=pf["recall"],
            f1=pf["f1"],
        )
        print("\n[detection]")
        print(f"  GT={n_gt} pred={n_pred} TP={tp} FP={n_fp} FN={n_miss}")
        print(f"  P={pf['precision']:.4f} R={pf['recall']:.4f} F1={pf['f1']:.4f}")

    ms = total_ms / n
    rep.speed = dict(total_ms=ms, fps=1000.0 / max(ms, 1e-9))
    for k, v in stage_sums.items():
        rep.speed[f"{k}_ms"] = v / n
    print("\n[speed]")
    for k, v in stage_sums.items():
        print(f"  {k:12s} {v / n:7.2f} ms/img")
    print(f"  {'total':12s} {ms:7.2f} ms/img  ({1000.0 / max(ms, 1e-9):.1f} FPS)")
    return rep
