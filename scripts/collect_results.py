#!/usr/bin/env python3
"""Collect every ``reports/*/summary.json`` into one comparison table.

Both the Python validators and the C++ harness write a ``summary.json`` with the
same top-level keys, so the collector is runtime-agnostic.

Usage::

    python scripts/collect_results.py --reports reports --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _flatten(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run = path.parent.name
    parts = run.split("_")
    runtime = parts[-1] if parts[-1] in {"py", "cxx"} else "?"
    scheme = run[: -(len(runtime) + 1)] if runtime != "?" else run
    seg = data.get("segmentation", {}) or {}
    det = data.get("detection", {}) or {}
    speed = data.get("speed", {}) or {}
    if det.get("precision") is None and "tp" in det:
        tp, fp, fn = int(det.get("tp", 0)), int(det.get("fp", 0)), int(det.get("fn", 0))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        det["precision"] = precision
        det["recall"] = recall
        det["f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        run=run,
        scheme=scheme,
        runtime=runtime,
        n_images=data.get("n_images", 0),
        precision=det.get("precision"),
        recall=det.get("recall"),
        f1=det.get("f1"),
        dice=seg.get("dice"),
        iou=seg.get("iou"),
        hd95=seg.get("hd95"),
        assd=seg.get("assd"),
        bf1=seg.get("bf1"),
        area_err=seg.get("area_err"),
        cls_acc=seg.get("cls_acc"),
        total_ms=speed.get("total_ms"),
        fps=speed.get("fps"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect scheme benchmark summaries")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.reports).rglob("summary.json")):
        row = _flatten(p)
        if row:
            rows.append(row)
    if not rows:
        print("no summary.json found")
        return 1
    df = pd.DataFrame(rows).sort_values(["scheme", "runtime"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    cols = [
        "scheme",
        "runtime",
        "n_images",
        "precision",
        "recall",
        "f1",
        "dice",
        "iou",
        "hd95",
        "bf1",
        "cls_acc",
        "total_ms",
        "fps",
    ]
    md = ["# Benchmark summary", "", df[cols].round(4).to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(df[cols].round(4).to_string(index=False))
    print(f"\n-> {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
