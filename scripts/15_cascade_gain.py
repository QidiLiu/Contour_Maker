"""检测-分割级联的真实收益分析: 用 B 的统计把「漏检导致的 0 分」折算掉。

    PYTHONPATH=src ./.conda/bin/python scripts/15_cascade_gain.py --eval reports/B_plus_refiner
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import REPORTS  # noqa: E402


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(REPORTS / "B_plus_refiner"))
    args = ap.parse_args()
    ev = Path(args.eval)

    df = pd.read_csv(ev / "per_image.csv")
    det = df[df["method"].str.endswith("__detstats")].copy()
    det["method"] = det["method"].str.replace("__detstats", "", regex=False)
    det = det.drop_duplicates(subset=["uid", "method"])

    rows = []
    for m, g in det.groupby("method"):
        tp = int(g["n_gt"].sum() - g["n_miss"].sum())
        fp = int(g["n_fp"].sum())
        fn = int(g["n_miss"].sum())
        p, r, f = prf(tp, fp, fn)
        rows.append(dict(method=m, tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f))
    tbl = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print("=== 目标级 检测/分割 匹配统计 (IoU>=0.5 计 TP) ===")
    print(tbl.round(4).to_string(index=False))

    # 只统计每个方法内与 GT 匹配上的目标 (排除命中率差异的影响)
    score = df[~df["method"].str.endswith("__detstats")].copy()
    print("\n=== 只统计命中目标的分割精度 (matched-only) ===")
    rows2 = []
    for m, g in score.groupby("method"):
        if m == "seg_refiner":
            continue
        h = tbl[tbl.method == m]
        rec = float(h["recall"].iloc[0]) if len(h) else float("nan")
        prec = float(h["precision"].iloc[0]) if len(h) else float("nan")
        f1 = float(h["f1"].iloc[0]) if len(h) else float("nan")
        n_gt = int(h["tp"].iloc[0] + h["fn"].iloc[0]) if len(h) else 0
        rows2.append(dict(method=m, n_gt=n_gt,
                          dice_matched=g["dice"].mean(), iou_matched=g["iou"].mean(),
                          hd95_matched=g["hd95"].mean(), recall=rec, precision=prec,
                          dice_x_f1=g["dice"].mean() * f1))
    out = pd.DataFrame(rows2).sort_values("dice_x_f1", ascending=False)
    print(out.round(4).to_string(index=False))
    out.to_csv(ev / "cascade_gain.csv", index=False)

    # seg_refiner 的量级: 与 B 在相同 uid 上配对比较
    a = score[score.method == "yolo26n_seg"].set_index("uid")["dice"]
    b = score[score.method == "seg_refiner"].set_index("uid")["dice"]
    j = pd.concat([a.rename("B"), b.rename("B_ref")], axis=1).dropna()
    print(f"\n=== 配对比较 (同 {len(j)} 个预测目标) ===")
    print(f"B      : dice={j.B.mean():.4f}")
    print(f"B+ref  : dice={j.B_ref.mean():.4f}  ({j.B_ref.mean()-j.B.mean():+.4f})")
    print(f"变差 {float((j.B_ref<j.B).mean())*100:.1f}% / 变好 {float((j.B_ref>j.B).mean())*100:.1f}%")
    print(f"\n-> {ev/'cascade_gain.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
