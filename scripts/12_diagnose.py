"""诊断: 检测框质量对方案 A 的影响 (框 IoU 分桶统计 Dice)。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import REPORTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(REPORTS / "eval_A5"))
    ap.add_argument("--eval-b", default=str(REPORTS / "eval_B"))
    args = ap.parse_args()

    a = pd.read_csv(Path(args.eval) / "per_image.csv")
    a = a[~a["method"].str.endswith("__detstats")]
    b = pd.read_csv(Path(args.eval_b) / "per_image.csv")
    b = b[~b["method"].str.endswith("__detstats")]

    print("=== 各方法 Dice 随时间/数据集 ===")
    print(a.groupby(["method", "dataset"])["dice"].agg(["count", "mean", "median"]).to_string())

    # 检测框 IoU 分桶 (用 refiner 与 GT 框版本对比, 反映框质量影响)
    det = pd.read_csv(Path(args.eval) / "detection_stats.csv")
    print("\n=== 检测统计 ===")
    print(det.to_string(index=False))

    # 结合 match_iou 与 dice
    print("\n=== refiner: 按目标尺度分桶 (GT 框对角线, px) ===")
    piv = a.pivot_table(index="uid", columns="method", values="dice")
    merged = piv.join(b.set_index("uid")["dice"].rename("yolo26n_seg"), how="outer")
    merged.to_csv(Path(args.eval) / "dice_per_image_wide.csv")
    print(merged.describe().to_string())
    print(f"\n-> {Path(args.eval) / 'dice_per_image_wide.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
