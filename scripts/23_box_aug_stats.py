"""框增强消融的统计检验: 与对照的配对 bootstrap (按图像重采样)。

    PYTHONPATH=src ./.conda/bin/python scripts/23_box_aug_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import REPORTS  # noqa: E402

GROUPS = ["ctrl", "C1C2", "C3C4", "C1C2C3C4", "p50_aug"]
REF = "runs/refiner_v3_A5_busi-ddti-tn3k/best.pt"


def load_eval(tag: str, use_main_baseline: bool = False) -> pd.DataFrame | None:
    if use_main_baseline:
        p = REPORTS / "final_A_vs_B" / "per_image.csv"
    else:
        p = REPORTS / f"boxaug_eval_{tag}" / "per_image.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[~df["method"].astype(str).str.endswith("__detstats")]
    m = "refiner_box" if (df["method"] == "refiner_box").any() else "refiner"
    g = df[df["method"] == m][["uid", "dataset", "dice", "iou", "hd95"]].copy()
    g["tag"] = tag
    return g


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """按 uid 配对, 对 (a - b) 的均值做 bootstrap 置信区间。"""
    j = a.merge(b, on="uid", suffixes=("_a", "_b"))
    if not len(j):
        return None
    diff = (j["dice_a"] - j["dice_b"]).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=n, mean_a=j["dice_a"].mean(), mean_b=j["dice_b"].mean(),
                delta=diff.mean(), ci_lo=lo, ci_hi=hi,
                p_win=float((diff > 0).mean()),
                p_better=float((boots > 0).mean()))


def main() -> int:
    base = load_eval("v3_baseline", use_main_baseline=True)
    ctrl = load_eval("ctrl")
    rows = []
    for tag in GROUPS:
        g = load_eval(tag)
        if g is None:
            continue
        for ref_name, ref in (("v3基线", base), ("ctrl", ctrl)):
            if ref is None or (ref_name == "ctrl" and tag == "ctrl"):
                continue
            st = paired_bootstrap(g, ref)
            if st:
                rows.append(dict(group=tag, vs=ref_name, **st))
    t = pd.DataFrame(rows)
    if not len(t):
        print("没有可比较的结果")
        return 1
    t.to_csv(REPORTS / "boxaug_stats.csv", index=False)
    pd.set_option("display.width", 160)
    print("=== 配对 bootstrap (按图像重采样, 2000 次) ===")
    print(t[["group", "vs", "n", "mean_a", "mean_b", "delta", "ci_lo", "ci_hi", "p_win"]]
          .round(4).to_string(index=False))
    print("\n判读: delta 的 95% CI 若跨 0 (ci_lo<0<ci_hi), 则该增强与参照无显著差异。")
    print(f"\n-> {REPORTS/'boxaug_stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
