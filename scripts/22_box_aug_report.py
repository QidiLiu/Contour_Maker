"""汇总框增强消融结果。

    PYTHONPATH=src ./.conda/bin/python scripts/22_box_aug_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import REPORTS, RUNS  # noqa: E402

GROUPS = {
    "ctrl": "对照: 无框增强 (box_init_prob=0.25)",
    "C1C2": "C1+C2: 边独立抖动 + 宽高缩放/平移",
    "C3C4": "C3+C4: 内部内缩 + 角点抖动",
    "C1C2C3C4": "C1+C2+C3+C4: 全部形状增强",
    "p50_aug": "box_init_prob=0.5 + 全部形状增强",
}
EXTRA = {"ctrl_v3": "主结果基线 (refiner_v3, 同配置独立训练)"}


def main() -> int:
    rows = []
    for tag, name in GROUPS.items():
        d = REPORTS / f"boxaug_eval_{tag}" / "per_image.csv"
        if not d.exists():
            continue
        df = pd.read_csv(d)
        df = df[~df["method"].astype(str).str.endswith("__detstats")]
        g = df[df["method"] == "refiner_box"]
        if not len(g):
            g = df[df["method"] == "refiner"]
        rows.append(dict(tag=tag, name=name, n=len(g), dice=g["dice"].mean(),
                         dice_std=g["dice"].std(), iou=g["iou"].mean(), hd95=g["hd95"].mean(),
                         assd=g["assd"].mean(), bf1=g["bf1"].mean(), area_err=g["area_err"].mean(),
                         fail=float((g["dice"] < 0.5).mean())))
    # 主结果基线 (refiner_v3 的 A-box)
    base = REPORTS / "final_A_vs_B" / "per_image.csv"
    if base.exists():
        df = pd.read_csv(base)
        df = df[~df["method"].astype(str).str.endswith("__detstats")]
        g = df[df["method"] == "refiner_box"]
        if len(g):
            rows.append(dict(tag="v3_baseline", name="主结果基线 A-box (refiner_v3)", n=len(g),
                             dice=g["dice"].mean(), dice_std=g["dice"].std(), iou=g["iou"].mean(),
                             hd95=g["hd95"].mean(), assd=g["assd"].mean(), bf1=g["bf1"].mean(),
                             area_err=g["area_err"].mean(), fail=float((g["dice"] < 0.5).mean())))
    t = pd.DataFrame(rows).sort_values("dice", ascending=False)

    # 训练期验证集曲线 (看增强是否让验证集更稳)
    hist = []
    for tag in GROUPS:
        h = RUNS / f"refiner_v6_{tag}_busi-ddti-tn3k" / "history.json"
        if not h.exists():
            continue
        import json
        recs = json.loads(h.read_text())
        if not recs:
            continue
        best = max(recs, key=lambda r: r.get("val_dice", -1))
        hist.append(dict(tag=tag, best_val_dice=best.get("val_dice"), best_epoch=best.get("epoch"),
                         rough_val_dice=best.get("val_rough_dice")))
    ht = pd.DataFrame(hist).sort_values("best_val_dice", ascending=False) if hist else pd.DataFrame()

    out = REPORTS / "boxaug_summary.csv"
    t.to_csv(out, index=False)
    print("=== 框增强消融 (端到端, 共享 test split, refiner 结构不变) ===")
    print(t[["name", "n", "dice", "iou", "hd95", "area_err", "fail"]].round(4).to_string(index=False))
    if len(ht):
        print("\n=== 训练期验证集 (注意: 不同增强的验证分布不同, 仅供参照) ===")
        print(ht.round(4).to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
