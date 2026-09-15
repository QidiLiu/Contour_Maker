"""汇总多次评估结果, 生成最终对比表 (Markdown + CSV + 图)。

    PYTHONPATH=src ./.conda/bin/python scripts/13_compare.py \
        --evals reports/v3_eval_A5 reports/v3_eval_A0 reports/v2_eval_A5 reports/v2_eval_A0 reports/eval_B \
        --out reports/final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import REPORTS, RUNS  # noqa: E402

METHOD_LABEL = {
    "rough_otsu": "A-0 仅 Otsu (YOLO 框)",
    "rough_otsu_gtbox": "A-0 仅 Otsu (GT 框)",
    "refiner": "A 完整 (Otsu+refiner, YOLO 框)",
    "refiner_gtbox": "A 完整 (GT 框)",
    "yolo26n_seg": "B YOLO26n-seg",
}
DATASET_LABEL = {"busi": "BUSI 乳腺", "tn3k": "TN3K 甲状腺", "ddti": "DDTI 甲状腺",
                 "tg3k": "TG3K 腺体"}


def load_eval(d: Path) -> pd.DataFrame | None:
    f = d / "per_image.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[~df["method"].astype(str).str.endswith("__detstats")].copy()
    df["eval"] = d.name
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", default=str(REPORTS / "final"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    frames = []
    for e in args.evals:
        d = load_eval(Path(e))
        if d is not None and len(d):
            frames.append(d)
    if not frames:
        print("没有可用结果")
        return 1
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "all_per_image.csv", index=False)

    # ---- 主对比表 (每个评估中方法级汇总)
    rows = []
    for (ev, m), g in all_df.groupby(["eval", "method"]):
        rows.append(dict(eval=ev, method=m, n=len(g), dice=g["dice"].mean(),
                         dice_std=g["dice"].std(), iou=g["iou"].mean(),
                         hd95=g["hd95"].mean(), assd=g["assd"].mean(),
                         bf1=g["bf1"].mean(), area_err=g["area_err"].mean(),
                         dice_median=g["dice"].median(),
                         fail_rate=float((g["dice"] < 0.5).mean())))
    main_tbl = pd.DataFrame(rows)
    main_tbl.to_csv(out / "compare_all_evals.csv", index=False)

    # ---- 选定"最终"评估做结论表
    primary = {}
    for pref in ("v3_eval_A5", "v3_eval_A0", "v2_eval_A5", "eval_A5", "eval_B"):
        d = Path(REPORTS / pref)
        if (d / "per_image.csv").exists() and pref not in primary:
            primary[pref] = d
    lines = ["# 方案 A vs 方案 B —— 对比结果", ""]
    for ev, d in primary.items():
        sub = load_eval(d)
        if sub is None:
            continue
        agg = sub.groupby("method").agg(
            n=("dice", "size"), dice=("dice", "mean"), dice_std=("dice", "std"),
            iou=("iou", "mean"), hd95=("hd95", "mean"), assd=("assd", "mean"),
            bf1=("bf1", "mean"), area_err=("area_err", "mean"),
            fail_rate=("dice", lambda s: float((s < 0.5).mean()))).reset_index()
        lines += [f"## 评估: `{ev}` (测试集 {int(agg['n'].max())} 个目标)", "",
                  "| 方法 | n | Dice ↑ | IoU ↑ | HD95(px) ↓ | ASSD(px) ↓ | 边界F1 ↑ | 面积误差 ↓ | Dice<0.5 比例 ↓ |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in agg.sort_values("dice", ascending=False).iterrows():
            lines.append(f"| {METHOD_LABEL.get(r['method'], r['method'])} | {int(r['n'])} | "
                         f"**{r['dice']:.4f}** | {r['iou']:.4f} | {r['hd95']:.2f} | {r['assd']:.2f} | "
                         f"{r['bf1']:.4f} | {r['area_err']:.3f} | {r['fail_rate']*100:.1f}% |")
        lines.append("")

    # ---- 增强消融图 (test Dice, 按评估标签分组)
    abl = main_tbl[main_tbl["method"] == "refiner"].copy()
    if len(abl):
        fig, ax = plt.subplots(figsize=(8, 4))
        for ev, g in abl.groupby("eval"):
            g = g.assign(level=g["eval"].str.extract(r"([A-Z]\d)"))
            ax.bar(g["eval"], g["dice"], label=ev)
        ax.set_ylabel("Test Dice")
        ax.set_title("contour-refiner test Dice by augmentation level")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig_compare.png", dpi=150)
        plt.close(fig)
        lines += ["![compare](fig_compare.png)", ""]

    # ---- 各数据集拆解 (取最新一次评估)
    if primary:
        ev = list(primary)[0]
        sub = load_eval(primary[ev])
        if sub is not None:
            p = sub.pivot_table(index="dataset", columns="method", values="dice", aggfunc="mean")
            lines += [f"## 分数据集 (评估 `{ev}`)", "", p.round(4).to_markdown(), ""]

    # ---- 训练记录
    hist_rows = []
    for d in sorted(RUNS.glob("refiner_*")):
        h = d / "history.json"
        if not h.exists():
            continue
        try:
            hist = json.loads(h.read_text())
        except Exception:
            continue
        if not hist:
            continue
        best = max(hist, key=lambda r: r.get("val_dice", -1))
        hist_rows.append(dict(run=d.name, best_val_dice=best.get("val_dice"),
                              best_epoch=best.get("epoch"),
                              rough_val_dice=best.get("val_rough_dice"), epochs=len(hist)))
    ht = pd.DataFrame(hist_rows).sort_values("best_val_dice", ascending=False)
    ht.to_csv(out / "training_runs.csv", index=False)
    lines += ["## 训练记录 (验证集)", "", ht.round(4).to_markdown(index=False), ""]

    (out / "COMPARISON.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))
    print(f"\n-> {out/'COMPARISON.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
