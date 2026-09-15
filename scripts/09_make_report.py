"""汇总评估结果, 生成对比报告 (Markdown + 图表)。

    PYTHONPATH=src ./.conda/bin/python scripts/09_make_report.py --eval reports/eval_A5 --out reports
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
from cm.config import REPORTS, ROOT, RUNS  # noqa: E402

METHOD_LABEL = {
    "rough_otsu": "A-粗: Otsu 粗糙轮廓 (YOLO 框)",
    "refiner": "A: Otsu+contour-refiner (YOLO 框)",
    "rough_otsu_gtbox": "A-粗: Otsu 粗糙轮廓 (GT 框)",
    "refiner_gtbox": "A: Otsu+contour-refiner (GT 框)",
    "yolo26n_seg": "B: YOLO26n-seg 端到端",
}
ORDER = ["rough_otsu", "refiner", "rough_otsu_gtbox", "refiner_gtbox", "yolo26n_seg"]
DATASET_LABEL = {"busi": "BUSI 乳腺", "tn3k": "TN3K 甲状腺", "ddti": "DDTI 甲状腺",
                 "tg3k": "TG3K 甲状腺腺体"}


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "-"
    return f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(REPORTS / "eval_A5"))
    ap.add_argument("--out", default=str(REPORTS))
    ap.add_argument("--title", default="YOLO26n + Otsu + contour-refiner  vs  YOLO26n-seg")
    args = ap.parse_args()
    ev = Path(args.eval)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    per_image = pd.read_csv(ev / "per_image.csv")
    score = per_image[~per_image["method"].str.endswith("__detstats")].copy()
    det = per_image[per_image["method"].str.endswith("__detstats")].copy()
    det["method"] = det["method"].str.replace("__detstats", "", regex=False)

    overall = score.groupby("method").agg(
        n=("dice", "size"), dice=("dice", "mean"), dice_std=("dice", "std"),
        iou=("iou", "mean"), hd95=("hd95", "mean"), assd=("assd", "mean"),
        bf1=("bf1", "mean"), area_err=("area_err", "mean")).reset_index()
    by_ds = score.groupby(["dataset", "method"]).agg(
        n=("dice", "size"), dice=("dice", "mean"), iou=("iou", "mean"),
        hd95=("hd95", "mean"), assd=("assd", "mean"), bf1=("bf1", "mean"),
        area_err=("area_err", "mean")).reset_index()
    det_sum = det.groupby("method").agg(
        n_gt=("n_gt", "sum"), n_pred=("n_pred", "sum"), n_miss=("n_miss", "sum"),
        n_fp=("n_fp", "sum")).reset_index()

    def sort_key(m):
        return ORDER.index(m) if m in ORDER else 99

    overall["_k"] = overall["method"].map(sort_key)
    overall = overall.sort_values("_k").drop(columns="_k")
    by_ds["_k"] = by_ds["method"].map(sort_key)
    by_ds = by_ds.sort_values(["dataset", "_k"]).drop(columns="_k")

    # ---------------- charts
    plt.rcParams["font.size"] = 9
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(overall))
    axes[0].bar(x, overall["dice"], yerr=overall["dice_std"].fillna(0), capsize=3,
                color=["#999", "#2b7bba", "#bbb", "#7fb3d5", "#d95f02"][: len(overall)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([METHOD_LABEL.get(m, m) for m in overall["method"]],
                            rotation=25, ha="right", fontsize=7)
    axes[0].set_ylabel("Dice")
    axes[0].set_title("总体 Dice (test)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, overall["hd95"], color=["#999", "#2b7bba", "#bbb", "#7fb3d5", "#d95f02"][: len(overall)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([METHOD_LABEL.get(m, m) for m in overall["method"]],
                            rotation=25, ha="right", fontsize=7)
    axes[1].set_ylabel("HD95 (px)")
    axes[1].set_title("总体 HD95 (越小越好)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_overall.png", dpi=150)
    plt.close(fig)

    if len(by_ds):
        piv = by_ds.pivot(index="dataset", columns="method", values="dice")
        ax = piv.plot(kind="bar", figsize=(9, 4), rot=15,
                      color=["#999", "#2b7bba", "#bbb", "#7fb3d5", "#d95f02"][: piv.shape[1]])
        ax.set_ylabel("Dice")
        ax.set_title("各数据集 Dice 对比")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(out / "fig_by_dataset.png", dpi=150)
        plt.close()

    # ---------------- augmentation ablation
    abl_rows = []
    for d in sorted(RUNS.glob("refiner_A*")):
        h = d / "history.json"
        cfg = d / "config.json"
        if not h.exists():
            continue
        try:
            hist = json.loads(h.read_text())
            c = json.loads(cfg.read_text()) if cfg.exists() else {}
        except Exception:
            continue
        if not hist:
            continue
        tag = d.name.replace("refiner_", "")
        if not (tag.startswith("A") and tag[1:2].isdigit()):
            continue
        best = max(hist, key=lambda r: r.get("val_dice", -1))
        abl_rows.append(dict(tag=tag, level=c.get("level"), best_val_dice=best.get("val_dice"),
                             best_epoch=best.get("epoch"), rough_dice=best.get("val_rough_dice"),
                             epochs=len(hist)))
    abl = pd.DataFrame(abl_rows)
    if len(abl):
        abl = abl.sort_values(["level", "tag"])
        abl.to_csv(out / "ablation_augmentation.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(abl["tag"], abl["best_val_dice"], "o-", label="refiner 精细化后")
        ax.axhline(float(abl["rough_dice"].mean()), ls="--", color="gray", label="粗糙轮廓 (Otsu) 平均")
        ax.set_xlabel("数据增强级别 / 配置")
        ax.set_ylabel("验证集 Dice")
        ax.set_title("A1-A5 增强消融")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "fig_ablation.png", dpi=150)
        plt.close()

    # ---------------- markdown report
    lines = [f"# {args.title}", ""]
    lines += [f"评估目录: `{ev}`  |  测试集样本数: {int(overall['n'].max()) if len(overall) else 0}", ""]
    lines += ["## 1. 总体结果 (测试集)", "",
              "| 方法 | n | Dice ↑ | IoU ↑ | HD95 (px) ↓ | ASSD (px) ↓ | 边界F1 ↑ | 面积相对误差 ↓ |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in overall.iterrows():
        lines.append(f"| {METHOD_LABEL.get(r['method'], r['method'])} | {int(r['n'])} | "
                     f"**{fmt(r['dice'])}** ± {fmt(r['dice_std'],3)} | {fmt(r['iou'])} | {fmt(r['hd95'],2)} | "
                     f"{fmt(r['assd'],2)} | {fmt(r['bf1'])} | {fmt(r['area_err'],3)} |")
    lines += ["", "## 2. 分数据集结果", "",
              "| 数据集 | 方法 | n | Dice ↑ | IoU ↑ | HD95 (px) ↓ | ASSD (px) ↓ |",
              "|---|---|---|---|---|---|---|"]
    for _, r in by_ds.iterrows():
        lines.append(f"| {DATASET_LABEL.get(r['dataset'], r['dataset'])} | {METHOD_LABEL.get(r['method'], r['method'])} | "
                     f"{int(r['n'])} | {fmt(r['dice'])} | {fmt(r['iou'])} | {fmt(r['hd95'],2)} | {fmt(r['assd'],2)} |")
    lines += ["", "## 3. 检测层面统计 (漏检/误检)", "",
              "| 方法 | GT 目标数 | 预测数 | 漏检 | 误检 |", "|---|---|---|---|---|"]
    for _, r in det_sum.iterrows():
        lines.append(f"| {METHOD_LABEL.get(r['method'], r['method'])} | {int(r['n_gt'])} | {int(r['n_pred'])} | "
                     f"{int(r['n_miss'])} | {int(r['n_fp'])} |")
    lines += ["", "![overall](fig_overall.png)", "", "![by dataset](fig_by_dataset.png)", ""]
    if len(abl):
        lines += ["## 4. A1-A5 数据增强消融", "",
                  "| 配置 | 增强级别 | 最佳 epoch | 验证 Dice (精细化后) | 验证 Dice (粗糙) |",
                  "|---|---|---|---|---|"]
        for _, r in abl.iterrows():
            lines.append(f"| {r['tag']} | {int(r['level']) if pd.notna(r['level']) else '-'} | {r['best_epoch']} | "
                         f"{fmt(r['best_val_dice'])} | {fmt(r['rough_dice'])} |")
        lines += ["", "![ablation](fig_ablation.png)", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"\n-> {out/'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
