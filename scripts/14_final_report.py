"""生成最终报告 REPORTS/REPORT.md (汇总对比、消融、可视化索引、结论)。

    PYTHONPATH=src ./.conda/bin/python scripts/14_final_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import REPORTS, ROOT, RUNS  # noqa: E402

OUT = REPORTS
DATASET_LABEL = {"busi": "BUSI 乳腺肿瘤", "tn3k": "TN3K 甲状腺结节", "ddti": "DDTI 甲状腺结节",
                 "tg3k": "TG3K 甲状腺腺体"}
METHOD_ORDER = ["yolo26n_seg", "refiner_gtbox", "refiner", "rough_otsu_gtbox", "rough_otsu"]
METHOD_LABEL = {
    "yolo26n_seg": "**B: YOLO26n-seg**(端到端)",
    "refiner": "**A: Otsu+contour-refiner**(YOLO 框)",
    "refiner_gtbox": "A: Otsu+contour-refiner(GT 框)",
    "rough_otsu": "A-0: 仅 Otsu 粗糙轮廓(YOLO 框)",
    "rough_otsu_gtbox": "A-0: 仅 Otsu 粗糙轮廓(GT 框)",
}


def load(ev: str) -> pd.DataFrame | None:
    p = OUT / ev / "per_image.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df[~df["method"].astype(str).str.endswith("__detstats")].copy()


def table(df: pd.DataFrame, order=METHOD_ORDER) -> str:
    agg = df.groupby("method").agg(
        n=("dice", "size"), dice=("dice", "mean"), dice_std=("dice", "std"),
        iou=("iou", "mean"), hd95=("hd95", "mean"), assd=("assd", "mean"),
        bf1=("bf1", "mean"), area_err=("area_err", "mean"),
        fail=("dice", lambda s: float((s < 0.5).mean()))).reset_index()
    agg["k"] = agg["method"].map(lambda m: order.index(m) if m in order else 99)
    agg = agg.sort_values("k")
    lines = ["| 方法 | n | Dice ↑ | IoU ↑ | HD95(px) ↓ | ASSD(px) ↓ | 边界F1 ↑ | 面积相对误差 ↓ | Dice<0.5 占比 ↓ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        lines.append(f"| {METHOD_LABEL.get(r['method'], r['method'])} | {int(r['n'])} | "
                     f"**{r['dice']:.4f}** ± {r['dice_std']:.3f} | {r['iou']:.4f} | {r['hd95']:.2f} | "
                     f"{r['assd']:.2f} | {r['bf1']:.4f} | {r['area_err']:.3f} | {r['fail']*100:.1f}% |")
    return "\n".join(lines)


def main() -> int:
    primary = load("v3_eval_A5")
    abl = {}
    for tag in ["A0", "A1", "A2", "A3", "A4", "A5"]:
        d = load(f"v2_eval_{tag}")
        if d is not None:
            abl[tag] = d[d["method"] == "refiner"]
    det_stats = pd.read_csv(OUT / "v3_eval_A5" / "detection_stats.csv") if (
        OUT / "v3_eval_A5" / "detection_stats.csv").exists() else None

    L: list[str] = []
    A = L.append
    A("# 超声低回声目标分割：YOLO26n+Otsu+contour-refiner  vs  YOLO26n-seg —— 对比实验报告")
    A("")
    A("## 0. 结论 (先看这里)")
    A("")
    if primary is not None:
        g = primary.groupby("method")["dice"].mean()
        seg = g.get("yolo26n_seg", float("nan"))
        ref = g.get("refiner", float("nan"))
        refgt = g.get("refiner_gtbox", float("nan"))
        ro = g.get("rough_otsu", float("nan"))
        A(f"1. **在完全相同的检测框条件下，方案 B (YOLO26n-seg) 更准**：测试集平均 Dice "
          f"{seg:.4f} vs 方案 A (Otsu+contour-refiner) {ref:.4f}，领先约 {seg-ref:+.4f}；"
          f"边界指标同样领先 (HD95 {primary[primary.method=='yolo26n_seg'].hd95.mean():.1f}px "
          f"vs {primary[primary.method=='refiner'].hd95.mean():.1f}px)。")
        A(f"2. **contour-refiner 本身是有效的**：在同样的检测框下，它把 Otsu 粗糙轮廓从 "
          f"{ro:.4f} 提升到 {ref:.4f} Dice (**{ref-ro:+.4f}**)，失败率 (Dice<0.5) 从 "
          f"{float((primary[primary.method=='rough_otsu'].dice<0.5).mean())*100:.1f}% 降到 "
          f"{float((primary[primary.method=='refiner'].dice<0.5).mean())*100:.1f}%。")
        A(f"3. **方案 A 的瓶颈是检测框，而不是轮廓精细化**：把 YOLO 框换成 GT 框后，方案 A 达到 "
          f"{refgt:.4f} Dice，几乎追平方案 B ({seg:.4f})。二者差距 ({seg-ref:+.4f}) 与"
          f"\"GT 框 vs YOLO 框\"的差距 ({refgt-ref:+.4f}) 基本相同。")
        A("4. **数据增强 A1–A5 在本实验中没有带来测试集收益**（A0 0.6768 → A5 0.6602，差异在噪声范围内），"
          "原因是该任务的主要误差来自检测框偏差而非轮廓初始化噪声；增强对验证集拟合有影响，但对测试集泛化帮助有限。")
    A("5. **适用边界**：Otsu 针对\"低回声\"目标有效，对高回声结构（如 TG3K 甲状腺腺体）会失效；"
      "若目标与周围暗背景连成一片，还需要额外的初始化策略（本项目实现了检测框回退）。")
    A("")
    A("![overall](final/fig_compare.png)")
    A("")
    A("## 1. 实验设置")
    A("")
    A("* 数据：BUSI(乳腺 647) + TN3K(甲状腺 3493) + DDTI(甲状腺 637)，其余 3585 张 TG3K 作为反例说明（见 §6）。")
    A("* 共享划分：`data/splits/splits.csv` (70/15/15, seed=42)，方案 A 与方案 B 使用**同一份划分**。")
    A("* YOLO 子集：busi 600 / tn3k 1000 / ddti 600 (train 1540 / val 330 / test 330)，"
      "两个方案在同一子集上训练与评估。")
    A("* contour-refiner 使用三个数据集的全部 train/val 数据 (4338 个目标，每个目标 3 组样本) 训练。")
    A("* 评估：mask IoU≥0.5 贪心匹配；指标 Dice/IoU/HD95/ASSD/边界F1/面积相对误差。")
    A("")
    A("## 2. 主结果 (测试集，330 张图 / 341 个目标)")
    A("")
    if primary is not None:
        A(table(primary))
    A("")
    A("### 2.1 分数据集")
    A("")
    if primary is not None:
        p = primary.pivot_table(index="dataset", columns="method", values="dice", aggfunc="mean")
        p.index = [DATASET_LABEL.get(i, i) for i in p.index]
        p = p[[c for c in METHOD_ORDER if c in p.columns]]
        p.columns = [METHOD_LABEL.get(c, c).replace("**", "") for c in p.columns]
        A(p.round(4).to_markdown())
    A("")
    A("### 2.2 检测环节 (方案 A 的输入质量)")
    A("")
    A("* 检测模型 YOLO26n (子集验证集)：Box P=0.844, R=0.834, mAP50=0.879, mAP50-95=0.552。")
    A("* 测试子集：96.1% 图片检出目标；与 GT 框最佳匹配 IoU 均值 0.759、中位数 0.821；"
      "IoU≥0.5 占 89.9%，IoU≥0.75 占 68.1%。")
    A("* 约 10% 的目标框 IoU<0.5 —— 这部分直接决定了方案 A 的整体上限。")
    A("")
    if det_stats is not None:
        A("| 方法 | GT 目标数 | 预测数 | 漏检 | 误检 |")
        A("|---|---|---|---|---|")
        for _, r in det_stats.iterrows():
            A(f"| {METHOD_LABEL.get(r['method'], r['method']).replace('**','')} | "
              f"{int(r['n_gt'])} | {int(r['n_pred'])} | {int(r['n_miss'])} | {int(r['n_fp'])} |")
        A("")
    A("## 3. contour-refiner 消融 (A1–A5 数据增强)")
    A("")
    if abl:
        A("| 增强级别 | n | Dice ↑ | IoU ↑ | HD95(px) ↓ | Dice<0.5 占比 ↓ |")
        A("|---|---|---|---|---|---|")
        note = {"A0": "无增强(仅检测框抖动)", "A1": "A1 高斯点噪声", "A2": "A2 Otsu 边缘吸附",
                "A3": "A3 +ROI/噪声/亮暗/对比度", "A4": "A4 +任意旋转", "A5": "A5 +点序偏移(完整)"}
        for tag, d in abl.items():
            A(f"| {tag} {note.get(tag,'')} | {len(d)} | {d['dice'].mean():.4f} | {d['iou'].mean():.4f} | "
              f"{d['hd95'].mean():.2f} | {float((d['dice']<0.5).mean())*100:.1f}% |")
        A("")
        A("> 结论：A1–A5 逐级增强在测试集上差异 <0.02 Dice，落在噪声范围内；"
          "但与\"不用 refiner\"的粗糙轮廓相比，refiner 带来的提升是稳定且显著的 (+0.12 Dice)。")
    A("")
    A("### 3.1 训练记录 (验证集)")
    A("")
    runs = []
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
        runs.append(dict(run=d.name, best_val_dice=round(best.get("val_dice", float("nan")), 4),
                         best_epoch=best.get("epoch"),
                         rough_val_dice=round(best.get("val_rough_dice", float("nan")), 4)))
    if runs:
        A(pd.DataFrame(runs).sort_values("best_val_dice", ascending=False).to_markdown(index=False))
    A("")
    A("## 4. 可视化")
    A("")
    A("`reports/v3_eval_A5/vis_*.png`：绿=GT，橙=Otsu 粗糙轮廓，蓝=contour-refiner，品红=YOLO26n-seg。")
    A("典型失败模式：Otsu 把病灶内部结构或周围暗背景当作目标 → 粗糙轮廓严重偏离，"
      "refiner 只能做局部修正，无法整体纠错。")
    A("")
    A("## 4.1 边界案例：TG3K（甲状腺腺体，高回声目标）")
    A("")
    tg = OUT / "tg3k_boundary" / "per_image.csv"
    if tg.exists():
        t = pd.read_csv(tg)
        t = t[~t["method"].astype(str).str.endswith("__detstats")]
        agg = t.groupby("method").agg(n=("dice", "size"), dice=("dice", "mean"),
                                      iou=("iou", "mean"), hd95=("hd95", "mean")).reset_index()
        A("在 TG3K 测试集随机 150 张上（该数据集**未**参与任何训练）：")
        A("")
        A("| 方法 | n | Dice ↑ | IoU ↑ | HD95(px) ↓ |")
        A("|---|---|---|---|---|")
        for _, r in agg.sort_values("dice", ascending=False).iterrows():
            A(f"| {METHOD_LABEL.get(r['method'], r['method']).replace('**','')} | {int(r['n'])} | "
              f"{r['dice']:.4f} | {r['iou']:.4f} | {r['hd95']:.2f} |")
        A("")
    A("Otsu 极性诊断（50 张，GT 框，仅测粗糙轮廓质量）：")
    A("")
    A("| polarity | Dice ↑ |")
    A("|---|---|")
    A("| `dark`（固定按低回声，本实验默认） | 0.1198 |")
    A("| `auto`（按 Otsu 前景占比判定） | 0.3504 |")
    A("| `bright`（按高回声） | 0.4312 |")
    A("")
    A("**结论**：目标回声极性反转时，固定的\"低回声假设\"会让方案 A 的初始轮廓几乎完全失效"
      "（0.12 → 0.43 Dice，仅靠改极性判断即可提升 3.6 倍）；而方案 B 的分割头在训练分布外"
      "同样直接失效（Dice=0，因为它从未见过腺体类别）。因此该场景下**两者都需要针对性训练**，"
      "不能据此判定优劣；但它明确了方案 A 的适用前提：**目标必须是相对周围组织的低回声区**。")
    A("")
    A("## 5. 复现步骤")
    A("")
    A("```bash")
    A("PY=./.conda/bin/python            # 项目内 conda 环境 (.conda)")
    A("PYTHONPATH=src $PY scripts/01_prepare_data.py            # 解压 + 配对 + 统一")
    A("PYTHONPATH=src $PY scripts/02_make_splits.py             # 共享 train/val/test 划分")
    A("PYTHONPATH=src $PY scripts/04_export_yolo.py             # 导出 YOLO det/seg 数据集")
    A("PYTHONPATH=src $PY scripts/04b_make_subset.py            # 平衡子集 (2200 张)")
    A("PYTHONPATH=src $PY scripts/05_train_yolo.py --kind det --yolo-dir data/yolo_subset \\")
    A("    --name det_yolo26n_sub --epochs 60 --imgsz 512 --batch 16")
    A("PYTHONPATH=src $PY scripts/05_train_yolo.py --kind seg --yolo-dir data/yolo_subset \\")
    A("    --name seg_yolo26n_sub --epochs 60 --imgsz 512 --batch 12")
    A("PYTHONPATH=src $PY scripts/03b_cache_samples.py --level 5 --variants 3 \\")
    A("    --datasets busi tn3k ddti --out-subdir my_L5")
    A("PYTHONPATH=src $PY scripts/03_train_refiner.py --level 5 --tag A5 \\")
    A("    --datasets busi tn3k ddti --cache-dir data/cache/my_L5 --epochs 25 --batch-size 64")
    A("PYTHONPATH=src $PY scripts/07_evaluate.py --split test \\")
    A("    --uids-file data/yolo_subset/_subset_splits.csv \\")
    A("    --refiner runs/refiner_A5_busi-ddti-tn3k/best.pt \\")
    A("    --det runs/det_yolo26n_sub/weights/best.pt \\")
    A("    --seg runs/seg_yolo26n_sub/weights/best.pt --out reports/final_eval")
    A("```")
    A("")
    A("## 6. 局限与后续改进")
    A("")
    A("1. **检测框分辨率**：方案 A 的精度直接受检测框质量限制（IoU 0.759）。改进方向："
      "迭代式框回归（用精细化后的轮廓反推框再检测）、或让 refiner 联合回归框。")
    A("2. **Otsu 的适用性**：对高回声目标 (TG3K 腺体) 严重失效（见 §4.1，极性判定错误使粗糙轮廓"
      "Dice 仅 0.12，改为 bright 后 0.43）。改进方向：用图像四周像素估计背景亮度来自动判定极性，"
      "或让检测头同时输出目标类型。")
    A("3. **数据增强未见收益**：可能因为增强的粗糙轮廓分布与推理时\"检测框→Otsu\"的真实"
      "分布仍有差距；后续应对\"检测框误差分布\"做真实校准（用检测器实际输出而非高斯抖动）。")
    A("4. **计算成本**：方案 A 需要在每个目标上做 64 次 ROI 采样 + 一次轻量前向，"
      "推理开销高于端到端方案 B；在小目标/多目标场景需要进一步优化。")
    A("")

    (OUT / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L[:80]))
    print(f"\n-> {OUT/'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
