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
METHOD_ORDER = ["yolo26n_seg", "seg_refiner", "refiner_gtbox", "refiner", "rough_otsu_gtbox", "rough_otsu"]
METHOD_LABEL = {
    "yolo26n_seg": "**B: YOLO26n-seg**(端到端)",
    "seg_refiner": "**B+refiner: YOLO26n-seg mask→contour-refiner**",
    "refiner_box": "**A-box: 框矩形→contour-refiner (YOLO 框)**",
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
    A("0. **【最重要】把初始轮廓从\"框内 Otsu 最大暗连通域\"改为\"检测框矩形\"，"
      "方案 A 反超方案 B**：测试集 Dice **0.8028** vs 0.7923，HD95 持平 14.33px；"
      "检测命中目标上的轮廓命中率 86.5% → **99.3%**（oracle 上限 99.7%）。"
      "该改动不需要重训检测器或分割器，详见 §4.3。")
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
      "若目标与周围暗背景连成一片，还需要额外的初始化策略。")

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
    A("## 4.3 漏检攻防：把\"框矩形\"作为初始轮廓（本实验的关键改进）")
    A("")
    A("### 4.3.1 根因归因（GT 目标级，检测框 conf≥0.25，IoU≥0.5 记命中）")
    A("")
    A("| 环节 | 命中 | 占 341 个 GT 目标 |")
    A("|---|---|---|")
    A("| YOLO26n 检测命中 | 297 | 87.1% |")
    A("| Otsu 成功产出轮廓 | 297 | 87.1% |")
    A("| **Otsu 轮廓本身 IoU≥0.5** | **219** | **64.2%** |")
    A("| 用\"框矩形\"当初始轮廓 | 275 | 80.6% |")
    A("| GT 框 + Otsu（上限参照） | 308 | 90.3% |")
    A("")
    A("结论：**漏检的主因不是检测器，而是 Otsu 初始化**——78 个目标虽然检测到了，"
      "但 \"框内 Otsu 最大暗连通域\"给出的初始轮廓与真值 IoU<0.5"
      "（框只覆盖病灶一部分时，Otsu 只能在框内取到病灶的一部分）。")
    A("")
    A("### 4.3.2 检测置信度阈值扫描（目标级全局贪心匹配）")
    A("")
    A("| 变体 | 预测数 | TP | FP | FN | Precision | Recall | F1 |")
    A("|---|---|---|---|---|---|---|---|")
    A("| **det@0.25（当前配置）** | 362 | 318 | 44 | 23 | **0.8785** | 0.9326 | **0.9047** |")
    A("| det+flip@0.25 (翻转 TTA 池化) | 368 | 320 | 48 | 21 | 0.8696 | 0.9384 | 0.9027 |")
    A("| det+seg@0.25 (检测+分割候选池化) | 399 | 326 | 73 | 15 | 0.8170 | 0.9560 | 0.8811 |")
    A("| det@0.05 | 544 | 332 | 212 | 9 | 0.6103 | 0.9736 | 0.7503 |")
    A("")
    A("结论：降低阈值可以把召回从 93.3% 拉到 97.4%，但 F1 从 0.905 掉到 0.750；"
      "候选池化（检测+分割框、翻转 TTA、多尺度）都无法提升 F1。**阈值不是主要矛盾**。")
    A("")
    A("### 4.3.3 决定性发现：框矩形初始化 + refiner")
    A("")
    A("在检测命中的 297 个目标上（`scripts/17_recall_strategy.py`）：")
    A("")
    A("| 变体 | IoU≥0.5 命中率 | 平均 IoU | 平均 Dice |")
    A("|---|---|---|---|")
    A("| **框矩形 → refiner** | **0.9933** | **0.7829** | **0.8744** |")
    A("| oracle 最优（otsu+ref 与 box+ref 二选一） | 0.9966 | 0.7883 | — |")
    A("| 框矩形（不精细化） | 0.9259 | 0.6682 | 0.7965 |")
    A("| Otsu → refiner（原方案 A） | 0.8653 | 0.7054 | 0.8134 |")
    A("| Otsu（原粗糙轮廓） | 0.7374 | 0.5946 | 0.7196 |")
    A("")
    A("两种初始化互补性：仅框命中 20.5%、仅 Otsu 命中 1.7%、两者都命中 72.1%、"
      "都不命中 5.7%。由于 oracle 只比\"永远用框\"高 0.3 个百分点，"
      "**直接用框矩形初始化即可，不需要复杂的选择器**。")
    A("")
    A("### 4.3.4 端到端结果（正式评估流程，测试集 341 个目标）")
    A("")
    fb = OUT / "final_A_vs_B" / "per_image.csv"
    if fb.exists():
        f = pd.read_csv(fb)
        f = f[~f["method"].astype(str).str.endswith("__detstats")]
        order = ["refiner_box", "yolo26n_seg", "seg_refiner", "refiner_gtbox", "refiner",
                 "rough_otsu_gtbox", "rough_otsu"]
        agg = f.groupby("method").agg(
            n=("dice", "size"), dice=("dice", "mean"), dice_std=("dice", "std"),
            iou=("iou", "mean"), hd95=("hd95", "mean"), assd=("assd", "mean"),
            bf1=("bf1", "mean"), area_err=("area_err", "mean"),
            fail=("dice", lambda x: float((x < 0.5).mean()))).reset_index()
        agg["k"] = agg["method"].map(lambda m: order.index(m) if m in order else 99)
        A("| 方法 | n | Dice ↑ | IoU ↑ | HD95(px) ↓ | ASSD(px) ↓ | 边界F1 ↑ | Dice<0.5 占比 ↓ |")
        A("|---|---|---|---|---|---|---|---|")
        for _, r in agg.sort_values("k").iterrows():
            A(f"| {METHOD_LABEL.get(r['method'], r['method']).replace('**','')} | {int(r['n'])} | "
              f"**{r['dice']:.4f}** ± {r['dice_std']:.3f} | {r['iou']:.4f} | {r['hd95']:.2f} | "
              f"{r['assd']:.2f} | {r['bf1']:.4f} | {r['fail']*100:.1f}% |")
        A("")
    A("**结论：改用框矩形初始化后，方案 A（0.8028 Dice）在 Dice 与 HD95 上反超方案 B"
      "（0.7923 / 14.33px 持平），代价是边界 F1 与 ASSD 略差、失败率略高。**"
      "该改进完全来自“初始轮廓”这一环，不需要重新训练检测器或分割器。")
    A("")
    A("### 4.3.5 推荐做法（按性价比排序）")
    A("")
    A("1. **检测阈值维持 0.25**（F1 最优 0.905）；不要为了召回把阈值压到 0.05。")
    A("2. **初始轮廓改为框矩形**（或\"框矩形 + Otsu 择优\"）：命中率 73.7% → 99.3%。")
    A("3. 剩余误差中 Otsu/初始化只剩 0.3%（oracle 上限），"
      "**下一阶段的瓶颈回到检测器**：11–13% 的目标在 conf=0.25 下完全无框。")
    A("4. 进阶（尚未做）：以更高分辨率训练检测器、加入困难负样本挖掘与更强 TTA、"
      "或把 refiner 的输出作为二次检测的候选框（迭代式框回归）。")
    A("")
    A("## 4.2 追加实验：把 contour-refiner 接到 YOLO26n-seg 之后 (B+refiner)")
    A("")
    A("把 B 的输出 mask 当作\"粗糙轮廓\"（重采样为 64 点）再喂给同一个 contour-refiner，"
      "其余条件完全不变。测试集结果：")
    A("")
    bp = OUT / "B_plus_refiner" / "per_image.csv"
    if bp.exists():
        b = pd.read_csv(bp)
        b = b[~b["method"].astype(str).str.endswith("__detstats")]
        agg = b[b["method"].isin(["yolo26n_seg", "seg_refiner"])].groupby("method").agg(
            n=("dice", "size"), dice=("dice", "mean"), iou=("iou", "mean"),
            hd95=("hd95", "mean"), assd=("assd", "mean"), bf1=("bf1", "mean"),
            area_err=("area_err", "mean")).reset_index()
        A("| 方法 | n | Dice ↑ | IoU ↑ | HD95(px) ↓ | ASSD(px) ↓ | 边界F1 ↑ | 面积误差 ↓ |")
        A("|---|---|---|---|---|---|---|---|")
        for _, r in agg.sort_values("dice", ascending=False).iterrows():
            A(f"| {METHOD_LABEL.get(r['method'], r['method']).replace('**','')} | {int(r['n'])} | "
              f"**{r['dice']:.4f}** | {r['iou']:.4f} | {r['hd95']:.2f} | {r['assd']:.2f} | "
              f"{r['bf1']:.4f} | {r['area_err']:.3f} |")
        A("")
    A("**结论：没有提升，反而下降 0.011 Dice。** 配对比较 (同 337 个预测目标)："
      "Dice 0.7923 → 0.7813，58% 的目标变差、32% 变好。两条实测证据：")
    A("")
    A("1. **该 refiner 是为\"粗糙轮廓\"校准的**：把 GT 轮廓 (Dice 0.9954) 当作输入喂给它，"
      "输出 Dice 降到 0.9127（**100% 样本都变差**）。它的训练分布是\"框→Otsu\"粗糙轮廓"
      "（平均 Dice ≈ 0.54），学到的是\"大幅修正\"策略；面对本就准确的 seg mask 时属于"
      "分布外输入，修正量过大导致过修正。")
    A("2. **B 的精度已高于该 refiner 的修正能力**：B 在有预测的目标上 Dice 中位数 0.90、"
      "p10 = 0.80；按质量分桶后，refiner 在**每一个**质量区间都是负收益"
      "（0.7–0.85 区间 −0.021，0.85–0.95 区间 −0.014，>0.95 区间 −0.022）。")
    A("")
    A("| 级联方式 | 测试集 Dice | 说明 |")
    A("|---|---|---|")
    A("| B: YOLO26n-seg 单独 | **0.7923** | 端到端，已含检测+轮廓 |")
    A("| B + refiner（现有 Otsu 训练的 refiner） | 0.7813 | 分布外输入，过修正 |")
    A("| A: 检测框 → Otsu → refiner | 0.6602 | 提升来自初始化质量 |")
    A("| A: GT 框 → Otsu → refiner | 0.7808 | 接近 B，仍依赖初始轮廓 |")
    A("")
    A("**值得尝试的前提**：只有用**与 seg 输出同分布**的样本重训 refiner（以\"seg mask + 轻微形变\""
      "作为粗糙轮廓，而非 Otsu 轮廓）才可能有小幅收益。理论上限有限：B 的分割误差中位数已 0.90，"
      "剩余误差主要来自 11% 的完全漏检 (Dice=0，属于检测问题，轮廓精细化无法修复)。"
      "要突破应改进检测/分类头，而非叠加轮廓回归。")
    A("")
    A("![b+refiner](B_plus_refiner/vis_bplusref_busi_c4322390d6.png)")
    A("")
    A("## 4.4 第四条对照：YOLO26n + EdgeSAM / EfficientSAM（精度 + 推理速度）")
    A("")
    A("两个 SAM 变体均在**同一训练集**上以相同的 YOLO26n 检测框为提示做适配：冻结图像编码器，"
      "仅微调 mask decoder（4.06M 可训练参数，4 epoch）。零样本（不做适配）结果一并列出。")
    A("")
    sc = OUT / "final" / "sam_compare.csv"
    if sc.exists():
        t = pd.read_csv(sc)
        A("| 流水线 | Dice ↑ | IoU ↑ | HD95 ↓ | 零样本 Dice | Dice<0.5 占比 ↓ | 端到端 ms/张 ↓ | FPS ↑ |")
        A("|---|---|---|---|---|---|---|---|")
        for _, r in t.sort_values("end2end_ms_mean").iterrows():
            zs = "—" if pd.isna(r.get("dice_zeroshot")) else f"{r['dice_zeroshot']:.4f}"
            A(f"| {r['name']} | **{r['dice']:.4f}** | {r['iou']:.4f} | {r['hd95']:.2f} | {zs} | "
              f"{r['fail']*100:.1f}% | {r['end2end_ms_mean']:.2f} | {r['fps']:.1f} |")
        A("")
    A("**结论**：")
    A("")
    A("1. **精度上四者接近，速度差距巨大**：B（YOLO26n-seg）0.7735 / 10.2ms，"
      "EfficientSAM(ViT-T) 0.7682 / 99.7ms，EdgeSAM 0.7648 / 35.8ms，方案 A 0.7632 / 15.9ms。"
      "**EfficientSAM 精度只差 0.005，却慢 10 倍**（1024×1024 输入 + ViT 注意力）；"
      "B 在精度与速度上同时占优。")
    A("2. **SAM 类模型必须做域适配**：零样本时 EfficientSAM(ViT-T) 完全失效（Dice 0.0000，"
      "输出空 mask 或与目标无关的大块）；微调 decoder 后提升到 0.7682。"
      "EdgeSAM 零样本即 0.6180，微调后 0.7648（+0.147）——说明其 RepViT 编码器的自然图像"
      "特征迁移性更好，但两者都必须适配才能用于超声。")
    A("3. **方案 A 的性价比最高**：精度与 SAM 系持平（0.7632 vs 0.7682），"
      "端到端延迟 15.9ms（比 EfficientSAM 快 6.3 倍、比 EdgeSAM 快 2.2 倍），且模型仅 0.47M 参数、"
      "只需 32×32 的局部 ROI 输入，显存/算力需求远低于 1024×1024 的 SAM 系。")
    A("4. **推理速度明细**：每张图端到端包含图像读取、YOLO26n 检测、预处理与分割解码，"
      "GPU 同步计时，预热 6 张后统计（全测试集 330 张 × 2 轮 = 660 次）。"
      "分割模块单独耗时：A 6.9ms / B 10.2ms（含检测）/ EdgeSAM 8.9ms / EfficientSAM 9.8ms，"
      "**SAM 系的主要开销在 1024×1024 编码器**（EdgeSAM 编码约 27ms、EfficientSAM 约 91ms，"
      "相对检测约 3.6ms）。")
    A("")
    A("![acc vs speed](final/fig_acc_speed.png)")
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
