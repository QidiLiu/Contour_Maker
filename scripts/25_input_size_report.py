"""输入尺寸对比的汇总图表。

    PYTHONPATH=src ./.conda/bin/python scripts/25_input_size_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import REPORTS  # noqa: E402

LABEL = {"A_box": "A-box (box+contour-refiner)", "B_seg": "B (YOLO26n-seg)",
         "SAM_edgesam": "YOLO26n+EdgeSAM"}


def main() -> int:
    out = REPORTS / "input_size"
    acc = pd.read_csv(out / "accuracy_speed.csv")
    spd = pd.read_csv(out / "speed.csv")
    acc["name"] = acc["method"].map(LABEL)
    spd["name"] = spd["method"].map(LABEL)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    sizes = sorted(acc["size"].unique())
    methods = ["A_box", "B_seg", "SAM_edgesam"]
    w = 0.25
    x = np.arange(len(sizes))

    for i, m in enumerate(methods):
        a = acc[acc["method"] == m].set_index("size").reindex(sizes)
        s = spd[spd["method"] == m].set_index("size").reindex(sizes)
        axes[0].bar(x + (i - 1) * w, a["dice"], w, label=LABEL[m])
        axes[1].bar(x + (i - 1) * w, s["end2end_ms_mean"], w, label=LABEL[m])
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{s}x{s}" for s in sizes])
    axes[0].set_ylabel("Dice"); axes[0].set_title("Segmentation accuracy vs input size")
    axes[0].grid(axis="y", alpha=0.3); axes[0].legend(fontsize=7)

    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{s}x{s}" for s in sizes])
    axes[1].set_ylabel("End-to-end latency (ms)")
    axes[1].set_title("Inference speed vs input size")
    axes[1].grid(axis="y", alpha=0.3); axes[1].legend(fontsize=7)

    # 检测指标 (三方案共用同一检测器, 每个尺寸只画一次)
    det = acc.drop_duplicates("size").set_index("size").reindex(sizes)
    axes[2].plot(x, det["det_precision"], "o-", label="Precision")
    axes[2].plot(x, det["det_recall"], "s-", label="Recall")
    axes[2].plot(x, det["det_f1"], "^-", label="F1")
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"{s}x{s}" for s in sizes])
    axes[2].set_ylabel("Detection (IoU>=0.5)")
    axes[2].set_title("Detection quality vs input size")
    axes[2].grid(alpha=0.3); axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig_input_size.png", dpi=150)
    plt.close(fig)

    print("=== 精度 ===")
    print(acc[["size", "name", "n", "dice", "iou", "hd95", "assd", "bf1", "fail"]].round(4).to_string(index=False))
    print("\n=== 速度 ===")
    print(spd[["size", "name", "end2end_ms_mean", "end2end_ms_median", "fps"]].round(2).to_string(index=False))
    print("\n=== 检测 (共用检测器) ===")
    print(acc.drop_duplicates("size")[["size", "det_precision", "det_recall", "det_f1"]].round(4).to_string(index=False))
    print(f"\n-> {out/'fig_input_size.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
