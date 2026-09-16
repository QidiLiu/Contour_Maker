"""汇总 EdgeSAM / EfficientSAM 对比: 精度 × 速度 联合表 + 图。

    PYTHONPATH=src ./.conda/bin/python scripts/20_sam_compare.py
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

LABEL = {
    "A_box": "A: YOLO26n + Otsu/box + contour-refiner",
    "B_seg": "B: YOLO26n-seg",
    "SAM_edgesam": "YOLO26n + EdgeSAM",
    "SAM_efficientvit-t": "YOLO26n+EfficientSAM(ViT-T)",
    "SAM_efficientvit-s": "YOLO26n+EfficientSAM(ViT-S)",
}


def main() -> int:
    ft = pd.read_csv(REPORTS / "bench_final_ft" / "accuracy_speed.csv")
    spd = pd.read_csv(REPORTS / "bench_speed_full" / "speed.csv")
    zs_path = REPORTS / "bench_zs" / "accuracy_speed.csv"
    zs = pd.read_csv(zs_path) if zs_path.exists() else None

    det = pd.read_csv(REPORTS / "bench_final_ft" / "per_image.csv")
    det = det[det["method"].str.endswith("__detstats")].copy()
    det["method"] = det["method"].str.replace("__detstats", "", regex=False)
    f1 = {}
    for m, g in det.groupby("method"):
        tp = int(g["n_gt"].sum() - g["n_miss"].sum())
        fp, fn = int(g["n_fp"].sum()), int(g["n_miss"].sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1[m] = 2 * p * r / (p + r) if p + r else 0.0

    ft = ft.rename(columns={"n_x": "n"})
    tbl = ft.drop(columns=[c for c in ("n_y", "end2end_ms_mean", "end2end_ms_median",
                                       "end2end_ms_std", "fps", "seg_only_ms_mean")
                           if c in ft.columns])
    tbl = tbl.merge(spd[["method", "end2end_ms_mean", "end2end_ms_median",
                         "end2end_ms_std", "fps"]], on="method")
    tbl["det_f1"] = tbl["method"].map(f1)
    tbl["dice_x_f1"] = tbl["dice"] * tbl["det_f1"]
    if zs is not None:
        zs_map = zs.set_index("method")["dice"].to_dict()
        tbl["dice_zeroshot"] = tbl["method"].map(zs_map)
    tbl["name"] = tbl["method"].map(lambda m: LABEL.get(m, m))
    cols = ["name", "n", "dice", "dice_std", "iou", "hd95", "assd", "bf1", "area_err", "fail",
            "dice_zeroshot", "det_f1", "dice_x_f1", "end2end_ms_mean", "end2end_ms_median",
            "end2end_ms_std", "fps"]
    tbl = tbl[[c for c in cols if c in tbl.columns]].sort_values("dice", ascending=False)
    out = REPORTS / "final"
    tbl.to_csv(out / "sam_compare.csv", index=False)

    # ---- 图: 精度-速度 散点
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in tbl.iterrows():
        ax.scatter(r["end2end_ms_mean"], r["dice"], s=90)
        ax.annotate(r["name"], (r["end2end_ms_mean"], r["dice"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("End-to-end latency per image (ms, RTX 4060 Ti)")
    ax.set_ylabel("Dice on test set")
    ax.set_title("Accuracy vs. inference speed (same training/test split)")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, tbl["end2end_ms_mean"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(out / "fig_acc_speed.png", dpi=150)
    plt.close(fig)

    print("=== 精度 × 速度 联合对比 ===")
    show = tbl[["name", "dice", "iou", "hd95", "dice_zeroshot", "fail", "end2end_ms_mean", "fps"]]
    print(show.round(4).to_string(index=False))
    print(f"\n-> {out/'sam_compare.csv'}, {out/'fig_acc_speed.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
