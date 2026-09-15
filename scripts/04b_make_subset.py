"""构建平衡子集 YOLO 数据集 (符号链接图像, 复制标签), 用于可承受时间内完成完整对比。

默认: busi 600 + tn3k 1000 + ddti 600 = 2200 张, 与主实验的 train/val/test 划分严格一致。

    PYTHONPATH=src ./.conda/bin/python scripts/04b_make_subset.py --n-busi 600 --n-tn3k 1000 --n-ddti 600
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import ROOT, SPLITS  # noqa: E402

SRC = ROOT / "data" / "yolo"
DST = ROOT / "data" / "yolo_subset"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-busi", type=int, default=600)
    ap.add_argument("--n-tn3k", type=int, default=1000)
    ap.add_argument("--n-ddti", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    quota = {"busi": args.n_busi, "tn3k": args.n_tn3k, "ddti": args.n_ddti}
    sp = pd.read_csv(SPLITS / "splits.csv")
    sp = sp[sp["dataset"].isin(quota)]
    rng = pd.Series(range(len(sp)))
    chosen = []
    for (ds, split), grp in sp.groupby(["dataset", "split"]):
        q = quota[ds]
        frac = {"train": 0.70, "val": 0.15, "test": 0.15}[split]
        n = max(1, int(round(q * frac)))
        n = min(n, len(grp))
        picked = grp.sample(n=n, random_state=args.seed)
        chosen.append(picked)
    sub = pd.concat(chosen)
    print(sub.groupby(["dataset", "split"]).size().to_string())

    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels_seg" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels_det" / split).mkdir(parents=True, exist_ok=True)
    for _, r in sub.iterrows():
        uid = r["uid"]
        for kind in ("det", "seg"):
            lab_src = SRC / f"labels_{kind}" / r["split"] / f"{uid}.txt"
            if lab_src.exists():
                shutil.copy2(lab_src, DST / f"labels_{kind}" / r["split"] / f"{uid}.txt")
        img_src = SRC / "images" / r["split"] / f"{uid}.png"
        if img_src.exists():
            (DST / "images" / r["split"] / f"{uid}.png").symlink_to(img_src.resolve())
    (DST / "_subset_splits.csv").write_text(sub.to_csv(index=False))
    (DST / "EXPORT_NOTE.md").write_text(
        "平衡子集数据集: 图像为符号链接, 标签为副本。\n"
        "训练脚本会把 labels_det 或 labels_seg 链接为 labels/。\n")
    n_img = sum(1 for _ in (DST / "images").rglob("*.png"))
    print(f"[subset] {n_img} 张 -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
