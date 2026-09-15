"""按数据集分层生成 train/val/test 划分 (全实验共用同一套划分, 保证公平对比)。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import SPLITS, UNIFIED  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(UNIFIED / "manifest.csv")
    rng = np.random.default_rng(args.seed)
    parts = []
    for (ds, mod), grp in df.groupby(["dataset", "modality"]):
        idx = rng.permutation(len(grp))
        n = len(grp)
        n_test = int(round(n * args.test))
        n_val = int(round(n * args.val))
        assign = np.array(["train"] * n, dtype=object)
        assign[idx[:n_test]] = "test"
        assign[idx[n_test:n_test + n_val]] = "val"
        g = grp.copy()
        g["split"] = assign
        parts.append(g)
        print(f"{ds:6s} total={n:5d} train={int((assign=='train').sum()):5d} "
              f"val={int((assign=='val').sum()):4d} test={int((assign=='test').sum()):4d}")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(SPLITS / "splits.csv", index=False)
    (SPLITS / "splits_meta.json").write_text(json.dumps(
        dict(seed=args.seed, val=args.val, test=args.test,
             counts=out.groupby(["split"]).size().to_dict()), indent=2))
    print(f"\n-> {SPLITS / 'splits.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
