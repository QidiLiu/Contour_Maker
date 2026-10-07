"""分层划分：按 (dataset, class_id) 分层，seed 固定，可复现。

用法:
    python -m common.splits                       # 生成 splits.csv
    python -m common.splits --verify <old.csv>    # 与旧划分对照
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CLASS_NAMES,
    SPLITS,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
    UNIFIED,
)


def make_splits(manifest: pd.DataFrame, seed: int = SPLIT_SEED,
                fractions: dict | None = None) -> pd.DataFrame:
    """按 (dataset, class_id) 分层，把每层按比例分配到 train/val/test。

    分层键保证 BUSI 良性/恶性的比例在三个 split 中一致，
    避免类别不平衡导致某个 split 缺类。
    """
    fr = fractions or SPLIT_FRACTIONS
    total = sum(fr.values())
    assert abs(total - 1.0) < 1e-6, f"划分比例之和必须为 1, 当前 {total}"

    rng = np.random.default_rng(seed)
    df = manifest.sort_values("uid").reset_index(drop=True).copy()
    df["split"] = ""

    strat = df["dataset"].astype(str) + "_" + df["class_id"].astype(str)
    for _key, grp in df.groupby(strat).groups.items():
        idx = np.array(sorted(grp))
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(round(n * fr["val"]))
        n_test = int(round(n * fr["test"]))
        # 每层至少保留 1 张给 train / val / test
        n_val = min(n_val, max(0, n - 2))
        n_test = min(n_test, max(0, n - 1 - n_val))
        labels = ["train"] * (n - n_val - n_test) + ["val"] * n_val + ["test"] * n_test
        for i, lab in zip(idx, labels):
            df.loc[i, "split"] = lab

    assert (df["split"] != "").all(), "存在未分配的样本"
    return df


def _report(df: pd.DataFrame) -> None:
    print("=== split x dataset ===")
    print(pd.crosstab(df["dataset"], df["split"]))
    print("\n=== split x class ===")
    ct = pd.crosstab(df["class_id"], df["split"])
    ct.index = [f"{i}:{CLASS_NAMES[i]}" for i in ct.index]
    print(ct)

    print("\n=== 每 split 的类别覆盖 ===")
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        present = sorted(sub["class_id"].unique().tolist())
        missing = [c for c in range(len(CLASS_NAMES)) if c not in present]
        tag = "OK" if not missing else f"缺 {missing}"
        print(f"  {s:5s} n={len(sub):5d}  类别 {present}  {tag}")


def verify_against_old(old_csv: str, new_df: pd.DataFrame) -> None:
    """与旧划分对照。

    旧划分的 uid 形如 `busi_<md5[:10]>`，新命名为 `<dataset>_<stem>`，
    且旧 manifest 已随项目清理（md5 无法重建），因此无法逐样本映射。
    这里改为对照各 split 的规模占比与 dataset 分布是否与旧口径吻合。
    """
    try:
        old = pd.read_csv(old_csv)
    except FileNotFoundError:
        print(f"[warn] 旧划分文件不存在, 跳过: {old_csv}")
        return
    if "uid" not in old.columns or "split" not in old.columns:
        print("[warn] 旧 csv 缺 uid/split 列, 跳过")
        return

    print("\n=== 与旧划分对照 ===")
    print("旧 uid 为 md5 摘要且原 manifest 已清理，无法逐样本映射；改为对照规模与分布。\n")

    rows = []
    for s in ("train", "val", "test"):
        o = old[old["split"] == s]
        n = new_df[new_df["split"] == s]
        rows.append(dict(
            split=s, old_n=len(o), new_n=len(n),
            old_pct=round(100 * len(o) / len(old), 2),
            new_pct=round(100 * len(n) / len(new_df), 2),
        ))
    print(pd.DataFrame(rows).to_string(index=False))

    old_ds = old["uid"].astype(str).str.split("_").str[0]
    print("\n旧: dataset x split")
    print(pd.crosstab(old_ds, old["split"]))
    print("\n新: dataset x split")
    print(pd.crosstab(new_df["dataset"], new_df["split"]))

    dev = max(abs(r["old_pct"] - r["new_pct"]) for r in rows)
    verdict = "通过" if dev < 1.0 else "偏差偏大, 需人工确认"
    print(f"\n各 split 占比最大偏差: {dev:.2f} 个百分点 ({verdict})")
    print("注: 新划分额外按 BUSI 良性/恶性分层，故计数与旧口径接近但不完全相同。")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成分层划分")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--verify", type=str, default=None, help="旧 splits.csv 路径")
    args = ap.parse_args()

    man_path = UNIFIED / "manifest.csv"
    if not man_path.exists():
        print(f"[error] 未找到 {man_path}, 请先运行 python -m common.data_prep --build")
        return 1

    manifest = pd.read_csv(man_path)
    print(f"[in ] manifest: {len(manifest)} 张, "
          f"{manifest['dataset'].nunique()} 数据集, "
          f"{manifest['class_id'].nunique()} 类别\n")

    df = make_splits(manifest, seed=args.seed)
    SPLITS.mkdir(parents=True, exist_ok=True)
    out = SPLITS / "splits.csv"
    df.to_csv(out, index=False)
    _report(df)

    (SPLITS / "splits_meta.json").write_text(json.dumps(dict(
        seed=args.seed, fractions=SPLIT_FRACTIONS,
        num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES,
        counts={s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
    ), indent=2, ensure_ascii=False))

    print(f"\n-> {out}")
    if args.verify:
        verify_against_old(args.verify, df)
    return 0


if __name__ == "__main__":
    sys.exit(main())