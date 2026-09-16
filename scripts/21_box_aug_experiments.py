"""框初始化数据增强消融 (refiner 结构不变)。

组别:
  ctrl       : 无框增强 (基线, box_init_prob=0.25, 与主结果 A-box 同配置)
  C1C2       : 边抖动 + 宽高缩放/平移
  C3C4       : 内部内缩 + 角点抖动
  C1C2C3C4   : 全部形状增强
  p50_aug    : box_init_prob=0.5 + 全部形状增强

每组: 生成样本缓存 -> 训练 (同 epoch/同超参) -> 在共享 test split 上端到端评估。
    PYTHONPATH=src ./.conda/bin/python scripts/21_box_aug_experiments.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import CONDA_PY, REPORTS, ROOT, RUNS  # noqa: E402

LOGS = ROOT / "logs"
SUB = "data/yolo_subset/_subset_splits.csv"
DET = "runs/det_yolo26n_sub/weights/best.pt"
SEG = "runs/seg_yolo26n_sub/weights/best.pt"
REFINER_MAIN = "runs/refiner_v3_A5_busi-ddti-tn3k/best.pt"
RUN_SUFFIX = "busi-ddti-tn3k"

GROUPS = [
    # (tag, 缓存子目录, 缓存额外参数, 训练超参)
    ("ctrl", "v6_ctrl", ["--box-init-prob", "0.25", "--box-aug", "0"], []),
    ("C1C2", "v6_c1c2", ["--box-init-prob", "0.25", "--box-aug", "1",
                         "--box-interior-offset", "0", "--box-corner-jitter", "0"], []),
    ("C3C4", "v6_c3c4", ["--box-init-prob", "0.25", "--box-aug", "1",
                         "--box-side-jitter", "0", "--box-scale-jitter", "0",
                         "--box-shift-jitter", "0"], []),
    ("C1C2C3C4", "v6_all", ["--box-init-prob", "0.25", "--box-aug", "1"], []),
    ("p50_aug", "v6_p50", ["--box-init-prob", "0.5", "--box-aug", "1"], []),
]


def run(cmd: list[str], log: Path) -> int:
    print(f"\n>>> {' '.join(cmd)}\n    log -> {log}", flush=True)
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n\n===== {time.strftime('%F %T')} :: {' '.join(cmd)}\n")
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"<<< exit={p.returncode}  {time.time()-t0:.0f}s", flush=True)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    for tag, cdir, cache_args, extra in GROUPS:
        if args.only and tag not in args.only:
            continue
        cache = f"data/cache/{cdir}"
        if not (ROOT / cache / "samples.parquet").exists():
            run([str(CONDA_PY), "scripts/03b_cache_samples.py", "--level", "5", "--variants", "3",
                 "--datasets", "busi", "tn3k", "ddti", "--out-subdir", cdir, *cache_args],
                LOGS / f"boxaug_cache_{tag}.log")
        run_name = f"refiner_v6_{tag}_{RUN_SUFFIX}"
        if not (ROOT / "runs" / run_name / "best.pt").exists():
            run([str(CONDA_PY), "scripts/03_train_refiner.py", "--level", "5", "--tag", f"v6_{tag}",
                 "--datasets", "busi", "tn3k", "ddti", "--cache-dir", cache,
                 "--batch-size", "64", "--num-workers", "4", "--epochs", str(args.epochs), *extra],
                LOGS / f"boxaug_train_{tag}.log")
        ev = f"reports/boxaug_eval_{tag}"
        if not (ROOT / ev / "summary_overall.csv").exists():
            run([str(CONDA_PY), "scripts/07_evaluate.py", "--split", "test", "--uids-file", SUB,
                 "--refiner", f"runs/{run_name}/best.pt", "--det", DET, "--seg", SEG,
                 "--imgsz", "512", "--device", "cuda", "--save-vis", "0", "--out", ev],
                LOGS / f"boxaug_eval_{tag}.log")
    print("\n完成. 汇总: PYTHONPATH=src ./.conda/bin/python scripts/22_box_aug_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
