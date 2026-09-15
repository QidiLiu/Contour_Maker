"""实验编排: 串行跑完 contour-refiner 的训练/消融。

GPU 资源有限 (RTX 4060 Ti 16GB), YOLO 训练与 refiner 训练串行执行, 避免显存竞争。

用法:
    PYTHONPATH=src ./.conda/bin/python scripts/08_run_experiments.py --stage ablations
    PYTHONPATH=src ./.conda/bin/python scripts/08_run_experiments.py --stage main
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import CONDA_PY, ROOT  # noqa: E402

LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

MAIN_DATASETS = ["busi", "tn3k", "ddti"]


def run(cmd: list[str], log_name: str) -> int:
    log = LOGS / log_name
    print(f"\n>>> {' '.join(cmd)}\n    log -> {log}", flush=True)
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n\n===== {time.strftime('%F %T')} :: {' '.join(cmd)}\n")
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"<<< exit={p.returncode} in {time.time()-t0:.0f}s", flush=True)
    return p.returncode


def train_refiner(level: int, tag: str, epochs: int, extra: list[str] | None = None,
                  datasets: list[str] | None = None, log: str = "refiner_train.log") -> int:
    cmd = [str(CONDA_PY), "scripts/03_train_refiner.py", "--level", str(level), "--tag", tag,
           "--epochs", str(epochs)]
    if datasets:
        cmd += ["--datasets", *datasets]
    cmd += extra or []
    return run(cmd, log)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["main", "ablations", "window", "seeds", "all"])
    ap.add_argument("--epochs-main", type=int, default=120)
    ap.add_argument("--epochs-abl", type=int, default=60)
    args = ap.parse_args()

    env_extra = ["--batch-size", "32", "--samples-per-target", "4"]

    if args.stage in ("ablations", "all"):
        # A0-A5 增强消融: 相同数据/网络/轮数, 唯一变量是增强级别
        for lvl in range(0, 6):
            train_refiner(lvl, f"A{lvl}", args.epochs_abl,
                          extra=env_extra + ["--feat-dim", "128"], datasets=MAIN_DATASETS)

    if args.stage in ("window", "all"):
        # ROI 窗口尺度消融 (window_ratio)
        for wr in (0.20, 0.35, 0.50):
            train_refiner(5, f"A5_wr{int(wr*100)}", max(40, args.epochs_abl // 2),
                          extra=env_extra + ["--window-ratio", str(wr)], datasets=MAIN_DATASETS)

    if args.stage in ("main", "all"):
        train_refiner(5, "A5", args.epochs_main, extra=env_extra, datasets=MAIN_DATASETS)

    if args.stage in ("seeds", "all"):
        for seed in (1, 2):
            train_refiner(5, f"A5_seed{seed}", args.epochs_abl,
                          extra=env_extra + ["--seed", str(seed)], datasets=MAIN_DATASETS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
