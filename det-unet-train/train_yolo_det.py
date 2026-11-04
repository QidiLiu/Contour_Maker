"""训练 YOLO26n 检测器（Det-UNet 第一级）。

检测器负责「目标在哪 + 是什么类别」，其框与类别将引导 MK-UNet 做前景/背景分割。

用法:
    python train_yolo_det.py --data ../data/yolo_det/data_det.yaml \
        --epochs 60 --imgsz 512 --batch 16 --name det_yolo26n
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KIND = "det"

from common.config import (  # noqa: E402
    RUNS,
    YOLO_BATCH_DET,
    YOLO_EPOCHS_DET,
    YOLO_IMGSZ,
    YOLO_MODEL_DET,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="训练 YOLO26n 检测器")
    ap.add_argument("--dataset", choices=["ultrasound", "ade20k"], default="ultrasound")
    ap.add_argument("--data", default=None)
    ap.add_argument("--model", default=None, help=f"预训练权重, 默认 {YOLO_MODEL_DET}")
    ap.add_argument("--name", default="det_yolo26n")
    ap.add_argument("--epochs", type=int, default=YOLO_EPOCHS_DET)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--batch", type=int, default=YOLO_BATCH_DET)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    # 预训练权重：优先项目内 weights/，否则用 ultralytics 的官方名字自动下载
    model_path = args.model
    if model_path is None:
        local = Path(__file__).resolve().parents[1] / "weights" / YOLO_MODEL_DET
        model_path = str(local) if local.exists() else YOLO_MODEL_DET

    if args.data is None:
        base = "ade20k/" if args.dataset == "ade20k" else ""
        args.data = f"../data/{base}yolo_{KIND}/data_{KIND}.yaml"
    print(f"[cfg ] dataset={args.dataset} data={args.data}")
    print(f"[cfg ] model={model_path}")

    model = YOLO(model_path)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS),
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        exist_ok=True,
        resume=args.resume,
        plots=True,
        val=True,
        deterministic=True,
        # 超声图像为灰度单通道，关闭颜色类增强；开启上下翻转（探头方向任意）
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        shear=5.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        erasing=0.0,
        optimizer="auto",
    )
    best = RUNS / args.name / "weights" / "best.pt"
    print(f"[done] -> {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
