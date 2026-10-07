"""训练 YOLO26n-seg（单阶段端到端实例分割）。

用法:
    python train_yolo_seg.py --data ../data/yolo_seg/data_seg.yaml \
        --epochs 60 --imgsz 512 --batch 12 --name seg_yolo26n
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    CLASS_NAMES,
    RUNS,
    YOLO_BATCH_SEG,
    YOLO_EPOCHS_SEG,
    YOLO_IMGSZ,
    YOLO_MODEL_SEG,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="训练 YOLO26n-seg")
    ap.add_argument("--data", default="../data/yolo_seg/data_seg.yaml")
    ap.add_argument("--model", default=None, help=f"预训练权重, 默认 {YOLO_MODEL_SEG}")
    ap.add_argument("--name", default="seg_yolo26n")
    ap.add_argument("--epochs", type=int, default=YOLO_EPOCHS_SEG)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--batch", type=int, default=YOLO_BATCH_SEG)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model_path = args.model
    if model_path is None:
        local = Path(__file__).resolve().parents[1] / "weights" / YOLO_MODEL_SEG
        model_path = str(local) if local.exists() else YOLO_MODEL_SEG

    print(f"[cfg ] classes={CLASS_NAMES}")
    print(f"[cfg ] model={model_path}")
    print(f"[cfg ] data={args.data}")

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
        # 灰度单通道：关闭颜色类增强；分割任务开 copy_paste
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        degrees=15.0, translate=0.1, scale=0.3, shear=5.0,
        perspective=0.0, flipud=0.5, fliplr=0.5,
        mosaic=0.5, mixup=0.0, copy_paste=0.1, erasing=0.0,
        optimizer="auto",
    )
    best = RUNS / args.name / "weights" / "best.pt"
    print(f"[done] -> {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())