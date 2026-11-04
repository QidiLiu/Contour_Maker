"""Train the RF-DETR-Seg-Nano scheme.

RF-DETR consumes COCO instance-segmentation annotations.  Build them with
``python -m common.coco_export --dataset ...`` first.

Usage::

    python train_rfdetr_seg.py --data ../data/coco --name rfdetr_seg_ultrasound
    python train_rfdetr_seg.py --dataset ade20k --name rfdetr_seg_ade20k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KIND = "rfdetr_seg"

from common.config import (  # noqa: E402
    ADE20K_CLASS_NAMES,
    ADE20K_COCO,
    CLASS_NAMES,
    COCO,
    RFDETR_BATCH,
    RFDETR_EPOCHS,
    RFDETR_GRAD_ACCUM,
    RFDETR_LR,
    RFDETR_RESOLUTION,
    RFDETR_WEIGHTS,
    RUNS,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train RF-DETR-Seg-Nano")
    ap.add_argument("--dataset", choices=["ultrasound", "ade20k"], default="ultrasound")
    ap.add_argument("--data", default=None, help="COCO root (train/ valid/ test/)")
    ap.add_argument("--weights", default=RFDETR_WEIGHTS, help="pretrained checkpoint name or path")
    ap.add_argument("--name", default="rfdetr_seg_ultrasound")
    ap.add_argument("--epochs", type=int, default=RFDETR_EPOCHS)
    ap.add_argument("--batch", type=int, default=RFDETR_BATCH)
    ap.add_argument("--grad-accum", type=int, default=RFDETR_GRAD_ACCUM)
    ap.add_argument("--lr", type=float, default=RFDETR_LR)
    ap.add_argument("--resolution", type=int, default=RFDETR_RESOLUTION)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--class-names", default=None)
    args = ap.parse_args()

    ade = args.dataset == "ade20k"
    data = Path(args.data) if args.data else (ADE20K_COCO if ade else COCO)
    class_names = (
        [c.strip() for c in args.class_names.split(",") if c.strip()]
        if args.class_names
        else (ADE20K_CLASS_NAMES if ade else CLASS_NAMES)
    )
    if not (data / "train" / "_annotations.coco.json").exists():
        print(
            f"[error] {data}/train/_annotations.coco.json not found; "
            "run `python -m common.coco_export` first"
        )
        return 1
    if args.resolution % 12:
        print("[error] RF-DETR-Seg-Nano resolution must be a multiple of 12")
        return 1

    from rfdetr import RFDETRSegNano

    model = RFDETRSegNano(
        pretrain_weights=args.weights,
        resolution=args.resolution,
        # class count is auto-aligned from the COCO categories by train()
    )
    out_dir = RUNS / args.name
    print(f"[cfg ] scheme=rfdetr-seg dataset={args.dataset} data={data}")
    print(
        f"[cfg ] resolution={args.resolution} epochs={args.epochs} batch={args.batch} "
        f"grad_accum={args.grad_accum} classes={class_names}"
    )
    print(f"[cfg ] output={out_dir}")

    model.train(
        dataset_dir=str(data),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        resolution=args.resolution,
        output_dir=str(out_dir),
        device=args.device,
        num_workers=args.workers,
        class_names=class_names,
        seed=args.seed,
        resume=args.resume,
        tensorboard=False,
        save_dataset_grids=False,
        run_test=False,
        eval_interval=1,
        progress_bar=True,
        notes={"scheme": "rfdetr-seg", "dataset": args.dataset},
    )
    print(f"[done] checkpoints under {out_dir}")
    for name in (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "last.pth",
    ):
        p = out_dir / name
        if p.exists():
            print(f"  {name}: {p}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
