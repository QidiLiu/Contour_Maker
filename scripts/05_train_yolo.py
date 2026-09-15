"""训练 YOLO26n (检测) / YOLO26n-seg (分割) baseline。

ultralytics 会在 images 的同级目录查找 labels/, 因此这里先把 data/yolo/labels_det
或 labels_seg 软链接为 data/yolo/labels, 训练完再恢复。

用法:
    PYTHONPATH=src ./.conda/bin/python scripts/05_train_yolo.py --kind det --model yolo26n.pt
    PYTHONPATH=src ./.conda/bin/python scripts/05_train_yolo.py --kind seg --model yolo26n-seg.pt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import ROOT, RUNS  # noqa: E402

YOLO_DIR = ROOT / "data" / "yolo"


def use_labels(kind: str, yolo_dir: Path) -> Path:
    """把 labels_<kind> 链接为 labels/。"""
    link = yolo_dir / "labels"
    target = yolo_dir / f"labels_{kind}"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            shutil.rmtree(link)
    link.symlink_to(target, target_is_directory=True)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["det", "seg"], required=True)
    ap.add_argument("--model", default=None, help="预训练权重, 例如 weights/yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--yolo-dir", default=str(YOLO_DIR), help="YOLO 数据集根目录")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    yolo_dir = Path(args.yolo_dir).resolve()
    model_name = args.model or str(ROOT / "weights" / ("yolo26n-seg.pt" if args.kind == "seg" else "yolo26n.pt"))
    name = args.name or f"{args.kind}_yolo26n"
    use_labels(args.kind, yolo_dir)

    # 临时 data yaml: 共享 images/, 标签目录通过软链 labels/ 指向对应 kind
    data_yaml = yolo_dir / f"_run_{args.kind}.yaml"
    data_yaml.write_text(
        f"path: {yolo_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n  0: lesion\n")

    from ultralytics import YOLO
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, project=str(RUNS), name=name, seed=args.seed,
        patience=args.patience, exist_ok=True, resume=args.resume, fraction=args.fraction,
        plots=True, val=True, deterministic=True,
        # 超声图像增强 (灰度单通道, 关掉颜色类增强)
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=15.0, translate=0.1, scale=0.3,
        shear=5.0, perspective=0.0, flipud=0.5, fliplr=0.5, mosaic=0.5, mixup=0.0,
        copy_paste=0.1 if args.kind == "seg" else 0.0, erasing=0.1, crop_fraction=1.0,
        optimizer="auto",
    )
    best = RUNS / name / "weights" / "best.pt"
    print(f"[done] kind={args.kind} -> {best}")
    _ = os.environ
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
