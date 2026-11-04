"""Export the RF-DETR-Seg-Nano scheme to ONNX (and optionally MNN).

The exported graph has three outputs, which is exactly what the C++
``rfdetr_seg_test`` pipeline expects::

    input   (1, 3, R, R)        RGB, mean/std normalised, R = resolution
    dets    (1, Q, 4)           normalised cx, cy, w, h (Q = num_queries)
    labels  (1, Q, C + 1)       class logits (sigmoid -> score)
    masks   (1, Q, R/4, R/4)    per-query mask logits

Usage::

    python export_onnx.py --weights ../runs/rfdetr_seg_ultrasound/checkpoint_best_total.pth
    python export_onnx.py --dataset ade20k \
        --weights ../runs/rfdetr_seg_ade20k/checkpoint_best_total.pth
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    ADE20K_CLASS_NAMES,
    ADE20K_NUM_CLASSES,
    CLASS_NAMES,
    MODELS,
    NUM_CLASSES,
    RFDETR_ONNX_OPSET,
    RFDETR_RESOLUTION,
    ROOT,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export RF-DETR-Seg-Nano to ONNX")
    ap.add_argument("--dataset", choices=["ultrasound", "ade20k"], default="ultrasound")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default=None, help="output directory (default models/)")
    ap.add_argument("--resolution", type=int, default=RFDETR_RESOLUTION)
    ap.add_argument("--opset", type=int, default=RFDETR_ONNX_OPSET)
    ap.add_argument("--mnn-convert", default=str(ROOT / "build" / "mnn-conv" / "MNNConvert"))
    ap.add_argument("--mnn-out", default=None)
    args = ap.parse_args()

    ade = args.dataset == "ade20k"
    out_dir = Path(args.out) if args.out else MODELS
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "rfdetr_seg_nano_ade20k" if ade else "rfdetr_seg_nano"
    class_names = ADE20K_CLASS_NAMES if ade else CLASS_NAMES
    num_classes = ADE20K_NUM_CLASSES if ade else NUM_CLASSES

    from rfdetr import RFDETRSegNano

    model = RFDETRSegNano(pretrain_weights=args.weights, resolution=args.resolution)
    onnx_path = model.export(
        output_dir=str(out_dir),
        output_name=stem,
        shape=(args.resolution, args.resolution),
        opset_version=args.opset,
        fp16=False,
        batch_size=1,
        dynamic_batch=False,
        verbose=False,
        notes={"scheme": "rfdetr-seg", "dataset": args.dataset},
    )
    print(f"[onnx] {onnx_path}")

    info = dict(
        scheme="rfdetr-seg",
        dataset=args.dataset,
        variant="RFDETRSegNano",
        resolution=args.resolution,
        opset=args.opset,
        classes=class_names,
        num_classes=num_classes,
        weights=str(args.weights),
        outputs=["dets (1,Q,4)", "labels (1,Q,C+1)", "masks (1,Q,R/4,R/4)"],
    )

    mnn_out = Path(args.mnn_out) if args.mnn_out else out_dir / f"{stem}.mnn"
    convert = Path(args.mnn_convert)
    if convert.exists():
        print(f"[mnn ] {convert} -> {mnn_out}")
        subprocess.run(
            [str(convert), "-f", "ONNX", "--modelFile", str(onnx_path), "--MNNModel", str(mnn_out)],
            check=True,
        )
        info["mnn"] = str(mnn_out)
    else:
        print(f"[warn] MNNConvert not found at {convert}; skipping .mnn conversion")

    (out_dir / f"{stem}_export_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
