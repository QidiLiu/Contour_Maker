"""导出 YOLO26n-seg 的 ONNX。

ultralytics 的分割头包含 mask prototype 与实例系数，ONNX 导出时结构较复杂。
这里关闭内置 NMS（nms=False），把 NMS 与掩码组合交给 C++ 侧，
以便与 Python 侧口径完全一致，也避免 MNNConvert 对内置 NMS 算子的兼容问题。

用法:
    python export_onnx.py --seg ../runs/seg_yolo26n/weights/best.pt --out ../models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    ONNX_OPSET,
    YOLO_IMGSZ,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 YOLO26n-seg ONNX")
    ap.add_argument("--seg", required=True)
    ap.add_argument("--out", default=None, help="输出目录，默认 ../models")
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--opset", type=int, default=ONNX_OPSET)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "yolo26n_seg.onnx"
    if target.exists():
        target.unlink()

    from ultralytics import YOLO

    model = YOLO(args.seg)
    print(f"[cfg ] imgsz={args.imgsz} opset={args.opset} classes={CLASS_NAMES}")

    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=False,
        dynamic=False,     # C++ 侧按固定尺寸预处理，固定 shape 兼容性最好
        half=False,         # MNN CPU/OpenCL 走 fp32
        nms=False,          # NMS 交给 C++ 侧
        device="cpu",
    )

    # ultralytics 会把 onnx 写到 weights 的上级目录，统一搬到 out_dir
    produced = Path(args.seg).parent.parent.parent / "yolo26n-seg.onnx"
    if produced.exists() and produced.resolve() != target.resolve():
        produced.replace(target)

    if not target.exists():
        cand = list(out_dir.glob("*seg*.onnx")) or list(out_dir.glob("*.onnx"))
        if cand:
            cand[0].replace(target)

    print(f"[done] -> {target}")
    (out_dir / "export_info.json").write_text(json.dumps(dict(
        imgsz=args.imgsz, opset=args.opset, num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES, nms="cxx-side",
    ), indent=2, ensure_ascii=False))
    print("下一步: 用 MNNConvert 转 .mnn，然后交给 one-stage-test/")
    return 0


if __name__ == "__main__":
    sys.exit(main())