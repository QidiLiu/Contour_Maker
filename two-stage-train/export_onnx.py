"""导出 ONNX：YOLO26n 检测器 + MK-UNet。

两份 ONNX 供 `two-stage-test/` 用 MNNConvert 转换为 .mnn。

导出后会做一次 PyTorch ↔ ONNXRuntime 数值一致性校验，
避免把有问题的模型带到 C++ 侧再排查。

用法:
    python export_onnx.py --det ../runs/det_yolo26n/weights/best.pt \
        --mkunet ../runs/mkunet_yolo26n/best.pt --out ../models
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ONNX_IOU_NMS, ONNX_OPSET, YOLO_IMGSZ  # noqa: E402
from common.mkunet import build_mkunet, count_params  # noqa: E402


# ---------------------------------------------------------------- MK-UNet
def export_mkunet(ckpt: Path, out: Path, opset: int, roi_size: int | None = None):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    variant = ck.get("variant", "MK_UNet")
    size = int(roi_size or ck.get("roi_size", 256))
    model = build_mkunet(variant, num_classes=1, in_channels=1).eval()
    model.load_state_dict(ck["model"])
    n = count_params(model)

    dummy = torch.randn(1, 1, size, size)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
    )
    print(f"[mkunet] {variant} params={n/1e6:.3f}M roi={size} -> {out}")

    _verify_onnx(out, model, dummy, tol=1e-3, name="mkunet")
    return dict(variant=variant, params=n, roi_size=size)


def _verify_onnx(onnx_path: Path, model: torch.nn.Module, dummy: torch.Tensor,
                 tol: float, name: str):
    """PyTorch 与 ONNXRuntime 输出一致性校验。

    提前发现导出问题，避免把有偏差的模型带到 C++ 侧再排查。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"[warn ] 未安装 onnxruntime, 跳过 {name} 的数值校验")
        return None
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
    with torch.no_grad():
        want = model(dummy).numpy()
    diff = float(np.abs(got - want).max())
    print(f"[verif] {name}: max|onnx - torch| = {diff:.3e} "
          f"{'OK' if diff <= tol else 'FAIL'}")
    if diff > tol:
        raise RuntimeError(f"{name} ONNX 数值偏差过大: {diff:.3e} > {tol}")
    return diff


# ---------------------------------------------------------------- YOLO det
def export_yolo(weights: Path, out: Path, imgsz: int, opset: int, kind: str):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out.parent.mkdir(parents=True, exist_ok=True)
    model.export(format="onnx", imgsz=imgsz, opset=opset,
                 simplify=False, dynamic=False, nms=False,
                 half=False, device="cpu")
    # ultralytics 把 onnx 写在权重同目录且与权重同名（best.pt -> best.onnx）。
    # 按 stem 精确定位，再兜底 glob。
    produced = out.with_suffix(".onnx") if out.suffix != ".onnx" else out
    for c in (Path(weights).with_suffix(".onnx"), Path(weights).parent / "best.onnx"):
        if c.exists() and c.resolve() != produced.resolve():
            c.replace(produced)
            break
    if not produced.exists():
        found = sorted(Path(weights).parent.glob("*.onnx"))
        if found:
            found[0].replace(produced)
    if not produced.exists():
        raise FileNotFoundError(f"未找到导出的 ONNX（权重目录 {Path(weights).parent}）")
    print(f"[{kind:5s}] -> {produced}")
    return produced


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 ONNX")
    ap.add_argument("--det", required=True)
    ap.add_argument("--mkunet", required=True)
    ap.add_argument("--out", default=None, help="输出目录，默认 ../models")
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--roi-size", type=int, default=None)
    ap.add_argument("--opset", type=int, default=ONNX_OPSET)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg ] out={out_dir} imgsz={args.imgsz} opset={args.opset} "
          f"nms_iou={ONNX_IOU_NMS}")

    info = export_mkunet(Path(args.mkunet), out_dir / "mkunet.onnx",
                         args.opset, args.roi_size)
    export_yolo(Path(args.det), out_dir / "yolo26n_det.onnx",
                args.imgsz, args.opset, "det")

    import json
    (out_dir / "export_info.json").write_text(json.dumps(
        dict(imgsz=args.imgsz, opset=args.opset, nms_iou=ONNX_IOU_NMS,
             mkunet=info), indent=2))
    print(f"\n[done] -> {out_dir}")
    print("下一步: 用 MNNConvert 转 .mnn，然后交给 two-stage-test/")
    return 0


if __name__ == "__main__":
    sys.exit(main())