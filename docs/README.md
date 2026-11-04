# Documentation

| Document | Contents |
|---|---|
| [datasets.md](datasets.md) | Ultrasound + ADE20K-10 sources, splits, mask formats, licenses |
| [architecture.md](architecture.md) | Three schemes, data flow, directory layout, design decisions |
| [deployment.md](deployment.md) | ONNX → MNN → C++, tensor contracts, CLI, OpenCL notes |
| [reproduce.md](reproduce.md) | Step-by-step commands for both datasets |
| [benchmarks.md](benchmarks.md) | Hardware, metric definitions, results, RF-DETR weight workaround |
| [verification.md](verification.md) | Engineering verification log (data, MK-UNet, PNG, cross-checks) |

## Scheme naming

| Scheme | Directory | Models |
|---|---|---|
| YOLO-Seg | `yolo-seg-train/`, `yolo-seg-test/` | YOLO26n-seg |
| Det-UNet | `det-unet-train/`, `det-unet-test/` | YOLO26n-det + MK-UNet |
| RF-DETR-Seg | `rfdetr-seg-train/`, `rfdetr-seg-test/` | RF-DETR-Seg-Nano |

## Quick links

- [Root README](../README.md)
- [Contributing](../CONTRIBUTING.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)
- [License](../LICENSE)
