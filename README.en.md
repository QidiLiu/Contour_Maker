# Contour_Maker

Three instance-segmentation schemes under one shared data/metric protocol,
trained in PyTorch and deployed in C++/MNN.

| Scheme | Models | Idea |
|---|---|---|
| **YOLO-Seg** | YOLO26n-seg | single-stage, end-to-end, fastest |
| **Det-UNet** | YOLO26n-det + MK-UNet | detector box → ROI refinement, finer boundaries |
| **RF-DETR-Seg** | RF-DETR-Seg-Nano | DETR queries, no NMS/anchor, strongest generalisation |

Evaluated on:

- **Ultrasound**: BUSI + TN3K + DDTI + TG3K, 4 classes, 8362 images.
- **ADE20K-10**: 10 indoor classes (bed, windowpane, cabinet, door, table, chair,
  armchair, lamp, bottle, pot), 4000/600/800 train/val/test.

Every scheme is measured in both **Python (PyTorch)** and **C++ (MNN, CPU)** with
the same IoU-0.5 matching, Dice/IoU/HD95/ASSD/boundary-F1 and end-to-end latency.

## Results

See [README.md](README.md#实测结果) (Chinese) and
[docs/benchmarks.md](docs/benchmarks.md) for the full protocol. Summary files
live in [results/](results/).

## Quick start

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install torch torchvision
VIRTUAL_ENV=$PWD/.venv uv pip install -e .
VIRTUAL_ENV=$PWD/.venv uv pip install "rfdetr[train,onnx]"

.venv/bin/python -m common.data_prep --all
.venv/bin/python -m common.splits
.venv/bin/python -m common.yolo_export --kind both
.venv/bin/python -m common.coco_export

cd yolo-seg-train && ../.venv/bin/python train_yolo_seg.py --name seg_yolo26n && cd ..
cd det-unet-train && ../.venv/bin/python train_yolo_det.py --name det_yolo26n && cd ..
cd rfdetr-seg-train && ../.venv/bin/python train_rfdetr_seg.py --name rfdetr_seg_ultrasound && cd ..

# export + MNN + C++ (see docs/deployment.md)
./scripts/run_all.sh export
./scripts/run_all.sh build
./scripts/run_all.sh test
```

## Documentation

- [Datasets](docs/datasets.md)
- [Architecture](docs/architecture.md)
- [Deployment (ONNX → MNN → C++)](docs/deployment.md)
- [Reproduce](docs/reproduce.md)
- [Benchmarks](docs/benchmarks.md)

## License

AGPL-3.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
