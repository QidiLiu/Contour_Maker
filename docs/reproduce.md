# 复现指南

以下命令均在仓库根目录执行。设：

```bash
ROOT=$PWD
PY=$ROOT/.venv/bin/python
MC=$ROOT/build/mnn-conv/MNNConvert
```

实测环境：Ubuntu 26.04 (WSL2)、Python 3.11、PyTorch 2.14.1+cu130、Ultralytics 8.4.174、
RF-DETR 1.11.2、MNN 3.4.0、RTX 4060 Ti 16 GB、12 核 CPU / 16 GB RAM。

## 0. 环境

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=$ROOT/.venv uv pip install torch torchvision
VIRTUAL_ENV=$ROOT/.venv uv pip install -e .
VIRTUAL_ENV=$ROOT/.venv uv pip install "rfdetr[train,onnx]"
VIRTUAL_ENV=$ROOT/.venv uv pip install pytest ruff
```

C++ 依赖与 MNN 构建见 [`deployment.md`](deployment.md)。

> RF-DETR 的预训练权重默认从 `storage.googleapis.com` 下载（国内可能很慢）。本项目的替代做法：
> 从 HF 镜像下载 `Roboflow/rf-detr-seg-nano` 的 `model.safetensors`，转成 rfdetr 的
> `{"model": state_dict}` 检查点，放到 `~/.roboflow/models/rf-detr-seg-nano.pt`。
> 详见 [`benchmarks.md`](benchmarks.md) 的「权重获取」。

## 1. 超声数据集

```bash
$PY -m common.data_prep --all       # 下载 481 MB + 解压 + manifest
$PY -m common.splits                # 分层划分 seed=42
$PY -m common.yolo_export --kind both
$PY -m common.coco_export           # RF-DETR 用
```

## 2. 超声三方案训练

### YOLO-Seg

```bash
cd "$ROOT/yolo-seg-train"
$PY train_yolo_seg.py --data "$ROOT/data/yolo_seg/data_seg.yaml" \
    --epochs 60 --imgsz 512 --batch 12 --name seg_yolo26n
$PY validate.py --seg "$ROOT/runs/seg_yolo26n/weights/best.pt" \
    --split test --out "$ROOT/reports/ultrasound_yolo_seg_py"
cd "$ROOT"
```

### Det-UNet

```bash
cd "$ROOT/det-unet-train"
$PY train_yolo_det.py --data "$ROOT/data/yolo_det/data_det.yaml" \
    --epochs 60 --imgsz 512 --batch 16 --name det_yolo26n
$PY build_roi_dataset.py --splits train --box-source det \
    --det "$ROOT/runs/det_yolo26n/weights/best.pt" --out "$ROOT/data/roi_cache/train"
$PY build_roi_dataset.py --splits val --box-source gt --out "$ROOT/data/roi_cache/val"
$PY train_mkunet.py --data "$ROOT/data/roi_cache/train" \
    --val-data "$ROOT/data/roi_cache/val" \
    --variant MK_UNet --roi-size 256 --epochs 60 --batch 16 --name mkunet_yolo26n
$PY validate.py --det "$ROOT/runs/det_yolo26n/weights/best.pt" \
    --mkunet "$ROOT/runs/mkunet_yolo26n/best.pt" --split test \
    --out "$ROOT/reports/ultrasound_det_unet_py"
cd "$ROOT"
```

### RF-DETR-Seg

```bash
cd "$ROOT/rfdetr-seg-train"
$PY train_rfdetr_seg.py --dataset ultrasound \
    --epochs 8 --batch 8 --resolution 384 --name rfdetr_seg_ultrasound
$PY validate.py --weights "$ROOT/runs/rfdetr_seg_ultrasound/checkpoint_best_total.pth" \
    --split test --out "$ROOT/reports/ultrasound_rfdetr_seg_py"
cd "$ROOT"
```

> RF-DETR 在 384²、batch 8 下约 2.3 it/s（RTX 4060 Ti），8 epoch 约 50 分钟。
> Nano 的输入边长必须是 12 的倍数；384 是接近 512 的折中值，504 也可训练但明显更慢。

## 3. ADE20K-10

```bash
$PY -m common.ade20k --all
$PY -m common.yolo_export --dataset ade20k --kind both
$PY -m common.coco_export --dataset ade20k
```

### YOLO-Seg

```bash
cd "$ROOT/yolo-seg-train"
$PY train_yolo_seg.py --dataset ade20k --epochs 20 --imgsz 512 --batch 12 \
    --name yolo_seg_ade20k
$PY validate.py --dataset ade20k --seg "$ROOT/runs/yolo_seg_ade20k/weights/best.pt" \
    --split test --out "$ROOT/reports/ade20k_yolo_seg_py"
cd "$ROOT"
```

### Det-UNet

```bash
cd "$ROOT/det-unet-train"
$PY train_yolo_det.py --dataset ade20k --epochs 20 --imgsz 512 --batch 16 \
    --name det_ade20k
$PY build_roi_dataset.py --dataset ade20k --splits train --box-source det \
    --det "$ROOT/runs/det_ade20k/weights/best.pt" \
    --out "$ROOT/data/ade20k/roi_cache/train"
$PY build_roi_dataset.py --dataset ade20k --splits val --box-source gt \
    --out "$ROOT/data/ade20k/roi_cache/val"
$PY train_mkunet.py --data "$ROOT/data/ade20k/roi_cache/train" \
    --val-data "$ROOT/data/ade20k/roi_cache/val" \
    --variant MK_UNet --roi-size 256 --epochs 20 --batch 16 --name mkunet_ade20k
$PY validate.py --dataset ade20k --det "$ROOT/runs/det_ade20k/weights/best.pt" \
    --mkunet "$ROOT/runs/mkunet_ade20k/best.pt" --split test \
    --out "$ROOT/reports/ade20k_det_unet_py"
cd "$ROOT"
```

### RF-DETR-Seg

```bash
cd "$ROOT/rfdetr-seg-train"
$PY train_rfdetr_seg.py --dataset ade20k --epochs 8 --batch 8 --resolution 384 \
    --name rfdetr_seg_ade20k
$PY validate.py --dataset ade20k \
    --weights "$ROOT/runs/rfdetr_seg_ade20k/checkpoint_best_total.pth" \
    --split test --out "$ROOT/reports/ade20k_rfdetr_seg_py"
cd "$ROOT"
```

## 4. 导出 ONNX / MNN

```bash
cd "$ROOT/yolo-seg-train"
$PY export_onnx.py --seg "$ROOT/runs/seg_yolo26n/weights/best.pt" --out "$ROOT/models"
cd "$ROOT/det-unet-train"
$PY export_onnx.py --det "$ROOT/runs/det_yolo26n/weights/best.pt" \
    --mkunet "$ROOT/runs/mkunet_yolo26n/best.pt" --out "$ROOT/models"
cd "$ROOT/rfdetr-seg-train"
$PY export_onnx.py --weights "$ROOT/runs/rfdetr_seg_ultrasound/checkpoint_best_total.pth" \
    --out "$ROOT/models"
cd "$ROOT"

$MC -f ONNX --modelFile models/yolo26n_seg.onnx --MNNModel models/yolo26n_seg.mnn
$MC -f ONNX --modelFile models/yolo26n_det.onnx --MNNModel models/yolo26n_det.mnn
$MC -f ONNX --modelFile models/mkunet.onnx      --MNNModel models/mkunet.mnn
$MC -f ONNX --modelFile models/rfdetr_seg_nano.onnx --MNNModel models/rfdetr_seg_nano.mnn
```

ADE20K 模型同理，把 `--out` 指向 `models/ade20k`。

## 5. C++ 构建与测试

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON
cmake --build build -j

./build/yolo-seg-test/yolo_seg_test --model models/yolo26n_seg.mnn \
    --image-dir data/unified/images --gt-dir data/unified/masks \
    --out-dir reports/ultrasound_yolo_seg_cxx
./build/det-unet-test/det_unet_test --det models/yolo26n_det.mnn \
    --mkunet models/mkunet.mnn --image-dir data/unified/images \
    --gt-dir data/unified/masks --out-dir reports/ultrasound_det_unet_cxx
./build/rfdetr-seg-test/rfdetr_seg_test --model models/rfdetr_seg_nano.mnn \
    --image-dir data/unified/images --gt-dir data/unified/masks \
    --resolution 384 --out-dir reports/ultrasound_rfdetr_seg_cxx
```

## 6. 汇总结果

```bash
$PY scripts/collect_results.py --reports reports --out results
```

生成 `results/summary.csv` 与 `results/summary.md`。
