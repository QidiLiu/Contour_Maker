# 部署：ONNX → MNN → C++

项目把训练（PyTorch）与部署（C++/MNN）分开：训练侧导出 ONNX，`MNNConvert` 转 `.mnn`，
三个 C++ 测试可执行文件只依赖 MNN 与自带的 PNG 编解码器。

## 1. 构建 MNN 与 MNNConvert

```bash
# MNN 源码（锁定 3.4.0；3.6 的公开头文件移除了本项目使用的部分 Session 接口）
git clone --depth 1 -b 3.4.0 https://github.com/alibaba/MNN.git third_party/MNN

# MNNConvert 默认不编译，需显式打开，并使用 MNN 自带 protobuf
cmake -S third_party/MNN -B build/mnn-conv -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMNN_OPENCL=ON -DMNN_BUILD_PROTOBUFFER=ON -DMNN_BUILD_CONVERTER=ON \
    -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_WIN_RUNTIME_MT=ON
cmake --build build/mnn-conv --target MNNConvert -j
# 产物: build/mnn-conv/MNNConvert
```

## 2. 三个方案的模型接口

### YOLO-Seg

```bash
cd yolo-seg-train
../.venv/bin/python export_onnx.py --seg ../runs/seg_yolo26n/weights/best.pt --out ../models
```

| 张量 | shape | 含义 |
|---|---|---|
| `images` | (1, 3, 512, 512) | 3 通道（灰度复制） |
| `output0` | (1, 300, 38) | `[x1,y1,x2,y2,score,class_id,coeff×32]`，letterbox 空间，已按 score 降序、未 NMS |
| `output1` | (1, 32, 128, 128) | 32 个 mask prototype |

### Det-UNet

```bash
cd det-unet-train
../.venv/bin/python export_onnx.py \
    --det ../runs/det_yolo26n/weights/best.pt \
    --mkunet ../runs/mkunet_yolo26n/best.pt --out ../models
```

| 张量 | shape | 含义 |
|---|---|---|
| `images` | (1, 3, 512, 512) | 检测输入 |
| `output0` | (1, 300, 6) | `[x1,y1,x2,y2,score,class_id]` |
| `input` | (1, 1, 256, 256) | MK-UNet 灰度 ROI |
| `logits` | (1, 1, 256, 256) | sigmoid 后按 0.5 二值化 |

### RF-DETR-Seg

```bash
cd rfdetr-seg-train
../.venv/bin/python export_onnx.py \
    --weights ../runs/rfdetr_seg_ultrasound/checkpoint_best_total.pth \
    --out ../models
```

| 张量 | shape | 含义 |
|---|---|---|
| `input` | (1, 3, R, R) | RGB，ImageNet mean/std；R = 384 或 504，须为 12 的倍数 |
| `dets` | (1, 100, 4) | 归一化 `cx,cy,w,h`（乘原图宽高即得像素坐标） |
| `labels` | (1, 100, C+1) | 类别 logits；最后一列为 background，取前 C 列的 sigmoid argmax |
| `masks` | (1, 100, R/4, R/4) | 每个 query 的 mask logits；双线性上采样到原图后 `> 0` |

RF-DETR 导出使用 opset 17，包含 `GridSample`、`LayerNormalization` 等算子；MNN 3.4 可以转换，
但建议转换后做一次数值抽检（见下文）。

## 3. ONNX → MNN

```bash
MC=build/mnn-conv/MNNConvert
$MC -f ONNX --modelFile models/yolo26n_seg.onnx --MNNModel models/yolo26n_seg.mnn
$MC -f ONNX --modelFile models/yolo26n_det.onnx --MNNModel models/yolo26n_det.mnn
$MC -f ONNX --modelFile models/mkunet.onnx      --MNNModel models/mkunet.mnn
$MC -f ONNX --modelFile models/rfdetr_seg_nano.onnx --MNNModel models/rfdetr_seg_nano.mnn
```

也可以直接让 `rfdetr-seg-train/export_onnx.py` 顺带完成转换（检测到 `MNNConvert` 时自动执行）。

## 4. 构建 C++

```bash
sudo apt-get install -y build-essential cmake ninja-build \
    libopenblas-dev opencl-headers ocl-icd-opencl-dev opencl-clhpp-headers

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON
cmake --build build -j
```

产物：

```
build/yolo-seg-test/yolo_seg_test
build/det-unet-test/det_unet_test
build/rfdetr-seg-test/rfdetr_seg_test
```

## 5. 运行

超声（120 张测试子集示例）：

```bash
# YOLO-Seg
./build/yolo-seg-test/yolo_seg_test \
    --model models/yolo26n_seg.mnn \
    --image-dir data/unified/images --gt-dir data/unified/masks \
    --imgsz 512 --conf 0.25 --num-classes 4 \
    --out-dir reports/ultrasound_yolo_seg_cxx

# Det-UNet
./build/det-unet-test/det_unet_test \
    --det models/yolo26n_det.mnn --mkunet models/mkunet.mnn \
    --image-dir data/unified/images --gt-dir data/unified/masks \
    --det-imgsz 512 --roi-size 256 --pad-ratio 0.15 \
    --out-dir reports/ultrasound_det_unet_cxx

# RF-DETR-Seg
./build/rfdetr-seg-test/rfdetr_seg_test \
    --model models/rfdetr_seg_nano.mnn \
    --image-dir data/unified/images --gt-dir data/unified/masks \
    --resolution 384 --conf 0.25 --num-classes 4 \
    --out-dir reports/ultrasound_rfdetr_seg_cxx
```

ADE20K（`--gt-instances` 启用逐实例类别）：

```bash
./build/yolo-seg-test/yolo_seg_test \
    --model models/ade20k/yolo26n_seg_ade20k.mnn \
    --image-dir data/ade20k/unified/images --gt-dir data/ade20k/unified/masks \
    --gt-instances data/ade20k/unified/instances \
    --imgsz 512 --num-classes 10 \
    --out-dir reports/ade20k_yolo_seg_cxx
```

所有可执行文件都支持 `--single <image>` 单图调试、`--max-images` 限制、`--quiet` 关闭进度。
`--out-dir` 下会写 `per_image.csv` 与 `summary.json`。

## 6. OpenCL 说明

MNN 已编译 OpenCL 支持（`-DMNN_OPENCL=ON`），但本机（WSL2 + NVIDIA）没有 OpenCL 设备：
NVIDIA 在 WSL2 下只透传 CUDA，`/etc/OpenCL/vendors` 为空。因此：

- 本仓库的所有实测速度均为 **CPU 后端**。
- `--backend opencl` 会在不可用时自动降级到 CPU（YOLO-Seg 支持该参数；Det-UNet/RF-DETR 当前统一走 CPU）。
- 真机 GPU 加速需在 ARM Mali/Adreno 等有 OpenCL 的设备上验证。

## 7. Python ↔ C++ 一致性

两套实现共享同一套几何参数（letterbox、ROI pad 0.15、square_pad 114）与指标定义。
RF-DETR 的 C++ 侧按官方后处理实现：`sigmoid(labels)` 取真实类别 argmax、`dets` 乘原图宽高、
`masks` 双线性上采样后 `> 0`。若需复核，建议先跑 `--max-images 50` 对比 Python/C++ 的 Dice 与
F1，再跑全量。
