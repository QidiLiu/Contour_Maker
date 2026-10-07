# Contour_Maker v2

超声图像病灶轮廓分割的两方案对比。**训练与验证**在 uv 管理的 PyTorch 环境，
**测试**在 C++ + MNN（OpenCL 可选）子项目中完成。

```
                    训练 / 验证  (uv venv, PyTorch + CUDA)
                    ──────────────────────────────────────
   data/  ──►  下载 → 解压 → 统一 → 分层划分 → 导出 YOLO
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
 two-stage-train/           one-stage-train/
   YOLO26n (det, 4类)         YOLO26n-seg (4类)
        │                          │
        ▼ 框 → ROI → 256²          │
   MK-UNet (1ch 二值分割)          │
        │                          │
        └────────► ONNX ◄──────────┘
                    │
              MNNConvert
                    │
                    ▼
                    .mnn

                    测试  (C++ / MNN)
                    ──────────────────────────────────────
        two-stage-test/            one-stage-test/
        yolo26n_det.mnn            yolo26n_seg.mnn
        + mkunet.mnn
              └────► 带类别的目标轮廓 (class_id, contour)
```

## 两个方案

| | **two-stage** | **one-stage** |
|---|---|---|
| 模型 | YOLO26n-det + MK-UNet | YOLO26n-seg |
| 分割输入域 | **目标框 ROI**，resize 到 256×256 | 整图，letterbox 到 512×512 |
| 前向次数 / 图 | 检测 1 次 + 每框 1 次 MK-UNet | 恒定 1 次 |
| 类别来源 | YOLO-det 检测头 | YOLO-seg 检测头 |
| 可干预性 | 高（框与 ROI 可替换、可加约束） | 低（端到端黑盒） |

两者的**类别体系完全一致**（4 类）：

| id | 名称 | 来源 |
|---|---|---|
| 0 | `breast_benign` | BUSI `benign/` |
| 1 | `breast_malignant` | BUSI `malignant/` |
| 2 | `thyroid_nodule` | TN3K + DDTI |
| 3 | `thyroid_gland` | TG3K（高回声腺体） |

## 目录

```
common/            共享 Python 库（config / 数据 / 划分 / 指标 / MK-UNet / YOLO 导出）
two-stage-train/   两阶段训练与验证（PyTorch）
one-stage-train/   单阶段训练与验证（PyTorch）
two-stage-test/    两阶段 C++ 推理（MNN）
one-stage-test/    单阶段 C++ 推理（MNN）
cxx-common/        两个 C++ 子项目共用的静态库（图像 IO / 几何 / 指标 / MNN 封装）
third_party/MNN/   MNN 源码（gitignored，CMake 内联编译）
weights/           ultralytics 预训练基座权重（入库，训练起点）
data/              数据（gitignored）
models/            导出的 ONNX（gitignored）与 **.mnn（入库）**
runs/              训练产物（gitignored）
reports/  logs/    评估结果与日志（gitignored）
```

> `runs/` 的**训练后**权重不入库（体积大），因此 `models/*.mnn` 是唯一入库的
> 训练产物，可直接用于 C++ 测试；`.onnx` 可由 `export_onnx.py` 重新生成。
> `weights/*.pt` 是训练**起点**（ultralytics 官方基座），体积小且下载源不稳定，
> 故一并入库以便离线复现。

## 环境准备

```bash
# 1) Python（uv 管理）
#    实测环境: Python 3.11 + torch 2.14.1+cu130 + ultralytics 8.4.174 (RTX 4060 Ti)
uv venv --python 3.11 .venv

# torch/torchvision：官方 cu128 索引对 cp311 且 torch>=2.7 无可用 wheel，
# 实测直接用 PyPI 默认源即可拿到带 CUDA 的构建（cu130）。
VIRTUAL_ENV=$PWD/.venv uv pip install torch torchvision

# 其余依赖（onnxscript 是 torch>=2.14 的 dynamo ONNX 导出器所需）
VIRTUAL_ENV=$PWD/.venv uv pip install \
    ultralytics onnx onnxruntime onnxslim onnxscript \
    numpy opencv-python-headless scipy scikit-image \
    pandas pillow tqdm pyyaml albumentations tabulate requests

# 2) C++（sudo 需交互密码，须你手动执行）
sudo apt-get install -y build-essential cmake ninja-build \
    libopenblas-dev opencl-headers ocl-icd-opencl-dev opencl-clhpp-headers

# 3) MNN 源码（锁定 3.4.0）
git clone --depth 1 -b 3.4.0 https://github.com/alibaba/MNN.git third_party/MNN
```

> **关于 OpenCL**：MNN 已编译进 OpenCL 支持（`-DMNN_OPENCL=ON`），
> 但 WSL2 + NVIDIA 环境**不提供 OpenCL 设备**（NVIDIA 在 WSL2 下只透传 CUDA），
> 因此本机只能验证 CPU 后端；`--backend opencl` 会自动探测并优雅降级。
> 真机 GPU 加速需在 ARM Mali/Adreno 设备上验证。

### 构建 MNNConvert（ONNX → MNN 必需）

`MNNConvert` **默认不编译**，需显式打开（`MNN_BUILD_CONVERTER` 默认 OFF），
并用 MNN 仓库自带的 protobuf：

```bash
cd third_party/MNN
cmake -S . -B ../../build/mnn-conv -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMNN_OPENCL=ON -DMNN_BUILD_PROTOBUFFER=ON -DMNN_BUILD_CONVERTER=ON \
    -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_WIN_RUNTIME_MT=ON
cmake --build ../../build/mnn-conv --target MNNConvert -j
# 产物: build/mnn-conv/MNNConvert  （后续命令中的 MNNConvert 指此路径）
```

## 完整流程

> 也可直接用 `./scripts/run_all.sh {data|train|export|build|test|all}` 一键执行；
> 长训练建议用 `./scripts/train_bg.sh {seg|det|mkunet}`（tmux 托管，避免会话断开被杀）。

```bash
PY=.venv/bin/python

# ---- 数据 ----
$PY -m common.data_prep --all       # 下载 481MB + 解压 + 统一 manifest
$PY -m common.splits                # 分层划分 (seed=42)
$PY -m common.yolo_export --kind both

# ---- 训练（两阶段）----
cd two-stage-train
$PY train_yolo_det.py --data ../data/yolo_det/data_det.yaml \
    --epochs 60 --imgsz 512 --batch 16 --name det_yolo26n
# ROI 训练集用检测框（匹配推理分布）；验证集用 GT 框（干净、稳定）
$PY build_roi_dataset.py --splits train --box-source det \
    --det ../runs/det_yolo26n/weights/best.pt --out ../data/roi_cache/train
$PY build_roi_dataset.py --splits val --box-source gt --out ../data/roi_cache/val
$PY train_mkunet.py --data ../data/roi_cache/train --val-data ../data/roi_cache/val \
    --variant MK_UNet --roi-size 256 --epochs 60 --batch 16 --name mkunet_yolo26n
$PY validate.py --det ../runs/det_yolo26n/weights/best.pt \
    --mkunet ../runs/mkunet_yolo26n/best.pt --split test

# ---- 训练（单阶段）----
cd ../one-stage-train
$PY train_yolo_seg.py --data ../data/yolo_seg/data_seg.yaml \
    --epochs 60 --imgsz 512 --batch 12 --name seg_yolo26n
$PY validate.py --seg ../runs/seg_yolo26n/weights/best.pt --split test

# ---- 导出 ONNX ----
$PY export_onnx.py --seg ../runs/seg_yolo26n/weights/best.pt --out ../models
cd ../two-stage-train
$PY export_onnx.py --det ../runs/det_yolo26n/weights/best.pt \
    --mkunet ../runs/mkunet_yolo26n/best.pt --out ../models

# ---- ONNX → MNN ----
cd ..
MC=build/mnn-conv/MNNConvert          # 见「构建 MNNConvert」
$MC -f ONNX --modelFile models/yolo26n_seg.onnx --MNNModel models/yolo26n_seg.mnn
$MC -f ONNX --modelFile models/yolo26n_det.onnx --MNNModel models/yolo26n_det.mnn
$MC -f ONNX --modelFile models/mkunet.onnx      --MNNModel models/mkunet.mnn

# ---- 构建 C++ ----
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON
cmake --build build -j

# ---- 测试 ----
./build/one-stage-test/one_stage_test --model models/yolo26n_seg.mnn \
    --image-dir data/unified/images --gt-dir data/unified/masks
./build/two-stage-test/two_stage_test --det models/yolo26n_det.mnn \
    --mkunet models/mkunet.mnn --image-dir data/unified/images --gt-dir data/unified/masks
```

### 实测结果

同一 120 张测试子集、C++/MNN、CPU 后端：

| 指标 | 单阶段 YOLO26n-seg | 两阶段 YOLO26n-det + MK-UNet |
|---|---|---|
| Precision | **0.8393** | 0.7642 |
| Recall | 0.7833 | 0.7833 |
| F1 | **0.8103** | 0.7737 |
| Dice | 0.8791 | **0.8877** |
| IoU | 0.7882 | **0.8031** |
| 边界 F1 | 0.6051 | **0.6438** |
| 端到端 | **37.3 ms** (26.8 FPS) | 47.4 ms (21.1 FPS) |

PyTorch 与 C++/MNN 的交叉校验：单阶段 F1 差 0.003，两阶段 Dice 差 0.002。
完整记录（含各环节验证与未验证项）见 [`VERIFICATION.md`](VERIFICATION.md)。

详见 [`two-stage-train/README.md`](two-stage-train/README.md)、
[`one-stage-train/README.md`](one-stage-train/README.md)。

## 数据集

来源 `us-segmentator/us-segmentation-dataset`（hf-mirror 可达）。

| key | 数据集 | 类别 | 图像数 |
|---|---|---|---|
| `busi` | BUSI | 0 / 1（良性/恶性） | 647 |
| `tn3k` | TN3K | 2 | 3493 |
| `ddti` | DDTI | 2 | 637 |
| `tg3k` | TG3K | 3 | 3585 |

BUSI 的 `normal` 类无病灶标注，已排除。类别从原包的 `benign/` `malignant/`
目录名恢复。

## 关键实现说明

### MK-UNet

vendor 自 [SLDGroup/MK-UNet](https://github.com/SLDGroup/MK-UNet)（ICCV 2025
CVAMD Workshop），见 `common/mkunet.py`。

相对官方实现的改动：

1. **移除 `timm` 依赖**（官方 README 提到的 `mmcv-full` 实际并未被代码使用）。
2. **`in_channels` 支持 1**（超声为灰度）。官方在 `forward` 里把 1 通道复制成
   3 通道，本项目直接用单通道，省 3 倍输入带宽。
3. 保留官方结构：5 级 MK-IRB 编码器 + maxpool、通道/空间注意力、
   GroupedAttentionGate 门控 skip、bilinear 上采样 + 逐元素相加。
4. 官方有 4 个深监督输出头，本项目只用最后一个 `p4`。

参数量实测 `MK_UNet` = **0.327M**（论文 0.316M，差异 3%），
`MK_UNet_S` = 0.095M，`MK_UNet_T` = 0.027M。

> **注意**：官方 `forward` 写作 `self.CA1(out)*out`，而 `ChannelAttention.forward`
> 本身已返回 `x * sigmoid(attn)`，等价于把激活平方。本项目按注意力门控的本意
> 直接使用模块返回值——否则经 5 级放大后激活会从 ~15 爆到 1e18，训练直接 NaN。

### ONNX 导出格式（C++ 侧解析依据）

ultralytics 对本任务的导出是**合并式**（头部自带 TopK，未做 NMS）：

| 模型 | 输入 | 输出 |
|---|---|---|
| `yolo26n_det.onnx` | `images` (1,3,512,512) | `output0` (1,300,6) = `[x1,y1,x2,y2,score,class_id]` |
| `yolo26n_seg.onnx` | `images` (1,3,512,512) | `output0` (1,300,38) = `[x1,y1,x2,y2,score,class_id,coeff×32]`<br>`output1` (1,32,128,128) 掩码原型 |
| `mkunet.onnx` | `input` (1,1,256,256) | `logits` (1,1,256,256) |

- **输入是 3 通道**（ultralytics 以 BGR 读图，灰度图会复制成 3 通道）。
  C++ 侧必须把灰度复制到 3 通道，否则报 channel mismatch。
- 坐标为 **letterbox 空间**的 xyxy，C++ 侧需反变换回原图。
- NMS 在 C++ 侧按类别独立执行。

### Python ↔ C++ 预处理一致性

`common/roi.py`（Python）与 `cxx-common/src/image.cpp`（C++）实现同一套
letterbox / ROI 裁剪 / square_pad 贴回逻辑，填充值同为 114。
这是 MNN 推理结果能与 Python 权重对齐的前提。

C++ 侧自带最小 PNG 编解码器（`cxx-common/src/png.cpp`），
使测试子项目除 MNN 外零外部依赖（目标平台无 libpng/zlib 开发头文件）。

## 已知限制

- **OpenCL 未在本机验证**：WSL2 + NVIDIA 无 OpenCL 设备，`--backend opencl`
  在本机会自动降级到 CPU。真机加速需在 ARM 设备上实测。
- **MNN 版本锁定 3.4.0**：MNN 3.6 的公开头文件已移除 `Session` 低层接口，
  本项目使用 `Express::Module` + `RuntimeManager`。
- **MK-UNet 的验证集用 GT 框**（训练用检测框），`val_dice` 偏乐观；
  真实推理 Dice 见上文实测表（0.8877）。
- 依赖单一来源：全部写在 `pyproject.toml`，无 `requirements.txt`。