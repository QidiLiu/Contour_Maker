# Contour_Maker

> 超声病灶与通用场景的**三方案实例分割**对比：训练在 PyTorch，部署在 C++/MNN。
> Three instance-segmentation schemes, one shared data/metric protocol: PyTorch training, C++/MNN deployment.
> [English README](README.en.md)

| 方案 | 模型 | 一句话 |
|---|---|---|
| **YOLO-Seg** | YOLO26n-seg | 单阶段端到端，最快 |
| **Det-UNet** | YOLO26n-det + MK-UNet | 检测框引导 ROI 精修，边界更细 |
| **RF-DETR-Seg** | RF-DETR-Seg-Nano | DETR 式查询，无 NMS/anchor |

三个方案在同一份数据、同一套预处理、同一套指标下训练/测试，并同时给出 **Python (PyTorch)**
与 **C++ (MNN)** 两条推理链路的结果。

---

## 目录

- [三方案](#三方案)
- [实测结果](#实测结果)
- [快速开始](#快速开始)
- [数据集](#数据集)
- [部署](#部署)
- [复现](#复现)
- [已知限制](#已知限制)
- [开源与许可](#开源与许可)

---

## 三方案

### YOLO-Seg（原 one-stage）

```
原图 ─► letterbox 512 ─► YOLO26n-seg ─► (box, class, coeff×32) + protos
                                              └─► 掩码组合 → 原图 → 轮廓
```

单模型、单次前向。适合对延迟敏感、目标不太小的场景。

### Det-UNet（原 two-stage）

```
原图 ─► YOLO26n-det ─► 框 + 类别
                         └─► 框外扩 15% → ROI 256² ─► MK-UNet(0.33M) ─► 贴回原图
```

检测与分割解耦：框可替换、可在 ROI 上加约束；ROI 固定分辨率带来更细边界。
多目标时 MK-UNet 逐框前向，耗时随目标数增长。

### RF-DETR-Seg（新增）

```
原图 ─► resize 384² RGB ─► DINOv2-windowed + deformable DETR ─► 100 queries
                                                                 ├─ dets
                                                                 ├─ labels (C+1)
                                                                 └─ masks (R/4)²
```

DETR 式查询，无 NMS、无 anchor；参数量 33.5M。MNN 转换后可跑 C++，输出
`dets/labels/masks` 三张量（详见 [部署](docs/deployment.md)）。

### 命名规则

| 旧名 | 新名 | 目录 |
|---|---|---|
| one-stage | **YOLO-Seg** | `yolo-seg-train/` / `yolo-seg-test/` |
| two-stage | **Det-UNet** | `det-unet-train/` / `det-unet-test/` |
| — | **RF-DETR-Seg** | `rfdetr-seg-train/` / `rfdetr-seg-test/` |

---

## 实测结果

> 完整协议、硬件与逐步命令见 [`docs/benchmarks.md`](docs/benchmarks.md)。
> 所有速度均为 **C++/MNN、CPU 后端、顺序单图推理**（含预处理/后处理）。

### 超声测试集（4 类，1256 张 test，C++/MNN CPU）

| 指标 | YOLO-Seg | Det-UNet | RF-DETR-Seg |
|---|---|---|---|
| Precision | **0.8994** | 0.8818 | 0.7679 |
| Recall | 0.8967 | 0.8877 | **0.9289** |
| F1 | **0.8981** | 0.8847 | 0.8408 |
| Dice | 0.9061 | 0.8889 | **0.9243** |
| IoU | 0.8326 | 0.8050 | **0.8630** |
| HD95 (px) | 9.51 | 11.34 | **7.86** |
| ASSD (px) | 3.19 | 3.94 | **2.61** |
| 边界 F1 | 0.5843 | 0.5021 | **0.6630** |
| 端到端 (ms) | **43.9** | 65.6 | 518.2 |
| FPS | **22.8** | 15.2 | 1.9 |

> 三方案同 test 集、同一 mask IoU 0.5 匹配。RF-DETR 的分割几何最好、召回最高，
> 但在 MNN CPU 上明显更慢；Python/PyTorch 侧的交叉校验与分阶段耗时见
> [`docs/benchmarks.md`](docs/benchmarks.md#4-超声结果)。

### ADE20K-10（10 类室内物体，800 张 test，C++/MNN CPU）

| 指标 | YOLO-Seg | Det-UNet | RF-DETR-Seg |
|---|---|---|---|
| P / R / F1 | TBD | TBD | TBD |
| Dice / IoU | TBD | TBD | TBD |
| 边界 F1 | TBD | TBD | TBD |
| cls_acc | TBD | TBD | TBD |
| 端到端 (ms) | TBD | TBD | TBD |

> 结果汇总文件：[`results/summary.csv`](results/summary.csv) / [`results/summary.md`](results/summary.md)（运行后生成）。

---

## 快速开始

```bash
# 1) Python 环境
uv venv --python 3.11 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install torch torchvision
VIRTUAL_ENV=$PWD/.venv uv pip install -e .
VIRTUAL_ENV=$PWD/.venv uv pip install "rfdetr[train,onnx]"

# 2) 超声数据 + 划分 + 导出
.venv/bin/python -m common.data_prep --all
.venv/bin/python -m common.splits
.venv/bin/python -m common.yolo_export --kind both
.venv/bin/python -m common.coco_export

# 3) 训练（任选一个）
cd yolo-seg-train  && ../.venv/bin/python train_yolo_seg.py --name seg_yolo26n && cd ..
cd det-unet-train  && ../.venv/bin/python train_yolo_det.py --name det_yolo26n && cd ..
cd rfdetr-seg-train && ../.venv/bin/python train_rfdetr_seg.py --name rfdetr_seg_ultrasound && cd ..

# 4) 导出 + C++/MNN
cd yolo-seg-train && ../.venv/bin/python export_onnx.py --seg ../runs/seg_yolo26n/weights/best.pt && cd ..
build/mnn-conv/MNNConvert -f ONNX --modelFile models/yolo26n_seg.onnx --MNNModel models/yolo26n_seg.mnn
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON && cmake --build build -j
./build/yolo-seg-test/yolo_seg_test --model models/yolo26n_seg.mnn \
    --list-file data/splits/test.txt --gt-dir data/unified/masks
```

C++ 依赖、MNN/MNNConvert 构建见 [`docs/deployment.md`](docs/deployment.md)；
两个数据集的完整训练命令见 [`docs/reproduce.md`](docs/reproduce.md)。

---

## 数据集

| profile | 内容 | 规模 |
|---|---|---|
| **Ultrasound** | BUSI + TN3K + DDTI + TG3K，4 类 | 8362 张，8922 实例 |
| **ADE20K-10** | ADE20K 室内物体，10 类 | 5400 张（4000/600/800），32599 实例 |

- 超声：BUSI 良性/恶性 + 甲状腺结节/腺体；分层划分 seed=42。
- ADE20K：具身智能/室内场景最常用的分割基准之一；选取 bed/windowpane/cabinet/door/table/
  chair/armchair/lamp/bottle/pot 十类；官方 validation 作为 test。
- ADE20K 使用实例 id mask + `instances/<uid>.csv` 类别 sidecar，支持多类多实例。

细节（来源、筛选规则、灰度说明、许可证）见 [`docs/datasets.md`](docs/datasets.md)。

---

## 部署

三个方案都导出 ONNX，再由 `MNNConvert` 转 `.mnn`；C++ 侧共用 `cxx-common`：

- 图像 IO：自带最小 PNG 编解码器（与 OpenCV 逐字节一致），无 libpng/zlib 依赖。
- 几何：letterbox / ROI / square_pad 与 Python `common/roi.py` 完全一致（填充 114）。
- 评估：`cxx-common/eval_harness.cpp` 与 `common/eval.py` 使用同一匹配/指标协议。
- 支持 `--gt-instances` 读取 ADE20K 的逐实例类别。

模型 I/O、转换命令、CLI 参数见 [`docs/deployment.md`](docs/deployment.md)。

---

## 复现

```bash
# 数据 → 训练 → 验证 → 导出 → MNN → C++ → 汇总
./scripts/run_all.sh all
./scripts/run_benchmarks.sh ultrasound   # Python + C++ 三方案对比
.venv/bin/python scripts/collect_results.py --reports reports --out results
```

长训练建议用 tmux 托管：

```bash
./scripts/train_bg.sh yolo_seg
./scripts/train_bg.sh det
./scripts/train_bg.sh mkunet
./scripts/train_bg.sh rfdetr
./scripts/train_bg.sh status
```

逐步命令见 [`docs/reproduce.md`](docs/reproduce.md)。

---

## 已知限制

- **OpenCL 未在本机验证**：WSL2 + NVIDIA 不提供 OpenCL 设备，`--backend opencl` 自动降级 CPU；
  真机加速需在 ARM Mali/Adreno 上验证。
- **RF-DETR 分辨率**：Nano 输入边长须为 12 的倍数；为控制单卡训练时间，默认用 384²，
  而 YOLO 方案为 512²。速度对比需结合分辨率解读。
- **ADE20K 使用灰度**：为与嵌入式单通道管线一致，彩色图统一转灰度，精度低于彩色训练。
- **ADE20K 为子集**：4000/600/800 的确定性采样，不代表完整 ADE20K 榜单。
- **Det-UNet 的 val_dice 偏乐观**：MK-UNet 验证集用 GT 框、训练集用检测框；
  真实指标以 test 端到端为准。
- **MNN 版本锁定 3.4.0**：更新的 MNN 公开头文件已变动，升级需重新验证。
- **RF-DETR 权重下载**：官方权重在 Google Storage 上较慢；项目提供从 HF 镜像
  `Roboflow/rf-detr-seg-nano` 的 `model.safetensors` 转换方案（见 benchmarks 文档）。

---

## 开源与许可

- 本项目代码：**AGPL-3.0**（训练链路使用 Ultralytics YOLO，AGPL-3.0）。
- 第三方：
  - [Ultralytics](https://github.com/ultralytics/ultralytics) — AGPL-3.0
  - [RF-DETR](https://github.com/roboflow/rf-detr) — Apache-2.0
  - [MNN](https://github.com/alibaba/MNN) — Apache-2.0
  - [MK-UNet](https://github.com/SLDGroup/MK-UNet) — MIT（仅重实现网络结构）
  - ADE20K — BSD-3-Clause（仅提供构建脚本，不再分发数据）
  - BUSI / TN3K / DDTI / TG3K — 见各自上游许可

完整说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`LICENSE`](LICENSE)。

如果这个项目对你有帮助，欢迎 Star / Issue / PR。贡献指南见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 引用

```bibtex
@software{contour_maker_2026,
  title  = {Contour_Maker: YOLO-Seg vs Det-UNet vs RF-DETR-Seg for instance segmentation},
  year   = {2026},
  url    = {https://github.com/<your-name>/Contour_Maker}
}
```

同时请引用上游工作：YOLO/Ultralytics、RF-DETR、MK-UNet、ADE20K。
