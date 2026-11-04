# 架构与实现

## 1. 三个方案

| 方案 | 英文名 / 目录 | 模型 | 分割输入域 | 每图前向 | 类别来源 |
|---|---|---|---|---|---|
| **YOLO-Seg** | `yolo-seg` | YOLO26n-seg | 整图 letterbox 512² | 1 次 | YOLO-seg 检测头 |
| **Det-UNet** | `det-unet` | YOLO26n-det + MK-UNet | 目标框 ROI → 256² | 1 + 每框 1 次 | YOLO-det 检测头 |
| **RF-DETR-Seg** | `rfdetr-seg` | RF-DETR-Seg-Nano | 整图 resize 384² / 504² | 1 次（DETR，无 NMS） | DETR 分类头 |

三个名字都直接描述「模型 + 分割方式」：YOLO-Seg / Det-UNet / RF-DETR-Seg。目录统一为 `<scheme>-train` 与 `<scheme>-test`。

### YOLO-Seg

```
原图 ──► letterbox 512 ──► YOLO26n-seg ──► (bbox, score, class, coeff×32)
                                                   │
                                       protos × coeff 组合
                                                   │
                                         掩码 → 原图 → 轮廓
```

- 单模型、单次前向，速度最快。
- 分割分辨率受整图 letterbox 限制，小目标边界较粗。
- 端到端黑盒，检测/分割不可分别干预。

### Det-UNet

```
原图 ──► YOLO26n-det ──► 框 + 类别
                           │
                 框外扩 15% → 裁剪 ROI → 256²
                           │
                    MK-UNet (1ch, 0.33M)
                           │
                    ROI mask 贴回原图
                           ▼
                 带类别的目标轮廓
```

- 检测与分割解耦：框可替换、可加约束，ROI 分辨率固定，边界更细。
- 多病灶时 MK-UNet 逐 ROI 前向，耗时线性增长。
- MK-UNet 只做前景/背景二值分割，类别完全来自检测器。

### RF-DETR-Seg

```
原图 ──► resize 384² RGB ──► DINOv2-windowed + deformable DETR decoder
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                 dets(Q,4)     labels(Q,C+1)   masks(Q,R/4,R/4)
                    │              │              │
                    └──── 阈值/argmax ──► 掩码上采样 → 原图 → 轮廓
```

- DETR 式查询：无 NMS、无 anchor，输出固定 Q=100 个查询。
- 分类头为 sigmoid 多标签；最后一列为 background，C++/Python 都只在真实类别上取 argmax。
- 掩码是 per-query logit（R/4 × R/4），双线性上采样到原图后按 `> 0` 二值化。
- 参数量 33.5M，约为 YOLO26n-seg 的 3 倍；Nano 是 RF-DETR-Seg 的最小变体。

---

## 2. 统一数据流

```
common/data_prep.py        超声：下载/解压/统一 → manifest.csv
common/splits.py           分层划分 → splits.csv
common/ade20k.py           ADE20K：parquet → 统一格式 + instance sidecar
common/yolo_export.py      → YOLO det/seg 数据集（per-instance class）
common/coco_export.py      → COCO（RF-DETR）
```

训练：

```
yolo-seg-train/     train_yolo_seg.py   → validate.py → export_onnx.py
det-unet-train/     train_yolo_det.py → build_roi_dataset.py → train_mkunet.py
                    → validate.py → export_onnx.py
rfdetr-seg-train/   train_rfdetr_seg.py → validate.py → export_onnx.py
```

评估：

```
common/eval.py      Python 统一评估循环（预测回调 + GT + 指标 + CSV/JSON）
cxx-common/eval_harness.cpp   C++ 统一评估循环（同一协议）
scripts/collect_results.py    汇总 reports/*/summary.json
```

Python 与 C++ 使用相同的 `letterbox / ROI / square_pad` 几何（`common/roi.py` 与
`cxx-common/src/image.cpp`），填充值同为 114，这是两侧指标能对齐的前提。

---

## 3. 目录结构

```
common/                 共享 Python 库
  config.py             路径、类别、超参、数据集常量
  data_prep.py          超声数据下载/解压/统一
  splits.py             分层划分（通用）
  ade20k.py             ADE20K 子集构建
  yolo_export.py        YOLO 导出（通用）
  coco_export.py        COCO 导出（RF-DETR）
  instances.py          GT 实例解码 + sidecar
  eval.py               Python 统一评估
  metrics.py            Dice/IoU/HD95/ASSD/BF1/面积误差
  roi.py                letterbox/ROI/掩码轮廓
  mkunet.py             MK-UNet（vendored，MIT）
yolo-seg-train/         YOLO-Seg 训练/验证/导出
det-unet-train/         Det-UNet 训练/验证/导出
rfdetr-seg-train/       RF-DETR-Seg 训练/验证/导出
yolo-seg-test/          YOLO-Seg C++/MNN 推理
det-unet-test/          Det-UNet C++/MNN 推理
rfdetr-seg-test/        RF-DETR-Seg C++/MNN 推理
cxx-common/             C++ 共享库：图像 IO/几何/指标/MNN 封装/评估
scripts/                一键脚本与结果汇总
docs/                   数据集、架构、部署、复现、基准文档
models/                 入库的 .mnn（超声三方案）
weights/                训练起点（YOLO 基座）
results/                可复现的汇总结果（CSV/Markdown）
```

---

## 4. 关键设计决定

### 4.1 为什么超声方案都用灰度

超声原图为单通道。YOLO/RF-DETR 的 3 通道输入由灰度复制得到，MK-UNet 直接用 1 通道。
优点：输入带宽最小、与 C++ 单通道图像 IO 一致。ADE20K 也统一转灰度，以保证三方案、
Python/C++ 的预处理完全一致；代价是丢失颜色信息，已在基准文档中说明。

### 4.2 多类多实例的 GT 表示

超声：一张图一个类别，mask 为二值，连通域 = 实例。
ADE20K：一张图多类别多实例，mask 为 8-bit 实例 id，类别在 `instances/<uid>.csv`。
`common/instances.py` 与 `cxx-common` 同时支持两种约定；评估时按实例 id 拆连通域并赋予类别。

### 4.3 检测框 vs GT 框

Det-UNet 的 MK-UNet 训练集用**检测框**（匹配推理分布），验证集用 **GT 框**（干净、稳定）。
这会让 `val_dice` 偏乐观；真实推理指标以 test 的端到端口径为准。

### 4.4 NMS

- YOLO-det：头部自带 TopK，但同框会跨类别重复，C++/Python 都按类别做 NMS。
- YOLO-seg：同上，导出时 `nms=False`，NMS 在推理侧。
- RF-DETR：DETR 查询天然去重，不做 NMS。

### 4.5 评估指标

- 检测：mask IoU ≥ 0.5 的贪心匹配 → TP/FP/FN → P/R/F1。
- 分割：匹配对的 Dice、IoU、HD95、ASSD、边界 F1（容差 2 px）、面积相对误差。
- 分类：匹配对的类别一致率 `cls_acc`。
- 速度：端到端 wall time（含预处理/后处理），C++ 侧另给分阶段耗时。
