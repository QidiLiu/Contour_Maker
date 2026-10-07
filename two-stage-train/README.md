# two-stage-train — YOLO26n + MK-UNet（两阶段）

```
原图 ──► YOLO26n(det, 4类) ──► 框 + 类别
                              │
                    ┌─────────┴─────────┐
                    ▼                   │
              裁剪 ROI + resize 256²    │
                    ▼                   │
              MK-UNet (1ch 二值分割)    │
                    ▼                   │
              掩码贴回原图 ─────────────┘
                    ▼
        带类别的目标轮廓 (class_id, contour)
```

## 与 one-stage 的差异

| | two-stage | one-stage |
|---|---|---|
| 模型数 | 2（YOLO26n-det + MK-UNet） | 1（YOLO26n-seg） |
| 分割输入域 | **目标框 ROI**（256×256，已裁剪） | 整图（letterbox 512） |
| 训练数据 | 检测器用 det 集；MK-UNet 用 ROI 集 | 全部用 seg 集 |
| 每个目标的分割前向 | 1 次（每框一次） | 1 次（整图一次解出全部实例） |
| 类别来源 | YOLO 检测头 | YOLO-seg 检测头 |

## 步骤

### 0. 前置：数据与划分

```bash
# 在仓库根目录执行一次
.venv/bin/python -m common.data_prep --all     # 下载 + 解压 + 统一 manifest
.venv/bin/python -m common.splits              # 分层划分 splits.csv
.venv/bin/python -m common.yolo_export --kind both
```

### 1. 训练 YOLO26n 检测器

```bash
.venv/bin/python train_yolo_det.py \
    --data ../data/yolo_det/data_det.yaml \
    --epochs 60 --imgsz 512 --batch 16 \
    --name det_yolo26n
```

输出 `runs/det_yolo26n/weights/best.pt`。

### 2. 构建 ROI 数据集

以检测框（或 GT 框）为引导裁剪 ROI，生成 MK-UNet 的训练对。

```bash
# 训练集：用检测框，让 MK-UNet 见到推理时的真实框噪声分布
.venv/bin/python build_roi_dataset.py \
    --splits train --box-source det \
    --det ../runs/det_yolo26n/weights/best.pt \
    --out ../data/roi_cache/train

# 验证集：用 GT 框（干净、稳定，便于模型选择）
.venv/bin/python build_roi_dataset.py \
    --splits val --box-source gt \
    --out ../data/roi_cache/val
```

> `--box-source det` 需要 `--det` 指定检测权重；`gt` 模式不需要。
> 注意两者要写到**不同目录**——`build_roi_dataset.py` 每次都会重写
> `index.csv`，写到同一目录会覆盖上一次的结果。
>
> 训练用 det 框、验证用 GT 框会让 `val_dice` 偏乐观（GT 框更干净）。
> 若要严格反映推理表现，验证集也应改用 `--box-source det`。

### 3. 训练 MK-UNet

```bash
.venv/bin/python train_mkunet.py \
    --data ../data/roi_cache/train \
    --val-data ../data/roi_cache/val \
    --variant MK_UNet --roi-size 256 \
    --epochs 60 --batch 16 \
    --name mkunet_yolo26n
```

输出 `runs/mkunet_yolo26n/best.pt`。

### 4. 验证（Python 侧，与 C++ 侧口径一致）

```bash
.venv/bin/python validate.py \
    --det ../runs/det_yolo26n/weights/best.pt \
    --mkunet ../runs/mkunet_yolo26n/best.pt \
    --split test --out ../reports/two_stage
```

### 5. 导出 ONNX

```bash
.venv/bin/python export_onnx.py \
    --det ../runs/det_yolo26n/weights/best.pt \
    --mkunet ../runs/mkunet_yolo26n/best.pt \
    --out ../models
```

产出 `models/yolo26n_det.onnx` 与 `models/mkunet.onnx`，
供 `two-stage-test/` 用 MNNConvert 转成 `.mnn`（转换命令见根 README「构建 MNNConvert」）。

## 依赖

见根目录 `pyproject.toml`（uv 管理）。本子项目不额外引入 MK-UNet 官方依赖：
`common/mkunet.py` 是 vendor 自 [SLDGroup/MK-UNet](https://github.com/SLDGroup/MK-UNet)
并移除了 `timm` 依赖的重实现（官方 README 提到的 `mmcv-full` 实际未被其代码使用）。
官方仓库为 MIT 许可，本项目仅重实现网络结构，未复制其训练脚本。