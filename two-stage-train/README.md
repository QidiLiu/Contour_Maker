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

### 2. 导出 ROI 训练集

以 GT 框（或检测框）为引导裁剪 ROI，生成 MK-UNet 的训练对。

```bash
.venv/bin/python build_roi_dataset.py \
    --det ../runs/det_yolo26n/weights/best.pt \
    --splits train val \
    --box-source gt        # gt | det  （gt 用于训练, det 用于匹配推理分布）
    --out ../data/roi_cache/train_gt
```

`--box-source det` 时需要真实检测框，才能让 MK-UNet 见到推理时的框噪声分布。

### 3. 训练 MK-UNet

```bash
.venv/bin/python train_mkunet.py \
    --data ../data/roi_cache/train_gt \
    --val-data ../data/roi_cache/val_gt \
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

产出 `models/yolo26n_det.onnx` 与 `models/mkunet.onnx`，供 `two-stage-test/` 用 MNNConvert 转换。

## 依赖

见根目录 `pyproject.toml`（uv 管理）。本子项目不额外引入 MK-UNet 官方依赖：
`common/mkunet.py` 是 vendor 自 [SLDGroup/MK-UNet](https://github.com/SLDGroup/MK-UNet) 并移除了 `timm` 依赖的重实现，许可证见 `common/mkunet.LICENSE`。