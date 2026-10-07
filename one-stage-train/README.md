# one-stage-train — YOLO26n-seg（单阶段端到端）

```
原图 ──► letterbox 512 ──► YOLO26n-seg ──► mask 原型 + 实例系数
                                        │
                              proto × coeff 组合 → 实例掩码
                                        │
                                   原图分辨率 + 类别
                                        ▼
                            带类别的目标轮廓
```

一次前向同时输出「目标在哪 + 是什么类别 + 边界形状」，不裁剪、不做第二级分割。

## 与 two-stage 的差异

| | one-stage | two-stage |
|---|---|---|
| 模型数 | 1 | 2 |
| 分割输入域 | 整图 letterbox 512 | 目标框 ROI 256×256 |
| 每图前向次数 | 恒定 1 次 | 检测 1 次 + 每框 1 次 MK-UNet |
| 类别来源 | YOLO-seg 检测头 | YOLO-det 检测头 |

## 步骤

### 0. 前置：数据与划分

```bash
# 在仓库根目录执行一次
.venv/bin/python -m common.data_prep --all
.venv/bin/python -m common.splits
.venv/bin/python -m common.yolo_export --kind both
```

### 1. 训练

```bash
.venv/bin/python train_yolo_seg.py \
    --data ../data/yolo_seg/data_seg.yaml \
    --epochs 60 --imgsz 512 --batch 12 \
    --name seg_yolo26n
```

输出 `runs/seg_yolo26n/weights/best.pt`。

### 2. 验证

```bash
.venv/bin/python validate.py \
    --seg ../runs/seg_yolo26n/weights/best.pt \
    --split test --out ../reports/one_stage
```

### 3. 导出 ONNX

```bash
.venv/bin/python export_onnx.py \
    --seg ../runs/seg_yolo26n/weights/best.pt \
    --out ../models
```

产出 `models/yolo26n_seg.onnx`，供 `one-stage-test/` 用 MNNConvert 转换。

## 注意事项

ultralytics 的分割头（mask prototype + 实例系数 + NMS）在 ONNX 导出时结构较复杂，
部分版本需要关闭内置 NMS（`nms=False`），在 C++ 侧自行做 NMS 与掩码组合，
以保证与 Python 侧口径一致。`export_onnx.py` 已按此配置导出。