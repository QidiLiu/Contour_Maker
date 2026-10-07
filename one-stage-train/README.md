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

YOLO26-seg 的头部（`Segment26`）**自带 TopK 选择**，因此 ONNX 导出既不是
「纯 logits」也不是标准的 YOLOv8 三输出，而是**两个输出**：

```
images  (1, 3, 512, 512)          # 3 通道（ultralytics 以 BGR 读图）
output0 (1, 300, 38)              # [x1,y1,x2,y2, score, class_id, coeff×32]
                                  #  已按 score 降序，未做 NMS
output1 (1, 32, 128, 128)         # 32 个掩码原型
```

`export_onnx.py` 以 `nms=False, dynamic=False` 导出，NMS 与
YOLACT 式掩码组合（`protos × coeffs`，框内裁剪）都放在 C++ 侧实现，
以保证两侧口径一致。详见 `one-stage-test/README.md`。