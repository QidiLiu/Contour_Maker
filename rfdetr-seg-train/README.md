# rfdetr-seg-train — RF-DETR-Seg-Nano

```
原图 ──► resize 384² RGB ──► DINOv2-windowed backbone
                                   │
                        deformable DETR decoder (100 queries)
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
           dets(Q,4)          labels(Q,C+1)        masks(Q,R/4,R/4)
              │                    │                    │
              └──── score > conf ──┴──► mask 上采样到原图 ──► 轮廓
```

单阶段、无 NMS/anchor。RF-DETR-Seg-Nano 参数量约 33.5M。

## 前置：COCO 数据

RF-DETR 只接受 COCO 格式：

```bash
cd "$ROOT"
.venv/bin/python -m common.coco_export                 # 超声 -> data/coco
.venv/bin/python -m common.coco_export --dataset ade20k # ADE20K -> data/ade20k/coco
```

目录布局为 `train/ valid/ test/`，每个 split 一份 `_annotations.coco.json`。

## 训练

```bash
ROOT=$(cd .. && pwd)
../.venv/bin/python train_rfdetr_seg.py --dataset ultrasound \
    --epochs 8 --batch 8 --resolution 384 --name rfdetr_seg_ultrasound
```

- `--resolution` 必须是 12 的倍数（Nano 的 `patch_size * num_windows = 12`）。
- 训练日志与 `summary.json` 在 `runs/<name>/`；最佳权重为
  `checkpoint_best_total.pth`（EMA 与 regular 的较优者）。
- 预训练权重默认 `rf-detr-seg-nano.pt`，rfdetr 会从官方源下载；国内可用 HF 镜像转换，
  见根目录 `docs/benchmarks.md`。

## 验证

```bash
../.venv/bin/python validate.py \
    --weights ../runs/rfdetr_seg_ultrasound/checkpoint_best_total.pth \
    --split test --out ../reports/ultrasound_rfdetr_seg_py
```

ADE20K：

```bash
../.venv/bin/python validate.py --dataset ade20k \
    --weights ../runs/rfdetr_seg_ade20k/checkpoint_best_total.pth \
    --split test --out ../reports/ade20k_rfdetr_seg_py
```

## 导出 ONNX / MNN

```bash
../.venv/bin/python export_onnx.py \
    --weights ../runs/rfdetr_seg_ultrasound/checkpoint_best_total.pth \
    --out ../models
# 自动调用 MNNConvert（若存在）
```

导出接口：

| 张量 | shape | 说明 |
|---|---|---|
| `input` | (1,3,R,R) | RGB，ImageNet mean/std |
| `dets` | (1,100,4) | 归一化 cx,cy,w,h |
| `labels` | (1,100,C+1) | 类别 logits，最后一列 background |
| `masks` | (1,100,R/4,R/4) | per-query mask logits |

C++ 推理见 [`../rfdetr-seg-test/README.md`](../rfdetr-seg-test/README.md)。
