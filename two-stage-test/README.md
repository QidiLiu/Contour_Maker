# two-stage-test —— 两阶段 C++ 推理（YOLO26n-det + MK-UNet + MNN）

```
原图 ──► YOLO26n(det) ──► 框 + 类别
                          │
                裁剪 ROI → resize 256²
                          │
                   MK-UNet (1ch)
                          │
                   掩码贴回原图
                          ▼
                带类别的目标轮廓
```

## 构建

与 `one-stage-test` 共用根 CMake：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON
cmake --build build -j
```

产物：`build/two-stage-test/two_stage_test`

## 准备模型

```bash
cd two-stage-train
../.venv/bin/python export_onnx.py \
    --det ../runs/det_yolo26n/weights/best.pt \
    --mkunet ../runs/mkunet_yolo26n/best.pt \
    --out ../models
cd ..

MNNConvert -f ONNX --modelFile models/yolo26n_det.onnx --MNNModel models/yolo26n_det.mnn
MNNConvert -f ONNX --modelFile models/mkunet.onnx     --MNNModel models/mkunet.mnn
```

## 运行

```bash
./build/two-stage-test/two_stage_test \
    --det models/yolo26n_det.mnn \
    --mkunet models/mkunet.mnn \
    --image-dir data/unified/images \
    --gt-dir data/unified/masks \
    --det-imgsz 512 --roi-size 256 --pad-ratio 0.15 --conf 0.25 \
    --out-dir reports/two_stage_cxx
```

单图调试：

```bash
./build/two-stage-test/two_stage_test --det models/yolo26n_det.mnn \
    --mkunet models/mkunet.mnn --single data/unified/images/busi_benign__100_.png
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--det` / `--mkunet` | 必需 | 两个 `.mnn` 模型 |
| `--image-dir` / `--list-file` / `--single` | — | 输入来源 |
| `--gt-dir` | — | GT 掩码目录，提供则评估 |
| `--out-dir` | — | 输出 `per_image.csv` |
| `--backend` | `cpu` | `cpu` / `opencl` |
| `--det-imgsz` | 512 | 检测输入尺寸 |
| `--roi-size` | 256 | MK-UNet ROI 尺寸（须与训练一致） |
| `--conf` | 0.25 | 置信度阈值 |
| `--nms-iou` | 0.7 | NMS IoU |
| `--num-classes` | 4 | 类别数 |
| `--pad-ratio` | 0.15 | 框外扩比例（**须与 Python 侧一致**） |
| `--thr` | 0.5 | MK-UNet 二值化阈值（sigmoid 之后） |
| `--min-area` | 50 | 最小连通域面积 |
| `--threads` / `--warmup` / `--max-images` | 4 / 3 / 0 | 运行控制 |

## 实现要点

- **逐 ROI 前向**：与 Python 侧 `validate.py` 一致，每个检测框调用一次 MK-UNet
  （batch=1），因此速度随病灶数线性增长。
- **两级 ROI 几何必须与 Python 侧逐像素一致**：
  1. 框按 `pad_ratio` 外扩 → clip 到图像范围
  2. 裁剪 ROI → 等比 resize 到 `roi_size` 后居中填充（填充值 114）
  3. MK-UNet 输出 sigmoid > `thr` → 二值 ROI 掩码
  4. 反解居中偏移 → resize 回 ROI 尺寸 → 平移贴回原图
  实现在 `cxx-common/src/image.cpp` 的 `CropRoi` / `ResizeRoi` / `PasteRoiMask`。
- 计时分四段输出（检测 / ROI 裁剪 / MK-UNet / 后处理），便于定位瓶颈。

## 类别表

| id | 名称 |
|---|---|
| 0 | `breast_benign` |
| 1 | `breast_malignant` |
| 2 | `thyroid_nodule` |
| 3 | `thyroid_gland` |

打印的 `class=` 即上述 id。