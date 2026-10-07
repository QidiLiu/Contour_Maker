# one-stage-test —— 单阶段 C++ 推理（YOLO26n-seg + MNN）

端到端一次前向同时得到「位置 + 类别 + 边界」，输出带类别的目标轮廓。

## 构建

```bash
# 仓库根目录
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON
cmake --build build -j
```

产物：`build/one-stage-test/one_stage_test`

## 准备模型

```bash
# 1) 训练子项目导出 ONNX（nms=False，NMS 交给 C++ 侧）
cd one-stage-train
../.venv/bin/python export_onnx.py --seg ../runs/seg_yolo26n/weights/best.pt --out ../models

# 2) ONNX -> MNN
cd ..
MNNConvert -f ONNX --modelFile models/yolo26n_seg.onnx --MNNModel models/yolo26n_seg.mnn
```

> MNNConvert 的构建方式见根 README。

## 运行

```bash
./build/one-stage-test/one_stage_test \
    --model models/yolo26n_seg.mnn \
    --image-dir data/unified/images \
    --gt-dir data/unified/masks \
    --imgsz 512 --conf 0.25 --num-classes 4 \
    --out-dir reports/one_stage_cxx
```

单图调试（打印每个实例的类别/分数/框/轮廓点数）：

```bash
./build/one-stage-test/one_stage_test --model models/yolo26n_seg.mnn \
    --single data/unified/images/busi_benign__100_.png
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 必需 | `.mnn` 模型路径 |
| `--image-dir` / `--list-file` | 二选一 | 图像目录 / 图像清单 |
| `--single` | — | 只推理单张图并打印详情 |
| `--gt-dir` | — | GT 掩码目录，提供则评估精度 |
| `--out-dir` | — | 输出 `per_image.csv` |
| `--backend` | `cpu` | `cpu` / `opencl`（不可用时自动降级） |
| `--imgsz` | 512 | 推理输入尺寸 |
| `--conf` | 0.25 | 置信度阈值 |
| `--nms-iou` | 0.7 | NMS IoU 阈值 |
| `--num-classes` | 4 | 类别数 |
| `--min-area` | 50 | 掩码最小连通域面积 |
| `--threads` | 4 | CPU 线程数 |
| `--warmup` | 3 | 预热次数 |
| `--max-images` | 0 | 最多处理张数（0 = 全部） |

## 输出

终端打印汇总（检测 P/R/F1、分割 Dice/IoU/HD95/ASSD/BF1、分阶段耗时），
`--out-dir` 下产出 `per_image.csv`（每图 GT/预测/漏检/误检数）。

## 实现要点

- 分割头输出由 **形状识别**（而非顺序假设）定位：`4 维 = protos`、
  `3 维且通道数 ≥ 4+nc = det`、其余 `3 维 = coeffs`。
- NMS 由 C++ 侧实现（`cxx-common/src/postprocess.cpp`），`YoloDetOut::indices`
  记录保留的**原始输出索引**，用于把每实例掩码系数对齐到解码后的实例顺序。
- 掩码按 YOLACT 方式组合（`protos × coeffs`），只在框内区域计算以省时间。
- 图像 IO 用自带的最小 PNG 编解码器（`cxx-common/src/png.cpp`），
  已验证与 OpenCV 逐字节一致。