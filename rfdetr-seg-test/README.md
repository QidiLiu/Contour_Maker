# rfdetr-seg-test — RF-DETR-Seg C++ 推理（MNN）

单阶段、DETR 式查询，无 NMS。C++ 侧按官方后处理实现：

1. `input`：原图 resize 到 R×R（R 为 12 的倍数），灰度复制到 3 通道并做 ImageNet 归一化。
2. `labels`：对前 C 列做 sigmoid，取 argmax；最后一列为 background。
3. `dets`：归一化 `cx,cy,w,h` → 乘原图宽高 → xyxy。
4. `masks`：每个 query 的 logit 图（R/4×R/4）双线性上采样到原图，`> 0` 二值化。
5. 取最大外轮廓；不做 NMS（DETR 查询天然去重）。

## 构建

根目录统一构建：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON
cmake --build build -j
```

产物：`build/rfdetr-seg-test/rfdetr_seg_test`

## 运行

```bash
./build/rfdetr-seg-test/rfdetr_seg_test \
    --model models/rfdetr_seg_nano.mnn \
    --image-dir data/unified/images \
    --gt-dir data/unified/masks \
    --resolution 384 --conf 0.25 --num-classes 4 \
    --out-dir reports/ultrasound_rfdetr_seg_cxx
```

ADE20K（逐实例类别）：

```bash
./build/rfdetr-seg-test/rfdetr_seg_test \
    --model models/ade20k/rfdetr_seg_nano_ade20k.mnn \
    --image-dir data/ade20k/unified/images \
    --gt-dir data/ade20k/unified/masks \
    --gt-instances data/ade20k/unified/instances \
    --resolution 384 --num-classes 10 \
    --out-dir reports/ade20k_rfdetr_seg_cxx
```

单图调试：

```bash
./build/rfdetr-seg-test/rfdetr_seg_test --model models/rfdetr_seg_nano.mnn \
    --single data/unified/images/busi_benign__100_.png --num-classes 4
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 必需 | `.mnn` 模型 |
| `--image-dir` / `--list-file` | 二选一 | 图像来源 |
| `--single` | — | 单图推理 |
| `--gt-dir` | — | GT mask 目录 |
| `--gt-instances` | — | `instance_id,class_id` sidecar 目录（ADE20K） |
| `--out-dir` | — | 输出 `per_image.csv` / `summary.json` |
| `--resolution` | 384 | 输入边长，须与导出时一致且为 12 的倍数 |
| `--conf` | 0.25 | sigmoid 置信度阈值 |
| `--num-classes` | 4 | 真实类别数（不含 background） |
| `--threads` / `--warmup` / `--max-images` | 4 / 3 / 0 | 运行控制 |
| `--quiet` | — | 关闭进度 |

## 与 Python 侧的差异

- Python 用 `RFDETRSegNano.predict()`（torchvision resize，antialias=False）；C++ 用等价的双线性
  resize。输入分辨率一致时 Dice 差异通常 < 0.01。
- C++ 的 `labels` 取前 C 列 sigmoid argmax；Python 的 postprocess 在 Q×(C+1) 上 top-k 后
  再取类别，但 background 分数通常很低，实际结果一致。
- MNN 的 `GridSample`/`LayerNormalization` 为 fp32 实现，与 ONNX Runtime 可能有 1e-3 量级差异；
  建议用 `--max-images 50` 先做交叉校验。
