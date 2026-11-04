# 基准协议与结果

## 1. 硬件与软件

| 项 | 配置 |
|---|---|
| GPU | NVIDIA RTX 4060 Ti 16 GB（WSL2） |
| CPU | 12 核，16 GB RAM |
| OS | Ubuntu 26.04 (WSL2, kernel 6.18) |
| Python | 3.11.16 |
| PyTorch | 2.14.1+cu130 |
| Ultralytics | 8.4.174 |
| RF-DETR | 1.11.2（`RFDETRSegNano`） |
| MNN | 3.4.0，CPU 后端 |

> 速度数字只在同一台机器、同一后端、顺序单图调用下可比。C++ 侧首次推理包含
> MNN 冷启动/算子准备，因此正式测量前都有 warmup。

## 2. 指标定义

匹配：预测 mask 与 GT mask 按 IoU 贪心匹配，阈值 **0.5**；匹配上的为 TP，未匹配 GT 为 FN，
未匹配预测为 FP。

| 指标 | 定义 |
|---|---|
| Precision / Recall / F1 | TP/(TP+FP), TP/(TP+FN) |
| Dice | 2·\|P∩G\| / (\|P\|+\|G\|) |
| IoU | \|P∩G\| / \|P∪G\| |
| HD95 | 双向表面距离的 95 分位（px） |
| ASSD | 双向平均表面距离（px） |
| 边界 F1 | 容差 2 px 的双向边界 F1 |
| 面积误差 | \|area(P)-area(G)\| / area(G) |
| cls_acc | 匹配对中类别一致的比例 |
| 端到端 | 单图 wall time，含预处理/后处理；C++ 另给分阶段 |

实现细节：Python 用 `scipy.ndimage.distance_transform_edt`，C++ 用等价的
Felzenszwalb 精确 EDT；两侧都用 8-邻域边界。因此同一 mask 的 HD95/ASSD/BF1 应在
1 px 量级内一致；表中剩余差异来自预测 mask 本身（NMS/掩码上采样），不是指标定义。

GT 解码：

- 超声：mask 为二值，连通域 = 实例，类别取 `splits.csv` 的 `class_id`。
- ADE20K：mask 为实例 id，类别取自 `instances/<uid>.csv`；同一 id 的多连通域会拆成多个实例。

## 3. 训练配置

| 方案 | 输入 | 超声 | ADE20K-10 |
|---|---|---|---|
| YOLO-Seg | letterbox 512² | 60 epoch, batch 12 | 15 epoch, batch 12 |
| Det-UNet | det 512² + ROI 256² | det 60 + MK-UNet 60 | det 15 + MK-UNet 15 |
| RF-DETR-Seg | resize 384² | 8 epoch, batch 8 | 6 epoch, batch 8 |

RF-DETR 的 Nano 输入边长必须是 12 的倍数。默认 384² 是在单卡 16 GB 上兼顾时间与精度的折中；
504² 也可训练，但约慢 2 倍。

## 4. 超声结果

### 4.1 C++/MNN（CPU，test 1256 张）

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
| 面积误差 | **0.1057** | 0.1310 | 0.0920 |
| 端到端 (ms) | **43.9** | 65.6 | 518.2 |
| FPS | **22.8** | 15.2 | 1.9 |

> RF-DETR 的 MNN CPU 推理显著慢于两个 CNN 方案（Transformer + 100 query 的掩码上采样）；
> 它的优势在分割几何（Dice/HD95/BF1）和召回。精度表来自与 Python 完全一致的 GT/匹配口径。

### 4.2 Python/PyTorch（同一 test，GPU）

| 指标 | YOLO-Seg | Det-UNet | RF-DETR-Seg |
|---|---|---|---|
| P / R / F1 | 0.8751 / 0.9072 / 0.8908 | 0.8462 / 0.9057 / 0.8749 | 0.7514 / **0.9296** / 0.8310 |
| Dice | 0.9186 | 0.8880 | **0.9242** |
| IoU | 0.8535 | 0.8032 | **0.8630** |
| HD95 (px) | 8.86 | 11.67 | **7.82** |
| BF1 | 0.6332 | 0.4971 | **0.6634** |
| cls_acc | **0.9975** | 0.9959 | 0.9919 |
| 端到端 (ms) | **30.2** | 38.6 | 46.1 |

**Python ↔ C++ 交叉校验**：

- RF-DETR 的 Dice/IoU/HD95/BF1 两侧差异 ≤ 0.05，说明 MNN 转换与 C++ 后处理正确。
- YOLO-Seg 的 Dice 差 0.012、BF1 差 0.049，主要来自 NMS 与掩码上采样实现（Python 双线性 vs
  C++ 最近邻/最大连通域）。
- Det-UNet 的 HD95/ASSD/BF1 两侧几乎一致（≤ 0.3 px / 0.005）。

**结论**：YOLO-Seg 在检测 F1 与速度上最好；RF-DETR-Seg 在分割几何（Dice/HD95/BF1）上最好，
但误检多（P 0.77）、CPU 推理慢；Det-UNet 位于两者之间，且可解释性/可干预性最强。

## 5. ADE20K-10 结果

> 800 张 test（官方 validation 子集），10 类室内物体，mask IoU 0.5 匹配。
> 该子集的检测难度显著高于超声：小目标多、类别多、每图平均 6 个实例；
> 15 epoch（RF-DETR 6 epoch）下检测召回是主要瓶颈。

### 5.1 C++/MNN（CPU）

| 方案 | P | R | F1 | Dice | IoU | HD95 | BF1 | cls_acc | ms/图 |
|---|---|---|---|---|---|---|---|---|---|
| YOLO-Seg | 0.6078 | 0.3114 | 0.4118 | 0.8508 | 0.7480 | 18.49 | 0.4571 | 0.9387 | 45.8 |
| Det-UNet | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RF-DETR-Seg | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.2 Python/PyTorch（同一 test，GPU）

| 方案 | P | R | F1 | Dice | IoU | HD95 | BF1 | cls_acc | ms/图 |
|---|---|---|---|---|---|---|---|---|---|
| YOLO-Seg | 0.5275 | 0.3995 | 0.4546 | 0.8822 | 0.7970 | 17.77 | 0.6208 | 0.9272 | 28.4 |
| Det-UNet | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RF-DETR-Seg | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

> YOLO-Seg 的 Python 与 C++ 在 ADE20K 上的差异比超声大：C++ 的 NMS/掩码组合
> （protos × coeff + 最近邻）在 10 类、小目标场景下更保守，FP 更少但 FN 更多。
> 超声（单类/大目标）两侧差异 ≤ 0.01，ADE20K 上建议以同一 runtime 的横向对比为准。

## 6. 权重获取（RF-DETR）

`rfdetr` 包默认从 `storage.googleapis.com` 下载 `rf-detr-seg-nano.pt`。该域名在国内通常很慢
（实测约 30–100 KB/s）。项目提供一键转换脚本：

```bash
python scripts/convert_rfdetr_weights.py
```

脚本会从 HF 镜像下载 `Roboflow/rf-detr-seg-nano` 的 `model.safetensors`，校验键名/形状，
写入 `~/.roboflow/models/rf-detr-seg-nano.pt`。之后 `RFDETRSegNano()` 与
`train_rfdetr_seg.py` 会直接加载它。

> 该文件的 MD5 与官方原文件不同，rfdetr 会打印一条“MD5 mismatch”警告；这是预期的
> （本地转换文件），不影响加载。若想消除警告，可在训练脚本里显式传 `--weights` 路径。

## 7. 解读注意事项

- **分辨率不同**：RF-DETR 384² vs YOLO 512²；参数量 33.5M vs ~3M 级。
- **灰度**：ADE20K 彩色图被转灰度，精度会低于彩色训练；三方案看到相同输入。
- **子集**：ADE20K-10 使用 4000/600/800 的确定性采样，不是完整榜单。
- **分类头**：RF-DETR 为 sigmoid 多标签 + background 列；C++/Python 都只在真实类别上取 argmax。
- **Det-UNet 验证**：MK-UNet 验证集用 GT 框，`val_dice` 偏乐观；表中用 test 端到端指标。
