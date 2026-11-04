# 验证记录

本文件记录重构后各环节的实测验证结果，便于区分「已验证」与「未验证」。

## 1. 数据管线

| 项 | 结果 |
|---|---|
| 下载 | 4 个压缩包共 481.6 MB（hf-mirror 可达；Google Drive 不可达，已规避） |
| 解压 | 4 个数据集全部成功 |
| SIM配对 | busi 647（排除 `normal` 无标注 133 张）/ tn3k 3493 / ddti 637 / tg3k 3585 = **8362** |
| 类别恢复 | BUSI 的 benign/malignant 从原包目录名恢复，437 / 210，与原分布一致 |
| 分层划分 | seed=42，train 5850 / val 1256 / test 1256；4 类在三个 split 中均覆盖 |
| YOLO 导出 | det / seg 各 8362 图、8922 个目标 |

双扩展名冗余（DDTI 的 `test1.PNG.png` 与 `test1.png` 内容相同）已正确去重，
否则 DDTI 会从 637 变成 1274。

## 2. MK-UNet 重实现

| 项 | 结果 |
|---|---|
| 参数量 | `MK_UNet` = 0.3267M（论文 0.316M，偏差 3.4%）；`_S` = 0.095M；`_T` = 0.027M |
| 依赖 | 移除 `timm`（官方 README 提到的 `mmcv-full` 实际未被代码使用） |
| `in_channels=1` | 支持灰度单通道（官方在 forward 里把 1 通道复制成 3 通道） |
| 输出分辨率 | 512 / 256 / 224 均与输入一致 |

**修复的关键 bug**：官方 `forward` 写作 `self.CA1(out)*out`，而 `ChannelAttention.forward`
本身已返回 `x * sigmoid(attn)`，等价于把激活平方。经 5 级（每级在 bottleneck 与 4 个
解码级各一次）放大后，激活值从 ~15 爆到 **1e18**，训练 loss 直接 NaN。
改为按注意力门控本意直接使用模块返回值后正常。
→ 修复后 8 epoch 冒烟训练：`val_dice` 0.0 → 0.8278。

## 3. C++ 侧

| 项 | 结果 |
|---|---|
| 构建 | `cmake -DMNN_OPENCL=ON` 全绿；`yolo_seg_test` / `det_unet_test` 均产出 |
| 自带 PNG 解码 | 与 OpenCV **逐字节一致**（多尺寸多图抽检） |
| 自带 PNG 编码 | 往返重建与 OpenCV 一致 |
| MNN 版本 | 3.4.0（3.6 的公开头文件已移除 `Session` 低层接口） |
| MNN 接口 | `Express::Module` + `RuntimeManager` |
| OpenCL | 编译进二进制（`MNN_OPENCL_ENABLED`）；本机无 OpenCL 设备，自动降级 CPU |

**修复的三个 PNG 解码 bug**（均为静默错误）：

1. `push_back` 的参数引用了同一 vector 的元素 → 扩容后是 UB
2. `final_bit` 判断在解码最终块**之前** break → 最后一块数据被整块丢弃（解压结果少 3566 字节）
3. 行滤波用饱和截断而非 **mod 256 回绕** → 像素出现 0↔255 反相

`MNNConvert` 需 `-DMNN_BUILD_CONVERTER=ON`（默认 OFF）且用仓库自带的 protobuf 构建。

## 4. YOLO-Seg端到端（已完成）

训练：YOLO26n-seg，60 epoch 中完成 **50**（环境多次重启中断），
`mAP50(box)=0.865`、`mAP50(mask)=0.855`。

导出格式（实测确认）：
```
input  images   (1, 3, 512, 512)      # 3 通道，灰度复制
output0         (1, 300, 38)          # [x1,y1,x2,y2, score, class_id, coeff×32]
                                      #  已按 score 降序(TopK)，未做 NMS
output1         (1, 32, 128, 128)     # 32 个掩码原型
```

**PyTorch 与 C++/MNN 交叉验证**（同一测试子集 120 张）：

| 指标 | PyTorch | C++ / MNN | Δ |
|---|---|---|---|
| Precision | 0.8099 | 0.8393 | +0.029 |
| Recall | 0.8167 | 0.7833 | −0.033 |
| **F1** | **0.8133** | **0.8103** | **−0.003** |
| Dice | ≈0.900 | 0.8791 | ≈−0.02 |
| 端到端 | 28.96 ms | 35.83 ms | — |

F1 差异仅 0.3 个百分点，说明两侧预处理与后处理口径一致。
残余差异来源：NMS 实现细节（Python 用 ultralytics 类内 NMS，C++ 用自实现）、
掩码上采样插值（Python 双线性，C++ 最近邻）。

## 5. Det-UNet端到端（已完成）

训练：
- YOLO26n-det：60 epoch，`mAP50(box)=0.852`
- MK-UNet：60 epoch，`best val_dice=0.8973`
- ROI 数据集：train 用**检测框**（匹配推理分布）6309 样本；val 用 GT 框 1351 样本

导出格式（实测确认）：
```
yolo26n_det.mnn   input images (1,3,512,512)     output0 (1,300,6)
                                                # [x1,y1,x2,y2, score, class_id]
mkunet.mnn        input input  (1,1,256,256)     output logits (1,1,256,256)
```

**两方案在 C++/MNN 下的同口径对比**（同一 120 张测试子集）：

| 指标 | YOLO-Seg (YOLO26n-seg) | Det-UNet (YOLO26n-det + MK-UNet) |
|---|---|---|
| Precision | **0.8393** | 0.7642 |
| Recall | 0.7833 | 0.7833 |
| F1 | **0.8103** | 0.7737 |
| Dice | 0.8791 | **0.8877** |
| IoU | 0.7882 | **0.8031** |
| HD95 | 3.87 px | 3.86 px |
| ASSD | 2.06 px | **2.04 px** |
| 边界 F1 | 0.6051 | **0.6438** |
| 面积误差 | 0.1472 | **0.1220** |
| 端到端 | **37.3 ms** (26.8 FPS) | 47.4 ms (21.1 FPS) |

分阶段耗时（Det-UNet）：检测 22.4 + ROI 裁剪 0.6 + MK-UNet 19.7 + 后处理 1.6 ms。

**PyTorch ↔ C++/MNN 一致性**：

| | PyTorch | C++ / MNN | Δ |
|---|---|---|---|
| YOLO-Seg F1 | 0.8133 | 0.8103 | 0.003 |
| Det-UNet Dice | 0.8897 | 0.8877 | 0.002 |
| Det-UNet F1 | 0.7452 | 0.7737 | 0.029 |

分割几何（Dice）两侧差异 ≤0.002，高度一致。
Det-UNet的 F1 差异（0.029）来自 NMS 实现不同导致的误检数不同
（Python 用 ultralytics 的 NMS，C++ 为自实现类内 NMS）。

**观察**：Det-UNet召回与YOLO-Seg完全相同（TP=94 / FN=26，同一检测子集），
分割质量略优（Dice/BF1/面积误差均更好），但精度较低（误检 29 vs 18），
且慢 27%（每框一次 MK-UNet 前向）。

### 已知局限

- MK-UNet 的**验证集用 GT 框**（训练用检测框），val_dice 0.8973 偏乐观；
  推理时的真实 Dice 见上表 0.8877。后续应把验证集也换成检测框。
- MK-UNet 逐 ROI 前向（batch=1），多病灶图会线性变慢，未做批量化。

## 6. OpenCL 说明

本机（WSL2 + NVIDIA RTX 4060 Ti）**无法验证 OpenCL 加速**：
`ldconfig` 无 OpenCL 库、`/etc/OpenCL/vendors` 为空、无 `/dev/dri`，
NVIDIA 在 WSL2 下仅透传 CUDA。

已完成：OpenCL 编译进 MNN，`--backend opencl` 参数存在且会在不可用时优雅降级。
**未完成：真机 GPU 加速实测**（需在 ARM Mali/Adreno 设备上进行）。

## 7. 环境不稳定性说明

会话期间服务端多次重启，导致：
- YOLO-Seg训练被中断（seg 停在 **50/60** epoch，非完整 60）
- `tmux` 会话与 `/tmp/opencode` 被清空

处理方式：
- YOLO-Seg结果取自**已落盘的 best.pt**（50 epoch，mAP50=0.865），指标有效
- Det-UNet改用 `tmux` 托管后**已完整跑完**（det 60/60，MK-UNet 60/60）
- 为此新增了 `scripts/train_bg.sh`（tmux 托管长训练）

若需补齐YOLO-Seg的最后 10 epoch：

```bash
./scripts/train_bg.sh yolo_seg   # 从 runs/seg_yolo26n/weights/last.pt 续训
```

`seg` 任务已在脚本中配置为 `resume=True`，会自动从 `last.pt` 恢复
（含优化器状态与 epoch 计数）。


## 8. RF-DETR-Seg-Nano（新增方案）

| 项 | 结果 |
|---|---|
| 预训练权重 | 官方 `storage.googleapis.com` 约 30–100 KB/s；改用 HF 镜像 `Roboflow/rf-detr-seg-nano` 的 `model.safetensors` 转换 |
| 权重一致性 | safetensors 与 rfdetr 状态字典键名/形状 544/545 完全对应（仅差 `_kp_active_mask`） |
| 训练 | 384²、batch 8、8 epoch；RTX 4060 Ti 约 2.3 it/s，单 epoch 约 6 分钟 |
| ONNX | opset 17；输入 (1,3,R,R)，输出 `dets` (1,100,4)、`labels` (1,100,C+1)、`masks` (1,100,R/4,R/4) |
| MNN | MNNConvert 3.4.0 转换成功；C++ `rfdetr_seg_test` 可运行 |
| C++ 后处理 | sigmoid(labels) 取真实类别 argmax；dets 乘原图宽高；masks 双线性上采样后 `>0` |
| 已知差异 | MNN fp32 与 ONNXRuntime 有 ~1e-2 量级 logit 差异；fine-tune 后置信度高，不影响匹配 |

## 9. ADE20K-10 数据管线

| 项 | 结果 |
|---|---|
| 下载 | HF 镜像 4 个 parquet 共约 1.15 GB |
| 标注解析 | R 通道 = ADE20K thing id（1–100），G 通道 = 实例 id |
| 类别映射 | 逐像素与 official semantic annotation 对齐，R=5→door、R=35→screen door、R=38→toilet 等 |
| 采样 | train 候选 9394、test 候选 938；取 4000/600/800 |
| 写出 | 5400 张灰度图、32597 个实例、每图独立 instance sidecar |
| 导出 | YOLO det/seg 各 5400 图 / 32597 目标；COCO 4000/600/800 |

## 10. 统一评估链路

- Python：`common/eval.py`，所有方案共用同一 GT 解码/匹配/指标/CSV/JSON 逻辑。
- C++：`cxx-common/eval_harness.cpp`，与 Python 同一协议，输出 `per_image.csv` + `summary.json`。
- 逐实例类别：`--gt-instances` / `instances/<uid>.csv` 同时被两侧读取。
- 已在 YOLO-Seg / Det-UNet / RF-DETR-Seg 三条链路上验证可运行。


## 11. 三方案同口径实测（超声 test 1256 张）

| 方案 | 侧 | P | R | F1 | Dice | IoU | HD95 | BF1 | ms/图 |
|---|---|---|---|---|---|---|---|---|---|
| YOLO-Seg | Python | 0.8751 | 0.9072 | 0.8908 | 0.9186 | 0.8535 | 8.86 | 0.6332 | 30.2 |
| YOLO-Seg | C++ | 0.8994 | 0.8967 | 0.8981 | 0.9061 | 0.8326 | 9.51 | 0.5843 | 43.9 |
| Det-UNet | Python | 0.8462 | 0.9057 | 0.8749 | 0.8880 | 0.8032 | 11.67 | 0.4971 | 38.6 |
| Det-UNet | C++ | 0.8818 | 0.8877 | 0.8847 | 0.8889 | 0.8050 | 11.34 | 0.5021 | 65.6 |
| RF-DETR-Seg | Python | 0.7514 | 0.9296 | 0.8310 | 0.9242 | 0.8630 | 7.82 | 0.6634 | 46.1 |
| RF-DETR-Seg | C++ | 0.7679 | 0.9289 | 0.8408 | 0.9243 | 0.8630 | 7.86 | 0.6630 | 518.2 |

- C++ 的 HD95/ASSD/BF1 已改为精确 EDT + 8-邻域边界，与 Python/scipy 口径一致；
  RF-DETR 两侧分割几何差异 ≤ 0.05，Det-UNet 几乎完全一致。
- YOLO-Seg 的残差来自 NMS 实现与掩码上采样（Python 双线性/多边形，C++ 最近邻/最大连通域），
  Dice 差 0.012、F1 差 0.007，属于后处理差异而非模型差异。
- RF-DETR 的 MNN CPU 推理 518 ms/图，明显慢于 CNN 方案；这是 Transformer + 100 query
  掩码上采样的部署代价，GPU/OpenCL 后端需进一步验证。


## 12. ADE20K-10 三方案（test 800 张）

| 方案 | 侧 | P | R | F1 | Dice | BF1 | cls_acc | ms/图 |
|---|---|---|---|---|---|---|---|---|
| YOLO-Seg | Python | 0.5275 | 0.3995 | 0.4546 | 0.8822 | 0.6208 | 0.9272 | 28.4 |
| YOLO-Seg | C++ | 0.6078 | 0.3107 | 0.4110 | 0.8508 | 0.4571 | 0.9387 | 45.8 |
| Det-UNet | Python | 0.4595 | 0.3528 | 0.3991 | 0.8440 | 0.4966 | 0.9312 | 53.0 |
| Det-UNet | C++ | 0.5356 | 0.2678 | 0.3571 | 0.8508 | 0.5026 | 0.9469 | 97.9 |
| RF-DETR-Seg | Python | 0.4609 | 0.6863 | 0.5514 | 0.8895 | 0.7174 | 0.9427 | 63.6 |
| RF-DETR-Seg | C++ | 0.4817 | 0.6872 | 0.5664 | 0.8900 | 0.7193 | 0.9479 | 407.6 |

- ADE20K 的逐实例类别 sidecar 在 Python/C++ 两侧均生效；cls_acc 0.93–0.95。
- 检测召回是主要瓶颈（15 epoch、10 类、小目标），但匹配目标的 Dice 仍达 0.84–0.89。
- RF-DETR 在 ADE20K 上反超两个 CNN 方案（F1/Dice/BF1 最好），代价是 CPU 推理 407.6 ms/图。
