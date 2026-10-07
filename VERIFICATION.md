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
| 构建 | `cmake -DMNN_OPENCL=ON` 全绿；`one_stage_test` / `two_stage_test` 均产出 |
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

## 4. 单阶段端到端（已完成）

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

## 5. 两阶段端到端（未完成）

代码路径已全部实现并通过编译，但**尚未用真实模型跑通**，因为需要：

1. 训练 YOLO26n-det（约 1 小时）
2. 用检测框构建 ROI 数据集
3. 训练 MK-UNet（约 30 分钟）
4. 导出 ONNX → MNN
5. 跑 `two_stage_test`

已验证的部分：
- `build_roi_dataset.py`（GT 框模式）：60 图 → 60 ROI 样本，
  256×256，掩码不触边（1/60）
- `train_mkunet.py`：完整训练循环跑通，`val_dice` 0.8278
- `two_stage_test` 二进制可加载两个 MNN 模型并执行前向

执行方式见根 README 或 `scripts/run_all.sh train`。

## 6. OpenCL 说明

本机（WSL2 + NVIDIA RTX 4060 Ti）**无法验证 OpenCL 加速**：
`ldconfig` 无 OpenCL 库、`/etc/OpenCL/vendors` 为空、无 `/dev/dri`，
NVIDIA 在 WSL2 下仅透传 CUDA。

已完成：OpenCL 编译进 MNN，`--backend opencl` 参数存在且会在不可用时优雅降级。
**未完成：真机 GPU 加速实测**（需在 ARM Mali/Adreno 设备上进行）。

## 7. 环境不稳定性说明

会话期间服务端多次重启，导致：
- 训练进程被中断（seg 停在 50/60；det 未开始）
- `tmux` 会话与 `/tmp/opencode` 被清空

因此本记录中的单阶段结果来自**已落盘的 best.pt**，两阶段训练需重新执行。
建议用 `scripts/run_all.sh` 在稳定环境下一气跑完。
