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

TBD（训练/测试完成后填入）。

## 5. ADE20K-10 结果

TBD（训练/测试完成后填入）。

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
