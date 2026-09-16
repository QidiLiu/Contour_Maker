# Contour_Maker

超声图像中**低回声目标**（血管横截面 / 乳腺肿瘤 / 甲状腺结节）的轮廓识别与分割对比实验。

对比两条技术路线：

| 方案 | 组成 | 说明 |
|---|---|---|
| **A（本方案）** | YOLO26n 检测框 → ROI 内 Otsu 自适应分割 → 最大轮廓作为**粗糙轮廓** → **contour-refiner** 深度学习模型精细化 | 用检测框限定局部，Otsu 给出初始轮廓，轻量模型逐点回归精准轮廓 |
| **B（baseline）** | YOLO26n-seg 端到端实例分割 | 直接输出实例 mask |

## 结果速览（测试集 330 张 / 341 个目标，BUSI+TN3K+DDTI）

| 方法 | Dice ↑ | IoU ↑ | HD95(px) ↓ | ASSD(px) ↓ | Dice<0.5 占比 ↓ |
|---|---|---|---|---|---|
| **A-box: 框矩形→contour-refiner（YOLO 框）** | **0.8028** | 0.7194 | **14.33** | 5.39 | **8.4%** |
| B: YOLO26n-seg（端到端） | 0.7923 | 0.7191 | 14.33 | **4.62** | 11.0% |
| A: Otsu+contour-refiner（GT 框，理想检测） | 0.7808 | 0.6937 | 15.59 | 5.43 | 9.7% |
| A: Otsu+contour-refiner（YOLO 框，原配置） | 0.6602 | 0.5833 | 16.83 | 5.69 | 23.0% |
| A-0: 仅 Otsu 粗糙轮廓（YOLO 框） | 0.5410 | 0.4617 | 21.27 | 7.16 | 33.8% |
| A-0: 仅 Otsu 粗糙轮廓（GT 框） | 0.6556 | 0.5683 | 20.39 | 6.51 | 21.7% |

**改进后的最终结论**：把初始轮廓从"框内 Otsu 最大暗连通域"改为**检测框矩形**（同一 refiner、
无需重训检测器/分割器），方案 A 达到 **0.8028 Dice，反超 YOLO26n-seg 的 0.7923**，HD95 持平
14.33px；检测命中目标上的轮廓命中率 **86.5% → 99.3%**（oracle 上限 99.7%）。

| 方法 | Dice ↑ | IoU ↑ | HD95(px) ↓ |
|---|---|---|---|
| **A-box: 框矩形→contour-refiner** | **0.8028** | 0.7194 | **14.33** |
| B: YOLO26n-seg | 0.7923 | 0.7191 | 14.33 |
| A: Otsu→contour-refiner（原配置） | 0.6602 | 0.5833 | 16.83 |
| A-0: 仅 Otsu 粗糙轮廓 | 0.5410 | 0.4617 | 21.27 |

漏检根因归因（341 个 GT 目标）：检测命中 87.1%、Otsu 出轮廓 87.1%、但 **Otsu 轮廓本身
IoU≥0.5 仅 64.2%**，而框矩形达 80.6% —— 说明**漏检主因是初始化而非检测器**；检测阈值扫描与
候选池化（检测+分割框 / 翻转 TTA / 多尺度）都无法提升 F1（0.905 已是阈值最优）。
详见 [`reports/REPORT.md`](reports/REPORT.md) §4.3 与 [`EXPERIMENT.md`](EXPERIMENT.md)。

**边界案例（TG3K 甲状腺腺体，高回声）**：Otsu 的"低回声"假设失效，粗糙轮廓 Dice 仅 0.12
（改用 `polarity=bright` 可到 0.43）；方案 B 在未训练过的该类别上同样输出空 mask。
说明方案 A 适用于**目标为相对周围组织的低回声区**这一前提。

**追加实验：把 refiner 接到 YOLO26n-seg 之后（B+refiner）** → **没有提升**：
Dice 0.7923 → 0.7813（配对比较，58% 目标变差）。原因是该 refiner 按"框→Otsu 粗糙轮廓"
（平均 Dice≈0.54）校准，面对本就准确的 seg mask 属分布外输入，会过修正——
把 GT 轮廓喂给它时 Dice 从 0.9954 掉到 0.9127（100% 变差）。详见 `reports/REPORT.md` §4.2。

## 四方案对比：精度 × 推理速度（同一数据集 / 同一划分 / 测试集）

| 流水线 | Dice ↑ | IoU ↑ | HD95(px) ↓ | 零样本 Dice | 端到端 ms/张 ↓ | FPS ↑ |
|---|---|---|---|---|---|---|
| **B: YOLO26n-seg** | **0.7735** | 0.7040 | 15.93 | — | **10.32** | **96.9** |
| YOLO26n + EfficientSAM (ViT-T，微调 decoder) | 0.7682 | 0.6947 | 16.83 | 0.0000 | 99.71 | 10.0 |
| YOLO26n + EdgeSAM (微调 decoder) | 0.7648 | 0.6899 | **15.46** | 0.6180 | 35.76 | 28.0 |
| A: YOLO26n + Otsu/框 + contour-refiner | 0.7632 | 0.6842 | 16.30 | — | 15.95 | 62.7 |

* 两个 SAM 变体在**同一训练集**上适配：冻结图像编码器，仅微调 mask decoder（4.06M 参数，4 epoch），
  提示框与其它方案共用同一个 YOLO26n 检测器。
* **零样本直接用于超声会失效**：EfficientSAM Dice 0.0000（输出空 mask/无关大块）、EdgeSAM 0.6180；
  微调后分别达到 0.7682 / 0.7648。
* **速度差距主要来自 1024×1024 编码器**：SAM 系分割模块约 9–10ms，但端到端 36ms（EdgeSAM）与
  100ms（EfficientSAM），B 仅 10.3ms、方案 A 16.0ms。B 在精度与速度上同时占优；
  EfficientSAM 精度只差 0.005 却慢近 10 倍。
* 速度口径：全测试集 330 张 × 2 轮 = 660 次，含图像读取、YOLO26n 检测、预处理与分割解码，
  GPU 同步计时，预热 6 张，RTX 4060 Ti 16GB。

## 针对"框初始化"的数据增强消融（refiner 结构不变）

新增三类形状级增强（`src/cm/box_aug.py`）：C1 边独立抖动、C2 宽高缩放+平移、C3 内部内缩、C4 角点抖动。
五组对照（同结构/同数据/同超参）：

| 组别 | Dice ↑ | ΔDice vs 对照 | 95% CI | 结论 |
|---|---|---|---|---|
| 对照：无框增强 | **0.7985** | — | — | 基线 |
| C1+C2（边抖动+缩放平移） | 0.7957 | −0.003 | [−0.011, +0.005] | 无显著差异 |
| box_init_prob=0.5 + 全部增强 | 0.7933 | −0.005 | [−0.013, +0.002] | 无显著差异 |
| 全部形状增强 C1+C2+C3+C4 | 0.7898 | −0.008 | [−0.020, +0.003] | 无提升 |
| C3+C4（内缩+角点抖动） | 0.7869 | −0.011 | [−0.024, +0.000] | 显著变差 |

**结论：框增强没有带来提升，激进的两种反而显著变差。** 推理时的初始轮廓就是检测器的真实框
（接近矩形、中等定位误差），而 C3/C4 制造的是推理时不会出现的形状（内缩轮廓、非矩形四边形），
反而稀释了有效样本。refiner 对框误差的鲁棒性已由 `bbox_jitter` + A3/A4 + A5 提供。
对照自身重复性 ±0.004 Dice，因此 <0.01 的"提升"都不可信。**最终保持 `box_aug=False`。**

## 五方案对比（同一数据集 / 同一划分 / 同一检测器，测试集 200 张）

| 方案 | 参数量 | Dice ↑ | IoU ↑ | HD95 ↓ | ASSD ↓ | Dice<0.5 ↓ | 端到端 ms ↓ | FPS ↑ |
|---|---|---|---|---|---|---|---|---|
| **B: YOLO26n-seg** | 3.13M | **0.7735** | **0.7040** | 15.93 | 5.10 | 13.4% | **9.72** | **102.8** |
| YOLO26n + LiteMedSAM（零样本） | 9.79M | 0.7715 | 0.7007 | 15.07 | **5.00** | 13.4% | 44.76 | 22.3 |
| YOLO26n + EdgeSAM（微调 decoder） | 9.58M | 0.7648 | 0.6899 | 15.46 | 5.54 | 13.4% | 34.52 | 29.0 |
| A-box（框 + contour-refiner） | **0.47M** | 0.7632 | 0.6842 | 16.30 | 6.12 | **12.9%** | 15.96 | 62.7 |
| YOLO26n + Swin-LiteMedSAM ⚠️ | 36.77M | 0.6000 | 0.5152 | 20.39 | 8.48 | 27.2% | 45.92 | 21.8 |

检测（框级 IoU≥0.5，所有方案共用）：GT 203 / 预测 219，P=0.8493 / R=0.9163 / **F1=0.8815**。

* **LiteMedSAM 是 SAM 系里精度最高的**（零样本 0.7715，ASSD/面积误差全场最优），但端到端 44.76ms，
  比 B 慢 4.6 倍。
* **⚠️ Swin-LiteMedSAM 的权重仅在 Google Drive，本机不可达且无公开镜像**：本项目只能用
  LiteMedSAM 迁移 18/254 个编码器键 + 在本数据训练 decoder（val Dice 0.844），
  因此 0.6000 是"半个随机编码器"的结果，**不代表该架构真实能力**。详见 `reports/REPORT.md` §4.8。
* 分割模块本身耗时都在 7–9ms，差异主要来自编码器：TinyViT@256 ≈ 9ms、RepViT@1024 ≈ 8ms、
  Swin@256 ≈ 9ms，但端到端还要叠加检测与每框一次的 prompt/decoder 调用。

## 输入尺寸对比：512×512 vs 256×256（同一检测器，测试集 120 张）

**检测（三方案共用 YOLO26n）**

| 输入尺寸 | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 512×512 | 108 | 25 | 12 | 0.8120 | **0.9000** | **0.8538** |
| 256×256 | 90 | 17 | 30 | **0.8411** | 0.7500 | 0.7930 |

**分割（原图分辨率下评估）与速度（端到端 ms/张）**

| 输入尺寸 | 方案 | Dice ↑ | IoU ↑ | HD95 ↓ | Dice<0.5 ↓ | ms/张 ↓ | FPS ↑ |
|---|---|---|---|---|---|---|---|
| 512 | A-box (框+refiner) | 0.7781 | 0.7008 | 18.54 | **11.7%** | 19.36 | 51.7 |
| 512 | B (YOLO26n-seg) | **0.7798** | **0.7129** | 18.51 | 13.3% | **11.61** | **86.1** |
| 512 | YOLO26n+EdgeSAM | 0.7710 | 0.6988 | **16.77** | 13.3% | 48.77 | 20.5 |
| 256 | A-box | 0.5761 | 0.5165 | 22.61 | 34.2% | 15.25 | 65.6 |
| 256 | B (YOLO26n-seg) | **0.6278** | **0.5672** | 21.23 | 29.2% | **10.15** | **98.5** |
| 256 | YOLO26n+EdgeSAM | 0.5785 | 0.5204 | 22.25 | 34.2% | 41.22 | 24.3 |

* **512 是优选工作点**：三种方案 Dice 都在 0.771–0.780；降到 256 后掉 0.15–0.20，失败率翻倍以上。
* **退化主因是检测**：256 下召回 0.90→0.75（漏检 +18 个目标），误差从检测框传递到分割。
* **速度收益不划算**：降到 256 只快 1.5–7.6ms，却损失 0.15–0.20 Dice。EdgeSAM 因固定 1024×1024
  编码器，输入尺寸几乎不影响其开销；A-box 只在框内取 64 张 32×32 ROI，对输入尺寸也不敏感。

## 初始化策略：用框还是用 Otsu？（同一 refiner、同一测试子集）

| 初始化策略 | Dice ↑ | IoU ↑ | HD95 ↓ | Dice<0.5 ↓ |
|---|---|---|---|---|
| **框矩形（当前正式配置）** | **0.7632** | 0.6842 | **16.30** | **12.9%** |
| Otsu / 框 边界证据择优 | 0.6919 | 0.6166 | 17.69 | 20.4% |
| Otsu 粗糙轮廓 | 0.6576 | 0.5837 | 18.75 | 23.9% |
| Otsu + 框回退 | 0.6576 | 0.5837 | 18.75 | 23.9% |

**原因（177 个检测命中目标）**：Otsu 初始轮廓的**真值覆盖率仅 0.718**（p10 仅 0.301，常只圈住病灶一部分），
而框矩形覆盖率 **0.967**（p10 0.912）。refiner 只能局部修正，无法补回初始轮廓完全没覆盖的区域 →
Otsu 的缺陷会一路传递（精细化后 0.8159 vs 框 0.8737）。框的收缩系数（0 / 0.05 / 0.1 / 0.2）几乎不影响结果
（0.8742 / 0.8752 / 0.8734 / 0.8700），**只要覆盖完整即可**。

**最终推荐配置**：`YOLO26n 检测框 → 框矩形初始轮廓(64 点) → contour-refiner`。

## contour-refiner 设计（严格按给定规格）

* 每个目标取 `P = 64` 个粗糙轮廓点（等弧长、逆时针、首尾相接）。
* 以每个粗糙点为中心，取边长为 `window = window_ratio × 粗糙轮廓外接框对角线` 的正方形 ROI；
  `64` 个 ROI 统一 resize 到 `32×32` 灰度，逐图 **Z-score** 归一化，拼成 `64×32×32` 输入矩阵。
* 输出 `64` 个精准轮廓点在各自 ROI 内的归一化坐标（`0~32` → `-1~1`）。
* 结构：共享 CNN backbone（逐 ROI 提特征）→ 点间 Transformer（闭合轮廓的全局上下文）→ 逐点回归头。
* 损失：Smooth-L1（坐标）+ 可选闭合轮廓 Laplacian 平滑项。

### 数据对生成

```
GT mask ──► 精准轮廓 (64 等距点, 逆时针)
YOLO 框 (+抖动) ──► ROI 内 Otsu ──► 最大轮廓 ──► 粗糙轮廓 (64 点)
粗糙点 ──弧长对应──► 精准点
64×(粗糙点局部 ROI) ──► 64×32×32 ──► contour-refiner ──► 64 个精准点
```

### 数据增强模式（逐级叠加，`--level` 控制）

| 级别 | 内容 |
|---|---|
| A1 | 边缘点坐标加 Gaussian 噪声 |
| A2 | 由 ROI 内 Otsu 阈值求最近边缘点 + 微量 Gaussian 噪声（A1 的 0~1/3） |
| A3 | A2 + 随机（0~1/3）ROI 范围变化、图像噪声、亮暗、对比度变化 |
| A4 | A3 + 随机（0~1/3）任意角度旋转 |
| A5 | A4 + 点序偏移（0~100%），平移 64 个 ROI 与轮廓点的序号起点，保持对应关系与逆时针顺序 |

另加 **检测框抖动**（`bbox_jitter`），模拟 YOLO26n 推理时的定位误差，保证训练输入分布与推理一致。

## 数据集

| key | 数据集 | 模态 | 目标 |
|---|---|---|---|
| `busi` | BUSI (Breast Ultrasound Images) | 乳腺 | 乳腺肿瘤 |
| `tn3k` | TN3K (Thyroid Nodule 3K) | 甲状腺 | 甲状腺结节 |
| `ddti` | DDTI (Thyroid Digital Database) | 甲状腺 | 甲状腺结节 |
| `tg3k` | TG3K | 甲状腺 | 甲状腺腺体（扩展） |

来源：`us-segmentator/us-segmentation-dataset`（镜像 `hf-mirror.com`）中的 `zips/Breast`、`zips/Thyroid`。

## 目录结构

```
src/cm/           核心库 (config / rough / geometry / roi / model / metrics)
scripts/          00 自检 -> 01 数据准备 -> 02 划分 -> 03 训练 refiner
                  04 导出 YOLO -> 05 训练 YOLO26n(-seg) -> 07 统一评估
data/raw          原始压缩包
data/interim      解压结果
data/unified      统一后的 images/ + masks/ + manifest.csv
data/splits       全实验共用的 train/val/test 划分
data/yolo         导出的 YOLO 数据集
runs/             训练输出 (权重、曲线)
reports/          评估结果、对比表、可视化
```

## 复现流程

```bash
# 0) 环境（已就绪）：项目内 conda 环境 ./.conda (Python 3.11 + torch cu128)
PY=./.conda/bin/python

# 1) 数据准备与划分
PYTHONPATH=src $PY scripts/01_prepare_data.py
PYTHONPATH=src $PY scripts/02_make_splits.py

# 2) 方案 A：训练 contour-refiner
PYTHONPATH=src $PY scripts/03_train_refiner.py --level 5 --tag A5

# 3) 方案 B：导出 YOLO 数据集并训练
PYTHONPATH=src $PY scripts/04_export_yolo.py
PYTHONPATH=src $PY scripts/05_train_yolo.py --kind det --model yolo26n.pt
PYTHONPATH=src $PY scripts/05_train_yolo.py --kind seg --model yolo26n-seg.pt

# 3.5) EdgeSAM / EfficientSAM 对照：微调 mask decoder + 精度/速度评测
PYTHONPATH=src $PY scripts/19_finetune_sam.py --variant edgesam --epochs 4 --lr 5e-4
PYTHONPATH=src $PY scripts/19_finetune_sam.py --variant efficientvit-t --epochs 4 --lr 5e-4
PYTHONPATH=src $PY scripts/18_bench_sam.py --limit 200 --sam edgesam efficientvit-t \
    --sam-ckpt weights/sam_finetuned --out reports/bench_final_ft
PYTHONPATH=src $PY scripts/20_sam_compare.py

# 4) 统一评估（同一 test split、同一指标）
PYTHONPATH=src $PY scripts/07_evaluate.py \
    --refiner runs/refiner_A5/best.pt \
    --det runs/det_yolo26n/weights/best.pt \
    --seg runs/seg_yolo26n/weights/best.pt \
    --out reports/eval_A5
```

## 评估指标

Dice、IoU、HD95、ASSD、边界 F1、面积相对误差；检测层面统计漏检/误检；
四个对照：`rough_otsu`（只有 Otsu）、`refiner`（完整方案 A）、`refiner_gtbox`（用 GT 框，隔离检测误差）、`yolo26n_seg`（方案 B）。
