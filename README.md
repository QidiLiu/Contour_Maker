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
