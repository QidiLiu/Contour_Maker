# 实验设计说明 (EXPERIMENT.md)

本文档说明对比实验的设计、对用户给定算法规格的实现映射、以及为使规格可执行而做出的
工程假设。所有假设都在代码中显式参数化，可用命令行开关复现或推翻。

---

## 1. 对比的两条技术路线

| 记号 | 方案 | 输入 | 输出 |
|---|---|---|---|
| **A（本方案）** | YOLO26n 检测框 → ROI 内 Otsu 自适应分割 → 最大轮廓（粗糙轮廓）→ contour-refiner 逐点精细化 | 超声灰度图 | 64 点多边形 → mask |
| **B（baseline）** | YOLO26n-seg 端到端实例分割 | 超声灰度图 | 实例 mask |

为**分离"检测误差"与"轮廓精细化误差"**，方案 A 额外评估两种框来源：
`GT 框`（理想检测）与 `YOLO26n 预测框`（真实检测）。检测框带来的整体影响可以由此分解。

## 2. 算法规格 → 实现映射

| 用户规格 | 实现 |
|---|---|
| 目标框区域计算 Otsu 自适应分割 | `src/cm/rough.py::otsu_adaptive_rough_contour`：框内高斯去噪 → Otsu 阈值 → 形态学开闭 → 取最大连通域 → 外轮廓（默认**不外扩框**，见 §7） |
| 找最大轮廓作为初始粗糙轮廓 | 同上，`cv2.findContours` + `max(contourArea)` |
| 实际输入是 64 张 ROI 图，每张属于 1 个粗糙轮廓点 | `src/cm/roi.py::build_sample`，`RefinerConfig.n_points = 64` |
| 64 个基本等距粗轮廓点、依次相邻、逆时针 | `geometry.resample_closed`（等弧长）+ `ensure_ccw` |
| 以每个轮廓点为中心、以轮廓尺度为边长取正方形 ROI | 窗口边长 `window = window_ratio × 粗糙轮廓外接框对角线`，默认 `window_ratio=0.35`（见 §4 假设 H1） |
| 统一 resize 为 32×32 灰度、Z-score 归一化、拼成 64×32×32 | `extract_patches(..., size=32)`，逐 ROI 独立 Z-score（`rcfg.zscore=True`） |
| 输出 64 个精准坐标（0~32 → -1~1） | 回归头 `tanh` 输出，监督信号为精准点在各自 ROI 内的归一化坐标 |
| head + backbone（经典且合适） | 共享权重 CNN backbone（depthwise-separable，32×32→2×2→GAP）→ 点间 Transformer Encoder（2 层，4 头）→ 逐点 MLP 回归头，约 1.4M 参数（`feat_dim=128`） |
| 由当前图像、粗糙轮廓点、精准轮廓点得到输入输出数据对 | 精准轮廓来自 GT mask；粗糙轮廓来自"检测框 + Otsu"流水线（含框抖动），与推理分布一致 |
| 求 loss + back-prop | Smooth-L1（坐标回归）+ 可选闭合轮廓 Laplacian 平滑项，AdamW + Cosine 退火 + AMP |

### 数据增强模式实现（`src/cm/roi.py`）

| 级别 | 实现要点 |
|---|---|
| A1 | `gaussian_noise_contour`：粗糙点坐标加 Gaussian 噪声，`σ = a1_sigma × window`（默认 `a1_sigma=0.10`） |
| A2 | `otsu_snap_contour`：以粗糙点为中心的小 ROI 内 Otsu → Canny 边缘 → 最近的边缘点；再叠加 `[0, 1/3]·σ_A1` 的微量 Gaussian 噪声。训练时三种模式混合（纯 A2 / A2+少量 A1 / 纯 A1），保证两种分布都见过 |
| A3 | `augment_image`：ROI 范围变化（窗口自适应 + 框抖动）、图像 Gaussian 噪声（`≤12` 灰阶）、亮暗（`β ≤ ±30`）、对比度（`α ≤ ±0.3`） |
| A4 | 在 A3 基础上对整图做任意角度旋转（`±180°×strength`），旋转后重新计算框与粗糙轮廓、重算弧长对应 |
| A5 | 在 A4 基础上对 64 个 ROI 与轮廓点整体滚动 `0~100%` 起始序号（`np.roll`），**保持前后顺序、逆时针方向与 ROI-点对应关系不变** |
| 额外 | `bbox_jitter`：检测框中心/尺度随机抖动，模拟 YOLO26n 定位误差，使训练输入分布与推理一致 |

## 3. 数据集

| key | 数据集 | 模态 | 目标类型 | 图像数(有效) |
|---|---|---|---|---|
| `busi` | BUSI Breast Ultrasound Images | 乳腺 | 乳腺肿瘤（低回声） | 647 |
| `tn3k` | TN3K Thyroid Nodule 3K | 甲状腺 | 甲状腺结节 | 3493 |
| `ddti` | DDTI Thyroid Digital Database | 甲状腺 | 甲状腺结节 | 637 |
| `tg3k` | TG3K | 甲状腺 | 甲状腺腺体（扩展实验） | 3585 |

来源：`us-segmentator/us-segmentation-dataset`（`hf-mirror.com` 镜像），原始压缩包存于 `data/raw/`。

**说明**：TG3K 的目标是**甲状腺腺体**，其回声特性与结节/肿瘤不同——腺体常为高回声、
周围为低回声，导致"框内 Otsu + 最大暗区"的粗糙轮廓策略在该数据集上失效
（实测粗糙 Dice ≈ 0.05，见 `reports/contour_check/`）。因此主对比使用
**BUSI + TN3K + DDTI** 三个"低回声目标"数据集；TG3K 作为反例在报告中说明方案 A 的适用边界。

### 数据划分
`scripts/02_make_splits.py` 按数据集分层随机划分 70/15/15（seed=42），
**方案 A 与方案 B 共用同一份 `data/splits/splits.csv`**，保证测试集完全一致。

### YOLO 训练子集
在 `busi/tn3k/ddti` 上取与划分一致的平衡子集（bus i 600 / tn3k 1000 / ddti 600，
共 2200 张：train 1540 / val 330 / test 330），用于在可承受时间内完成完整对比实验。
contour-refiner 则使用这三个数据集的**全部** train/val 数据训练。

## 4. 实现假设（可配置、可验证）

- **H1 ROI 窗口尺度**："以轮廓尺度为边长" 取 `window = window_ratio × 粗糙轮廓外接框对角线`，
  默认 `0.35`；`--window-ratio` 可调，另有 `{0.20, 0.35, 0.50}` 的消融实验。
- **H2 点对应关系**：粗糙点 ↔ 精准点采用**弧长对应**（GT 轮廓按弧长重采样，起始点对齐到
  粗糙首点的最近点），保证一一对应、同序同向；`geometry.radial_correspondence` 提供径向对应对照。
- **H3 GT 轮廓起点**：取轮廓最高点（并列取最左）作为确定性起点，避免起点歧义。
  该歧义由 A5 的点序偏移增强进一步消除。
- **H4 训练时窗口与推理一致性**：训练时若某点的精准点落在窗口外，会自适应放大该点窗口
  （最多 `max_expand=4` 倍）。推理时只用固定窗口，因此模型对此类"粗糙点偏离较远"的
  情形天然具备一定鲁棒性；实测落在窗口外的精准点比例 <2%（`scripts/00_sanity_check.py`）。
- **H5 粗糙轮廓来源**：训练时用"抖动后的 GT 框 + Otsu"，而非完美 GT 框，以匹配推理时的
  YOLO 框误差水平（`bbox_jitter=0.10`，对应框 IoU 约 0.8~0.95）。

## 5. 评估协议

- 统一的测试集（`splits.csv` 的 `test`），统一的目标匹配规则：mask IoU ≥ 0.5 的贪心匹配。
- 指标：**Dice、IoU、HD95、ASSD、边界 F1（2px 容差）、面积相对误差**；
  另记录检测层面的漏检/误检数量、以及轮廓点到 GT 轮廓的平均像素距离（点级诊断）。
- 五路对照：
  1. `rough_otsu`：YOLO 框 + Otsu 最大轮廓（方案 A 去掉 refiner）
  2. `refiner`：**完整方案 A**
  3. `rough_otsu_gtbox` / 4. `refiner_gtbox`：用 GT 框，隔离检测误差
  5. `yolo26n_seg`：**方案 B**

## 7. 关键工程发现：粗糙轮廓提取中的"外扩陷阱"

最初实现按常规做法把检测框**外扩 15%** 再在 ROI 内做 Otsu，理由是"保证病灶边缘落在 ROI 内"。
但这在超声图像上适得其反：低回声病灶周围常有大片暗背景（DDTI 尤其明显），外扩后
**病灶与背景在阈值分割中连成一片**，"最大暗连通域"退化成背景区域，其轮廓贴着框边、
甚至贴着图像边界。

在 val 集上用 GT 框做的算法对比（`mean Dice`，越高越好）：

| 策略 | BUSI | DDTI | TN3K | 平均 |
|---|---|---|---|---|
| 外扩 15% + 最大暗连通域（初版） | 0.738 | 0.566 | 0.635 | 0.640 |
| **不外扩 + 最大暗连通域（最终采用）** | **0.833** | **0.669** | **0.728** | **0.734** |
| 不外扩 + 排除贴边连通域 | 0.008 | 0.039 | 0.052 | 0.046 |
| 外扩 5% + 中心偏好 | 0.793 | 0.641 | 0.700 | 0.705 |

用 YOLO26n **检测框**（框 IoU 均值 0.759）复测同样结论：`pad=0` 使粗糙轮廓平均 Dice
由 **0.607 → 0.678**。因此最终版本 `pad_ratio=0`；同时"排除贴边连通域"这种看似合理的
改进反而灾难性失败（病灶本身常常贴边/贴框），故不予采用。

该发现说明：在"检测框 + Otsu + 轮廓精细化"范式中，**粗糙轮廓的质量是整体精度的主要瓶颈**，
contour-refiner 只能在其基础上做局部修正。

## 8. 复现命令

见 `README.md`。关键脚本：

```
scripts/01_prepare_data.py      # 解压/配对/统一
scripts/02_make_splits.py       # 共享划分
scripts/03b_cache_samples.py    # 生成 (粗糙, 精准) 样本缓存 (加速训练)
scripts/03_train_refiner.py     # 训练 contour-refiner (--cache-dir 使用缓存)
scripts/04_export_yolo.py       # 导出 YOLO det/seg 数据集
scripts/04b_make_subset.py      # 平衡子集
scripts/05_train_yolo.py        # 训练 YOLO26n / YOLO26n-seg
scripts/07_evaluate.py          # 统一评估
scripts/09_make_report.py       # 生成报告与图表
```
