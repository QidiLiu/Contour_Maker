# Contour_Maker

超声图像中**低回声目标**（乳腺肿瘤 / 甲状腺结节 / 血管横截面）的轮廓识别与分割对比实验。
在**同一数据划分、同一检测器、同一评估脚本**下对比 6 条技术流水线，同时比较精度与推理速度。

> **文档地图**
> * 本文件 —— 项目总览、结论速览、推荐配置、交付物清单、完整复现流程、评估协议
> * [`reports/REPORT.md`](reports/REPORT.md) —— 详细实验报告（全部对照、消融、边界案例、局限）
> * [`EXPERIMENT.md`](EXPERIMENT.md) —— 实验设计与工程决策（算法规格映射、全部工程发现）

---

## 1. 结论速览

### 1.1 五条流水线总表（同一数据集/划分/检测器，测试集 200 张，原图分辨率下评估）

| 流水线 | 参数量 | Dice ↑ | IoU ↑ | HD95 ↓ | ASSD ↓ | 边界F1 ↑ | Dice<0.5 ↓ | 端到端 ms ↓ | FPS ↑ |
|---|---|---|---|---|---|---|---|---|---|
| **B: YOLO26n-seg** | 3.13M | **0.7735** | **0.7040** | 15.93 | 5.10 | **0.3850** | 13.4% | **9.72** | **102.8** |
| YOLO26n + LiteMedSAM（官方权重，零样本） | 9.79M | 0.7715 | 0.7007 | **15.07** | **5.00** | 0.3799 | 13.4% | 44.76 | 22.3 |
| YOLO26n + EdgeSAM（微调 decoder） | 9.58M | 0.7648 | 0.6899 | 15.46 | 5.54 | 0.3445 | 13.4% | 34.52 | 29.0 |
| **A-box: 检测框 → 框矩形初始轮廓 → contour-refiner** | **0.47M** | 0.7632 | 0.6842 | 16.30 | 6.12 | 0.3132 | **12.9%** | 15.96 | 62.7 |
| YOLO26n + EfficientSAM (ViT-T，微调 decoder) | 10.2M | 0.7682 | 0.6947 | 16.83 | 5.37 | 0.3648 | 13.4% | 99.71 | 10.0 |
| YOLO26n + Swin-LiteMedSAM ⚠️ | 36.77M | 0.6000 | 0.5152 | 20.39 | 8.48 | 0.1796 | 27.2% | 45.92 | 21.8 |

检测（框级 IoU≥0.5，所有流水线共用同一 YOLO26n）：GT 203 / 预测 219，TP 186 / FP 33 / FN 17
→ **Precision 0.8493 / Recall 0.9163 / F1 0.8815**。

⚠️ Swin-LiteMedSAM 官方权重仅托管于 Google Drive，本机不可达且无公开镜像；表内为其架构在本数据上的
**自训版本**（Swin 编码器冻结 + decoder 自训），不代表该架构真实能力，详见
[`reports/REPORT.md`](reports/REPORT.md) §4.8。

### 1.2 核心发现（按重要性）

1. **contour-refiner 的价值在于"修烂轮廓"，不在于"修好轮廓"**：同一检测框下把 Otsu 粗糙轮廓
   （覆盖率仅 0.718）提升 **0.5410 → 0.6602 Dice**；而把初始轮廓换成**检测框矩形**（覆盖率 0.967）
   后达到 **0.8028**，反超 B 的 0.7923。
2. **瓶颈是初始化质量而非精细化能力**：Otsu 常只圈住病灶的一部分，refiner 只能局部微调、
   补不回未覆盖区域。Otsu/框 oracle 二选一的上限只比"永远用框"高 0.3 个百分点（99.66% vs 99.33%）。
3. **把 refiner 接在 YOLO26n-seg 之后没有收益**（0.7923 → 0.7813，58% 目标变差）：
   该 refiner 按"粗糙轮廓分布"校准，对已准确的 seg mask 属于分布外输入，会过修正
   （把 GT 轮廓喂给它，Dice 0.9954 → 0.9127）。
4. **SAM 系模型必须做域适配**：零样本 EfficientSAM 完全失效（Dice 0.0000）、EdgeSAM 0.6180；
   适配后 0.7682 / 0.7648。LiteMedSAM 零样本即 0.7715，是 SAM 系最优，但端到端慢 4.6 倍。
5. **数据增强没有收益**：A1–A5 逐级增强、框形状增强 C1–C4、box_init_prob 加倍，
   测试集差异均 <0.02 且 95% CI 跨 0；本实验 run-to-run 噪声 **±0.004 Dice**，更小的差异不可信。
6. **输入 512 明显优于 256**：256 下检测召回 0.90 → 0.75、Dice 掉 0.15–0.20，速度只快 1.5–7.6ms。

### 1.3 推荐配置

| 需求 | 推荐 | 理由 |
|---|---|---|
| 精度与速度综合最优 | **B: YOLO26n-seg** | Dice 0.7735 / 9.72ms / 103 FPS，两项均领先 |
| 极致轻量（边缘设备 / 低算力） | **A-box** | 仅 0.47M 参数（B 的 15%），Dice 只低 0.010，16ms |
| 边界最贴合（ASSD 优先） | **LiteMedSAM** | ASSD 5.00、面积误差 0.134 全场最优，但慢 4.6 倍 |

方案 A-box 的完整定义：`YOLO26n 检测框 → 框矩形初始轮廓（64 点，等弧长，逆时针）
→ contour-refiner（64×32×32 ROI → 64 点回归）`。

---

## 2. contour-refiner 设计（严格按给定规格实现，结构未改动）

* 每个目标取 `P = 64` 个粗糙轮廓点（等弧长、逆时针、首尾相接）。
* 以每个粗糙点为中心，取边长为 `window = window_ratio × 粗糙轮廓外接框对角线` 的正方形 ROI
  （默认 `window_ratio = 0.35`）；64 个 ROI 统一 resize 到 `32×32` 灰度，逐 ROI **Z-score** 归一化，
  拼成 `64×32×32` 输入矩阵。
* 输出 64 个精准轮廓点在各自 ROI 内的归一化坐标（`0~32` → `-1~1`）。
* 结构：共享权重 CNN backbone（depthwise-separable，逐 ROI 提特征）→ 点间 Transformer
  （2 层 4 头，闭合轮廓全局上下文）→ 逐点回归头；`feat_dim=128`，0.47M 参数。
* 损失：Smooth-L1（坐标）+ 可选闭合轮廓 Laplacian 平滑项；AdamW + Cosine 退火 + AMP。

```
GT mask   ──► 精准轮廓 (64 等距点, 逆时针)
YOLO 框(+抖动) ──► 框矩形 或 ROI 内 Otsu ──► 粗糙轮廓 (64 点)
粗糙点 ──弧长对应──► 精准点
64×(粗糙点局部 ROI) ──► 64×32×32 ──► contour-refiner ──► 64 个精准点
```

### 数据增强（逐级叠加，`--level` 控制）

| 级别 | 内容 | 针对分支 |
|---|---|---|
| A1 | 边缘点坐标加 Gaussian 噪声 | Otsu |
| A2 | ROI 内 Otsu 最近边缘点 + 微量 Gaussian 噪声（A1 的 0~1/3） | Otsu |
| A3 | A2 + 随机（0~1/3）ROI 范围变化、图像噪声、亮暗、对比度变化 | 全局 |
| A4 | A3 + 随机（0~1/3）任意角度旋转 | 全局 |
| A5 | A4 + 点序偏移（0~100%），保持对应关系与逆时针顺序 | 全局 |
| `bbox_jitter` | 检测框中心/尺度抖动，模拟 YOLO26n 定位误差 | 全局 |
| C1–C4 | 边独立抖动 / 宽高缩放平移 / 内部内缩 / 角点抖动（`box_aug`，**默认关闭**） | 框 |

---

## 3. 数据集与划分

| key | 数据集 | 模态 | 目标 | 有效图像数 |
|---|---|---|---|---|
| `busi` | BUSI Breast Ultrasound Images | 乳腺 | 乳腺肿瘤（低回声） | 647 |
| `tn3k` | TN3K Thyroid Nodule 3K | 甲状腺 | 甲状腺结节 | 3493 |
| `ddti` | DDTI Thyroid Digital Database | 甲状腺 | 甲状腺结节 | 637 |
| `tg3k` | TG3K（边界案例） | 甲状腺 | 甲状腺**腺体**（高回声） | 3585 |

来源：`us-segmentator/us-segmentation-dataset`（镜像 `hf-mirror.com`），原始压缩包在 `data/raw/`。

* **共享划分**：`data/splits/splits.csv`，按数据集分层 70/15/15（seed=42），全部方案共用。
* **YOLO 平衡子集**：`busi 600 / tn3k 1000 / ddti 600`（train 1540 / val 330 / test 330），
  用于检测/分割/SAM 系训练与评测。
* **contour-refiner**：用三个数据集的全部 train/val（4338 个目标）训练。

---

## 4. 目录结构

```
src/cm/                 核心库
  config.py             全局配置 (路径/划分/RefinerConfig/AugConfig)
  rough.py              Otsu 自适应分割 + 框矩形回退
  box_aug.py            框形状增强 C1-C4 (默认关闭)
  geometry.py           重采样/点对应/方向统一/mask<->polygon
  roi.py                ROI 采样 + A1-A5 增强 + 样本构建
  model.py              contour-refiner (CNN backbone + Transformer + 回归头)
  metrics.py            Dice/IoU/HD95/ASSD/边界F1/面积误差
  sam_models.py         EdgeSAM / EfficientSAM 适配层
  litemedsam.py         LiteMedSAM / Swin-LiteMedSAM 适配层
scripts/                00-26 全流程脚本 (数据 -> 缓存 -> 训练 -> 评测 -> 报告)
data/raw                原始压缩包        data/interim   解压结果
data/unified            统一 images/masks + manifest.csv
data/splits             共享 train/val/test 划分
data/yolo, yolo_subset  导出的 YOLO 数据集 / 平衡子集
runs/                   训练输出 (权重、history.json、config.json)
reports/                评估结果、对比表、可视化、REPORT.md
weights/                预训练与自训权重 (yolo26n*, edge_sam, lite_medsam, sam_finetuned/)
third_party/            EdgeSAM / EfficientSAM / LiteMedSAM / Swin_LiteMedSAM 官方代码
```

---

## 5. 交付物清单

| 内容 | 路径 |
|---|---|
| 详细报告 | `reports/REPORT.md` |
| 五方案对比表 / 图 | `reports/final/all_five_methods.csv`、`reports/final/fig_acc_speed_five.png` |
| 逐图逐目标明细 | `reports/bench_litemedsam/per_image.csv`（五方案）、`reports/v3_eval_A5/per_image.csv`（A/B 全测试集） |
| 速度明细 | `reports/bench_speed_full/speed.csv`（660 次计时） |
| 输入尺寸对比 | `reports/input_size/accuracy_speed.csv`、`fig_input_size.png` |
| 消融（增强/初始化） | `reports/boxaug_summary.csv`、`reports/boxaug_stats.csv`、`reports/init_strategy_analysis.csv` |
| 检测器/分割器权重 | `runs/det_yolo26n_sub/weights/best.pt`、`runs/seg_yolo26n_sub/weights/best.pt` |
| contour-refiner 权重 | `runs/refiner_v3_A5_busi-ddti-tn3k/best.pt`（正式配置）等 18 个 run（含 A0–A5 增强消融、C1–C4 框增强消融、seed 复现） |
| SAM 自训权重 | `weights/sam_finetuned/{edgesam,efficientvit-t}_decoder.pt`、`swin_litemedsam_head.pt` |
| 可视化 | `reports/final_A_vs_B/vis_*.png`、`reports/v3_eval_A5/vis_*.png`（绿=GT，橙=粗糙，蓝=refiner，品红=seg/SAM） |

---

## 6. 完整复现流程

环境：项目内 conda 环境 `./.conda`（Python 3.11 + torch 2.11.0+cu128，RTX 4060 Ti 16GB）。

```bash
PY=./.conda/bin/python
export PYTHONPATH=src
export YOLO_CONFIG_DIR=$PWD/.cache/ultralytics

# ---- 1) 数据准备 ----
$PY scripts/01_prepare_data.py                       # 解压 + image/mask 配对 + 统一
$PY scripts/02_make_splits.py                        # 共享 70/15/15 划分 (seed=42)
$PY scripts/04_export_yolo.py                        # 导出 YOLO det/seg 数据集
$PY scripts/04b_make_subset.py                       # 平衡子集 (2200 张)

# ---- 2) 训练 YOLO26n 检测器 (方案 A/B/SAM 系共用) ----
$PY scripts/05_train_yolo.py --kind det --yolo-dir data/yolo_subset \
    --name det_yolo26n_sub --epochs 60 --imgsz 512 --batch 16
$PY scripts/05_train_yolo.py --kind seg --yolo-dir data/yolo_subset \
    --name seg_yolo26n_sub --epochs 60 --imgsz 512 --batch 12

# ---- 3) 方案 A: contour-refiner ----
$PY scripts/03b_cache_samples.py --level 5 --variants 3 --datasets busi tn3k ddti \
    --out-subdir v5_mix_L5_v3                        # 生成 (粗糙, 精准) 样本缓存
$PY scripts/03c_verify_cache.py                      # 缓存完整性校验
$PY scripts/03_train_refiner.py --level 5 --tag A5 --datasets busi tn3k ddti \
    --cache-dir data/cache/v5_mix_L5_v3 --epochs 25 --batch-size 64
# 增强消融 (A0-A4) 与框增强消融 (C1-C4):
$PY scripts/21_box_aug_experiments.py ; $PY scripts/22_box_aug_report.py
$PY scripts/23_box_aug_stats.py                      # 配对 bootstrap 显著性检验

# ---- 4) 统一评估 (A-box / B / SAM 系, 同一 test split) ----
$PY scripts/18_bench_sam.py --limit 200 --warmup 5 --repeats 2 \
    --sam edgesam efficientvit-t litemedsam swin_litemedsam \
    --sam-ckpt weights/sam_finetuned --out reports/bench_litemedsam
$PY scripts/24_input_size_bench.py --sizes 512 256 --limit 120   # 输入尺寸对比
$PY scripts/20_sam_compare.py ; $PY scripts/25_input_size_report.py

# ---- 5) 生成报告 ----
$PY scripts/14_final_report.py                       # -> reports/REPORT.md
```

**SAM 系适配（可选，用于复现 SAM 变体结果）**

```bash
$PY scripts/19_finetune_sam.py --variant edgesam --epochs 4 --lr 5e-4
$PY scripts/19_finetune_sam.py --variant efficientvit-t --epochs 4 --lr 5e-4
$PY scripts/26_train_swin_litemedsam.py --epochs 8 --stride 3   # Swin 变体 (官方权重不可得)
```

---

## 7. 评估协议

* **测试集**：`data/splits/splits.csv` 的 `test`，全部方案共用；SAM 系对比用前 200 张子集。
* **目标匹配**：mask IoU ≥ 0.5 的贪心匹配；未匹配的 GT 计漏检、未匹配的预测计误检。
* **分割指标**：Dice、IoU、HD95、ASSD、边界 F1（2px 容差）、面积相对误差、Dice<0.5 失败率。
* **检测指标**：框级 IoU ≥ 0.5 的 Precision / Recall / F1（三方案共用同一检测器）。
* **速度口径**：每张图端到端 wall-clock（含图像预处理、检测、分割解码），GPU 同步计时，
  预热后统计；报告 mean / median / std / FPS。
* **显著性**：跨配置比较用**按图像配对的 bootstrap**（2000 次重采样），给出 ΔDice 的 95% CI。
  本实验 run-to-run 噪声约 ±0.004 Dice。
