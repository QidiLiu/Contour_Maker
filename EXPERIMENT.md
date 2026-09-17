# 实验设计与工程决策 (EXPERIMENT.md)

本文档记录：① 对比实验的设计与对给定算法规格的实现映射；② 为使规格可执行而做出的工程假设；
③ 实验中**实测发现并修正**的全部工程问题（含失败方案，便于复现与避坑）。

> 结果与数字见 [`reports/REPORT.md`](reports/REPORT.md)；总览与复现流程见 [`README.md`](README.md)。
> 所有假设与增强都在代码中显式参数化，可用命令行开关复现或推翻。

---

## 1. 对比的流水线

| 记号 | 流水线 | 分割模块 | 训练条件 |
|---|---|---|---|
| **A-box** | YOLO26n 框 → 框矩形初始轮廓 → **contour-refiner** | 自研 0.47M | refiner 在三数据集全部 train/val 上训练 |
| **A-otsu** | YOLO26n 框 → 框内 Otsu 粗糙轮廓 → contour-refiner | 同上 | 同上（对照，已弃用） |
| **B** | YOLO26n-seg 端到端实例分割 | YOLO26n-seg 3.13M | 平衡子集 1540 张，60 epoch |
| **B+refiner** | B 的 mask 当粗轮廓 → contour-refiner | 复用 A 的 refiner | 无额外训练 |
| **+EdgeSAM** | YOLO26n 框提示 EdgeSAM | 9.58M | 冻结编码器，微调 mask decoder（同训练集） |
| **+EfficientSAM** | YOLO26n 框提示 EfficientSAM(ViT-T) | 10.2M | 同上 |
| **+LiteMedSAM** | YOLO26n 框提示 LiteMedSAM(TinyViT) | 9.79M | **官方权重零样本** |
| **+Swin-LiteMedSAM** | YOLO26n 框提示 Swin-LiteMedSAM | 36.77M | ⚠️ 官方权重不可得，本项目自训 head（见 §7.5） |

为**分离"检测误差"与"轮廓精细化误差"**，方案 A 额外评估两种框来源：`GT 框`（理想检测）
与 `YOLO26n 预测框`（真实检测）。

---

## 2. 算法规格 → 实现映射

| 给定规格 | 实现 |
|---|---|
| 目标框区域计算 Otsu 自适应分割 | `src/cm/rough.py::otsu_adaptive_rough_contour`：框内高斯去噪 → Otsu → 形态学开闭 → 最大连通域 → 外轮廓（**不外扩框**，见 §7.1；支持 `polarity=dark/bright/auto`） |
| 找最大轮廓作为初始粗糙轮廓 | 同上，`cv2.findContours` + `max(contourArea)`；另有 `box_rough_contour` 提供框矩形初始化 |
| **实际输入是 64 张 ROI 图，每张属于 1 个粗糙轮廓点** | `src/cm/roi.py::build_sample`，`RefinerConfig.n_points = 64` |
| **64 个基本等距粗轮廓点、依次相邻、逆时针** | `geometry.resample_closed`（等弧长，`endpoint=False`）+ `ensure_ccw` |
| 以每个轮廓点为中心、以轮廓尺度为边长取正方形 ROI | `window = window_ratio × 粗糙轮廓外接框对角线`，默认 `0.35`（假设 H1） |
| 统一 resize 为 32×32 灰度、Z-score 归一化、拼成 64×32×32 | `extract_patches(..., size=32)`，逐 ROI 独立 Z-score（`rcfg.zscore=True`） |
| 输出 64 个精准坐标（0~32 → -1~1） | 回归头 `tanh`，监督 = 精准点在各自 ROI 内的归一化坐标 |
| head + backbone（经典且合适） | 共享权重 CNN backbone（depthwise-separable 32×32→2×2→GAP）→ 点间 Transformer Encoder（2 层 4 头）→ 逐点 MLP 回归头 |
| 由当前图像、粗糙轮廓点、精准轮廓点得到输入输出数据对 | 精准轮廓来自 GT mask；粗糙轮廓来自"检测框 + 框矩形/Otsu"流水线（含框抖动） |
| 求 loss + back-prop | Smooth-L1（坐标）+ 可选闭合轮廓 Laplacian 平滑项；AdamW + Cosine 退火 + AMP |

### 2.1 数据增强实现（`src/cm/roi.py`、`src/cm/box_aug.py`）

| 级别 | 实现要点 | 作用分支 |
|---|---|---|
| A1 | `gaussian_noise_contour`：粗糙点坐标加 Gaussian 噪声，`σ = a1_sigma × window`（默认 0.10） | Otsu |
| A2 | `otsu_snap_contour`：小 ROI 内 Otsu → Canny → 最近边缘点；再叠加 `[0,1/3]·σ_A1` 微量噪声；训练时三种模式混合（纯 A2 / A2+少量 A1 / 纯 A1） | Otsu |
| A3 | `augment_image`：窗口自适应、图像 Gaussian 噪声（≤12 灰阶）、亮暗（β≤±30）、对比度（α≤±0.3） | 全局 |
| A4 | A3 + 整图任意角度旋转（±180°×strength），旋转后重算框、粗糙轮廓与弧长对应 | 全局 |
| A5 | A4 + 64 个 ROI 与轮廓点整体滚动 `0~100%` 起始序号（`np.roll`），保持顺序/逆时针/对应关系 | 全局 |
| 额外 | `bbox_jitter=0.10`：框中心/尺度抖动，模拟 YOLO26n 定位误差 | 全局 |
| C1 | 四条边独立随机平移 → 模拟"只框住一部分" | 框 |
| C2 | 宽高独立缩放 + 整体平移 | 框 |
| C3 | 轮廓在框内随机内缩（不贴边） | 框 |
| C4 | 四角独立抖动 → 任意四边形 | 框 |

C1–C4 由 `AugConfig.box_aug` 控制，**默认关闭**（实测无收益，见 §7.4）。

---

## 3. 数据集与划分

| key | 数据集 | 模态 | 目标类型 | 有效图像数 |
|---|---|---|---|---|
| `busi` | BUSI Breast Ultrasound Images | 乳腺 | 乳腺肿瘤（低回声） | 647 |
| `tn3k` | TN3K Thyroid Nodule 3K | 甲状腺 | 甲状腺结节 | 3493 |
| `ddti` | DDTI Thyroid Digital Database | 甲状腺 | 甲状腺结节 | 637 |
| `tg3k` | TG3K | 甲状腺 | 甲状腺**腺体**（高回声，边界案例） | 3585 |

来源：`us-segmentator/us-segmentation-dataset`（`hf-mirror.com` 镜像），原始压缩包在 `data/raw/`。
配对逻辑支持 5 种目录布局（同目录 `_mask`、平行 `img/`+`label/` 目录、双扩展名如 `test1.PNG.png` 等）。

* **共享划分**：`scripts/02_make_splits.py` 按数据集分层 70/15/15（seed=42），
  全部方案共用 `data/splits/splits.csv`，保证测试集完全一致。
* **YOLO 平衡子集**：`scripts/04b_make_subset.py` 取 busi 600 / tn3k 1000 / ddti 600
  （train 1540 / val 330 / test 330），用于在可承受时间内完成完整对比。
* **contour-refiner**：使用三数据集**全部** train/val（4338 个目标，每目标 3~4 组样本）。

---

## 4. 实现假设（可配置、可验证）

* **H1 ROI 窗口尺度**："以轮廓尺度为边长" 取 `window = window_ratio × 粗糙轮廓外接框对角线`，
  默认 `0.35`。依据：median 目标对角线约 147px → 窗口 ≈ 51px，能覆盖边界搜索范围。
* **H2 点对应关系**：粗糙点 ↔ 精准点用**弧长对应**（GT 轮廓按弧长重采样，起始点对齐粗糙首点最近点），
  保证一一对应、同序同向；`geometry.radial_correspondence` 提供径向对应对照。
* **H3 GT 轮廓起点**：取轮廓最高点（并列取最左）作为确定性起点；起点歧义由 A5 点序偏移进一步消除。
* **H4 训练窗口与推理一致性**：训练时若精准点落在窗口外，自适应放大该点窗口（最多 `max_expand=4` 倍）。
  实测落在窗口外的精准点比例 <2%，自适应后目标点全部落在 `[-1,1]` 内。
* **H5 粗糙轮廓来源**：训练时用"抖动后的框"，而非完美 GT 框，以匹配推理时的 YOLO 框误差水平。
* **H6 训练/推理同分布**：缓存生成与推理都走同一套 `build_sample`/`refine_contour` 代码路径，
  避免"训练用 32×32 局部 ROI、推理用别的东西"这类隐性偏差。

---

## 5. 评估协议

* **测试集**：`splits.csv` 的 `test`；SAM 系横向对比用其前 200 张子集（覆盖 busi/ddti/tn3k）。
* **目标匹配**：mask IoU ≥ 0.5 的贪心匹配；未匹配 GT 记漏检、未匹配预测记误检。
* **分割指标**：Dice、IoU、HD95、ASSD、边界 F1（2px 容差）、面积相对误差、Dice<0.5 失败率。
* **检测指标**：框级 IoU ≥ 0.5 的 Precision / Recall / F1。
* **速度口径**：每张图端到端 wall-clock（含图像预处理、YOLO 检测、分割解码），GPU 同步计时，
  预热后统计 mean / median / std / FPS。CPU 图像预处理移出 GPU 计时区间。
* **显著性**：跨配置比较用**按图像配对的 bootstrap**（2000 次重采样）给出 ΔDice 的 95% CI。

### 多路对照（`scripts/07_evaluate.py` 的 variants）

| variant | 含义 |
|---|---|
| `refiner_box` | **A-box（正式配置）**：框矩形初始化 + refiner |
| `refiner` | A-otsu：Otsu 初始化 + refiner（对照） |
| `rough_otsu` / `rough_otsu_gtbox` | 仅 Otsu 粗糙轮廓（YOLO 框 / GT 框） |
| `refiner_gtbox` | 隔离检测误差：GT 框 + Otsu + refiner |
| `yolo26n_seg` | B：端到端实例分割 |
| `seg_refiner` | B+refiner：seg mask 当粗轮廓再精细化 |

---

## 6. 关键工程发现（实测）

### 6.1 Otsu 的"外扩陷阱"

常规做法会把检测框**外扩 15%** 再在 ROI 内做 Otsu（"保证边缘落在 ROI 内"）。在超声上适得其反：
低回声病灶周围常有大片暗背景（DDTI 尤其明显），外扩后**病灶与背景在阈值分割中连成一片**，
"最大暗连通域"退化成背景，轮廓贴着框边甚至图像边界。

GT 框条件下的算法对比（mean Dice）：

| 策略 | BUSI | DDTI | TN3K | 平均 |
|---|---|---|---|---|
| 外扩 15% + 最大暗连通域（初版） | 0.738 | 0.566 | 0.635 | 0.640 |
| **不外扩 + 最大暗连通域（最终采用）** | **0.833** | **0.669** | **0.728** | **0.734** |
| 不外扩 + 排除贴边连通域 | 0.008 | 0.039 | 0.052 | 0.046 |
| 外扩 5% + 中心偏好 | 0.793 | 0.641 | 0.700 | 0.705 |

用检测框（框 IoU 均值 0.759）复测同样结论：`pad=0` 使粗糙轮廓平均 Dice 由 **0.607 → 0.678**。
"排除贴边连通域"这种看似合理的改进反而灾难性失败（病灶本身常常贴边/贴框），不予采用。
→ 最终 `pad_ratio = 0`。

### 6.2 初始化质量才是精度天花板（覆盖率为王）

同一 refiner、同一测试子集，只换初始轮廓（`scripts/17_recall_strategy.py`、`scripts/23_box_aug_stats.py`）：

| 初始化策略 | Dice | 初始轮廓**真值覆盖率** | 覆盖率 p10 |
|---|---|---|---|
| **框矩形** | **0.7632** | **0.967** | 0.912 |
| Otsu / 框 边界证据择优 | 0.6919 | — | — |
| Otsu 粗糙轮廓 | 0.6576 | 0.718 | **0.301** |

Otsu 平均只覆盖 72% 的病灶像素、最差 10% 只覆盖 30%（框未完整包住病灶、或框内阈值把病灶切掉一块）。
**refiner 只能把轮廓往外/往里挪，补不回初始轮廓完全没覆盖的区域**，因此 Otsu 的缺口会一路传递
（精细化后 0.8159 vs 框 0.8737）。框的收缩系数（0/0.05/0.1/0.2 → 0.8742/0.8752/0.8734/0.8700）
几乎无影响：**只要覆盖完整即可**。

### 6.3 漏检归因：不是检测器的锅

逐级归因（341 个 GT 目标，conf≥0.25，IoU≥0.5 记命中）：

| 环节 | 命中 | 占比 |
|---|---|---|
| YOLO26n 检测命中 | 297 | 87.1% |
| Otsu 成功产出轮廓 | 297 | 87.1% |
| **Otsu 轮廓本身 IoU≥0.5** | **219** | **64.2%** |
| 用框矩形当初始轮廓 | 275 | 80.6% |
| GT 框 + Otsu（上限参照） | 308 | 90.3% |

即 **78 个目标"检测到了但 Otsu 初始轮廓不合格"**。排除掉的方案（均实测）：
降低阈值（0.25→0.05：召回 93.3%→97.4% 但 **F1 0.905→0.750**）、检测+分割框池化（F1 0.881）、
翻转 TTA 池化（0.903）、多尺度池化（0.849）、Otsu/框启发式择优（0.6919 Dice）。
`det@0.25` 的 F1=0.9047 已是阈值最优区间。

### 6.4 框形状增强无效（负结果）

五组对照（同结构/同数据/同超参，配对 bootstrap）：

| 组别 | Dice | ΔDice vs 对照 | 95% CI | 判定 |
|---|---|---|---|---|
| 对照：无框增强 | 0.7985 | — | — | 基线 |
| C1+C2 | 0.7957 | −0.003 | [−0.011, +0.005] | 无显著差异 |
| box_init_prob=0.5 + 全部增强 | 0.7933 | −0.005 | [−0.013, +0.002] | 无提升 |
| C1+C2+C3+C4 | 0.7898 | −0.008 | [−0.020, +0.003] | 无提升 |
| C3+C4 | 0.7869 | −0.011 | [−0.024, +0.000] | 显著变差 |

推理时的初始轮廓就是检测器的真实框（接近矩形、中等定位误差），而 C3/C4 制造的是推理时不会出现的
形状（内缩轮廓、非矩形四边形），反而稀释有效样本。鲁棒性已由 `bbox_jitter` + A3/A4 + A5 提供。
**对照自身重复性 ±0.004 Dice** → 本实验任何 <0.01 的"提升"都不可信。最终 `box_aug=False`。

### 6.5 SAM 系适配：必须域适配，且速度受限

| 模型 | 零样本 Dice | 适配后 Dice | 端到端 ms |
|---|---|---|---|
| EfficientSAM (ViT-T) | **0.0000**（空 mask/无关大块） | 0.7682 | 99.7 |
| EdgeSAM | 0.6180 | 0.7648 | 35.8 |
| LiteMedSAM (TinyViT) | **0.7715**（官方权重，未适配） | — | 44.8 |

三者分割模块本身都在 8–10ms，差异来自图像编码器（RepViT@1024 ≈ 27ms、TinyViT@256 ≈ 9ms）与
每框一次 prompt/decoder 调用。EfficientSAM/EdgeSAM 用**冻结编码器 + 仅微调 mask decoder**（4.06M）
在同训练集上适配。

### 6.6 Swin-LiteMedSAM 官方权重不可得（记录）

* 官方 repo `RuochenGao/Swin_LiteMedSAM` 的 checkpoint 仅托管在 Google Drive，本机不可达；
  GitHub Releases 为空，hf-mirror 无镜像，Zenodo 无。
* 结构比对：prompt encoder 与 LiteMedSAM **完全同构**（17/17 键可迁移），
  Swin 版 mask decoder 键名体系不同（**0 交集**），Swin encoder 仅 patch_embed 等 **18/254** 键可迁移。
* 采用的替代方案：Swin encoder 迁移可对上的键后**冻结**，prompt encoder + mask decoder（26.2M）
  在本数据集上用 YOLO26n 框提示训练 8 epoch（val Dice@256 = 0.844）。
* 该结果（Dice 0.6000）**不代表 Swin-LiteMedSAM 的真实能力**，仅作架构参考。

### 6.7 输入尺寸：512 优于 256

| 输入尺寸 | 检测 Recall | A-box | B | EdgeSAM | 端到端 ms |
|---|---|---|---|---|---|
| 512×512 | **0.9000** | 0.7781 | **0.7798** | 0.7710 | 19.4 / 11.6 / 48.8 |
| 256×256 | 0.7500 | 0.5761 | 0.6278 | 0.5785 | 15.3 / 10.2 / 41.2 |

256 下检测召回掉 0.15（漏检 +18 个目标），三种方案 Dice 掉 0.15–0.20，速度只快 1.5–7.6ms。
**退化主因是检测**；EdgeSAM 因固定 1024×1024 编码器几乎不受输入尺寸影响。

---

## 7. 踩坑与修正记录（复现时最常见）

| 问题 | 现象 | 修正 |
|---|---|---|
| `masks.xy` 坐标空间 | 显式传 `imgsz` 时 `masks.xy` 是 **letterbox 坐标**，直接用会整体偏移 | 改用 `masks.data`，裁掉填充区后映射回原图（Dice 0.35 → 0.957） |
| `boxes.xyxy` 坐标空间 | ultralytics 的框**已是原图坐标**（`orig_shape`），再做反向 letterbox → 检测 TP 5/33 | 直接用；缩放比只用来自行推导 mask 的映射 |
| 退化框导致 patch 为空 | 窗口完全出界时 `np.pad` 抛 "can't extend empty axis" | `extract_patches` 对空 patch 用常量填充 |
| 每图多目标 vs 单 prompt | Swin decoder 要求 prompt 数与 image embedding 一一对应 | 每图采样 1 个目标构造成对样本 |
| CUDA + DataLoader fork | `Cannot re-initialize CUDA in forked subprocess` | `num_workers=0`（在线编码场景） |
| mmap/np 数组转 tensor | `Trying to resize storage that is not resizable`（collate 报错） | 先 `np.array(..., copy=True)` 拷进内存 |
| Swin 多尺度特征 | 4 个尺度空间尺寸不同，`np.stack` 直接失败 | 分尺度存 `fs0..fs3.npy`（最终改为在线编码免落盘） |
| 缓存体积 | 4 尺度特征缓存约 13GB | 改为在线编码训练，不落盘 |

---

## 8. 脚本清单

```
数据处理   01_prepare_data.py   02_make_splits.py   04_export_yolo.py   04b_make_subset.py
自检       00_sanity_check.py   00b_vis_contours.py 03c_verify_cache.py
方案 A     03b_cache_samples.py 03_train_refiner.py
方案 B     05_train_yolo.py (--kind det|seg)
SAM 适配   19_finetune_sam.py (EdgeSAM/EfficientSAM)   26_train_swin_litemedsam.py
统一评估   07_evaluate.py       18_bench_sam.py (精度+速度)   24_input_size_bench.py
消融实验   21_box_aug_experiments.py  22_box_aug_report.py  23_box_aug_stats.py
          16_recall_fix.py  17_recall_strategy.py  15_cascade_gain.py  12_diagnose.py
可视化     10_make_vis.py   11_failure_gallery.py
报告       13_compare.py   14_final_report.py   20_sam_compare.py   25_input_size_report.py
```
