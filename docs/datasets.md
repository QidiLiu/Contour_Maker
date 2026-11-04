# 数据集

项目包含两个数据集 profile：

1. **Ultrasound** —— 超声病灶分割，4 类，用于原始任务与三方案的主对比。
2. **ADE20K-10** —— ADE20K 室内物体实例分割子集，用于验证三方案在通用（具身智能常用）场景上的泛化能力。

所有数据都统一成同一种中间格式，训练/评估脚本只依赖该格式，不再关心原始数据集的目录布局。

---

## 1. Ultrasound（4 类超声病灶）

来源：Hugging Face 镜像 [`us-segmentator/us-segmentation-dataset`](https://hf-mirror.com/datasets/us-segmentator/us-segmentation-dataset)（`hf-mirror.com` 可达；Google Drive 不可达）。

| key | 数据集 | 类别 | 图像数 |
|---|---|---|---|
| `busi` | BUSI | 0 良性 / 1 恶性 | 647（排除 `normal` 133 张无标注图） |
| `tn3k` | TN3K | 2 甲状腺结节 | 3493 |
| `ddti` | DDTI | 2 甲状腺结节 | 637 |
| `tg3k` | TG3K | 3 甲状腺腺体 | 3585 |

- BUSI 的 `benign/`、`malignant/` 目录名用于恢复类别；`normal/` 没有病灶标注，被排除。
- DDTI 上游同时存在 `test1.PNG.png` 与 `test1.png`，已按去扩展名 key 去重（否则 DDTI 会翻倍）。
- 划分：按 `(dataset, class_id)` 分层，`seed=42`，train/val/test = 70/15/15，共 8362 张。

类别表：

| id | 名称 | 来源 |
|---|---|---|
| 0 | `breast_benign` | BUSI `benign/` |
| 1 | `breast_malignant` | BUSI `malignant/` |
| 2 | `thyroid_nodule` | TN3K + DDTI |
| 3 | `thyroid_gland` | TG3K |

---

## 2. ADE20K-10（通用 / 具身智能场景）

[ADE20K](https://sceneparsing.csail.mit.edu/) 是场景理解与具身智能（室内导航、机器人感知）最常用的分割基准之一。官方 2017 版提供 **instance segmentation** 标注：PNG 的 **R 通道 = ADE20K thing 类索引（1–100）**，**G 通道 = 实例 id**，B 通道为 0。

### 2.1 选出的 10 类

从 100 个 thing 类中选取与室内操作/导航相关、且样本量足够的 10 类：

| 本项目 class id | ADE20K thing id | 名称 |
|---|---|---|
| 0 | 1 | `bed` |
| 1 | 2 | `windowpane` |
| 2 | 3 | `cabinet` |
| 3 | 5 | `door` |
| 4 | 6 | `table` |
| 5 | 8 | `chair` |
| 6 | 14 | `armchair` |
| 7 | 19 | `lamp` |
| 8 | 64 | `bottle` |
| 9 | 81 | `pot` |

### 2.2 子集与划分

官方 train 共 20210 张、validation 2000 张。为了在单卡 16 GB 上可复现地训练三个方案，项目使用确定性采样的子集：

| split | 来源 | 图像数 | 实例数 |
|---|---|---|---|
| train | 官方 train | 4000 | 24223 |
| val | 官方 train（与 train 不重叠） | 600 | 3597 |
| test | 官方 validation | 800 | 4779 |

- 采样种子 `42`；只保留至少包含 1 个上述类别实例的图像。
- 实例面积 < 200 px 的丢弃；每图最多 200 个实例。
- 官方 validation 被保留为 **test**，因此 test 从未参与训练/调参。

### 2.3 灰度与 mask 编码

嵌入式管线是单通道灰度输入，因此 ADE20K 图像统一转灰度（`cv2.COLOR_BGR2GRAY`），三个方案都看到同样的灰度输入。这是与彩色训练的一个明确差异，见 [`benchmarks.md`](benchmarks.md) 的局限说明。

多类别 + 多实例不能用一个二值 mask 表示，因此使用：

```
unified/images/<uid>.png        灰度图
unified/masks/<uid>.png         8-bit 实例 id（0 = 背景，1..N = 实例）
unified/instances/<uid>.csv     instance_id,class_id
```

`instances/<uid>.csv` 是 Python 与 C++ 共用的类别 sidecar。对超声数据集没有该文件时，mask 按二值连通域解释，类别取 `splits.csv` 的 `class_id`。

### 2.4 获取与构建

```bash
# 下载 ADE20K instance parquet（HF 镜像，约 1.1 GB）
.venv/bin/python -m common.ade20k --download
# 扫描 + 采样 + 写统一格式（约 1–2 分钟）
.venv/bin/python -m common.ade20k --build

# YOLO / COCO 导出
.venv/bin/python -m common.yolo_export --dataset ade20k --kind both
.venv/bin/python -m common.coco_export --dataset ade20k
```

下载源默认走 `hf-mirror.com`（可用 `HF_ENDPOINT=https://huggingface.co` 覆盖）。

许可证：ADE20K 为 **BSD-3-Clause**；本项目只分发脚本，不重新分发图像/标注。

---

## 3. 统一清单格式

`data/unified/manifest.csv`（超声）与 `data/ade20k/unified/manifest.csv`：

| 列 | 说明 |
|---|---|
| `uid` | 全局唯一 id，如 `busi_benign__100_`、`ade20k_ADE_train_00003517` |
| `dataset` | `busi` / `tn3k` / `ddti` / `tg3k` / `ade20k` |
| `modality` | `breast` / `thyroid` / `indoor` |
| `class_id` | 图像主类别（兼容旧脚本；多类图像的逐实例类别在 sidecar 中） |
| `class_source` | 类别来源说明 |
| `image` / `mask` | 相对 `unified/` 的路径 |
| `height` / `width` | 原图尺寸 |
| `split` | `train` / `val` / `test` |

`data/splits/splits.csv` 与 `data/ade20k/splits/splits.csv` 为同结构 + `split` 列的清单。
