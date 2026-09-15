"""全局配置: 路径、数据划分、模型与训练超参。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
UNIFIED = DATA / "unified"
SPLITS = DATA / "splits"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
CONDA_PY = ROOT / ".conda" / "bin" / "python"

for _p in (RAW, INTERIM, UNIFIED, SPLITS, RUNS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- datasets
@dataclass
class DatasetSpec:
    """一个原始数据集在 data/raw 下的布局描述。"""

    key: str                 # 唯一标识, 例如 busi
    name: str                # 展示名
    modality: str            # breast | thyroid
    zip_name: str            # data/raw 下的压缩包
    inner_root: str | None = None   # 解压后需要深入的子目录(自动探测为 None)


DATASETS: list[DatasetSpec] = [
    DatasetSpec("busi", "BUSI (Breast Ultrasound Images)", "breast", "BUSI.zip"),
    DatasetSpec("ddti", "DDTI (Thyroid Digital Database)", "thyroid", "ThyroidNodule-DDTI.zip"),
    DatasetSpec("tn3k", "TN3K (Thyroid Nodule 3K)", "thyroid", "ThyroidNodule-TN3K.zip"),
    DatasetSpec("tg3k", "TG3K (Thyroid Gland 3K)", "thyroid", "ThyroidNodule-TG3K.zip"),
]


# ---------------------------------------------------------------- contour refiner
@dataclass
class RefinerConfig:
    """contour-refiner 的输入输出规格 (对应用户给定的算法描述)。"""

    n_points: int = 64          # 每个目标的轮廓点数 = ROI 图数量
    roi_size: int = 32          # 每张 ROI 统一 resize 到 32x32 灰度
    window_ratio: float = 0.35  # ROI 窗口边长 = window_ratio * 粗糙轮廓外接框对角线
    adaptive_window: bool = True  # 训练时若精准点落在窗口外, 自动放大该点窗口 (最多 max_expand 倍)
    max_expand: float = 4.0
    zscore: bool = True         # ROI 灰度做 Z-score 归一化 (每张 ROI 独立)
    # 架构
    backbone: str = "cnn_transformer"   # 共享 CNN backbone + 点间 Transformer 上下文
    feat_dim: int = 128
    n_blocks: int = 4           # CNN stage 数
    n_heads: int = 4
    n_tx_layers: int = 2
    dropout: float = 0.0
    # 训练
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "smooth_l1"     # smooth_l1 | l1 | l2
    samples_per_target: int = 4  # 每个目标每轮采样多少组 (粗糙, 精准) 数据对
    seed: int = 0


# ---------------------------------------------------------------- augmentation
@dataclass
class AugConfig:
    """数据增强模式 A1-A5 (逐级叠加) + 检测框抖动。

    A1: 边缘点坐标 Gaussian 噪声
    A2: ROI 内 Otsu 最近边缘点坐标 + 微量 Gaussian 噪声 (A1 的 0~1/3)
    A3: A2 + 随机 ROI 范围变化 / 图像噪声 / 亮暗 / 对比度  (0~1/3 强度)
    A4: A3 + 随机旋转 (任意角度, 0~1/3 概率强度)
    A5: A4 + 点序偏移 (0~100% 起始点平移)
    bbox_jitter: 检测框抖动, 模拟 YOLO26n 检测框误差 (推理时的真实输入分布)
    """

    level: int = 5                      # 1..5, 累积启用 A1..A5
    a1_sigma: float = 0.10              # 归一化坐标下的标准差
    a2_ratio: float = 1.0 / 3.0         # A2 噪声幅度 = a1_sigma * [0, ratio]
    a3_strength: float = 1.0 / 3.0      # A3 各扰动幅度上限
    a4_strength: float = 1.0 / 3.0      # A4 旋转概率/幅度强度
    a5_max_shift: float = 1.0           # A5 起始点偏移上限 (1.0 = 100%)
    bbox_jitter: float = 0.10           # 检测框中心/尺度相对抖动上限 (0.10 -> 约 IoU 0.8~0.95)
    box_init_prob: float = 0.25         # 用检测框矩形本身作为初始轮廓的样本比例
    box_fallback: bool = True           # Otsu 分割失败时回退到框矩形

    def enabled(self, level: int) -> bool:
        return self.level >= level


# ---------------------------------------------------------------- evaluation
IOU_MATCH_THRESHOLD = 0.5   # 检测框匹配阈值
HD95_PERCENTILE = 95
