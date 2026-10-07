"""全局配置：路径、类别表、数据集源、模型与评估常量。

被 two-stage-train/ 与 one-stage-train/ 共同 import。
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"              # 原始压缩包
INTERIM = DATA / "interim"      # 解压结果
UNIFIED = DATA / "unified"      # 统一后的 images/ masks/ + manifest.csv
SPLITS = DATA / "splits"        # splits.csv
YOLO_DET = DATA / "yolo_det"    # 两阶段: 检测训练数据
YOLO_SEG = DATA / "yolo_seg"    # 单阶段: 分割训练数据
ROI_CACHE = DATA / "roi_cache"  # 两阶段: MK-UNet 的 ROI 训练缓存

RUNS = ROOT / "runs"
LOGS = ROOT / "logs"

for _p in (RAW, INTERIM, UNIFIED, SPLITS, YOLO_DET, YOLO_SEG, ROI_CACHE, RUNS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 类别表
# 4 类, 按病灶性质细分。
# BUSI 的良性/恶性从原包的 benign/ malignant/ 目录名恢复
# (HF 镜像保留了目录结构, 无需 dataset.xlsx)。
CLASS_NAMES: list[str] = [
    "breast_benign",     # 0
    "breast_malignant",  # 1
    "thyroid_nodule",    # 2  TN3K + DDTI
    "thyroid_gland",     # 3  TG3K (高回声腺体)
]
NUM_CLASSES = len(CLASS_NAMES)

CLS_BREAST_BENIGN = 0
CLS_BREAST_MALIGNANT = 1
CLS_THYROID_NODULE = 2
CLS_THYROID_GLAND = 3

# ---------------------------------------------------------------- 数据集源
# 统一镜像: us-segmentator/us-segmentation-dataset (hf-mirror 可达)。
HF_REPO = "us-segmentator/us-segmentation-dataset"
HF_ENDPOINT = "https://hf-mirror.com"


class DatasetSpec:
    """单个数据集的压缩包与目录/类别映射规则。"""

    def __init__(
        self,
        key: str,
        modality: str,          # breast | thyroid
        zip_path: str,          # 仓库内路径
        class_id: int,          # 该数据集所有目标所属的类别
        class_from_dirname: bool = False,   # 类别需从解压后的目录名解析
        expected_images: int = 0,
    ):
        self.key = key
        self.modality = modality
        self.zip_path = zip_path
        self.class_id = class_id
        self.class_from_dirname = class_from_dirname
        self.expected_images = expected_images

    @property
    def url(self) -> str:
        return f"{HF_ENDPOINT}/datasets/{HF_REPO}/resolve/main/{self.zip_path}"

    @property
    def zip_name(self) -> str:
        return self.zip_path.rsplit("/", 1)[-1]


DATASETS: list[DatasetSpec] = [
    DatasetSpec("busi", "breast", "zips/Breast/BUSI.zip", CLS_BREAST_BENIGN,
                class_from_dirname=True, expected_images=647),
    DatasetSpec("tn3k", "thyroid", "zips/Thyroid/ThyroidNodule-TN3K.zip",
                CLS_THYROID_NODULE, expected_images=3493),
    DatasetSpec("ddti", "thyroid", "zips/Thyroid/ThyroidNodule-DDTI.zip",
                CLS_THYROID_NODULE, expected_images=637),
    DatasetSpec("tg3k", "thyroid", "zips/Thyroid/ThyroidNodule-TG3K.zip",
                CLS_THYROID_GLAND, expected_images=3585),
]

DATASET_BY_KEY = {d.key: d for d in DATASETS}

# BUSI 目录名 -> class_id
BUSI_DIRNAME_TO_CLASS = {
    "benign": CLS_BREAST_BENIGN,
    "malignant": CLS_BREAST_MALIGNANT,
}

# BUSI 的 normal 类没有病灶标注(其 _mask 为全黑)，不属于本任务的 4 类，必须排除。
BUSI_EXCLUDE_DIRNAMES = {"normal"}

# ---------------------------------------------------------------- 划分
SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------- 图像/掩码处理
MIN_MASK_AREA = 50        # 小于该面积的连通域视为噪声丢弃 (px)
MASK_THRESHOLD = 127      # 掩码二值化阈值

# ---------------------------------------------------------------- YOLO
YOLO_MODEL_DET = "yolo26n.pt"
YOLO_MODEL_SEG = "yolo26n-seg.pt"
YOLO_IMGSZ = 512
YOLO_EPOCHS_DET = 60
YOLO_EPOCHS_SEG = 60
YOLO_BATCH_DET = 16
YOLO_BATCH_SEG = 12
YOLO_CONF = 0.25          # 推理置信度阈值
YOLO_IOU_NMS = 0.7        # NMS IoU 阈值

# ---------------------------------------------------------------- MK-UNet (两阶段第二级)
MKUNET_VARIANT = "MK_UNet"
MKUNET_IN_CHANNELS = 1    # 超声为灰度
MKUNET_ROI_SIZE = 256     # 目标框 ROI resize 到的正方形边长
MKUNET_EPOCHS = 60
MKUNET_BATCH = 16
MKUNET_LR = 1e-3
MKUNET_WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------- 评估
IOU_MATCH_THRESHOLD = 0.5   # mask/box 匹配阈值
HD95_PERCENTILE = 95
BOUNDARY_F1_TOL = 2.0        # px

# ---------------------------------------------------------------- 导出
ONNX_OPSET = 12             # MNNConvert 对新 opset 支持有限, 保守取 12
ONNX_IOU_NMS = 0.7