"""Global configuration: paths, class tables, dataset sources and constants.

The repository ships two dataset profiles:

* ``ultrasound`` -- the original 4-class breast/thyroid lesion dataset
  (BUSI, TN3K, DDTI, TG3K).  Its paths are the module-level constants below and
  are the defaults for every training/validation script.
* ``ade20k`` -- a 10-class indoor-object subset of ADE20K, used as the
  general-purpose (embodied-AI) generalisation benchmark.  Its paths live under
  :data:`ADE20K`.

Scripts that need to work on both datasets accept explicit ``--unified`` /
``--splits-csv`` / ``--classes`` arguments; the values below are only defaults.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"  # downloaded archives of the ultrasound datasets
INTERIM = DATA / "interim"  # extracted ultrasound archives
UNIFIED = DATA / "unified"  # unified images/ masks/ + manifest.csv
SPLITS = DATA / "splits"  # splits.csv
YOLO_DET = DATA / "yolo_det"  # det-unet: detection training data
YOLO_SEG = DATA / "yolo_seg"  # yolo-seg: segmentation training data
ROI_CACHE = DATA / "roi_cache"  # det-unet: MK-UNet ROI training cache
COCO = DATA / "coco"  # COCO-format export used by RF-DETR

RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
WEIGHTS = ROOT / "weights"
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"

# ---------------------------------------------------------------- ADE20K
ADE20K = DATA / "ade20k"
ADE20K_RAW = ADE20K / "raw"
ADE20K_UNIFIED = ADE20K / "unified"
ADE20K_SPLITS = ADE20K / "splits"
ADE20K_YOLO_DET = ADE20K / "yolo_det"
ADE20K_YOLO_SEG = ADE20K / "yolo_seg"
ADE20K_ROI_CACHE = ADE20K / "roi_cache"
ADE20K_COCO = ADE20K / "coco"

for _p in (
    RAW,
    INTERIM,
    UNIFIED,
    SPLITS,
    YOLO_DET,
    YOLO_SEG,
    ROI_CACHE,
    COCO,
    RUNS,
    LOGS,
    MODELS,
    WEIGHTS,
    REPORTS,
    RESULTS,
    ADE20K,
    ADE20K_RAW,
    ADE20K_UNIFIED,
    ADE20K_SPLITS,
    ADE20K_YOLO_DET,
    ADE20K_YOLO_SEG,
    ADE20K_ROI_CACHE,
    ADE20K_COCO,
):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- ultrasound classes
# 4 classes, split by lesion nature.
# BUSI benign/malignant are recovered from the original ``benign/`` /
# ``malignant/`` directory names (the HF mirror keeps that layout).
CLASS_NAMES: list[str] = [
    "breast_benign",  # 0
    "breast_malignant",  # 1
    "thyroid_nodule",  # 2  TN3K + DDTI
    "thyroid_gland",  # 3  TG3K (hyperechoic gland)
]
NUM_CLASSES = len(CLASS_NAMES)

CLS_BREAST_BENIGN = 0
CLS_BREAST_MALIGNANT = 1
CLS_THYROID_NODULE = 2
CLS_THYROID_GLAND = 3

# ---------------------------------------------------------------- ADE20K classes
# ADE20K "thing" class indices (the R channel of annotations_instance) mapped to
# a compact, manipulation/navigation-relevant indoor subset.  The index is the
# original ADE20K thing-class id, so the mapping in common/ade20k.py is stable.
ADE20K_CLASSES: list[dict] = [
    {"ade_id": 1, "name": "bed"},
    {"ade_id": 2, "name": "windowpane"},
    {"ade_id": 3, "name": "cabinet"},
    {"ade_id": 5, "name": "door"},
    {"ade_id": 6, "name": "table"},
    {"ade_id": 8, "name": "chair"},
    {"ade_id": 14, "name": "armchair"},
    {"ade_id": 19, "name": "lamp"},
    {"ade_id": 64, "name": "bottle"},
    {"ade_id": 81, "name": "pot"},
]
ADE20K_CLASS_NAMES: list[str] = [c["name"] for c in ADE20K_CLASSES]
ADE20K_ADE_ID_TO_CLASS: dict[int, int] = {c["ade_id"]: i for i, c in enumerate(ADE20K_CLASSES)}
ADE20K_NUM_CLASSES = len(ADE20K_CLASS_NAMES)

# ADE20K subset sizes.  Sampling is deterministic (seed below) and documented
# in docs/datasets.md.  Keep these small enough to train three schemes on a
# single 16 GB consumer GPU.
ADE20K_TRAIN_IMAGES = 4000
ADE20K_VAL_IMAGES = 600
ADE20K_TEST_IMAGES = 800
ADE20K_SPLIT_SEED = 42
ADE20K_MIN_INSTANCE_AREA = 200
ADE20K_MAX_INSTANCES = 200

# ---------------------------------------------------------------- dataset sources
# Unified mirror: us-segmentator/us-segmentation-dataset (hf-mirror reachable).
HF_REPO = "us-segmentator/us-segmentation-dataset"
HF_ENDPOINT = "https://hf-mirror.com"

# ADE20K instance segmentation (converted parquet on the HF mirror).
ADE20K_HF_REPO = "zhoubolei/scene_parse_150"
ADE20K_HF_REVISION = "refs/convert/parquet"
ADE20K_INSTANCE_FILES = {
    "train": ["train/0000.parquet", "train/0001.parquet"],
    "validation": ["validation/0000.parquet"],
}


class DatasetSpec:
    """Archive and directory/class mapping rule for one ultrasound source."""

    def __init__(
        self,
        key: str,
        modality: str,  # breast | thyroid
        zip_path: str,  # path inside the HF repo
        class_id: int,  # class id shared by every target of this source
        class_from_dirname: bool = False,  # class recovered from the extracted path
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
    DatasetSpec(
        "busi",
        "breast",
        "zips/Breast/BUSI.zip",
        CLS_BREAST_BENIGN,
        class_from_dirname=True,
        expected_images=647,
    ),
    DatasetSpec(
        "tn3k",
        "thyroid",
        "zips/Thyroid/ThyroidNodule-TN3K.zip",
        CLS_THYROID_NODULE,
        expected_images=3493,
    ),
    DatasetSpec(
        "ddti",
        "thyroid",
        "zips/Thyroid/ThyroidNodule-DDTI.zip",
        CLS_THYROID_NODULE,
        expected_images=637,
    ),
    DatasetSpec(
        "tg3k",
        "thyroid",
        "zips/Thyroid/ThyroidNodule-TG3K.zip",
        CLS_THYROID_GLAND,
        expected_images=3585,
    ),
]

DATASET_BY_KEY = {d.key: d for d in DATASETS}

# BUSI directory name -> class_id
BUSI_DIRNAME_TO_CLASS = {
    "benign": CLS_BREAST_BENIGN,
    "malignant": CLS_BREAST_MALIGNANT,
}

# BUSI's normal class has no lesion annotation (all-black mask); it is not one
# of the 4 task classes and must be excluded.
BUSI_EXCLUDE_DIRNAMES = {"normal"}

# ---------------------------------------------------------------- splits
SPLIT_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------- image/mask processing
MIN_MASK_AREA = 50  # connected components below this area are discarded (px)
MASK_THRESHOLD = 127  # binarisation threshold for 8-bit masks

# ---------------------------------------------------------------- YOLO
YOLO_MODEL_DET = "yolo26n.pt"
YOLO_MODEL_SEG = "yolo26n-seg.pt"
YOLO_IMGSZ = 512
YOLO_EPOCHS_DET = 60
YOLO_EPOCHS_SEG = 60
YOLO_BATCH_DET = 16
YOLO_BATCH_SEG = 12
YOLO_CONF = 0.25  # inference confidence threshold
YOLO_IOU_NMS = 0.7  # NMS IoU threshold

# ---------------------------------------------------------------- MK-UNet (det-unet stage 2)
MKUNET_VARIANT = "MK_UNet"
MKUNET_IN_CHANNELS = 1  # ultrasound is grayscale
MKUNET_ROI_SIZE = 256  # square side that box ROIs are resized to
MKUNET_EPOCHS = 60
MKUNET_BATCH = 16
MKUNET_LR = 1e-3
MKUNET_WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------- RF-DETR
# Nano is the only variant used in the benchmark.  504 is the nearest multiple
# of patch_size * num_windows (12) to the 512 used by the YOLO schemes.
RFDETR_VARIANT = "RFDETRSegNano"
RFDETR_RESOLUTION = 504
RFDETR_EPOCHS = 40
RFDETR_BATCH = 4
RFDETR_GRAD_ACCUM = 4
RFDETR_LR = 1e-4
RFDETR_WEIGHTS = "rf-detr-seg-nano.pt"

# ---------------------------------------------------------------- evaluation
IOU_MATCH_THRESHOLD = 0.5  # mask/box matching threshold
HD95_PERCENTILE = 95
BOUNDARY_F1_TOL = 2.0  # px

# ---------------------------------------------------------------- export
ONNX_OPSET = 12  # MNNConvert has limited support for new opsets; keep 12 for YOLO
ONNX_IOU_NMS = 0.7
RFDETR_ONNX_OPSET = 17
