#!/usr/bin/env bash
# Run the three schemes (Python + C++/MNN) on one dataset and collect results.
#
# Usage:
#   ./scripts/run_benchmarks.sh ultrasound
#   ./scripts/run_benchmarks.sh ade20k
#   MAX_IMAGES=120 ./scripts/run_benchmarks.sh ultrasound   # quick subset
#
# Prerequisites: trained checkpoints, exported .mnn models and a C++ build.
set -euo pipefail

DATASET="${1:-ultrasound}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
MAX_IMAGES="${MAX_IMAGES:-0}"

if [[ "$DATASET" == "ade20k" ]]; then
  UNIFIED="$ROOT/data/ade20k/unified"
  SPLITS="$ROOT/data/ade20k/splits/splits.csv"
  INSTANCES="$ROOT/data/ade20k/unified/instances"
  NC=10
  SEG_W="$ROOT/runs/yolo_seg_ade20k/weights/best.pt"
  DET_W="$ROOT/runs/det_ade20k/weights/best.pt"
  MK_W="$ROOT/runs/mkunet_ade20k/best.pt"
  RF_W="$ROOT/runs/rfdetr_seg_ade20k/checkpoint_best_total.pth"
  SEG_M="$ROOT/models/ade20k/yolo26n_seg.mnn"
  DET_M="$ROOT/models/ade20k/yolo26n_det.mnn"
  MK_M="$ROOT/models/ade20k/mkunet.mnn"
  RF_M="$ROOT/models/ade20k/rfdetr_seg_nano_ade20k.mnn"
  EXTRA_PY=(--dataset ade20k --splits-csv "$SPLITS" --unified "$UNIFIED" --gt-instances "$INSTANCES")
  EXTRA_CXX=(--gt-instances "$INSTANCES")
else
  UNIFIED="$ROOT/data/unified"
  SPLITS="$ROOT/data/splits/splits.csv"
  NC=4
  SEG_W="$ROOT/runs/seg_yolo26n/weights/best.pt"
  DET_W="$ROOT/runs/det_yolo26n/weights/best.pt"
  MK_W="$ROOT/runs/mkunet_yolo26n/best.pt"
  RF_W="$ROOT/runs/rfdetr_seg_ultrasound/checkpoint_best_total.pth"
  SEG_M="$ROOT/models/yolo26n_seg.mnn"
  DET_M="$ROOT/models/yolo26n_det.mnn"
  MK_M="$ROOT/models/mkunet.mnn"
  RF_M="$ROOT/models/rfdetr_seg_nano.mnn"
  EXTRA_PY=()
  EXTRA_CXX=()
fi

MAX_PY=()
MAX_CXX=()
if [[ "$MAX_IMAGES" != "0" ]]; then
  MAX_PY=(--max-images "$MAX_IMAGES")
  MAX_CXX=(--max-images "$MAX_IMAGES")
fi

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

say "Python: YOLO-Seg"
cd "$ROOT/yolo-seg-train"
"$PY" validate.py "${EXTRA_PY[@]}" --seg "$SEG_W" --split test "${MAX_PY[@]}" \
  --out "$ROOT/reports/${DATASET}_yolo_seg_py"

say "Python: Det-UNet"
cd "$ROOT/det-unet-train"
"$PY" validate.py "${EXTRA_PY[@]}" --det "$DET_W" --mkunet "$MK_W" --split test \
  "${MAX_PY[@]}" --out "$ROOT/reports/${DATASET}_det_unet_py"

say "Python: RF-DETR-Seg"
cd "$ROOT/rfdetr-seg-train"
"$PY" validate.py "${EXTRA_PY[@]}" --weights "$RF_W" --resolution 384 --split test \
  "${MAX_PY[@]}" --out "$ROOT/reports/${DATASET}_rfdetr_seg_py"

say "C++: YOLO-Seg"
"$ROOT/build/yolo-seg-test/yolo_seg_test" \
  --model "$SEG_M" --image-dir "$UNIFIED/images" --gt-dir "$UNIFIED/masks" \
  "${EXTRA_CXX[@]}" --imgsz 512 --num-classes "$NC" "${MAX_CXX[@]}" \
  --out-dir "$ROOT/reports/${DATASET}_yolo_seg_cxx"

say "C++: Det-UNet"
"$ROOT/build/det-unet-test/det_unet_test" \
  --det "$DET_M" --mkunet "$MK_M" --image-dir "$UNIFIED/images" --gt-dir "$UNIFIED/masks" \
  "${EXTRA_CXX[@]}" --det-imgsz 512 --roi-size 256 --num-classes "$NC" "${MAX_CXX[@]}" \
  --out-dir "$ROOT/reports/${DATASET}_det_unet_cxx"

say "C++: RF-DETR-Seg"
"$ROOT/build/rfdetr-seg-test/rfdetr_seg_test" \
  --model "$RF_M" --image-dir "$UNIFIED/images" --gt-dir "$UNIFIED/masks" \
  "${EXTRA_CXX[@]}" --resolution 384 --num-classes "$NC" "${MAX_CXX[@]}" \
  --out-dir "$ROOT/reports/${DATASET}_rfdetr_seg_cxx"

say "Collect"
"$PY" "$ROOT/scripts/collect_results.py" --reports "$ROOT/reports" --out "$ROOT/results"
