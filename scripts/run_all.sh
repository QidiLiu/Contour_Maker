#!/usr/bin/env bash
# End-to-end pipeline: data -> train -> validate -> export -> MNN -> C++ test.
#
# Usage:
#   ./scripts/run_all.sh data       # download/unify/split/export the ultrasound data
#   ./scripts/run_all.sh train      # train YOLO-Seg + Det-UNet + RF-DETR-Seg
#   ./scripts/run_all.sh validate   # Python validation for the three schemes
#   ./scripts/run_all.sh export     # ONNX export + MNN conversion
#   ./scripts/run_all.sh build      # build the C++ binaries
#   ./scripts/run_all.sh test       # run the three C++ binaries
#   ./scripts/run_all.sh all        # everything above
#
# Environment: uv venv + C++ toolchain prepared first (see README.md).
# The ADE20K benchmark is documented in docs/reproduce.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
MNN_CONVERT="${MNN_CONVERT:-$ROOT/build/mnn-conv/MNNConvert}"
STAGE="${1:-all}"
JOBS="${JOBS:-8}"

# ultrasound run names (kept stable so existing checkpoints keep working)
SEG_RUN="${SEG_RUN:-seg_yolo26n}"
DET_RUN="${DET_RUN:-det_yolo26n}"
MKUNET_RUN="${MKUNET_RUN:-mkunet_yolo26n}"
RFDETR_RUN="${RFDETR_RUN:-rfdetr_seg_ultrasound}"

mkdir -p "$ROOT/logs" "$ROOT/models" "$ROOT/reports"

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

# ---------------------------------------------------------------- data
do_data() {
  say "Ultrasound data"
  "$PY" -m common.data_prep --download
  "$PY" -m common.data_prep --extract
  "$PY" -m common.data_prep --build
  "$PY" -m common.splits
  "$PY" -m common.yolo_export --kind both
  "$PY" -m common.coco_export
}

# ---------------------------------------------------------------- training
do_train_seg() {
  say "Train YOLO-Seg (YOLO26n-seg)"
  cd "$ROOT/yolo-seg-train"
  "$PY" train_yolo_seg.py \
      --data "$ROOT/data/yolo_seg/data_seg.yaml" \
      --model "$ROOT/weights/yolo26n-seg.pt" \
      --epochs "${SEG_EPOCHS:-60}" --imgsz 512 --batch 12 \
      --name "$SEG_RUN" 2>&1 | tee "$ROOT/logs/train_seg.log"
  cd "$ROOT"
}

do_train_det() {
  say "Train Det-UNet stage 1 (YOLO26n-det)"
  cd "$ROOT/det-unet-train"
  "$PY" train_yolo_det.py \
      --data "$ROOT/data/yolo_det/data_det.yaml" \
      --model "$ROOT/weights/yolo26n.pt" \
      --epochs "${DET_EPOCHS:-60}" --imgsz 512 --batch 16 \
      --name "$DET_RUN" 2>&1 | tee "$ROOT/logs/train_det.log"
  cd "$ROOT"
}

do_train_mkunet() {
  say "Build ROI cache + train MK-UNet"
  cd "$ROOT/det-unet-train"
  "$PY" build_roi_dataset.py --splits train --box-source det \
      --det "$ROOT/runs/$DET_RUN/weights/best.pt" \
      --out "$ROOT/data/roi_cache/train"
  "$PY" build_roi_dataset.py --splits val --box-source gt \
      --out "$ROOT/data/roi_cache/val"
  "$PY" train_mkunet.py \
      --data "$ROOT/data/roi_cache/train" \
      --val-data "$ROOT/data/roi_cache/val" \
      --variant MK_UNet --roi-size 256 \
      --epochs "${MKUNET_EPOCHS:-60}" --batch 16 \
      --name "$MKUNET_RUN" 2>&1 | tee "$ROOT/logs/train_mkunet.log"
  cd "$ROOT"
}

do_train_rfdetr() {
  say "Train RF-DETR-Seg-Nano"
  cd "$ROOT/rfdetr-seg-train"
  "$PY" train_rfdetr_seg.py --dataset ultrasound \
      --epochs "${RFDETR_EPOCHS:-8}" --batch "${RFDETR_BATCH:-8}" \
      --resolution "${RFDETR_RESOLUTION:-384}" \
      --name "$RFDETR_RUN" 2>&1 | tee "$ROOT/logs/train_rfdetr.log"
  cd "$ROOT"
}

do_train() {
  do_train_seg
  do_train_det
  do_train_mkunet
  do_train_rfdetr
}

# ---------------------------------------------------------------- Python validation
do_validate() {
  say "Python validation (YOLO-Seg)"
  cd "$ROOT/yolo-seg-train"
  "$PY" validate.py --seg "$ROOT/runs/$SEG_RUN/weights/best.pt" \
      --split test --out "$ROOT/reports/ultrasound_yolo_seg_py"
  cd "$ROOT"

  say "Python validation (Det-UNet)"
  cd "$ROOT/det-unet-train"
  "$PY" validate.py --det "$ROOT/runs/$DET_RUN/weights/best.pt" \
      --mkunet "$ROOT/runs/$MKUNET_RUN/best.pt" \
      --split test --out "$ROOT/reports/ultrasound_det_unet_py"
  cd "$ROOT"

  say "Python validation (RF-DETR-Seg)"
  cd "$ROOT/rfdetr-seg-train"
  "$PY" validate.py --weights "$ROOT/runs/$RFDETR_RUN/checkpoint_best_total.pth" \
      --resolution "${RFDETR_RESOLUTION:-384}" --split test \
      --out "$ROOT/reports/ultrasound_rfdetr_seg_py"
  cd "$ROOT"
}

# ---------------------------------------------------------------- export
do_export() {
  say "Export ONNX (YOLO-Seg)"
  cd "$ROOT/yolo-seg-train"
  "$PY" export_onnx.py --seg "$ROOT/runs/$SEG_RUN/weights/best.pt" \
      --out "$ROOT/models"
  cd "$ROOT"

  say "Export ONNX (Det-UNet)"
  cd "$ROOT/det-unet-train"
  "$PY" export_onnx.py --det "$ROOT/runs/$DET_RUN/weights/best.pt" \
      --mkunet "$ROOT/runs/$MKUNET_RUN/best.pt" \
      --out "$ROOT/models"
  cd "$ROOT"

  say "Export ONNX + MNN (RF-DETR-Seg)"
  cd "$ROOT/rfdetr-seg-train"
  "$PY" export_onnx.py --weights "$ROOT/runs/$RFDETR_RUN/checkpoint_best_total.pth" \
      --out "$ROOT/models" --mnn-convert "$MNN_CONVERT"
  cd "$ROOT"

  say "ONNX -> MNN (YOLO / MK-UNet)"
  do_mnnconvert
  "$MNN_CONVERT" -f ONNX --modelFile "$ROOT/models/yolo26n_seg.onnx" \
      --MNNModel "$ROOT/models/yolo26n_seg.mnn"
  "$MNN_CONVERT" -f ONNX --modelFile "$ROOT/models/yolo26n_det.onnx" \
      --MNNModel "$ROOT/models/yolo26n_det.mnn"
  "$MNN_CONVERT" -f ONNX --modelFile "$ROOT/models/mkunet.onnx" \
      --MNNModel "$ROOT/models/mkunet.mnn"
}

# ---------------------------------------------------------------- build / test
do_build() {
  say "Build C++"
  cmake -S "$ROOT" -B "$ROOT/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON
  cmake --build "$ROOT/build" -j "$JOBS"
}

do_mnnconvert() {
  if [ ! -x "$MNN_CONVERT" ]; then
    say "Build MNNConvert"
    cmake -S "$ROOT/third_party/MNN" -B "$ROOT/build/mnn-conv" -G Ninja \
          -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON \
          -DMNN_BUILD_PROTOBUFFER=ON -DMNN_BUILD_CONVERTER=ON \
          -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_WIN_RUNTIME_MT=ON
    cmake --build "$ROOT/build/mnn-conv" --target MNNConvert -j "$JOBS"
  fi
}

do_test() {
  [ -x "$ROOT/build/yolo-seg-test/yolo_seg_test" ] || do_build

  say "C++ test (YOLO-Seg)"
  "$ROOT/build/yolo-seg-test/yolo_seg_test" \
      --model "$ROOT/models/yolo26n_seg.mnn" \
      --list-file "$ROOT/data/splits/test.txt" \
      --gt-dir "$ROOT/data/unified/masks" \
      --imgsz 512 --conf 0.25 --num-classes 4 \
      --out-dir "$ROOT/reports/ultrasound_yolo_seg_cxx"

  say "C++ test (Det-UNet)"
  "$ROOT/build/det-unet-test/det_unet_test" \
      --det "$ROOT/models/yolo26n_det.mnn" \
      --mkunet "$ROOT/models/mkunet.mnn" \
      --list-file "$ROOT/data/splits/test.txt" \
      --gt-dir "$ROOT/data/unified/masks" \
      --det-imgsz 512 --roi-size 256 --pad-ratio 0.15 \
      --out-dir "$ROOT/reports/ultrasound_det_unet_cxx"

  say "C++ test (RF-DETR-Seg)"
  "$ROOT/build/rfdetr-seg-test/rfdetr_seg_test" \
      --model "$ROOT/models/rfdetr_seg_nano.mnn" \
      --list-file "$ROOT/data/splits/test.txt" \
      --gt-dir "$ROOT/data/unified/masks" \
      --resolution "${RFDETR_RESOLUTION:-384}" --conf 0.25 --num-classes 4 \
      --out-dir "$ROOT/reports/ultrasound_rfdetr_seg_cxx"
}

case "$STAGE" in
  data)     do_data ;;
  train)    do_train ;;
  validate) do_validate ;;
  export)   do_export ;;
  build)    do_build ;;
  test)     do_test ;;
  all)      do_data; do_train; do_validate; do_export; do_build; do_test ;;
  *) echo "unknown stage: $STAGE (data/train/validate/export/build/test/all)" >&2; exit 2 ;;
esac

say "done"
