#!/usr/bin/env bash
# 端到端流水线：数据 -> 训练 -> 导出 -> MNN -> C++ 测试
#
# 用法:
#   ./scripts/run_all.sh data      # 只做数据准备
#   ./scripts/run_all.sh train     # 训练两阶段 + 单阶段
#   ./scripts/run_all.sh export    # 导出 ONNX 并转 MNN
#   ./scripts/run_all.sh test      # 跑 C++ 测试并对比
#   ./scripts/run_all.sh all       # 全部
#
# 环境: 需先完成 uv venv 与 C++ 工具链准备（见 README.md）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
# MNNConvert 默认取本项目构建的产物；也可用 MNN_CONVERT 环境变量覆盖
MNN_CONVERT="${MNN_CONVERT:-$ROOT/build/mnn-conv/MNNConvert}"
STAGE="${1:-all}"
JOBS="${JOBS:-8}"

mkdir -p "$ROOT/logs" "$ROOT/models" "$ROOT/reports"

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 数据
do_data() {
  say "数据准备"
  "$PY" -m common.data_prep --download
  "$PY" -m common.data_prep --extract
  "$PY" -m common.data_prep --build
  "$PY" -m common.splits
  "$PY" -m common.yolo_export --kind both
}

# ---------------------------------------------------------------- 训练
do_train_seg() {
  say "训练 YOLO26n-seg（单阶段）"
  cd "$ROOT/one-stage-train"
  "$PY" train_yolo_seg.py \
      --data "$ROOT/data/yolo_seg/data_seg.yaml" \
      --model "$ROOT/weights/yolo26n-seg.pt" \
      --epochs "${SEG_EPOCHS:-60}" --imgsz 512 --batch 12 \
      --name seg_yolo26n 2>&1 | tee "$ROOT/logs/train_seg.log"
  cd "$ROOT"
}

do_train_det() {
  say "训练 YOLO26n-det（两阶段第一级）"
  cd "$ROOT/two-stage-train"
  "$PY" train_yolo_det.py \
      --data "$ROOT/data/yolo_det/data_det.yaml" \
      --model "$ROOT/weights/yolo26n.pt" \
      --epochs "${DET_EPOCHS:-60}" --imgsz 512 --batch 16 \
      --name det_yolo26n 2>&1 | tee "$ROOT/logs/train_det.log"
  cd "$ROOT"
}

do_train_mkunet() {
  say "构建 ROI 数据集 + 训练 MK-UNet"
  cd "$ROOT/two-stage-train"
  # 训练集用检测框（匹配推理分布）；验证集用 GT 框（干净、稳定）
  "$PY" build_roi_dataset.py --splits train --box-source det \
      --det "$ROOT/runs/det_yolo26n/weights/best.pt" \
      --out "$ROOT/data/roi_cache/train"
  "$PY" build_roi_dataset.py --splits val --box-source gt \
      --out "$ROOT/data/roi_cache/val"
  "$PY" train_mkunet.py \
      --data "$ROOT/data/roi_cache/train" \
      --val-data "$ROOT/data/roi_cache/val" \
      --variant MK_UNet --roi-size 256 \
      --epochs "${MKUNET_EPOCHS:-60}" --batch 16 \
      --name mkunet_yolo26n 2>&1 | tee "$ROOT/logs/train_mkunet.log"
  cd "$ROOT"
}

do_train() {
  do_train_seg
  do_train_det
  do_train_mkunet
}

# ---------------------------------------------------------------- 验证（PyTorch）
do_validate() {
  say "PyTorch 侧验证"
  cd "$ROOT/one-stage-train"
  "$PY" validate.py --seg "$ROOT/runs/seg_yolo26n/weights/best.pt" \
      --split test --out "$ROOT/reports/one_stage"
  cd "$ROOT/two-stage-train"
  "$PY" validate.py --det "$ROOT/runs/det_yolo26n/weights/best.pt" \
      --mkunet "$ROOT/runs/mkunet_yolo26n/best.pt" \
      --split test --out "$ROOT/reports/two_stage"
  cd "$ROOT"
}

# ---------------------------------------------------------------- 导出
do_export() {
  say "导出 ONNX"
  cd "$ROOT/one-stage-train"
  "$PY" export_onnx.py --seg "$ROOT/runs/seg_yolo26n/weights/best.pt" \
      --out "$ROOT/models"
  cd "$ROOT/two-stage-train"
  "$PY" export_onnx.py --det "$ROOT/runs/det_yolo26n/weights/best.pt" \
      --mkunet "$ROOT/runs/mkunet_yolo26n/best.pt" \
      --out "$ROOT/models"
  cd "$ROOT"

  say "ONNX -> MNN"
  do_mnnconvert
  "$MNN_CONVERT" -f ONNX --modelFile "$ROOT/models/yolo26n_seg.onnx" \
      --MNNModel "$ROOT/models/yolo26n_seg.mnn"
  "$MNN_CONVERT" -f ONNX --modelFile "$ROOT/models/yolo26n_det.onnx" \
      --MNNModel "$ROOT/models/yolo26n_det.mnn"
  "$MNN_CONVERT" -f ONNX --modelFile "$ROOT/models/mkunet.onnx" \
      --MNNModel "$ROOT/models/mkunet.mnn"
}

# ---------------------------------------------------------------- 构建 + 测试
do_build() {
  say "构建 C++"
  cmake -S "$ROOT" -B "$ROOT/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON
  cmake --build "$ROOT/build" -j "$JOBS"
}

do_mnnconvert() {
  # MNNConvert 默认不编译（MNN_BUILD_CONVERTER 默认 OFF），且需用 MNN 自带 protobuf
  if [ ! -x "$MNN_CONVERT" ]; then
    say "构建 MNNConvert"
    cmake -S "$ROOT/third_party/MNN" -B "$ROOT/build/mnn-conv" -G Ninja \
          -DCMAKE_BUILD_TYPE=Release -DMNN_OPENCL=ON \
          -DMNN_BUILD_PROTOBUFFER=ON -DMNN_BUILD_CONVERTER=ON \
          -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_WIN_RUNTIME_MT=ON
    cmake --build "$ROOT/build/mnn-conv" --target MNNConvert -j "$JOBS"
  fi
}

do_test() {
  [ -x "$ROOT/build/one-stage-test/one_stage_test" ] || do_build
  say "单阶段 C++ 测试"
  "$ROOT/build/one-stage-test/one_stage_test" \
      --model "$ROOT/models/yolo26n_seg.mnn" \
      --image-dir "$ROOT/data/unified/images" \
      --gt-dir "$ROOT/data/unified/masks" \
      --imgsz 512 --conf 0.25 --num-classes 4 \
      --out-dir "$ROOT/reports/one_stage_cxx"

  say "两阶段 C++ 测试"
  "$ROOT/build/two-stage-test/two_stage_test" \
      --det "$ROOT/models/yolo26n_det.mnn" \
      --mkunet "$ROOT/models/mkunet.mnn" \
      --image-dir "$ROOT/data/unified/images" \
      --gt-dir "$ROOT/data/unified/masks" \
      --det-imgsz 512 --roi-size 256 --pad-ratio 0.15 \
      --out-dir "$ROOT/reports/two_stage_cxx"
}

case "$STAGE" in
  data)    do_data ;;
  train)   do_train ;;
  validate) do_validate ;;
  export)  do_export ;;
  build)   do_build ;;
  test)    do_test ;;
  all)     do_data; do_train; do_validate; do_export; do_build; do_test ;;
  *) echo "未知阶段: $STAGE (可选 data/train/validate/export/build/test/all)" >&2; exit 2 ;;
esac

say "完成"