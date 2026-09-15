#!/usr/bin/env bash
# 最终实验流水线 (v2, pad_ratio=0 的改进粗糙轮廓):
#   1) 生成 A0-A5 样本缓存 (缺失时自动补)
#   2) 训练 refiner: A5 (主) -> A0-A4 (增强消融)
#   3) 每个模型训练完立即在共享 test split 上评估
set -u
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
export PYTHONPATH=src
EPOCHS=${EPOCHS:-25}
TAGDIR=v2
SUB=data/yolo_subset/_subset_splits.csv
DATASETS="busi tn3k ddti"
RUN_SUFFIX="busi-ddti-tn3k"     # 03_train_refiner.py 会在 tag 后追加排序后的数据集名

echo "[v2] 开始 $(date)"

gen_cache () {   # $1 = level
  local L=$1 D="data/cache/${TAGDIR}_L${L}_v3"
  if [ -f "$D/samples.parquet" ]; then echo "[v2] 缓存 L$L 已存在"; return; fi
  rm -rf "$D"
  echo "[v2] 生成缓存 L$L $(date)"
  nice -n 5 $PY scripts/03b_cache_samples.py --level "$L" --variants 3 \
      --datasets $DATASETS --out-subdir "${TAGDIR}_L${L}_v3" > "logs/v2_cache_L${L}.log" 2>&1
}

train_eval () {   # $1=level  $2=tag
  local L=$1 TAG=$2
  local D="data/cache/${TAGDIR}_L${L}_v3"
  local RUN="runs/refiner_v2_${TAG}_${RUN_SUFFIX}"
  gen_cache "$L"
  if [ ! -f "$D/samples.parquet" ]; then echo "[v2] 跳过 $TAG (缓存生成失败)"; return; fi
  if [ ! -f "$RUN/best.pt" ]; then
    echo "[v2] 训练 $TAG ($(date))"
    $PY scripts/03_train_refiner.py --level "$L" --tag "v2_${TAG}" --datasets $DATASETS \
        --cache-dir "$D" --batch-size 64 --num-workers 4 --epochs "$EPOCHS" \
        > "logs/v2_refiner_${TAG}.log" 2>&1
  fi
  if [ -f "$RUN/best.pt" ] && [ ! -f "reports/v2_eval_${TAG}/summary_overall.csv" ]; then
    echo "[v2] 评估 $TAG ($(date))"
    $PY scripts/07_evaluate.py --split test --uids-file "$SUB" \
        --refiner "$RUN/best.pt" \
        --det runs/det_yolo26n_sub/weights/best.pt \
        --seg runs/seg_yolo26n_sub/weights/best.pt \
        --imgsz 512 --device cuda --save-vis 0 \
        --out "reports/v2_eval_${TAG}" > "logs/v2_eval_${TAG}.log" 2>&1
    echo "[v2] $TAG 评估完成 ($(date))"
  fi
}

if [ "${1:-all}" = "eval-only" ]; then
  shift
  for TAG in "$@"; do
    case "$TAG" in A0) L=0;; A1) L=1;; A2) L=2;; A3) L=3;; A4) L=4;; A5) L=5;; esac
    train_eval "$L" "$TAG"
  done
  exit 0
fi

for pair in "5 A5" "0 A0" "1 A1" "2 A2" "3 A3" "4 A4"; do
  train_eval $pair
done
echo "[v2] 全部完成 $(date)"
