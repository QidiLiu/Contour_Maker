#!/usr/bin/env bash
# v3: 在 v2 基础上加入"检测框回退 + 框初始化"训练, 消除 Otsu 失败导致的整图零预测。
set -u
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
export PYTHONPATH=src
EPOCHS=${EPOCHS:-25}
SUB=data/yolo_subset/_subset_splits.csv
cdir() { echo "data/cache/v3_L${1}_v3"; }

# 等 v2 流水线结束, 避免 GPU 争用
while ps -eo cmd | grep -q "[r]un_experiments_v2.sh"; do sleep 30; done
echo "[v3] 开始 $(date)"

for L in 0 5; do
  D=$(cdir $L)
  if [ -f "$D/samples.parquet" ]; then echo "[v3] 缓存 L$L 已存在"; continue; fi
  rm -rf "$D"
  nice -n 5 $PY scripts/03b_cache_samples.py --level $L --variants 3 \
      --datasets busi tn3k ddti --out-subdir "v3_L${L}_v3" > "logs/v3_cache_L${L}.log" 2>&1
done
echo "[v3] 缓存完成 $(date)"

for pair in "5 A5" "0 A0"; do
  set -- $pair; L=$1; TAG=$2
  D=$(cdir $L); RUN="runs/refiner_v3_${TAG}_busi-ddti-tn3k"
  [ -f "$D/samples.parquet" ] || { echo "[v3] 跳过 $TAG (无缓存)"; continue; }
  if [ ! -f "$RUN/best.pt" ]; then
    echo "[v3] 训练 $TAG ($(date))"
    $PY scripts/03_train_refiner.py --level "$L" --tag "v3_${TAG}" --datasets busi tn3k ddti \
        --cache-dir "$D" --batch-size 64 --num-workers 4 --epochs "$EPOCHS" \
        > "logs/v3_refiner_${TAG}.log" 2>&1
  fi
  if [ -f "$RUN/best.pt" ] && [ ! -f "reports/v3_eval_${TAG}/summary_overall.csv" ]; then
    echo "[v3] 评估 $TAG ($(date))"
    $PY scripts/07_evaluate.py --split test --uids-file "$SUB" \
        --refiner "$RUN/best.pt" --seg runs/seg_yolo26n_sub/weights/best.pt \
        --imgsz 512 --device cuda --save-vis 12 \
        --out "reports/v3_eval_${TAG}" > "logs/v3_eval_${TAG}.log" 2>&1
    echo "[v3] $TAG 完成 ($(date))"
  fi
done
echo "[v3] 全部完成 $(date)"
