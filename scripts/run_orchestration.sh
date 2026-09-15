#!/usr/bin/env bash
# 串行编排: 检测 -> 分割 -> (等 CPU 缓存生成完) -> refiner 训练与消融
set -u
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
export PYTHONPATH=src
export YOLO_CONFIG_DIR=/home/ben/Dev/Contour_Maker/.cache/ultralytics

EPOCHS=${EPOCHS:-30}
COMMON="--batch-size 64 --num-workers 4 --epochs $EPOCHS"

# 1) 等检测训练结束, 再训练 YOLO26n-seg
while pgrep -f "05_train_yolo.py --kind det" > /dev/null; do sleep 20; done
if [ ! -f runs/seg_yolo26n_sub/weights/best.pt ]; then
  echo "[orch] 启动 YOLO26n-seg 训练 $(date)"
  $PY scripts/05_train_yolo.py --kind seg --yolo-dir data/yolo_subset --name seg_yolo26n_sub \
      --epochs 60 --imgsz 512 --batch 12 --workers 6 --patience 20 >> logs/train_seg_sub.log 2>&1
  echo "[orch] seg 训练结束 $(date)"
fi

# 2) 等缓存生成完成
while pgrep -f "03b_cache_samples.py" > /dev/null; do sleep 20; done
echo "[orch] 缓存就绪, 开始 refiner 主实验 $(date)"

CACHE="data/cache/L5_v3_wr35_p64"
if [ ! -f "$CACHE/samples.parquet" ]; then echo "[orch] 未找到 A5 缓存, 退出"; exit 1; fi

$PY scripts/03_train_refiner.py --level 5 --tag A5 --datasets busi tn3k ddti \
    --cache-dir $CACHE $COMMON >> logs/refiner_A5.log 2>&1
echo "[orch] A5 完成 $(date)"
$PY scripts/03_train_refiner.py --level 5 --tag A5_seed1 --seed 1 --datasets busi tn3k ddti \
    --cache-dir $CACHE $COMMON >> logs/refiner_A5_seed1.log 2>&1
echo "[orch] A5_seed1 完成 $(date)"

# 3) 增强消融: A0-A4 (与 A5 完全相同的网络/数据/轮数, 唯一变量是增强级别)
for L in 0 1 2 3 4; do
  CDIR="data/cache/L${L}_v3_wr35_p64"
  for _ in $(seq 1 120); do
    [ -f "$CDIR/samples.parquet" ] && break
    sleep 20
  done
  if [ -f "$CDIR/samples.parquet" ]; then
    $PY scripts/03_train_refiner.py --level $L --tag A$L --datasets busi tn3k ddti \
        --cache-dir $CDIR $COMMON >> logs/refiner_A$L.log 2>&1
    echo "[orch] A$L 完成 $(date)"
  else
    echo "[orch] 跳过 A$L (缓存缺失)"
  fi
done
echo "[orch] 全部完成 $(date)"
