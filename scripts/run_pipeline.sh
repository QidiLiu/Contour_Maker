#!/usr/bin/env bash
# 串行流水线: 等 YOLO 训练结束后依次训练 contour-refiner (主实验 + A1-A5 消融)。
set -u
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
export PYTHONPATH=src

echo "[pipeline] 等待现有 YOLO 训练结束..."
while pgrep -f "scripts/05_train_yolo.py" > /dev/null; do sleep 20; done
echo "[pipeline] YOLO 训练已结束, 开始 refiner 主实验 $(date)"

COMMON="--batch-size 32 --samples-per-target 4 --num-workers 4"

# 主实验: A5 (完整增强), 三个数据集
$PY scripts/03_train_refiner.py --level 5 --tag A5 --epochs 80 \
    --datasets busi tn3k ddti $COMMON >> logs/refiner_A5.log 2>&1
echo "[pipeline] A5 完成 $(date)"

# 增强消融 A0-A4 (与 A5 相同数据/网络/轮数)
for L in 0 1 2 3 4; do
  $PY scripts/03_train_refiner.py --level $L --tag A$L --epochs 80 \
      --datasets busi tn3k ddti $COMMON >> logs/refiner_A$L.log 2>&1
  echo "[pipeline] A$L 完成 $(date)"
done

# 第二个 seed, 检验稳健性
$PY scripts/03_train_refiner.py --level 5 --tag A5_seed1 --epochs 80 --seed 1 \
    --datasets busi tn3k ddti $COMMON >> logs/refiner_A5_seed1.log 2>&1
echo "[pipeline] 全部完成 $(date)"
