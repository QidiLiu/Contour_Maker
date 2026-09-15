#!/usr/bin/env bash
# 用修正后的评估器重跑所有需要补齐的评估 (统一带 --det 与 --seg, 保证五路对照齐全)。
set -u
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
export PYTHONPATH=src
SUB=data/yolo_subset/_subset_splits.csv
DET=runs/det_yolo26n_sub/weights/best.pt
SEG=runs/seg_yolo26n_sub/weights/best.pt

eval_one () {   # $1=run目录  $2=输出名  $3=save-vis
  local RUN=$1 OUTN=$2 VIS=${3:-0}
  [ -f "$RUN/best.pt" ] || { echo "[re] 跳过 $OUTN (无权重)"; return; }
  rm -rf "reports/$OUTN"
  echo "[re] 评估 $OUTN $(date)"
  $PY scripts/07_evaluate.py --split test --uids-file "$SUB" \
      --refiner "$RUN/best.pt" --det "$DET" --seg "$SEG" \
      --imgsz 512 --device cuda --save-vis "$VIS" \
      --out "reports/$OUTN" > "logs/${OUTN}.log" 2>&1
}

O=${OUTDIR:-reports}
_=$O
eval_one runs/refiner_v3_A5_busi-ddti-tn3k v3_eval_A5 12
eval_one runs/refiner_v3_A0_busi-ddti-tn3k v3_eval_A0 0
for TAG in A0 A1 A2 A3 A4 A5; do
  eval_one "runs/refiner_v2_${TAG}_busi-ddti-tn3k" "v2_eval_${TAG}" 0
done
echo "[re] 全部完成 $(date)"
