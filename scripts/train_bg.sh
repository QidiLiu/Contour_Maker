#!/usr/bin/env bash
# 用 tmux 跑长训练，避免 SSH/会话断开导致进程被杀。
#
# 用法:
#   ./scripts/train_bg.sh seg      # 单阶段 YOLO26n-seg
#   ./scripts/train_bg.sh det      # 两阶段 YOLO26n-det
#   ./scripts/train_bg.sh mkunet   # MK-UNet
#   ./scripts/train_bg.sh status   # 查看状态
#   ./scripts/train_bg.sh attach seg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
SESSION="cm_train"
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics"
mkdir -p "$ROOT/logs"

cmd_for() {
  case "$1" in
    seg) cat <<EOF
cd $ROOT/one-stage-train && exec $PY -c "
from ultralytics import YOLO
YOLO('$ROOT/runs/seg_yolo26n/weights/last.pt').train(resume=True)
"
EOF
    ;;
    det) cat <<EOF
cd $ROOT/two-stage-train && exec $PY train_yolo_det.py \\
  --data $ROOT/data/yolo_det/data_det.yaml --model $ROOT/weights/yolo26n.pt \\
  --epochs \${DET_EPOCHS:-60} --imgsz 512 --batch 16 --name det_yolo26n
EOF
    ;;
    mkunet) cat <<EOF
cd $ROOT/two-stage-train && exec $PY train_mkunet.py \\
  --data $ROOT/data/roi_cache/train --val-data $ROOT/data/roi_cache/val \\
  --variant MK_UNet --roi-size 256 --epochs \${MKUNET_EPOCHS:-60} --batch 16 \\
  --name mkunet_yolo26n
EOF
    ;;
    *) echo "未知任务: $1" >&2; exit 2 ;;
  esac
}

case "${1:-status}" in
  seg|det|mkunet)
    task="$1"
    tmux has-session -t "$SESSION.$task" 2>/dev/null && {
      echo "任务 $task 已在运行 (tmux: $SESSION.$task)"; exit 0; }
    cmd_for "$task" > "$ROOT/logs/$task.sh"
    tmux new-session -d -s "$SESSION.$task" "bash $ROOT/logs/$task.sh 2>&1 | tee -a $ROOT/logs/train_$task.log"
    echo "已启动 $task -> tmux session '$SESSION.$task'  日志 logs/train_$task.log"
    ;;
  status)
    tmux ls 2>/dev/null || echo "没有运行中的训练"
    echo "--- results.csv ---"
    for f in "$ROOT"/runs/*/results.csv; do
      [ -f "$f" ] || continue
      n=$("$PY" -c "import csv,sys;print(len({int(float(r['epoch'])) for r in csv.DictReader(open('$f'))}))")
      echo "  $(basename $(dirname $f)): $n epochs"
    done
    ;;
  attach)
    tmux attach -t "$SESSION.${2:-seg}"
    ;;
  stop)
    tmux kill-session -t "$SESSION.${2:-seg}" 2>/dev/null && echo "已停止 ${2:-seg}"
    ;;
  *) echo "用法: $0 {seg|det|mkunet|status|attach <task>|stop <task>}" >&2; exit 2 ;;
esac