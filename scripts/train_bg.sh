#!/usr/bin/env bash
# Run long trainings inside tmux so SSH/session disconnects do not kill them.
#
# Usage:
#   ./scripts/train_bg.sh yolo_seg    # YOLO-Seg (YOLO26n-seg)
#   ./scripts/train_bg.sh det         # Det-UNet stage 1 (YOLO26n-det)
#   ./scripts/train_bg.sh mkunet      # Det-UNet stage 2 (MK-UNet)
#   ./scripts/train_bg.sh rfdetr      # RF-DETR-Seg-Nano
#   ./scripts/train_bg.sh status
#   ./scripts/train_bg.sh attach <task>
#   ./scripts/train_bg.sh stop <task>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
SESSION="cm_train"
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics"
mkdir -p "$ROOT/logs"

cmd_for() {
  case "$1" in
    yolo_seg) cat <<EOF
cd $ROOT/yolo-seg-train && exec $PY train_yolo_seg.py \\
  --data $ROOT/data/yolo_seg/data_seg.yaml --model $ROOT/weights/yolo26n-seg.pt \\
  --epochs \${SEG_EPOCHS:-60} --imgsz 512 --batch 12 --name seg_yolo26n
EOF
    ;;
    det) cat <<EOF
cd $ROOT/det-unet-train && exec $PY train_yolo_det.py \\
  --data $ROOT/data/yolo_det/data_det.yaml --model $ROOT/weights/yolo26n.pt \\
  --epochs \${DET_EPOCHS:-60} --imgsz 512 --batch 16 --name det_yolo26n
EOF
    ;;
    mkunet) cat <<EOF
cd $ROOT/det-unet-train && exec $PY train_mkunet.py \\
  --data $ROOT/data/roi_cache/train --val-data $ROOT/data/roi_cache/val \\
  --variant MK_UNet --roi-size 256 --epochs \${MKUNET_EPOCHS:-60} --batch 16 \\
  --name mkunet_yolo26n
EOF
    ;;
    rfdetr) cat <<EOF
cd $ROOT/rfdetr-seg-train && exec $PY train_rfdetr_seg.py \\
  --dataset ultrasound --epochs \${RFDETR_EPOCHS:-8} --batch \${RFDETR_BATCH:-8} \\
  --resolution \${RFDETR_RESOLUTION:-384} --name rfdetr_seg_ultrasound
EOF
    ;;
    *) echo "unknown task: $1" >&2; exit 2 ;;
  esac
}

case "${1:-status}" in
  yolo_seg|det|mkunet|rfdetr)
    task="$1"
    tmux has-session -t "$SESSION.$task" 2>/dev/null && {
      echo "task $task already running (tmux: $SESSION.$task)"; exit 0; }
    cmd_for "$task" > "$ROOT/logs/$task.sh"
    tmux new-session -d -s "$SESSION.$task" \
      "bash $ROOT/logs/$task.sh 2>&1 | tee -a $ROOT/logs/train_$task.log"
    echo "started $task -> tmux '$SESSION.$task'  log logs/train_$task.log"
    ;;
  status)
    tmux ls 2>/dev/null || echo "no running training"
    echo "--- results.csv ---"
    for f in "$ROOT"/runs/*/results.csv; do
      [ -f "$f" ] || continue
      n=$("$PY" -c "import csv,sys;print(len({int(float(r['epoch'])) for r in csv.DictReader(open('$f'))}))")
      echo "  $(basename "$(dirname "$f")"): $n epochs"
    done
    ;;
  attach) tmux attach -t "$SESSION.${2:-yolo_seg}" ;;
  stop)   tmux kill-session -t "$SESSION.${2:-yolo_seg}" 2>/dev/null && echo "stopped ${2:-yolo_seg}" ;;
  *) echo "usage: $0 {yolo_seg|det|mkunet|rfdetr|status|attach <task>|stop <task>}" >&2; exit 2 ;;
esac
