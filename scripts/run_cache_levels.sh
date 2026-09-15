#!/usr/bin/env bash
# 串行生成 A0-A5 全部增强级别的样本缓存 (3 变体), 带锁避免并发重复。
set -u
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
export PYTHONPATH=src
exec 9>/tmp/contour_maker_cache.lock
if ! flock -n 9; then echo "[cache-all] 另一个缓存生成任务在运行, 退出"; exit 1; fi

mkdir -p data/cache logs
for L in 0 1 2 3 4 5; do
  D="data/cache/L${L}_v3_wr35_p64"
  if [ -f "$D/samples.parquet" ]; then echo "[cache-all] L$L 已存在, 跳过"; continue; fi
  rm -rf "$D"
  echo "[cache-all] 生成 L$L $(date)"
  nice -n 5 $PY scripts/03b_cache_samples.py --level $L --variants 3 \
      --datasets busi tn3k ddti > "logs/cache_A${L}.log" 2>&1
  if [ -f "$D/samples.parquet" ]; then
    echo "[cache-all] L$L 完成 $(date)"
  else
    echo "[cache-all] L$L 失败, 见 logs/cache_A${L}.log"
  fi
done
echo "[cache-all] 全部完成 $(date)"
