#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
COORD='data/raw/初始时刻各节点的坐标值.txt'
DISP='data/raw/各时刻各节点的位移结果.txt'
OUT=${OUTPUT_DIR:-outputs/v01}
RANK=${RANK:-10}
if [[ ! -f "$DISP" ]]; then
    echo "[缺少原始文件] 请先上传完整时间历程位移TXT到：$PWD/$DISP" >&2
    echo '注意：初始坐标文件不是位移文件，PPT截图也不能替代真实数值。' >&2
    exit 2
fi
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
"$PYTHON" src/rom_pod_v1.py \
  --coordinates "$COORD" \
  --displacements "$DISP" \
  --time-map data/time_mapping.csv \
  --output "$OUT" --expected-times 101 --expected-nodes 267801 --rank "$RANK" "$@"
