#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
TMP_DIR='outputs/synthetic_fixture'
"$PYTHON" scripts/make_toy_data.py --output "$TMP_DIR/raw"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
"$PYTHON" src/rom_pod_v1.py \
  --coordinates "$TMP_DIR/raw/坐标_合成测试.txt" \
  --displacements "$TMP_DIR/raw/位移_合成测试.txt" \
  --time-map "$TMP_DIR/raw/time_mapping.csv" \
  --output "outputs/synthetic" --synthetic --rank 2 --node-ids '2,17' \
  --feature-chunk 4
"$PYTHON" - "outputs/synthetic/summary.json" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f:
    s=json.load(f)
print('[SYNTHETIC] overall relative L2 reconstruction error =',s['overall_relative_L2_reconstruction_error'])
assert s['n_nodes']==5 and s['n_time_steps']==6
assert s['overall_relative_L2_reconstruction_error'] < 1e-5
assert s['effective_rank']==2
print('[SYNTHETIC] PASS: 对低秩合成数据的重建误差符合预期')
PY
"$PYTHON" -m pytest -q tests
