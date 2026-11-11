#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_dir=time_interpolation_mlp_v1
cache_dir=data/prepared_neural_time_v1
run_dir=outputs/neural_time_interpolation_v1
python_bin=$PYTHON
mkdir -p "$run_dir"

case "$1" in
  prepare)
    "$python_bin" -u "$source_dir/cache.py" \
      --archive data/raw/updated_cases_2026-11-01/*.part01.rar \
      --coordinates "$cache_dir/source_coordinates.txt" \
      --time-map time_rom_v1/time_mapping.csv \
      --extracted data/extracted_v1 --previous-cache data/prepared_v1 --out "$cache_dir"
    ;;
  experiment)
    if [[ ! -f "$run_dir/predictions/baseline_timing.json" ]]; then
      printf '%s\n' 'STAGE baseline prediction'
      "$python_bin" -u "$source_dir/predict.py" batch --cache "$cache_dir" --out "$run_dir/predictions"
    fi
    resume=()
    if [[ -f "$run_dir/training/latest.pt" ]]; then resume=(--resume); fi
    printf '%s\n' 'STAGE MLP training'
    "$python_bin" -u "$source_dir/train.py" --cache "$cache_dir" --out "$run_dir/training" \
      --epochs 32 --nodes-per-window 4096 "${resume[@]}"
    printf '%s\n' 'STAGE MLP prediction'
    "$python_bin" -u "$source_dir/predict.py" batch --cache "$cache_dir" --out "$run_dir/predictions" \
      --checkpoint "$run_dir/training/best.pt"
    printf '%s\n' 'STAGE scoring'
    "$python_bin" -u "$source_dir/score.py" --cache "$cache_dir" --predictions "$run_dir/predictions" \
      --training "$run_dir/training" --out "$run_dir"
    printf '%s\n' 'STAGE complete'
    ;;
  *)
    printf '%s\n' 'Usage: PYTHON=/path/to/python bash time_interpolation_mlp_v1/run.sh prepare|experiment'
    exit 2
    ;;
esac
