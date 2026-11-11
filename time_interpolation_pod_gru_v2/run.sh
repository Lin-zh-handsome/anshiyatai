#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${PYTHON:-python}"
src=time_interpolation_pod_gru_v2
out=outputs/neural_time_interpolation_v2
export OPENBLAS_NUM_THREADS="${POD_THREADS:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
mkdir -p "$out"

train_rank() {
    local rank=$1
    shift
    local folder="$out/training/rank$rank"
    local resume=()
    if [[ -f "$folder/result.json" && ( "$rank" != 32 || -f "$folder/validation_full_field.json" ) ]]; then
        return
    fi
    if [[ -f "$folder/latest.pt" ]]; then
        resume=(--resume)
    fi
    "$py" -u "$src/fit_gru.py" --rank "$rank" "$@" "${resume[@]}"
}

case "${1:-all}" in
  pod)
    "$py" -u "$src/pod.py"
    ;;
  first)
    if [[ ! -f "$out/pod/fit.json" ]]; then
        "$py" -u "$src/pod.py"
    fi
    train_rank 32 --confirm-full-field
    ;;
  all)
    if [[ ! -f "$out/pod/fit.json" ]]; then
        "$py" -u "$src/pod.py"
    fi
    train_rank 32 --confirm-full-field
    # Keep at most two small independent runs active on the selected GPU.
    train_rank 8 > "$out/rank8.log" 2>&1 &
    first=$!
    train_rank 16 > "$out/rank16.log" 2>&1 &
    second=$!
    first_status=0
    second_status=0
    wait "$first" || first_status=$?
    wait "$second" || second_status=$?
    if [[ "$first_status" != 0 || "$second_status" != 0 ]]; then
        tail -12 "$out/rank8.log" "$out/rank16.log"
        exit 1
    fi
    train_rank 64 > "$out/rank64.log" 2>&1
    "$py" -u "$src/evaluate.py"
    ;;
  evaluate)
    "$py" -u "$src/evaluate.py"
    ;;
  *)
    echo "Usage: bash $src/run.sh pod|first|all|evaluate"
    exit 2
    ;;
esac
