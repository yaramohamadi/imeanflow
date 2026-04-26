#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage: bash scripts/train_plain_jit_h16_taylor_dataset.sh <dataset> <run_label> [extra main_jit.py args...]

Example:
  bash scripts/train_plain_jit_h16_taylor_dataset.sh caltech101 baseline
  bash scripts/train_plain_jit_h16_taylor_dataset.sh food101 lr1p25e6 --config.training.batch_size=64

Datasets:
  caltech101
  artbench10
  cub200
  food101
  stanfordcars
EOF
  exit 1
fi

DATASET_NAME="$1"
shift
RUN_LABEL="$1"
shift

export DATASET_NAME
export RUN_LABEL

bash scripts/train_plain_jit_caltech_h16_taylor.sh "$@"
