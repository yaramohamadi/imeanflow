#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/run_caltech_plain_sit_transportv_taylor.sh <run_label> [extra main_sit.py args...]

Example:
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_caltech_plain_sit_transportv_taylor.sh transportv

This script:
  1) runs Caltech plain SiT locally on Taylor
  2) initializes the backbone from the ImageNet DiT checkpoint
  3) trains the Transport-v variant with Diff2Flow state/time alignment
EOF
  exit 1
fi

RUN_LABEL="$1"
shift

export CONFIG_MODE="${CONFIG_MODE:-caltech_plain_sit_transportv}"
export WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-caltech101_plain_sit_transportv_${RUN_LABEL}}"

bash scripts/run_caltech_plain_sit_ditinit_taylor.sh "$RUN_LABEL" "$@"
