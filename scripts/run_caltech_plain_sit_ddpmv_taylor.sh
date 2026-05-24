#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/run_caltech_plain_sit_ddpmv_taylor.sh <run_label> [extra main_sit.py args...]

Example:
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_caltech_plain_sit_ddpmv_taylor.sh ddpmv

This script:
  1) runs Caltech plain SiT locally on Taylor
  2) initializes the backbone from the ImageNet DiT checkpoint
  3) trains the DDPM-v variant with the native DDPM velocity wrapper and no Diff2Flow alignment
EOF
  exit 1
fi

RUN_LABEL="$1"
shift

export CONFIG_MODE="${CONFIG_MODE:-caltech_plain_sit_ddpmv}"
export WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-caltech101_plain_sit_ddpmv_${RUN_LABEL}}"

bash scripts/run_caltech_plain_sit_ditinit_taylor.sh "$RUN_LABEL" "$@"
