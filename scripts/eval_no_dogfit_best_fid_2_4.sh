#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/eval_no_dogfit_best_fid_2_4.sh <run_dir_or_best_fid_dir>

Example:
  bash scripts/eval_no_dogfit_best_fid_2_4.sh \
    files/logs/finetuning/caltech_nodogfit_no_dogfit_20260327_212924_tnh8cs
EOF
  exit 1
fi

BEST_FID_ROOT="$1"
CONFIG_MODE="${CONFIG_MODE:-caltech_finetune}"
USE_WANDB="${USE_WANDB:-True}"

if [[ -n "${MODEL_STR:-}" ]]; then
  export MODEL_STR
fi

MODEL_USE_DOGFIT=False \
  TARGET_USE_NULL_CLASS=True \
  CLASS_DROPOUT_PROB=0.0 \
  CONFIG_MODE="$CONFIG_MODE" \
  USE_WANDB="$USE_WANDB" \
  bash scripts/eval_best_fid_steps.sh "$BEST_FID_ROOT" 2 4
