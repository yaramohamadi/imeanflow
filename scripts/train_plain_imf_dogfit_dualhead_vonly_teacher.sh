#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: DATASET_NAME=caltech101 VC_TARGET_SOURCE=ema bash scripts/train_plain_imf_dogfit_dualhead_vonly_teacher.sh <run_label> [extra main.py args...]

Examples:
  DATASET_NAME=caltech101 VC_TARGET_SOURCE=ema bash scripts/train_plain_imf_dogfit_dualhead_vonly_teacher.sh dogfit_caltech
  DATASET_NAME=cub200 VC_TARGET_SOURCE=ema bash scripts/train_plain_imf_dogfit_dualhead_vonly_teacher.sh dogfit_cub

This script:
  1) starts from a native iMF checkpoint
  2) keeps the trainable model in dual-head mode
  3) uses DogFit with EMA or online v_c
  4) stores frozen teacher/source copies as shared+v only
EOF
  exit 1
fi

RUN_LABEL="$1"
shift

DATASET_NAME="${DATASET_NAME:-caltech101}" \
DATASET_ROOT="${DATASET_ROOT:-}" \
FID_CACHE_REF="${FID_CACHE_REF:-}" \
FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-}" \
LOAD_FROM="${LOAD_FROM:-}" \
PYTHON="${PYTHON:-python3}" \
USE_WANDB="${USE_WANDB:-True}" \
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}" \
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}" \
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}" \
LOG_DIR="${LOG_DIR:-files/logs}" \
WANDB_PROJECT="${WANDB_PROJECT:-plain_imf_dogfit_finetune}" \
WANDB_NAME="${WANDB_NAME:-}" \
bash scripts/train_plain_imf_dogfit_finetune.sh "$RUN_LABEL" \
  --config.model.use_auxiliary_v_head=True \
  --config.model.use_v_only_teacher_source_copies=True \
  --config.logging.wandb_notes="native iMF DogFit dual-head trainable model with v-only frozen teacher/source copies" \
  "$@"
