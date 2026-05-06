#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage: bash scripts/train_sit_dmf_no_dogfit_dataset.sh <dataset_name> <run_label> [extra main.py args...]

Examples:
  bash scripts/train_sit_dmf_no_dogfit_dataset.sh caltech101 sit_plain
  USE_WANDB=True bash scripts/train_sit_dmf_no_dogfit_dataset.sh cub200 sit_plain

This wrapper:
  1) runs the SiT-DMF no-DogFit, no-EMA setup across datasets
  2) forces in-training FID selection to use 4 sampling steps
  3) keeps the final best-checkpoint eval enabled
  4) logs the final eval runs to W&B by default
  5) evaluates the final best checkpoint at 1, 2, 4, and 250 steps by default
EOF
  exit 1
fi

DATASET_NAME="$1"
shift
RUN_LABEL="$1"
shift
EXTRA_ARGS=("$@")

BACKBONE="${BACKBONE:-sit}"
USE_WANDB="${USE_WANDB:-True}"
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-True}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 4 250}"
WANDB_PROJECT="${WANDB_PROJECT:-dmf_no_dogfit}"

exec env \
  BACKBONE="$BACKBONE" \
  DATASET_NAME="$DATASET_NAME" \
  USE_WANDB="$USE_WANDB" \
  FINAL_EVAL_USE_WANDB="$FINAL_EVAL_USE_WANDB" \
  RUN_FINAL_BEST_FID_EVAL="$RUN_FINAL_BEST_FID_EVAL" \
  FINAL_EVAL_STEPS="$FINAL_EVAL_STEPS" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  bash scripts/run_dmf_no_dogfit_taylor.sh "$RUN_LABEL" \
    --config.training.force_metric_num_steps=4 \
    --config.sampling.num_steps=4 \
    "${EXTRA_ARGS[@]}"
