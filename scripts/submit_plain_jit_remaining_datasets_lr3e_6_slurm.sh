#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS="${DATASETS:-artbench10 cub200 food101 stanfordcars}"
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
RUN_LABEL_SUFFIX="${RUN_LABEL_SUFFIX:-h16_lr3e_6}"
WANDB_PROJECT="${WANDB_PROJECT:-plain_jit_finetune}"
USE_WANDB="${USE_WANDB:-True}"
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 50}"

DATASETS="$DATASETS" \
RUN_LABEL_SUFFIX="$RUN_LABEL_SUFFIX" \
WANDB_PROJECT="$WANDB_PROJECT" \
USE_WANDB="$USE_WANDB" \
RUN_FINAL_BEST_FID_EVAL="$RUN_FINAL_BEST_FID_EVAL" \
FINAL_EVAL_STEPS="$FINAL_EVAL_STEPS" \
bash scripts/submit_jit_h16_slurm.sh \
  --config.training.learning_rate="$LEARNING_RATE" \
  "$@"
