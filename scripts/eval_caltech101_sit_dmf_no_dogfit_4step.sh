#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-files/logs/finetuning/caltech101_SiT_DMF_plain_meanflow_sit_plain_20260506_002755_e6qju8}"

CONFIG_MODE="${CONFIG_MODE:-caltech_sit_dmf_finetune}"
PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-False}"
EVAL_PLATFORM="${EVAL_PLATFORM:-gpu}"

exec env \
  CONFIG_MODE="$CONFIG_MODE" \
  PYTHON="$PYTHON" \
  USE_WANDB="$USE_WANDB" \
  EVAL_PLATFORM="$EVAL_PLATFORM" \
  bash scripts/eval_best_fid_steps.sh "$RUN_DIR" 4
