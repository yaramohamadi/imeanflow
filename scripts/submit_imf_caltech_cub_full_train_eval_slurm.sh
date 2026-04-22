#!/usr/bin/env bash
set -euo pipefail

SWEEP_LABEL="${1:-plain_4step_best_imf_caltech_cub_full_eval}"
shift || true

SUBMIT_SLURM="${SUBMIT_SLURM:-True}" \
USE_WANDB="${USE_WANDB:-True}" \
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}" \
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}" \
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-True}" \
BACKBONES=imf \
DATASETS="${DATASETS:-caltech101 cub200}" \
SLURM_MEM="${SLURM_MEM:-64G}" \
SLURM_TIME_IMF="${SLURM_TIME_IMF:-36:00:00}" \
bash scripts/sweep_plain_finetune_datasets.sh "$SWEEP_LABEL" \
  --config.logging.wandb_max_retries="${WANDB_MAX_RETRIES:-5}" \
  --config.logging.wandb_retry_cooldown_seconds="${WANDB_RETRY_COOLDOWN_SECONDS:-600}" \
  --config.logging.wandb_eval_replay_buffer_size="${WANDB_EVAL_REPLAY_BUFFER_SIZE:-100}" \
  "$@"
