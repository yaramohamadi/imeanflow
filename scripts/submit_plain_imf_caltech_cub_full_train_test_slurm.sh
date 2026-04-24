#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ge 1 && "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/submit_plain_imf_caltech_cub_full_train_test_slurm.sh [sweep_label] [extra config overrides...]

Submits the current plain iMF full finetune + final best-checkpoint eval sweep for:
  - caltech101
  - cub200

Defaults:
  SUBMIT_SLURM=True
  USE_WANDB=True
  RUN_FINAL_BEST_FID_EVAL=True
  FINAL_EVAL_STEPS="1 2 250"
  FINAL_EVAL_USE_WANDB=True
  DATASETS="caltech101 cub200"
  BACKBONES="imf"
  SLURM_MEM=64G
  SLURM_TIME_IMF=24:00:00

Examples:
  bash scripts/submit_plain_imf_caltech_cub_full_train_test_slurm.sh

  SUBMIT_SLURM=DryRun \
  bash scripts/submit_plain_imf_caltech_cub_full_train_test_slurm.sh my_sweep \
    --config.training.max_train_steps=30000
EOF
  exit 0
fi

SWEEP_LABEL="${1:-plain_imf_caltech_cub_full_train_test}"
if [[ $# -gt 0 ]]; then
  shift
fi

SUBMIT_SLURM="${SUBMIT_SLURM:-True}" \
USE_WANDB="${USE_WANDB:-True}" \
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}" \
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 250}" \
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-True}" \
BACKBONES="${BACKBONES:-imf}" \
DATASETS="${DATASETS:-caltech101 cub200}" \
SLURM_MEM="${SLURM_MEM:-64G}" \
SLURM_TIME_IMF="${SLURM_TIME_IMF:-24:00:00}" \
WANDB_PROJECT_IMF="${WANDB_PROJECT_IMF:-plain_imf_finetune}" \
bash scripts/sweep_plain_finetune_datasets.sh "$SWEEP_LABEL" \
  --config.logging.wandb_max_retries="${WANDB_MAX_RETRIES:-5}" \
  --config.logging.wandb_retry_cooldown_seconds="${WANDB_RETRY_COOLDOWN_SECONDS:-600}" \
  --config.logging.wandb_eval_replay_buffer_size="${WANDB_EVAL_REPLAY_BUFFER_SIZE:-100}" \
  "$@"
