#!/usr/bin/env bash
set -euo pipefail

SWEEP_LABEL="${1:-plain_4step_best_all}"
shift || true

SUBMIT_SLURM="${SUBMIT_SLURM:-True}" \
USE_WANDB="${USE_WANDB:-True}" \
BACKBONES="${BACKBONES:-sit dit}" \
DATASETS="${DATASETS:-caltech101 artbench10 cub200 food101 stanfordcars}" \
SLURM_TIME_SIT="${SLURM_TIME_SIT:-11:00:00}" \
SLURM_TIME_DIT="${SLURM_TIME_DIT:-08:00:00}" \
bash scripts/sweep_plain_finetune_datasets.sh "$SWEEP_LABEL" "$@"
