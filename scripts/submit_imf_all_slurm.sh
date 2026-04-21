#!/usr/bin/env bash
set -euo pipefail

SWEEP_LABEL="${1:-plain_4step_best_imf_all}"
shift || true

SUBMIT_SLURM="${SUBMIT_SLURM:-True}" \
USE_WANDB="${USE_WANDB:-True}" \
BACKBONES=imf \
DATASETS="${DATASETS:-caltech101 artbench10 cub200 food101 stanfordcars}" \
SLURM_MEM="${SLURM_MEM:-64G}" \
SLURM_TIME_IMF="${SLURM_TIME_IMF:-20:00:00}" \
bash scripts/sweep_plain_finetune_datasets.sh "$SWEEP_LABEL" "$@"
