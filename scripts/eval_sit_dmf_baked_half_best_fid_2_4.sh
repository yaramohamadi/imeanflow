#!/usr/bin/env bash
set -euo pipefail

BEST_FID_DIR="${1:-files/logs/finetuning/caltech_SiT_DMF_fixed_guidance_no_context_unguided_sampling_baked_half/best_fid}"
CONFIG_MODE="${CONFIG_MODE:-caltech_sit_dmf_finetune}"
PYTHON="${PYTHON:-.venv/bin/python}"
USE_WANDB="${USE_WANDB:-False}"

CONFIG_MODE="$CONFIG_MODE" \
PYTHON="$PYTHON" \
USE_WANDB="$USE_WANDB" \
MODEL_STR="imfSiT_DMF_XL_2" \
MODEL_USE_DOGFIT="False" \
TARGET_USE_NULL_CLASS="True" \
bash scripts/eval_best_fid_steps.sh "$BEST_FID_DIR" 2 4
