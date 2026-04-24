#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-files/logs/finetuning/plain_JiT_finetune_caltech101_caltech_h16_20260424_010107_jam3fd}"
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1 2 50}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-caltech101_plain_jit_h16_best_fid}"
DRY_RUN="${DRY_RUN:-False}"

if [[ ! -d "$RUN_DIR/best_fid" ]]; then
  echo "ERROR: best_fid directory not found under run dir: $RUN_DIR" >&2
  exit 2
fi

cmd=(
  sbatch
  --job-name=plain_jit_caltech_best_eval
  --export=ALL,RUN_DIR="$RUN_DIR",FINAL_EVAL_STEPS="$FINAL_EVAL_STEPS",USE_WANDB="$USE_WANDB",WANDB_NAME_PREFIX="$WANDB_NAME_PREFIX"
  scripts/eval_plain_jit_best_fid_slurm.sbatch
)

if [[ "${DRY_RUN,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
