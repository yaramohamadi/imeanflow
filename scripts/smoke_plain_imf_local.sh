#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/smoke_plain_imf_local.sh [run_label]

Runs a local iMF smoke test:
  1) trains for 10 optimizer steps
  2) at step 5, samples 100 images and computes 4-step FID
  3) saves best_fid/ from that mid-train FID
  4) after training, evaluates best_fid/ with 1 sampling step

Useful env overrides:
  PYTHON=.venv/bin/python
  DATASET_NAME=caltech101
  DATASET_ROOT=/path/to/latents
  FID_CACHE_REF=/path/to/fid_stats.npz
  FD_DINO_CACHE_REF=               # empty disables FD-DINO
  LOAD_FROM=/path/to/iMF-XL-2-full
  LOG_DIR=files/logs/smoke
EOF
  exit 1
fi

RUN_LABEL="${1:-smoke10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

USE_WANDB="${USE_WANDB:-False}" \
RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-True}" \
FINAL_EVAL_STEPS="${FINAL_EVAL_STEPS:-1}" \
FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-False}" \
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-100}" \
FORCE_FID_STEPS="${FORCE_FID_STEPS:-5}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" \
SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-16}" \
SAMPLE_LOG_EVERY="${SAMPLE_LOG_EVERY:-1}" \
SAMPLE_FIRST_DEVICE_ONLY="${SAMPLE_FIRST_DEVICE_ONLY:-False}" \
LOG_DIR="${LOG_DIR:-files/logs/smoke}" \
WANDB_PROJECT="${WANDB_PROJECT:-plain_imf_smoke}" \
WANDB_NAME="${WANDB_NAME:-plain_imf_${RUN_LABEL}}" \
bash scripts/train_plain_imf_finetune.sh "$RUN_LABEL" \
  --config.training.max_train_steps=10 \
  --config.training.num_epochs=1 \
  --config.training.log_per_step=1 \
  --config.training.sample_per_step=0 \
  --config.training.save_best_fid_only=True \
  --config.training.save_eval_checkpoint_per_fid=False \
  --config.sampling.num_steps=4 \
  --config.fid.num_images_to_log=0
