#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/smoke_plain_imf_dogfit_dualhead_vonly_teacher_local.sh [run_label]

Runs a short local smoke test for:
  - native iMF start
  - dual-head trainable model
  - DogFit + EMA v_c
  - v-only frozen teacher/source copies

Useful env overrides:
  PYTHON=.venv/bin/python
  DATASET_NAME=caltech101
  DATASET_ROOT=/home/ens/AT74470/datasets/caltech-101_processed_latents
  FID_CACHE_REF=/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz
  FD_DINO_CACHE_REF=
  LOAD_FROM=/home/ens/AT74470/imeanflow/files/weights/iMF-XL-2-full
  LOG_DIR=files/logs/smoke
EOF
  exit 1
fi

RUN_LABEL="${1:-smoke_dualhead_vonly}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export USE_WANDB="${USE_WANDB:-False}"
export PYTHON="${PYTHON:-.venv/bin/python}"
export DATASET_NAME="${DATASET_NAME:-caltech101}"
export DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_processed_latents}"
export FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
export FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-}"
export LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/iMF-XL-2-full}"
export RUN_FINAL_BEST_FID_EVAL="${RUN_FINAL_BEST_FID_EVAL:-False}"
export LOG_DIR="${LOG_DIR:-files/logs/smoke}"
export WANDB_PROJECT="${WANDB_PROJECT:-plain_imf_dogfit_smoke}"
export WANDB_NAME="${WANDB_NAME:-plain_imf_dogfit_dualhead_vonly_${RUN_LABEL}}"

bash scripts/train_plain_imf_dogfit_dualhead_vonly_teacher.sh "$RUN_LABEL" \
  --config.training.max_train_steps=2 \
  --config.training.num_epochs=1 \
  --config.training.log_per_step=1 \
  --config.training.sample_per_step=0 \
  --config.training.fid_per_step=0 \
  --config.training.preview_at_step_zero=False \
  --config.training.debug_log_during_train=False \
  --config.training.grad_accum_steps=1 \
  --config.training.batch_size=1 \
  --config.fid.num_images_to_log=0 \
  --config.fid.sample_device_batch_size=1 \
  --config.fid.device_batch_size=1 \
  --config.training.capture_source_from_load=True \
  --config.training.use_ema=True \
  --config.training.fid_use_online_only=True
