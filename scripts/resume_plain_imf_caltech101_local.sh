#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"

export BACKBONE=imf
export DATASET_NAME=caltech101
export DATASET_ROOT=/scratch/ymbahram/datasets/caltech-101_processed_latents
export FID_CACHE_REF=/scratch/ymbahram/fid_stats/caltech-101-fid_stats.npz
export FD_DINO_CACHE_REF=/scratch/ymbahram/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz
export LOAD_FROM=files/logs/sweeps/plain_imf_plain_4step_best_imf_20260420_173923/finetuning/plain_iMF_finetune_caltech101_plain_4step_best_imf_caltech101_imf_20260420_180555_vf6kzl/latest_eval
export LOG_DIR=files/logs/sweeps/plain_imf_plain_4step_best_imf_20260420_173923

export USE_WANDB="${USE_WANDB:-True}"
export FINAL_EVAL_USE_WANDB="${FINAL_EVAL_USE_WANDB:-${USE_WANDB}}"
export WANDB_PROJECT=plain_imf_finetune
export WANDB_NAME="${WANDB_NAME:-caltech101_plain_imf_plain_4step_best_imf_resume_15000_to_30000_local}"

if [[ "${REQUIRE_GPU:-True}" =~ ^([Tt]rue|1|[Yy]es|[Yy]|[Oo]n)$ ]]; then
  "$PYTHON" - <<'PY'
import sys
import jax

devices = jax.devices()
print("JAX devices:", devices)
if not any(getattr(device, "platform", "") == "gpu" for device in devices):
    sys.exit("ERROR: no JAX GPU device is visible. Run inside a GPU allocation or set REQUIRE_GPU=False intentionally.")
PY
fi

bash scripts/train_plain_imf_finetune.sh plain_4step_best_imf_caltech101_imf_resume_15000_to_30000_local \
  --config.partial_load=False \
  --config.training.max_train_steps=30000 \
  "$@"
