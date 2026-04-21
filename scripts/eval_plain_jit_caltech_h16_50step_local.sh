#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"

export DATASET_NAME=caltech101
export DATASET_ROOT=/home/ymbahram/scratch/datasets/caltech-101_images
export FID_CACHE_REF=/home/ymbahram/scratch/fid_stats/caltech-101-fid_stats.npz
export FD_DINO_CACHE_REF=/home/ymbahram/scratch/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz

export USE_WANDB="${USE_WANDB:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-plain_jit_finetune}"
export WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-caltech101_plain_jit_50step_eval}"

export HALF_PRECISION="${HALF_PRECISION:-True}"
export HALF_PRECISION_DTYPE="${HALF_PRECISION_DTYPE:-float16}"
export SAMPLING_HALF_PRECISION="${SAMPLING_HALF_PRECISION:-False}"
export SAMPLING_HALF_PRECISION_DTYPE="${SAMPLING_HALF_PRECISION_DTYPE:-float16}"
export OPTIMIZER="${OPTIMIZER:-adamw}"
export OPTIMIZER_MU_DTYPE="${OPTIMIZER_MU_DTYPE:-float16}"

BEST_FID_ROOT="${BEST_FID_ROOT:-files/logs/finetuning/plain_JiT_finetune_caltech101_caltech_h16_20step_smoke_local_20260421_131110_kg8d90/best_fid}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_IMAGES_TO_LOG="${NUM_IMAGES_TO_LOG:-16}"

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

CONFIG_MODE=plain_jit_finetune \
  PYTHON="$PYTHON" \
  USE_WANDB="$USE_WANDB" \
  WANDB_NAME_PREFIX="$WANDB_NAME_PREFIX" \
  bash scripts/eval_best_fid_steps_plain_jit.sh "$BEST_FID_ROOT" 50 -- \
    --config.model.model_str=flaxJiT_H_16 \
    --config.dataset.root="$DATASET_ROOT" \
    --config.fid.cache_ref="$FID_CACHE_REF" \
    --config.fd_dino.cache_ref="$FD_DINO_CACHE_REF" \
    --config.fid.num_samples="$NUM_SAMPLES" \
    --config.fid.num_images_to_log="$NUM_IMAGES_TO_LOG" \
    --config.training.half_precision="$HALF_PRECISION" \
    --config.training.half_precision_dtype="$HALF_PRECISION_DTYPE" \
    --config.sampling.half_precision="$SAMPLING_HALF_PRECISION" \
    --config.sampling.half_precision_dtype="$SAMPLING_HALF_PRECISION_DTYPE" \
    --config.training.optimizer="$OPTIMIZER" \
    --config.training.optimizer_mu_dtype="$OPTIMIZER_MU_DTYPE" \
    "$@"
