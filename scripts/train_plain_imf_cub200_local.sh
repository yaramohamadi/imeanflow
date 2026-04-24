#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"

export DATASET_NAME="${DATASET_NAME:-cub200}"
export DATASET_ROOT="${DATASET_ROOT:-/scratch/ymbahram/datasets/cub-200-2011_processed_latents}"
export FID_CACHE_REF="${FID_CACHE_REF:-/scratch/ymbahram/fid_stats/cub-200-2011_processed-fid_stats.npz}"
export FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/scratch/ymbahram/fdd_stats/cub-200-2011-fd_dino-vitb14_stats.npz}"
export LOAD_FROM="${LOAD_FROM:-/scratch/ymbahram/weights/iMF-XL-2-full}"

export USE_WANDB="${USE_WANDB:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-plain_imf_finetune}"
export RUN_LABEL="${RUN_LABEL:-cub200_local}"
export WANDB_NAME="${WANDB_NAME:-cub200_plain_imf_${RUN_LABEL}}"

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

bash scripts/train_plain_imf_finetune.sh "$RUN_LABEL" "$@"
