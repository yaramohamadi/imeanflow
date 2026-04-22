#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  source .venv/bin/activate
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/$USER-matplotlib}"
export PYTHON="${PYTHON:-.venv/bin/python}"

export DATASET_NAME="${DATASET_NAME:-caltech101}"
export DATASET_ROOT="${DATASET_ROOT:-/home/ens/AT74470/datasets/caltech-101_images}"
export FID_CACHE_REF="${FID_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fid_stats/caltech-101-fid_stats.npz}"
export FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-/home/ens/AT74470/imeanflow/files/fdd_stats/caltech-101-fd_dino-vitb14_stats.npz}"
export LOAD_FROM="${LOAD_FROM:-/home/ens/AT74470/imeanflow/files/weights/pMF-H-16-full}"
export LOAD_ZIP="${LOAD_ZIP:-/home/ens/AT74470/imeanflow/files/weights/pMF-H-16-full.zip}"

export USE_WANDB="${USE_WANDB:-True}"
export WANDB_PROJECT="${WANDB_PROJECT:-plain_pmf_finetune}"
export RUN_LABEL="${RUN_LABEL:-caltech_h16_20step_smoke_taylor}"
export WANDB_NAME="${WANDB_NAME:-caltech101_plain_pmf_${RUN_LABEL}}"

export HALF_PRECISION="${HALF_PRECISION:-False}"
export SAMPLING_HALF_PRECISION="${SAMPLING_HALF_PRECISION:-True}"
export SAMPLE_DEVICE_BATCH_SIZE="${SAMPLE_DEVICE_BATCH_SIZE:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-auto}"
export PREVIEW_AT_STEP0="${PREVIEW_AT_STEP0:-False}"

for path in "$PYTHON" "$DATASET_ROOT" "$FID_CACHE_REF" "$FD_DINO_CACHE_REF"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required local path does not exist: $path" >&2
    exit 2
  fi
done

if [[ "${REQUIRE_GPU:-True}" =~ ^([Tt]rue|1|[Yy]es|[Yy]|[Oo]n)$ ]]; then
  "$PYTHON" - <<'PY'
import sys
import jax

devices = jax.devices()
print("JAX devices:", devices)
if not any(getattr(device, "platform", "") == "gpu" for device in devices):
    sys.exit("ERROR: no JAX GPU device is visible. Run inside a GPU session or set REQUIRE_GPU=False intentionally.")
PY
fi

bash scripts/train_plain_pmf_finetune.sh "$RUN_LABEL" \
  --config.training.max_train_steps=20 \
  --config.training.sample_per_step=10 \
  --config.training.fid_per_step=10 \
  --config.training.force_fid_per_step=10 \
  --config.training.force_metric_num_steps=1 \
  --config.training.preview_num_steps='(1,)' \
  --config.training.preview_guidance_scales='(7,)' \
  --config.fid.num_samples=100 \
  --config.fid.num_images_to_log=16 \
  --config.training.preview_at_step0="$PREVIEW_AT_STEP0" \
  "$@"
